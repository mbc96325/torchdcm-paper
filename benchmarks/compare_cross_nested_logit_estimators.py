from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from torchdcm import CrossNest, CrossNestedLogit
from compare_mnl_estimators import load_biogeme_swissmetro, make_initial_values, spec_with_initials


@dataclass
class BackendResult:
    backend: str
    available: bool
    seconds: float | None = None
    estimate_seconds: float | None = None
    covariance_seconds: float | None = None
    loglike: float | None = None
    params: dict[str, float] | None = None
    covariance: np.ndarray | None = None
    probabilities: np.ndarray | None = None
    message: str = ""


def swissmetro_cross_nests(lambda_public: float = 0.8, lambda_private: float = 0.9) -> dict[str, CrossNest]:
    return {
        "PUBLIC": CrossNest({"TRAIN": 0.7, "SM": 0.8, "CAR": 0.0}, init=lambda_public),
        "PRIVATE": CrossNest({"TRAIN": 0.3, "SM": 0.2, "CAR": 1.0}, init=lambda_private),
    }


def default_params(names: list[str]) -> dict[str, float]:
    values = {
        "ASC_TRAIN": 0.3,
        "B_TIME": -1.0,
        "B_COST": -1.2,
        "ASC_CAR": 0.6,
        "LAMBDA_PUBLIC": 0.75,
        "LAMBDA_PRIVATE": 0.9,
    }
    return {name: values[name] for name in names}


def run_torch_fixed(data, spec, nests, params: dict[str, float]) -> BackendResult:
    model = CrossNestedLogit(spec, nests)
    compiled = model.compile(data)
    vector = torch.as_tensor([params[name] for name in compiled.free_names], dtype=torch.float64)
    start = time.perf_counter()
    ll = model.loglike(vector, data, compiled)
    probabilities = model.predict_proba(data, vector, compiled)
    seconds = time.perf_counter() - start
    return BackendResult(
        backend="torchdcm_fixed",
        available=True,
        seconds=seconds,
        loglike=float(ll.detach().cpu()),
        params={name: params[name] for name in compiled.free_names},
        probabilities=probabilities.detach().cpu().numpy(),
    )


def run_torch_fit(data, spec, nests, max_iter: int) -> BackendResult:
    model = CrossNestedLogit(spec, nests, max_iter=max_iter)
    start = time.perf_counter()
    result = model.fit(data, max_iter=max_iter)
    seconds = time.perf_counter() - start
    return BackendResult(
        backend="torchdcm_fit",
        available=True,
        seconds=seconds,
        estimate_seconds=None,
        covariance_seconds=None,
        loglike=result.loglike,
        params=dict(zip(result.param_names, result.values)),
        covariance=result.cov_params().detach().cpu().numpy(),
        probabilities=result.predict_proba(data),
    )


def write_replay_inputs(df, alternatives, params: dict[str, float], directory: Path):
    code_by_alt = {alt: i + 1 for i, alt in enumerate(alternatives)}
    wide_df = df.copy().reset_index(drop=True)
    wide_df["choice_code"] = wide_df["choice"].map(code_by_alt).astype(int)
    for column in wide_df.select_dtypes(include=["bool"]).columns:
        wide_df[column] = wide_df[column].astype(int)
    csv_path = directory / "data.csv"
    spec_path = directory / "spec.json"
    wide_df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    spec = {
        "model_name": f"cross_nested_fixed_{len(df)}",
        "alternatives": alternatives,
        "choice_col": "choice_code",
        "parameters": params,
        "utility": {
            alt: {
                "code": code_by_alt[alt],
                "asc": f"ASC_{alt.upper()}" if f"ASC_{alt.upper()}" in params else None,
                "time": f"time_{alt.lower()}",
                "cost": f"cost_{alt.lower()}",
                "availability": f"avail_{alt.lower()}",
            }
            for alt in alternatives
        },
        "nests": {
            "PUBLIC": {
                "lambda_param": "LAMBDA_PUBLIC",
                "allocations": {"TRAIN": 0.7, "SM": 0.8, "CAR": 0.0},
            },
            "PRIVATE": {
                "lambda_param": "LAMBDA_PRIVATE",
                "allocations": {"TRAIN": 0.3, "SM": 0.2, "CAR": 1.0},
            },
        },
    }
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return csv_path, spec_path


def run_apollo_fixed(df, alternatives, params: dict[str, float]) -> BackendResult:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return BackendResult(backend="apollo_r_fixed", available=False, message="Rscript not found")
    script = Path(__file__).resolve().parent / "apollo" / "R" / "run_cross_nested_fixed.R"
    if not script.exists():
        return BackendResult(backend="apollo_r_fixed", available=False, message=f"Missing script: {script}")
    with tempfile.TemporaryDirectory(prefix="torchdcm_apollo_cnl_") as tmp:
        tmp_path = Path(tmp)
        data_path, spec_path = write_replay_inputs(df, alternatives, params, tmp_path)
        output_path = tmp_path / "apollo_result.json"
        cmd = [
            rscript,
            str(script),
            "--data",
            str(data_path),
            "--spec",
            str(spec_path),
            "--output",
            str(output_path),
        ]
        env = os.environ.copy()
        r_user_lib = str(Path.home() / "R" / "site-library")
        existing_r_lib = env.get("R_LIBS_USER")
        env["R_LIBS_USER"] = r_user_lib if not existing_r_lib else f"{r_user_lib}:{existing_r_lib}"
        start = time.perf_counter()
        proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
        seconds = time.perf_counter() - start
        if proc.returncode != 0:
            return BackendResult(
                backend="apollo_r_fixed",
                available=False,
                seconds=seconds,
                message=(proc.stderr or proc.stdout).strip(),
            )
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        return BackendResult(
            backend="apollo_r_fixed",
            available=True,
            seconds=seconds,
            loglike=float(payload["loglike"]),
            params=params,
            probabilities=np.asarray(payload["probabilities"], dtype=float),
        )


def run_biogeme_fixed(df, alternatives, params: dict[str, float]) -> BackendResult:
    try:
        import biogeme.biogeme as bio
        import biogeme.database as db
        from biogeme.expressions import Variable, exp, log
    except ImportError as exc:
        return BackendResult(backend="biogeme_fixed", available=False, message=f"Biogeme not found: {exc}")

    code_by_alt = {alt: i + 1 for i, alt in enumerate(alternatives)}
    wide_df = df.copy().reset_index(drop=True)
    wide_df["choice_code"] = wide_df["choice"].map(code_by_alt).astype(int)
    for alt in alternatives:
        wide_df[f"avail_{alt.lower()}"] = wide_df[f"avail_{alt.lower()}"].astype(int)
    allocations = {
        "PUBLIC": {"TRAIN": 0.7, "SM": 0.8, "CAR": 0.0},
        "PRIVATE": {"TRAIN": 0.3, "SM": 0.2, "CAR": 1.0},
    }
    lambda_params = {"PUBLIC": "LAMBDA_PUBLIC", "PRIVATE": "LAMBDA_PRIVATE"}

    try:
        database = db.Database("torchdcm_cross_nested_fixed", wide_df.drop(columns=["choice"]))
        av = {alt: Variable(f"avail_{alt.lower()}") for alt in alternatives}
        utility = {
            "TRAIN": params.get("ASC_TRAIN", 0.0)
            + params["B_TIME"] * Variable("time_train")
            + params["B_COST"] * Variable("cost_train"),
            "SM": params["B_TIME"] * Variable("time_sm") + params["B_COST"] * Variable("cost_sm"),
            "CAR": params.get("ASC_CAR", 0.0)
            + params["B_TIME"] * Variable("time_car")
            + params["B_COST"] * Variable("cost_car"),
        }
        log_s = {}
        log_g_terms = []
        for nest_name, nest_allocations in allocations.items():
            lam = float(params[lambda_params[nest_name]])
            s_expr = 0
            for alt in alternatives:
                alpha = float(nest_allocations[alt])
                if alpha > 0:
                    s_expr = s_expr + alpha * av[alt] * exp(utility[alt] / lam)
            log_s[nest_name] = log(s_expr)
            log_g_terms.append(exp(lam * log_s[nest_name]))
        g_expr = log_g_terms[0]
        for expression in log_g_terms[1:]:
            g_expr = g_expr + expression

        formulas = {}
        for alt in alternatives:
            numerator = 0
            for nest_name, nest_allocations in allocations.items():
                alpha = float(nest_allocations[alt])
                if alpha <= 0:
                    continue
                lam = float(params[lambda_params[nest_name]])
                numerator = numerator + alpha * av[alt] * exp(utility[alt] / lam) * exp((lam - 1.0) * log_s[nest_name])
            formulas[f"prob_{alt.lower()}"] = numerator / g_expr
        start = time.perf_counter()
        biogeme = bio.BIOGEME(database, formulas)
        biogeme.model_name = "torchdcm_cross_nested_fixed"
        simulated = biogeme.simulate({})
        seconds = time.perf_counter() - start
    except Exception as exc:
        return BackendResult(backend="biogeme_fixed", available=False, message=f"{type(exc).__name__}: {exc}")

    prob_matrix = np.column_stack([simulated[f"prob_{alt.lower()}"].to_numpy(dtype=float) for alt in alternatives])
    chosen_idx = wide_df["choice_code"].to_numpy(dtype=int) - 1
    chosen = prob_matrix[np.arange(len(prob_matrix)), chosen_idx]
    return BackendResult(
        backend="biogeme_fixed",
        available=True,
        seconds=seconds,
        loglike=float(np.log(np.clip(chosen, np.finfo(float).tiny, None)).sum()),
        params=params,
        probabilities=prob_matrix.reshape(-1),
    )


def run_biogeme_estimate(df, alternatives, names: list[str], initial_values: dict[str, float]) -> BackendResult:
    try:
        import biogeme.biogeme as bio
        import biogeme.database as db
        from biogeme.expressions import Beta, Elem, Variable, exp, log
        from biogeme.results_processing.variance_covariance import EstimateVarianceCovariance
    except ImportError as exc:
        return BackendResult(backend="biogeme", available=False, message=f"Biogeme not found: {exc}")

    code_by_alt = {alt: i + 1 for i, alt in enumerate(alternatives)}
    wide_df = df.copy().reset_index(drop=True)
    wide_df["choice_code"] = wide_df["choice"].map(code_by_alt).astype(int)
    for column in wide_df.select_dtypes(include=["bool"]).columns:
        wide_df[column] = wide_df[column].astype(int)
    allocations = {
        "PUBLIC": {"TRAIN": 0.7, "SM": 0.8, "CAR": 0.0},
        "PRIVATE": {"TRAIN": 0.3, "SM": 0.2, "CAR": 1.0},
    }
    lambda_params = {"PUBLIC": "LAMBDA_PUBLIC", "PRIVATE": "LAMBDA_PRIVATE"}

    try:
        database = db.Database("torchdcm_cross_nested_estimate", wide_df.drop(columns=["choice"]))
        choice = Variable("choice_code")
        av = {alt: Variable(f"avail_{alt.lower()}") for alt in alternatives}
        betas = {
            name: (
                Beta(name, initial_values[name], 0.0001, 1.0, 0)
                if name.startswith("LAMBDA_")
                else Beta(name, initial_values.get(name, 0.0), None, None, 0)
            )
            for name in names
        }
        utility = {
            "TRAIN": betas["ASC_TRAIN"] + betas["B_TIME"] * Variable("time_train") + betas["B_COST"] * Variable("cost_train"),
            "SM": betas["B_TIME"] * Variable("time_sm") + betas["B_COST"] * Variable("cost_sm"),
            "CAR": betas["ASC_CAR"] + betas["B_TIME"] * Variable("time_car") + betas["B_COST"] * Variable("cost_car"),
        }
        log_s = {}
        log_g_terms = []
        for nest_name, nest_allocations in allocations.items():
            lam = betas[lambda_params[nest_name]]
            s_expr = 0
            for alt in alternatives:
                alpha = float(nest_allocations[alt])
                if alpha > 0:
                    s_expr = s_expr + alpha * av[alt] * exp(utility[alt] / lam)
            log_s[nest_name] = log(s_expr)
            log_g_terms.append(exp(lam * log_s[nest_name]))
        g_expr = log_g_terms[0]
        for expression in log_g_terms[1:]:
            g_expr = g_expr + expression

        prob_by_code = {}
        for alt in alternatives:
            numerator = 0
            for nest_name, nest_allocations in allocations.items():
                alpha = float(nest_allocations[alt])
                if alpha <= 0:
                    continue
                lam = betas[lambda_params[nest_name]]
                numerator = numerator + alpha * av[alt] * exp(utility[alt] / lam) * exp((lam - 1.0) * log_s[nest_name])
            prob_by_code[code_by_alt[alt]] = numerator / g_expr
        logprob = log(Elem(prob_by_code, choice))
        biogeme = bio.BIOGEME(database, logprob)
        biogeme.model_name = f"torchdcm_cross_nested_estimate_{len(df)}"
        biogeme.biogeme_parameters.set_value("save_iterations", False)
        start = time.perf_counter()
        estimate_start = time.perf_counter()
        estimates = biogeme.estimate()
        estimate_seconds = time.perf_counter() - estimate_start
        covariance_start = time.perf_counter()
        covariance = None
        covariance_message = ""
        try:
            covariance = estimates.get_variance_covariance_matrix(EstimateVarianceCovariance.RAO_CRAMER)
            if hasattr(covariance, "loc"):
                covariance = covariance.loc[names, names]
            covariance = np.asarray(covariance, dtype=float)
            if not np.isfinite(covariance).all():
                covariance_message = "covariance unavailable: non-finite values"
                covariance = None
        except Exception as exc:
            covariance_message = f"covariance unavailable: {type(exc).__name__}: {exc}"
        covariance_seconds = time.perf_counter() - covariance_start
        total_seconds = time.perf_counter() - start
        params = estimates.get_beta_values()
        return BackendResult(
            backend="biogeme",
            available=True,
            seconds=total_seconds,
            estimate_seconds=estimate_seconds,
            covariance_seconds=covariance_seconds,
            loglike=float(estimates.final_log_likelihood),
            params={name: float(params[name]) for name in names},
            covariance=covariance,
            message=covariance_message,
        )
    except Exception as exc:
        return BackendResult(backend="biogeme", available=False, message=f"{type(exc).__name__}: {exc}")


def _params_to_vector(names: list[str], params: dict[str, float]) -> torch.Tensor:
    return torch.as_tensor([params[name] for name in names], dtype=torch.float64)


def _predict_probabilities(data, spec, nests, names: list[str], params: dict[str, float]) -> np.ndarray:
    model = CrossNestedLogit(spec, nests)
    return model.predict_proba(data, _params_to_vector(names, params)).detach().cpu().numpy()


def attach_probability_and_diffs(results: list[BackendResult], reference: str, data, spec, nests, names: list[str]) -> None:
    ref = next(result for result in results if result.backend == reference and result.available)
    for result in results:
        if not result.available:
            continue
        result.probabilities = _predict_probabilities(data, spec, nests, names, result.params)
    ref = next(result for result in results if result.backend == reference and result.available)
    for result in results:
        if not result.available:
            continue
        result.ll_diff = result.loglike - ref.loglike  # type: ignore[attr-defined]
        result.max_abs_param_diff = max(abs(result.params[name] - ref.params[name]) for name in names)  # type: ignore[attr-defined]
        result.max_abs_probability_diff = float(np.max(np.abs(result.probabilities - ref.probabilities)))  # type: ignore[attr-defined]
        if result.covariance is not None and ref.covariance is not None:
            result.max_abs_covariance_diff = float(np.max(np.abs(result.covariance - ref.covariance)))  # type: ignore[attr-defined]
            diag = np.diag(result.covariance)
            ref_diag = np.diag(ref.covariance)
            if np.isfinite(diag).all() and np.isfinite(ref_diag).all() and (diag > 0).all() and (ref_diag > 0).all():
                se = np.sqrt(diag)
                ref_se = np.sqrt(ref_diag)
                result.max_abs_se_diff = float(np.max(np.abs(se - ref_se)))  # type: ignore[attr-defined]
                beta = np.asarray([result.params[name] for name in names], dtype=float)
                ref_beta = np.asarray([ref.params[name] for name in names], dtype=float)
                result.max_abs_t_diff = float(np.max(np.abs(beta / se - ref_beta / ref_se)))  # type: ignore[attr-defined]
            else:
                result.max_abs_se_diff = None  # type: ignore[attr-defined]
                result.max_abs_t_diff = None  # type: ignore[attr-defined]
        else:
            result.max_abs_covariance_diff = None  # type: ignore[attr-defined]
            result.max_abs_se_diff = None  # type: ignore[attr-defined]
            result.max_abs_t_diff = None  # type: ignore[attr-defined]


def print_results(results: list[BackendResult], reference: str, n_obs: int, mode: str):
    ref = next(result for result in results if result.backend == reference and result.available)
    print("case: biogeme_swissmetro_cross_nested")
    print(f"mode: {mode}")
    print(f"n_obs: {n_obs}")
    print("alignment:")
    if mode == "fixed":
        print("  benchmark_mode: fixed_likelihood_replay")
        print("  estimated_backend: none")
    elif mode == "full-estimation":
        print("  benchmark_mode: full_estimation")
        print("  estimated_backend: each backend estimates independently")
    else:
        print("  benchmark_mode: torchdcm_full_estimation_then_fixed_replay")
        print("  estimated_backend: torchdcm")
    print("  data_source: biogeme.data.swissmetro/data/swissmetro.dat")
    print("  model: Cross-Nested Logit with fixed allocation weights")
    print("  allocations: PUBLIC={TRAIN:.7, SM:.8}, PRIVATE={TRAIN:.3, SM:.2, CAR:1}")
    print("  lambda_constraints: lambdas in [0.0001, 1]")
    print("  parameters: shared across replay backends")
    print(f"  reference: {reference}")
    print()
    print(
        f"{'backend':<18}{'available':>10}{'total_s':>12}{'estimate_s':>12}{'cov_s':>12}"
        f"{'loglike':>18}{'ll_diff':>14}{'param_diff':>14}{'prob_diff':>14}{'cov_diff':>14}{'se_diff':>14}{'t_diff':>14}"
    )
    for result in results:
        if not result.available:
            print(
                f"{result.backend:<18}{str(result.available):>10}"
                f"{'':>12}{'':>12}{'':>12}{'':>18}{'':>14}{'':>14}{'':>14}{'':>14}{'':>14}{'':>14}  {result.message}"
            )
            continue
        print(
            f"{result.backend:<18}{str(result.available):>10}"
            f"{_fmt_seconds(result.seconds):>12}"
            f"{_fmt_seconds(result.estimate_seconds):>12}"
            f"{_fmt_seconds(result.covariance_seconds):>12}"
            f"{result.loglike:>18.10f}"
            f"{getattr(result, 'll_diff', result.loglike - ref.loglike):>14.3e}"
            f"{_fmt_optional(getattr(result, 'max_abs_param_diff', None)):>14}"
            f"{_fmt_optional(getattr(result, 'max_abs_probability_diff', None)):>14}"
            f"{_fmt_optional(getattr(result, 'max_abs_covariance_diff', None)):>14}"
            f"{_fmt_optional(getattr(result, 'max_abs_se_diff', None)):>14}"
            f"{_fmt_optional(getattr(result, 'max_abs_t_diff', None)):>14}"
        )
    print()
    for result in results:
        if result.available and result.params is not None:
            print(f"{result.backend} params:")
            for name, value in result.params.items():
                print(f"  {name}: {value:.12g}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-obs", type=int, default=500)
    parser.add_argument("--mode", choices=["fixed", "fit-replay", "full-estimation"], default="fixed")
    parser.add_argument("--max-iter", type=int, default=40)
    args = parser.parse_args()

    df, data, base_spec, alternatives = load_biogeme_swissmetro(args.n_obs)
    initial_values = make_initial_values(base_spec.parameter_names, mode="zero", seed=20260704, scale=0.1)
    spec = spec_with_initials(base_spec, initial_values)
    nests = swissmetro_cross_nests()
    names = [*base_spec.parameter_names, "LAMBDA_PUBLIC", "LAMBDA_PRIVATE"]

    if args.mode == "fixed":
        params = default_params(names)
        torch_result = run_torch_fixed(data, spec, nests, params)
        apollo_result = run_apollo_fixed(df, alternatives, params)
        biogeme_result = run_biogeme_fixed(df, alternatives, params)
        results = [torch_result, apollo_result, biogeme_result]
        reference = "torchdcm_fixed"
        attach_probability_and_diffs(results, reference, data, spec, nests, names)
    else:
        if args.mode == "full-estimation":
            torch_result = run_torch_fit(data, spec, nests, args.max_iter)
            estimate_initial = {name: 0.0 for name in base_spec.parameter_names}
            estimate_initial["LAMBDA_PUBLIC"] = 0.8
            estimate_initial["LAMBDA_PRIVATE"] = 0.9
            biogeme_result = run_biogeme_estimate(df, alternatives, names, estimate_initial)
            results = [torch_result, biogeme_result]
            reference = "torchdcm_fit"
            attach_probability_and_diffs(results, reference, data, spec, nests, names)
        else:
            torch_result = run_torch_fit(data, spec, nests, args.max_iter)
            params = torch_result.params
            apollo_result = run_apollo_fixed(df, alternatives, params)
            biogeme_result = run_biogeme_fixed(df, alternatives, params)
            results = [torch_result, apollo_result, biogeme_result]
            reference = "torchdcm_fit"
            attach_probability_and_diffs(results, reference, data, spec, nests, names)
    print_results(results, reference, len(df), args.mode)


def _fmt_optional(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.3e}"


def _fmt_seconds(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


if __name__ == "__main__":
    main()
