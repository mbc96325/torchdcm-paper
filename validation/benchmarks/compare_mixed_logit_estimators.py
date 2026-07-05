from __future__ import annotations

import argparse
import csv
import json
import os
import math
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from torchdcm import ErrorComponent, ErrorComponentsLogit, MixedLogit, RandomCoefficient
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
    probabilities: np.ndarray | None = None
    message: str = ""


def default_params(names: list[str], sigma: float, random_names: list[str], correlated: bool) -> dict[str, float]:
    params = {
        "ASC_TRAIN": 0.3,
        "B_TIME": -1.0,
        "B_COST": -1.2,
        "ASC_CAR": 0.6,
    } | {name: 0.0 for name in names if name not in {"ASC_TRAIN", "B_TIME", "B_COST", "ASC_CAR"}}
    for name in random_names:
        params.setdefault(name, 0.0)
        params[f"SIGMA_{name}"] = sigma
    if correlated:
        for row, row_name in enumerate(random_names):
            for col in range(row):
                params[f"CHOL_{row_name}__{random_names[col]}"] = 0.25
    return params


def write_apollo_inputs(
    df,
    alternatives,
    params,
    draws: np.ndarray,
    panel: bool,
    random_names: list[str],
    error_component_public: bool,
    directory: Path,
):
    code_by_alt = {alt: i + 1 for i, alt in enumerate(alternatives)}
    wide_df = df.copy().reset_index(drop=True)
    wide_df["choice_code"] = wide_df["choice"].map(code_by_alt).astype(int)
    wide_df["person_id"] = wide_df["ID"]
    for column in wide_df.select_dtypes(include=["bool"]).columns:
        wide_df[column] = wide_df[column].astype(int)
    csv_path = directory / "data.csv"
    spec_path = directory / "spec.json"
    draws_path = directory / "draws.csv"
    wide_df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    np.savetxt(draws_path, draws, delimiter=",", header=",".join(random_names), comments="")
    spec = {
        "model_name": f"apollo_mixed_fixed_{len(df)}",
        "alternatives": alternatives,
        "choice_col": "choice_code",
        "panel": panel,
        "panel_id_col": "person_id",
        "parameters": params,
        "correlated": bool("CHOL_" in " ".join(params)),
        "random_coefficients": [{"name": name, "distribution": "normal"} for name in random_names],
        "error_components": (
            [{"parameter": "EC_PUBLIC", "loadings": {"TRAIN": 1.0, "SM": 1.0, "CAR": 0.0}}]
            if error_component_public
            else None
        ),
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
    }
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return csv_path, spec_path, draws_path


def run_torch_fixed(
    data,
    spec,
    params: dict[str, float],
    draws: torch.Tensor,
    panel: bool,
    random_names: list[str],
    correlated: bool,
    error_component_public: bool,
) -> BackendResult:
    model = make_torch_model(spec, params, draws, panel, random_names, correlated, error_component_public)
    compiled = model.compile(data)
    names = compiled.free_names
    vector = torch.as_tensor([params[name] for name in names], dtype=torch.float64)
    start = time.perf_counter()
    ll = model.loglike(vector, data, compiled)
    probabilities = model.predict_proba(data, vector, compiled)
    seconds = time.perf_counter() - start
    return BackendResult(
        backend="torchdcm_fixed",
        available=True,
        seconds=seconds,
        loglike=float(ll.detach().cpu()),
        params={name: params[name] for name in names},
        probabilities=probabilities.detach().cpu().numpy(),
    )


def run_torch_fit(
    data,
    spec,
    draws: torch.Tensor,
    panel: bool,
    max_iter: int,
    random_names: list[str],
    correlated: bool,
    error_component_public: bool,
) -> BackendResult:
    params = {f"SIGMA_{name}": 0.1 for name in random_names}
    model = make_torch_model(spec, params, draws, panel, random_names, correlated, error_component_public, max_iter=max_iter)
    data = data.to(device=model.device, dtype=model.dtype)
    compiled = model.compile(data)
    internal_initial = torch.cat(
        [
            compiled.free_initial,
            model._sigma_to_internal(compiled.sigma_initial[~compiled.sigma_is_fixed]),
            compiled.chol_offdiag_initial,
        ]
    )
    internal_params = internal_initial.clone().detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [internal_params],
        max_iter=max_iter,
        tolerance_grad=model.tolerance_grad,
        line_search_fn=model.line_search_fn,
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        natural = model._internal_to_natural(internal_params, compiled)
        loss = -model.loglike(natural, data, compiled)
        loss.backward()
        return loss

    estimate_start = time.perf_counter()
    optimizer.step(closure)
    estimate_seconds = time.perf_counter() - estimate_start
    final_internal = internal_params.detach().clone().requires_grad_(True)
    final_natural = model._internal_to_natural(final_internal, compiled)
    ll = model.loglike(final_natural, data, compiled)
    covariance_start = time.perf_counter()
    hessian_internal = torch.autograd.functional.hessian(
        lambda p: model.loglike(model._internal_to_natural(p, compiled), data, compiled),
        final_internal,
    )
    cov_internal = torch.linalg.pinv(-hessian_internal.detach(), hermitian=True)
    transform_jac = model._natural_jacobian(final_internal.detach(), compiled)
    _ = transform_jac @ cov_internal @ transform_jac.T
    covariance_seconds = time.perf_counter() - covariance_start
    return BackendResult(
        backend="torchdcm_fit",
        available=True,
        seconds=estimate_seconds + covariance_seconds,
        estimate_seconds=estimate_seconds,
        covariance_seconds=covariance_seconds,
        loglike=float(ll.detach().cpu()),
        params=dict(zip(compiled.free_names, final_natural.detach().cpu().numpy())),
        probabilities=model.predict_proba(data, final_natural.detach(), compiled).detach().cpu().numpy(),
    )


def make_torch_model(
    spec,
    params: dict[str, float],
    draws: torch.Tensor,
    panel: bool,
    random_names: list[str],
    correlated: bool,
    error_component_public: bool,
    **kwargs,
):
    regular_random_names = [name for name in random_names if name != "EC_PUBLIC"]
    regular_random_coefficients = [
        RandomCoefficient(name, sigma_init=params[f"SIGMA_{name}"])
        for name in regular_random_names
    ]
    if error_component_public:
        return ErrorComponentsLogit(
            spec,
            [ErrorComponent("PUBLIC", ["TRAIN", "SM"], sigma_init=params["SIGMA_EC_PUBLIC"])],
            random_coefficients=regular_random_coefficients,
            draws=draws,
            panel=panel,
            correlated=correlated,
            **kwargs,
        )
    return MixedLogit(
        spec,
        regular_random_coefficients,
        draws=draws,
        panel=panel,
        correlated=correlated,
        **kwargs,
    )


def run_apollo_fixed(
    df,
    alternatives,
    params,
    draws: np.ndarray,
    panel: bool,
    random_names: list[str],
    error_component_public: bool,
) -> BackendResult:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return BackendResult(backend="apollo_r_fixed", available=False, message="Rscript not found")
    script = Path(__file__).resolve().parent / "apollo" / "R" / "run_mixed_fixed.R"
    if not script.exists():
        return BackendResult(backend="apollo_r_fixed", available=False, message=f"Missing script: {script}")
    with tempfile.TemporaryDirectory(prefix="torchdcm_apollo_mixed_") as tmp:
        tmp_path = Path(tmp)
        data_path, spec_path, draws_path = write_apollo_inputs(
            df, alternatives, params, draws, panel, random_names, error_component_public, tmp_path
        )
        output_path = tmp_path / "apollo_result.json"
        cmd = [
            rscript,
            str(script),
            "--data",
            str(data_path),
            "--spec",
            str(spec_path),
            "--draws",
            str(draws_path),
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


def _biogeme_random_parameter(name: str, draw_index: int, random_names: list[str], params: dict[str, float], correlated: bool, variable):
    if name not in random_names:
        return float(params[name])
    row = random_names.index(name)
    expression = float(params[name])
    for col in range(row + 1):
        if not correlated and col != row:
            continue
        col_name = random_names[col]
        if col == row:
            coefficient = float(params[f"SIGMA_{name}"])
        else:
            coefficient = float(params[f"CHOL_{name}__{col_name}"])
        expression = expression + coefficient * variable(f"DRAW_{col_name}_{draw_index}")
    return expression


def run_biogeme_fixed(
    df,
    alternatives,
    params,
    draws: np.ndarray,
    panel: bool,
    random_names: list[str],
    correlated: bool,
    error_component_public: bool,
) -> BackendResult:
    try:
        import biogeme.biogeme as bio
        import biogeme.database as db
        import biogeme.models as models
        from biogeme.expressions import Variable
    except ImportError as exc:
        return BackendResult(backend="biogeme_fixed", available=False, message=f"Biogeme not found: {exc}")

    code_by_alt = {alt: i + 1 for i, alt in enumerate(alternatives)}
    wide_df = df.copy().reset_index(drop=True)
    wide_df["choice_code"] = wide_df["choice"].map(code_by_alt).astype(int)
    wide_df["person_id"] = wide_df["ID"]
    for alt in alternatives:
        wide_df[f"avail_{alt.lower()}"] = wide_df[f"avail_{alt.lower()}"].astype(int)
    draw_columns = {
        f"DRAW_{name}_{draw_index}": float(draws[draw_index, random_index])
        for draw_index in range(draws.shape[0])
        for random_index, name in enumerate(random_names)
    }
    if draw_columns:
        import pandas as pd

        wide_df = pd.concat([wide_df, pd.DataFrame(draw_columns, index=wide_df.index)], axis=1)

    try:
        biogeme_df = wide_df.drop(columns=["choice"])
        database = db.Database("torchdcm_mixed_fixed", biogeme_df)
        choice = Variable("choice_code")
        av = {code_by_alt[alt]: Variable(f"avail_{alt.lower()}") for alt in alternatives}
        chosen_probs_by_draw = []
        alt_probs_by_draw = {alt: [] for alt in alternatives}
        for draw_index in range(draws.shape[0]):
            b_time = _biogeme_random_parameter("B_TIME", draw_index, random_names, params, correlated, Variable)
            b_cost = _biogeme_random_parameter("B_COST", draw_index, random_names, params, correlated, Variable)
            ec_public = (
                _biogeme_random_parameter("EC_PUBLIC", draw_index, random_names, params, correlated, Variable)
                if error_component_public
                else 0.0
            )
            v = {
                code_by_alt["TRAIN"]: float(params.get("ASC_TRAIN", 0.0))
                + b_time * Variable("time_train")
                + b_cost * Variable("cost_train")
                + ec_public,
                code_by_alt["SM"]: b_time * Variable("time_sm") + b_cost * Variable("cost_sm") + ec_public,
                code_by_alt["CAR"]: float(params.get("ASC_CAR", 0.0))
                + b_time * Variable("time_car")
                + b_cost * Variable("cost_car"),
            }
            chosen_probs_by_draw.append(models.logit(v, av, choice))
            for alt in alternatives:
                alt_probs_by_draw[alt].append(models.logit(v, av, code_by_alt[alt]))

        def average(expressions):
            total = expressions[0]
            for expression in expressions[1:]:
                total = total + expression
            return total / float(len(expressions))

        formulas = {"chosen_prob": average(chosen_probs_by_draw)}
        for draw_index, expression in enumerate(chosen_probs_by_draw):
            formulas[f"chosen_prob_draw_{draw_index}"] = expression
        for alt, expressions in alt_probs_by_draw.items():
            formulas[f"prob_{alt.lower()}"] = average(expressions)

        start = time.perf_counter()
        biogeme = bio.BIOGEME(database, formulas)
        biogeme.model_name = "torchdcm_mixed_fixed_shared_draws"
        simulated = biogeme.simulate({})
        seconds = time.perf_counter() - start
    except Exception as exc:
        return BackendResult(backend="biogeme_fixed", available=False, message=f"{type(exc).__name__}: {exc}")

    if panel:
        loglike = 0.0
        draw_cols = [f"chosen_prob_draw_{draw_index}" for draw_index in range(draws.shape[0])]
        draw_probs = simulated[draw_cols].clip(lower=np.finfo(float).tiny).to_numpy(dtype=float)
        person_ids = wide_df["person_id"].to_numpy()
        _, person_inverse = np.unique(person_ids, return_inverse=True)
        for person_index in range(int(person_inverse.max()) + 1):
            draw_log_probs = np.log(draw_probs[person_inverse == person_index]).sum(axis=0)
            max_log = float(draw_log_probs.max())
            loglike += max_log + math.log(float(np.exp(draw_log_probs - max_log).mean()))
    else:
        chosen_prob = simulated["chosen_prob"].clip(lower=np.finfo(float).tiny).to_numpy(dtype=float)
        loglike = float(np.log(chosen_prob).sum())

    probabilities = np.column_stack([simulated[f"prob_{alt.lower()}"].to_numpy(dtype=float) for alt in alternatives]).reshape(-1)
    return BackendResult(
        backend="biogeme_fixed",
        available=True,
        seconds=seconds,
        loglike=loglike,
        params=params,
        probabilities=probabilities,
    )


def make_draws(n_draws: int, seed: int, n_random: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    half = (n_draws + 1) // 2
    base = torch.randn((half, n_random), generator=generator, dtype=torch.float64)
    return torch.cat([base, -base], dim=0)[:n_draws]


def print_results(
    results: list[BackendResult],
    reference: str,
    n_obs: int,
    n_draws: int,
    panel: bool,
    mode: str,
    random_names: list[str],
    correlated: bool,
    error_component_public: bool,
):
    ref = next(result for result in results if result.backend == reference and result.available)
    print("case: biogeme_swissmetro_mixed_logit")
    print(f"mode: {mode}")
    print(f"n_obs: {n_obs}")
    print(f"n_draws: {n_draws}")
    print(f"panel: {panel}")
    print(f"random_coefficients: {random_names}")
    print(f"correlated: {correlated}")
    print(f"error_component_public: {error_component_public}")
    print("alignment:")
    if mode == "fixed":
        print("  benchmark_mode: fixed_likelihood_replay")
        print("  estimated_backend: none")
    else:
        print("  benchmark_mode: torchdcm_full_estimation_then_fixed_replay")
        print("  estimated_backend: torchdcm")
    print("  data_source: biogeme.data.swissmetro/data/swissmetro.dat")
    print("  model: Mixed Logit with normal random coefficients")
    if error_component_public:
        print("  error_component: EC_PUBLIC loading 1 on TRAIN/SM and 0 on CAR")
    print("  draws: shared antithetic standard-normal draw matrix")
    print("  covariance: Cholesky lower triangular replay when correlated=True")
    print("  parameters: shared across replay backends")
    print("  probabilities: averaged over the same draws and row order")
    print(f"  reference: {reference}")
    print()
    print(
        f"{'backend':<18}{'available':>10}{'total_s':>12}{'estimate_s':>12}{'cov_s':>12}"
        f"{'loglike':>18}{'ll_diff':>14}{'prob_diff':>14}"
    )
    for result in results:
        if not result.available:
            print(
                f"{result.backend:<18}{str(result.available):>10}"
                f"{'':>12}{'':>12}{'':>12}{'':>18}{'':>14}{'':>14}  {result.message}"
            )
            continue
        ll_diff = result.loglike - ref.loglike
        prob_diff = None
        if result.probabilities is not None and ref.probabilities is not None:
            prob_diff = float(np.max(np.abs(result.probabilities - ref.probabilities)))
        print(
            f"{result.backend:<18}{str(result.available):>10}"
            f"{_fmt_seconds(result.seconds):>12}"
            f"{_fmt_seconds(result.estimate_seconds):>12}"
            f"{_fmt_seconds(result.covariance_seconds):>12}"
            f"{result.loglike:>18.10f}"
            f"{ll_diff:>14.3e}{_fmt_optional(prob_diff):>14}"
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
    parser.add_argument("--n-draws", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--panel", action="store_true")
    parser.add_argument("--random-cost", action="store_true")
    parser.add_argument("--correlated", action="store_true")
    parser.add_argument("--error-component-public", action="store_true")
    parser.add_argument("--mode", choices=["fixed", "fit", "fit-replay"], default="fixed")
    parser.add_argument("--max-iter", type=int, default=40)
    args = parser.parse_args()

    df, data, base_spec, alternatives = load_biogeme_swissmetro(args.n_obs)
    initial_values = make_initial_values(base_spec.parameter_names, mode="zero", seed=args.seed, scale=0.1)
    spec = spec_with_initials(base_spec, initial_values)
    random_names = ["B_TIME", *([] if not args.random_cost else ["B_COST"])]
    if args.error_component_public:
        random_names.append("EC_PUBLIC")
    if args.correlated and len(random_names) < 2:
        raise ValueError("--correlated requires at least two random coefficients. Use --random-cost.")
    draws = make_draws(args.n_draws, args.seed, len(random_names))

    if args.mode == "fixed":
        params = default_params(base_spec.parameter_names, args.sigma, random_names, args.correlated)
        torch_result = run_torch_fixed(
            data, spec, params, draws, args.panel, random_names, args.correlated, args.error_component_public
        )
        apollo_result = run_apollo_fixed(
            df, alternatives, params, draws.detach().cpu().numpy(), args.panel, random_names, args.error_component_public
        )
        biogeme_result = run_biogeme_fixed(
            df,
            alternatives,
            params,
            draws.detach().cpu().numpy(),
            args.panel,
            random_names,
            args.correlated,
            args.error_component_public,
        )
        results = [torch_result, apollo_result, biogeme_result]
        reference = "torchdcm_fixed"
    elif args.mode in {"fit", "fit-replay"}:
        torch_result = run_torch_fit(
            data, spec, draws, args.panel, args.max_iter, random_names, args.correlated, args.error_component_public
        )
        params = torch_result.params
        apollo_result = run_apollo_fixed(
            df, alternatives, params, draws.detach().cpu().numpy(), args.panel, random_names, args.error_component_public
        )
        biogeme_result = run_biogeme_fixed(
            df,
            alternatives,
            params,
            draws.detach().cpu().numpy(),
            args.panel,
            random_names,
            args.correlated,
            args.error_component_public,
        )
        results = [torch_result, apollo_result, biogeme_result]
        reference = "torchdcm_fit"
    display_mode = "fit-replay" if args.mode == "fit" else args.mode
    print_results(
        results,
        reference,
        len(df),
        args.n_draws,
        args.panel,
        display_mode,
        random_names,
        args.correlated,
        args.error_component_public,
    )


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
