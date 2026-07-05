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
import pandas as pd
import torch

from torchdcm import Beta, ChoiceDataset, Nest, NestedLogit, UtilitySpec
from torchdcm.spec.expressions import Expression, Term


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
    wtp: float | None = None
    wtp_se: float | None = None
    elasticities: dict[str, np.ndarray] | None = None
    message: str = ""


ALTERNATIVES = ["TRAIN", "SM", "CAR"]


def load_biogeme_swissmetro(n_obs: int | None = None):
    try:
        import biogeme.data.swissmetro as swissmetro
    except ImportError as exc:
        raise RuntimeError("Biogeme is required to load the official Swissmetro testing data.") from exc

    raw = pd.read_csv(Path(swissmetro.__file__).resolve().parent / "data" / "swissmetro.dat", sep="\t")
    df = raw.loc[raw["CHOICE"] != 0].copy()
    if n_obs is not None:
        df = df.head(n_obs).copy()
    df["obs_id"] = np.arange(len(df))
    df["choice"] = df["CHOICE"].map({1: "TRAIN", 2: "SM", 3: "CAR"})
    df["time_train"] = df["TRAIN_TT"] / 100.0
    df["time_sm"] = df["SM_TT"] / 100.0
    df["time_car"] = df["CAR_TT"] / 100.0
    df["cost_train"] = df["TRAIN_CO"] * (df["GA"] == 0) / 100.0
    df["cost_sm"] = df["SM_CO"] * (df["GA"] == 0) / 100.0
    df["cost_car"] = df["CAR_CO"] / 100.0
    df["avail_train"] = (df["TRAIN_AV"] * (df["SP"] != 0)).astype(bool)
    df["avail_sm"] = df["SM_AV"].astype(bool)
    df["avail_car"] = (df["CAR_AV"] * (df["SP"] != 0)).astype(bool)
    data = ChoiceDataset.from_wide(
        df,
        alternatives=ALTERNATIVES,
        choice="choice",
        variables={
            "time": {"TRAIN": "time_train", "SM": "time_sm", "CAR": "time_car"},
            "cost": {"TRAIN": "cost_train", "SM": "cost_sm", "CAR": "cost_car"},
        },
        availability={"TRAIN": "avail_train", "SM": "avail_sm", "CAR": "avail_car"},
        obs_id="obs_id",
        individual_id="ID",
    )
    spec = UtilitySpec()
    spec.utility("TRAIN", Beta("ASC_TRAIN") + Beta("B_TIME", init=-0.01) * "time" + Beta("B_COST", init=-0.1) * "cost")
    spec.utility("SM", Beta("B_TIME", init=-0.01) * "time" + Beta("B_COST", init=-0.1) * "cost")
    spec.utility("CAR", Beta("ASC_CAR") + Beta("B_TIME", init=-0.01) * "time" + Beta("B_COST", init=-0.1) * "cost")
    return df, data, spec, ALTERNATIVES


def swissmetro_nests(lambda_init: float, fixed_lambda: bool = False) -> dict[str, Nest]:
    return {
        "PUBLIC": Nest(["TRAIN", "SM"], init=lambda_init, fixed=fixed_lambda),
        "PRIVATE": Nest(["CAR"], init=1.0, fixed=True),
    }


def spec_with_initials(spec: UtilitySpec, initial_values: dict[str, float]) -> UtilitySpec:
    registry: dict[str, Beta] = {}

    def convert_beta(beta: Beta) -> Beta:
        if beta.name not in registry:
            registry[beta.name] = Beta(
                beta.name,
                init=initial_values.get(beta.name, beta.init),
                fixed=beta.fixed,
            )
        return registry[beta.name]

    new_spec = UtilitySpec()
    for alt, expr in spec.utilities.items():
        new_spec.utility(
            alt,
            Expression(
                [
                    Term(
                        parameter=convert_beta(term.parameter),
                        variable=term.variable,
                        multiplier=term.multiplier,
                    )
                    for term in expr.terms
                ]
            ),
        )
    return new_spec


def make_initial_values(names: list[str], mode: str, seed: int, scale: float, lambda_init: float) -> dict[str, float]:
    beta_names = [name for name in names if not name.startswith("LAMBDA_")]
    if mode == "zero":
        values = {name: 0.0 for name in beta_names}
    elif mode == "random":
        rng = np.random.default_rng(seed)
        values = {name: float(value) for name, value in zip(beta_names, rng.normal(0.0, scale, len(beta_names)))}
    else:
        raise ValueError(f"Unknown initialization mode: {mode}")
    values["LAMBDA_PUBLIC"] = lambda_init
    return values


def run_torchdcm(data, spec, nests) -> BackendResult:
    model = NestedLogit(spec, nests)
    data = data.to(device=model.device, dtype=model.dtype)
    compiled = model.compile(data)
    internal_initial = torch.cat(
        [
            compiled.free_initial,
            model._lambda_to_internal(compiled.lambda_initial[~compiled.lambda_is_fixed]),
        ]
    )
    internal_params = internal_initial.clone().detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [internal_params],
        max_iter=model.max_iter,
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
    loglike = float(model.loglike(final_natural, data, compiled).detach().cpu())
    covariance_start = time.perf_counter()
    hessian_internal = torch.autograd.functional.hessian(
        lambda p: model.loglike(model._internal_to_natural(p, compiled), data, compiled),
        final_internal,
    )
    cov_internal = torch.linalg.pinv(-hessian_internal.detach(), hermitian=True)
    transform_jac = model._natural_jacobian(final_internal.detach(), compiled)
    covariance = transform_jac @ cov_internal @ transform_jac.T
    covariance_seconds = time.perf_counter() - covariance_start
    return BackendResult(
        backend="torchdcm",
        available=True,
        seconds=estimate_seconds + covariance_seconds,
        estimate_seconds=estimate_seconds,
        covariance_seconds=covariance_seconds,
        loglike=loglike,
        params=dict(zip(compiled.free_names, final_natural.detach().cpu().numpy())),
        covariance=covariance.detach().cpu().numpy(),
    )


def run_biogeme_nested(df, alternatives, names, initial_values, lambda_min: float) -> BackendResult:
    try:
        import biogeme.biogeme as bio
        import biogeme.database as db
        from biogeme import models
        from biogeme.expressions import Beta, Variable
        from biogeme.nests import NestsForNestedLogit, OneNestForNestedLogit
        from biogeme.results_processing.variance_covariance import EstimateVarianceCovariance
    except ImportError as exc:
        return BackendResult(backend="biogeme", available=False, message=str(exc))

    code_by_alt = {alt: i + 1 for i, alt in enumerate(alternatives)}
    wide_df = df.copy()
    wide_df["choice_code"] = wide_df["choice"].map(code_by_alt).astype(int)
    wide_df = wide_df.drop(columns=["choice"])
    bool_columns = wide_df.select_dtypes(include=["bool"]).columns
    wide_df[bool_columns] = wide_df[bool_columns].astype(int)
    database = db.Database("torchdcm_nl_compare", wide_df)
    chosen_alt = Variable("choice_code")

    betas = {}
    for name in names:
        if name == "LAMBDA_PUBLIC":
            betas[name] = Beta(name, initial_values[name], lambda_min, 1.0, 0)
        else:
            betas[name] = Beta(name, initial_values.get(name, 0.0), None, None, 0)
    utility = {}
    av = {}
    for alt_name, code in code_by_alt.items():
        time_var = Variable(f"time_{alt_name.lower()}")
        cost_var = Variable(f"cost_{alt_name.lower()}")
        utility[code] = betas.get(f"ASC_{alt_name.upper()}", 0) + betas["B_TIME"] * time_var + betas["B_COST"] * cost_var
        av[code] = Variable(f"avail_{alt_name.lower()}")

    nests = NestsForNestedLogit(
        choice_set=list(code_by_alt.values()),
        tuple_of_nests=(
            OneNestForNestedLogit(1.0 / betas["LAMBDA_PUBLIC"], [code_by_alt["TRAIN"], code_by_alt["SM"]], name="PUBLIC"),
            OneNestForNestedLogit(1.0, [code_by_alt["CAR"]], name="PRIVATE"),
        ),
    )
    logprob = models.lognested(utility, av, nests, chosen_alt)
    biogeme = bio.BIOGEME(database, logprob)
    biogeme.model_name = f"torchdcm_nl_compare_{len(df)}"
    biogeme.biogeme_parameters.set_value("save_iterations", False)

    start = time.perf_counter()
    estimate_start = time.perf_counter()
    estimates = biogeme.estimate()
    estimate_seconds = time.perf_counter() - estimate_start
    covariance_start = time.perf_counter()
    covariance = estimates.get_variance_covariance_matrix(EstimateVarianceCovariance.RAO_CRAMER)
    covariance_seconds = time.perf_counter() - covariance_start
    if hasattr(covariance, "loc"):
        covariance = covariance.loc[names, names]
    total_seconds = time.perf_counter() - start
    return BackendResult(
        backend="biogeme",
        available=True,
        seconds=total_seconds,
        estimate_seconds=estimate_seconds,
        covariance_seconds=covariance_seconds,
        loglike=estimates.final_log_likelihood,
        params=estimates.get_beta_values(),
        covariance=np.asarray(covariance, dtype=float),
    )


def write_apollo_inputs(df, alternatives: list[str], names: list[str], initial_values: dict[str, float], directory: Path):
    code_by_alt = {alt: i + 1 for i, alt in enumerate(alternatives)}
    wide_df = df.copy()
    wide_df["choice_code"] = wide_df["choice"].map(code_by_alt).astype(int)
    wide_df = wide_df.drop(columns=["choice"])
    for column in wide_df.select_dtypes(include=["bool"]).columns:
        wide_df[column] = wide_df[column].astype(int)

    csv_path = directory / "data.csv"
    spec_path = directory / "spec.json"
    wide_df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    lambda_min = 0.0001
    lambda_init = initial_values["LAMBDA_PUBLIC"]
    scaled_lambda = np.clip((lambda_init - lambda_min) / (1.0 - lambda_min), 1e-12, 1.0 - 1e-12)
    apollo_parameters = {
        name: value for name, value in initial_values.items() if name != "LAMBDA_PUBLIC"
    }
    apollo_parameters["RAW_LAMBDA_PUBLIC"] = float(np.log(scaled_lambda / (1.0 - scaled_lambda)))
    spec = {
        "model_name": f"apollo_nl_{len(df)}",
        "alternatives": alternatives,
        "choice_col": "choice_code",
        "parameters": apollo_parameters,
        "utility": {
            alt: {
                "code": code_by_alt[alt],
                "asc": f"ASC_{alt.upper()}" if f"ASC_{alt.upper()}" in names else None,
                "time": f"time_{alt.lower()}",
                "cost": f"cost_{alt.lower()}",
                "availability": f"avail_{alt.lower()}",
            }
            for alt in alternatives
        },
        "nests": {
            "PUBLIC": {
                "alternatives": ["TRAIN", "SM"],
                "lambda_param": "LAMBDA_PUBLIC",
                "lambda_raw_param": "RAW_LAMBDA_PUBLIC",
                "lambda_min": lambda_min,
            },
            "PRIVATE": {"alternatives": ["CAR"], "lambda_param": None, "lambda_value": 1.0},
        },
    }
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return csv_path, spec_path


def run_apollo_nested(df, alternatives, names, initial_values) -> BackendResult:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return BackendResult(backend="apollo", available=False, message="Rscript not found")
    script = Path(__file__).resolve().parent / "apollo" / "R" / "run_nl.R"
    if not script.exists():
        return BackendResult(backend="apollo", available=False, message=f"Missing Apollo script: {script}")

    with tempfile.TemporaryDirectory(prefix="torchdcm_apollo_nl_") as tmp:
        tmp_path = Path(tmp)
        data_path, spec_path = write_apollo_inputs(df, alternatives, names, initial_values, tmp_path)
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
                backend="apollo",
                available=False,
                seconds=seconds,
                message=(proc.stderr or proc.stdout).strip(),
            )
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        return BackendResult(
            backend="apollo",
            available=True,
            seconds=seconds,
            estimate_seconds=payload.get("timing", {}).get("estimate_seconds"),
            covariance_seconds=payload.get("timing", {}).get("covariance_seconds"),
            loglike=float(payload["loglike"]),
            params={key: float(value) for key, value in payload["estimates"].items()},
            covariance=_as_float_array_or_none(payload.get("covariance")),
            message=f"apollo_version={payload.get('apollo_version')}",
        )


def params_to_vector(names: list[str], params: dict[str, float]) -> torch.Tensor:
    return torch.as_tensor([params[name] for name in names], dtype=torch.float64)


def _as_float_array_or_none(value) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(array).all():
        return None
    return array


def predict_probabilities(data, spec, nests, names: list[str], params: dict[str, float]) -> np.ndarray:
    model = NestedLogit(spec, nests)
    vector = params_to_vector(names, params)
    return model.predict_proba(data, vector).detach().cpu().numpy()


def wtp_with_delta(names: list[str], params: dict[str, float], covariance: np.ndarray | None) -> tuple[float | None, float | None]:
    if "B_TIME" not in params or "B_COST" not in params:
        return None, None
    wtp = -params["B_TIME"] / params["B_COST"]
    if covariance is None:
        return float(wtp), None
    idx_time = names.index("B_TIME")
    idx_cost = names.index("B_COST")
    grad = np.zeros(len(names), dtype=float)
    beta_time = params["B_TIME"]
    beta_cost = params["B_COST"]
    grad[idx_time] = -1.0 / beta_cost
    grad[idx_cost] = beta_time / (beta_cost**2)
    variance = float(grad @ covariance @ grad)
    return float(wtp), float(np.sqrt(max(variance, 0.0)))


def direct_elasticities(data, probabilities: np.ndarray, params: dict[str, float]) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    if "B_TIME" in params and "time" in data.x_alt:
        x = data.x_alt["time"].detach().cpu().numpy()
        values["time"] = params["B_TIME"] * x * (1.0 - probabilities)
    if "B_COST" in params and "cost" in data.x_alt:
        x = data.x_alt["cost"].detach().cpu().numpy()
        values["cost"] = params["B_COST"] * x * (1.0 - probabilities)
    return values


def attach_post_estimation(results: list[BackendResult], data, spec, nests, names: list[str]) -> None:
    for result in results:
        if not result.available:
            continue
        result.probabilities = predict_probabilities(data, spec, nests, names, result.params)
        result.wtp, result.wtp_se = wtp_with_delta(names, result.params, result.covariance)
        result.elasticities = direct_elasticities(data, result.probabilities, result.params)


def compare_to_reference(results: list[BackendResult], reference: str):
    ref = next(result for result in results if result.backend == reference and result.available)
    for result in results:
        if not result.available:
            continue
        result.ll_diff = result.loglike - ref.loglike  # type: ignore[attr-defined]
        shared = sorted(set(result.params) & set(ref.params))
        result.max_abs_param_diff = max(abs(result.params[name] - ref.params[name]) for name in shared)  # type: ignore[attr-defined]
        result.max_abs_probability_diff = float(np.max(np.abs(result.probabilities - ref.probabilities)))  # type: ignore[attr-defined]
        if result.covariance is not None and ref.covariance is not None:
            result.max_abs_covariance_diff = float(np.max(np.abs(result.covariance - ref.covariance)))  # type: ignore[attr-defined]
            result.max_abs_se_diff = float(
                np.max(np.abs(np.sqrt(np.diag(result.covariance)) - np.sqrt(np.diag(ref.covariance))))
            )  # type: ignore[attr-defined]
        else:
            result.max_abs_covariance_diff = None  # type: ignore[attr-defined]
            result.max_abs_se_diff = None  # type: ignore[attr-defined]
        result.wtp_diff = None if result.wtp is None or ref.wtp is None else result.wtp - ref.wtp  # type: ignore[attr-defined]
        result.wtp_se_diff = (
            None if result.wtp_se is None or ref.wtp_se is None else result.wtp_se - ref.wtp_se
        )  # type: ignore[attr-defined]
        elasticity_diffs = {}
        for key in sorted(set(result.elasticities or {}) & set(ref.elasticities or {})):
            elasticity_diffs[key] = float(np.max(np.abs(result.elasticities[key] - ref.elasticities[key])))
        result.max_abs_elasticity_diff = elasticity_diffs  # type: ignore[attr-defined]


def run_case(n_obs: int, data_seed: int, initial: str, init_seed: int, init_scale: float, lambda_init: float):
    del data_seed
    df, data, base_spec, alternatives = load_biogeme_swissmetro(n_obs)
    base_nests = swissmetro_nests(lambda_init=lambda_init)
    names = [*base_spec.parameter_names, "LAMBDA_PUBLIC"]
    initial_values = make_initial_values(names, initial, init_seed, init_scale, lambda_init)
    spec = spec_with_initials(base_spec, initial_values)
    nests = swissmetro_nests(lambda_init=initial_values["LAMBDA_PUBLIC"])

    results = [
        run_torchdcm(data, spec, nests),
        run_biogeme_nested(df, alternatives, names, initial_values, lambda_min=0.0001),
        run_apollo_nested(df, alternatives, names, initial_values),
    ]
    attach_post_estimation(results, data, spec, base_nests, names)
    compare_to_reference(results, reference="torchdcm")
    return initial_values, results, len(df)


def print_results(n_obs: int, initial: str, initial_values: dict[str, float], results: list[BackendResult]) -> None:
    print("case: biogeme_swissmetro_nested")
    print(f"n_obs: {n_obs}")
    print("alignment:")
    print("  benchmark_mode: full_estimation")
    print("  data_source: biogeme.data.swissmetro/data/swissmetro.dat")
    print("  model: Nested Logit with PUBLIC={TRAIN, SM}, PRIVATE={CAR}")
    print("  lambda_constraints: LAMBDA_PUBLIC in [0.0001, 1], PRIVATE fixed to 1")
    print("  initial_values: shared across TorchDCM, Biogeme, Apollo")
    print("  covariance: classic inverse observed information where available")
    print("  reference: torchdcm")
    print(f"initial: {initial}")
    print("nests:")
    print("  PUBLIC: TRAIN, SM")
    print("  PRIVATE: CAR (lambda fixed to 1)")
    print("initial_values:")
    for name, value in initial_values.items():
        print(f"  {name}: {value:.12g}")
    print()
    print(
        f"{'backend':<12}{'available':>10}{'total_s':>12}{'estimate_s':>12}{'cov_s':>12}"
        f"{'loglike':>18}{'ll_diff':>14}{'max_param_diff':>18}"
    )
    for result in results:
        if result.available:
            print(
                f"{result.backend:<12}{str(result.available):>10}"
                f"{_fmt_seconds(result.seconds):>12}"
                f"{_fmt_seconds(result.estimate_seconds):>12}"
                f"{_fmt_seconds(result.covariance_seconds):>12}"
                f"{result.loglike:>18.10f}"
                f"{getattr(result, 'll_diff'):>14.3e}"
                f"{getattr(result, 'max_abs_param_diff'):>18.3e}"
            )
        else:
            print(
                f"{result.backend:<12}{str(result.available):>10}"
                f"{'':>12}{'':>12}{'':>12}{'':>18}{'':>14}{'':>18}  {result.message}"
            )
    print()
    print(
        f"{'backend':<12}{'prob_diff':>14}{'cov_diff':>14}{'se_diff':>14}"
        f"{'wtp':>14}{'wtp_diff':>14}{'wtp_se':>14}{'wtp_se_diff':>14}"
        f"{'elas_time':>14}{'elas_cost':>14}"
    )
    for result in results:
        if not result.available:
            continue
        elas = getattr(result, "max_abs_elasticity_diff")
        print(
            f"{result.backend:<12}"
            f"{getattr(result, 'max_abs_probability_diff'):>14.3e}"
            f"{_fmt_optional(getattr(result, 'max_abs_covariance_diff')):>14}"
            f"{_fmt_optional(getattr(result, 'max_abs_se_diff')):>14}"
            f"{_fmt_optional(result.wtp):>14}"
            f"{_fmt_optional(getattr(result, 'wtp_diff')):>14}"
            f"{_fmt_optional(result.wtp_se):>14}"
            f"{_fmt_optional(getattr(result, 'wtp_se_diff')):>14}"
            f"{_fmt_optional(elas.get('time')):>14}"
            f"{_fmt_optional(elas.get('cost')):>14}"
        )


def _fmt_optional(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.3e}"


def _fmt_seconds(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-obs", type=int, default=500)
    parser.add_argument("--data-seed", type=int, default=3)
    parser.add_argument("--initial", choices=["zero", "random"], default="zero")
    parser.add_argument("--init-seed", type=int, default=20260704)
    parser.add_argument("--init-scale", type=float, default=0.1)
    parser.add_argument("--lambda-init", type=float, default=0.8)
    args = parser.parse_args()

    initial_values, results, actual_n_obs = run_case(
        n_obs=args.n_obs,
        data_seed=args.data_seed,
        initial=args.initial,
        init_seed=args.init_seed,
        init_scale=args.init_scale,
        lambda_init=args.lambda_init,
    )
    print_results(actual_n_obs, args.initial, initial_values, results)


if __name__ == "__main__":
    main()
