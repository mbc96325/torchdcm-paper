from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from benchmark_runtime import (
    configure_single_thread_cpu,
    estimation_covariance_total,
    runtime_policy_metadata,
)

if __name__ == "__main__":
    configure_single_thread_cpu(configure_torch=True)

import numpy as np
import pandas as pd
import torch

from torchdcm import Beta, ChoiceDataset, MixedLogit, RandomCoefficient, UtilitySpec
from compare_biogeme_public_mnl import CASE_BUILDERS as PUBLIC_CASE_BUILDERS
from compare_mixed_logit_estimators import make_draws
from compare_mnl_estimators import load_biogeme_swissmetro, make_initial_values, spec_with_initials
from run_mlogit_dataset_battery import DEFAULT_DATASETS as MLOGIT_DATASETS
from run_mlogit_dataset_battery import run_r_reference


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "generated"
APOLLO_MIXED_SCRIPT = ROOT / "benchmarks" / "apollo" / "R" / "run_generic_mixed_estimate.R"


@dataclass
class MixedCase:
    case: str
    dataset_id: str
    source: str
    model_name: str
    df: pd.DataFrame
    data: ChoiceDataset
    spec: UtilitySpec
    alternatives: list[str]
    choice_col: str
    utility_terms: dict[str, list[tuple[str, str | None]]]
    availability_columns: dict[str, str]
    parameter_names: list[str]
    random_names: list[str]
    sigma_init: float


@dataclass
class BackendResult:
    backend: str
    available: bool
    total_s: float | None = None
    estimate_s: float | None = None
    covariance_s: float | None = None
    loglike: float | None = None
    params: dict[str, float] | None = None
    covariance: np.ndarray | None = None
    probabilities: np.ndarray | None = None
    message: str = ""


def public_case_to_mixed(case_key: str, n_obs: int | None, sigma_init: float) -> MixedCase:
    base = PUBLIC_CASE_BUILDERS[case_key](n_obs)
    random_names = choose_random_parameters(base.parameter_names)
    return MixedCase(
        case=base.case,
        dataset_id=base.dataset_id,
        source=base.source,
        model_name=f"{base.model_name.replace(' MNL', '')} mixed logit",
        df=base.df.copy(),
        data=base.data,
        spec=base.spec,
        alternatives=base.alternatives,
        choice_col="choice",
        utility_terms=utility_terms_from_public_case(base),
        availability_columns=base.availability_columns,
        parameter_names=base.parameter_names,
        random_names=random_names,
        sigma_init=sigma_init,
    )


def swissmetro_case(n_obs: int | None, sigma_init: float) -> MixedCase:
    df, data, base_spec, alternatives = load_biogeme_swissmetro(n_obs or 10719)
    initial_values = make_initial_values(base_spec.parameter_names, mode="zero", seed=20260704, scale=0.1)
    spec = spec_with_initials(base_spec, initial_values)
    utility_terms = {
        "TRAIN": [("ASC_TRAIN", None), ("B_TIME", "time_train"), ("B_COST", "cost_train")],
        "SM": [("B_TIME", "time_sm"), ("B_COST", "cost_sm")],
        "CAR": [("ASC_CAR", None), ("B_TIME", "time_car"), ("B_COST", "cost_car")],
    }
    return MixedCase(
        case="swissmetro",
        dataset_id="biogeme_swissmetro",
        source="biogeme.data.swissmetro/data/swissmetro.dat",
        model_name="Swissmetro mixed logit",
        df=df.copy(),
        data=data,
        spec=spec,
        alternatives=alternatives,
        choice_col="choice",
        utility_terms=utility_terms,
        availability_columns={alt: f"avail_{alt.lower()}" for alt in alternatives},
        parameter_names=base_spec.parameter_names,
        random_names=["B_TIME", "B_COST"],
        sigma_init=sigma_init,
    )


def mlogit_case_to_mixed(dataset: str, sigma_init: float) -> MixedCase:
    df, ref = run_r_reference(dataset)
    if df is None or not ref.get("available"):
        raise RuntimeError(ref.get("message", f"Could not export mlogit::{dataset}"))
    variables_raw = ref["variables"]
    variables = [variables_raw] if isinstance(variables_raw, str) else list(variables_raw)
    if len(variables) < 2:
        raise RuntimeError(f"Need at least two observed variables for 2+ random coefficients; found {variables}.")
    alternatives = list(pd.unique(df["alt"]))
    data = ChoiceDataset.from_long(
        df,
        obs_id="obs_id",
        alt_id="alt",
        choice="choice",
        variables=variables,
        availability="availability" if "availability" in df.columns else None,
        alt_order=alternatives,
    )
    spec = UtilitySpec()
    parameter_names = [f"B_{variable.upper()}" for variable in variables]
    for alt in alternatives:
        utility = None
        for variable, parameter in zip(variables, parameter_names):
            term = Beta(parameter) * variable
            utility = term if utility is None else utility + term
        spec.utility(alt, utility)
    wide_df, alt_columns, choice_col = long_to_wide(df, variables)
    utility_terms = {
        str(alt): [(f"B_{variable.upper()}", str(columns[variable])) for variable in variables]
        for alt, columns in alt_columns.items()
    }
    availability_columns = {str(alt): str(columns["availability"]) for alt, columns in alt_columns.items()}
    return MixedCase(
        case=f"mlogit_{dataset}",
        dataset_id=f"mlogit_{dataset}",
        source=f"R mlogit::{dataset}",
        model_name=f"mlogit::{dataset} mixed logit",
        df=wide_df,
        data=data,
        spec=spec,
        alternatives=[str(alt) for alt in alternatives],
        choice_col=choice_col,
        utility_terms=utility_terms,
        availability_columns=availability_columns,
        parameter_names=parameter_names,
        random_names=parameter_names[: min(4, len(parameter_names))],
        sigma_init=sigma_init,
    )


def choose_random_parameters(parameter_names: list[str]) -> list[str]:
    observed = [name for name in parameter_names if not name.startswith("ASC_")]
    chosen = observed[:4]
    if len(chosen) < 2:
        chosen.extend([name for name in parameter_names if name not in chosen][: 2 - len(chosen)])
    if len(chosen) < 2:
        raise RuntimeError(f"Need at least two free parameters for 2+ random coefficients: {parameter_names}")
    return chosen[:4]


def utility_terms_from_public_case(base) -> dict[str, list[tuple[str, str | None]]]:
    terms: dict[str, list[tuple[str, str | None]]] = {}
    for alt in base.alternatives:
        alt_terms: list[tuple[str, str | None]] = []
        asc_name = f"ASC_{alt.upper()}"
        if asc_name in base.parameter_names:
            alt_terms.append((asc_name, None))
        for feature, columns in base.feature_columns.items():
            alt_terms.append((param_for_feature(feature), columns[alt]))
        terms[alt] = alt_terms
    return terms


def param_for_feature(feature: str) -> str:
    return {
        "trip_time": "B_TRIP_TIME",
        "fare": "B_FARE",
        "legroom": "B_LEGROOM",
        "access_time": "B_ACCESS_TIME",
        "search_time": "B_SEARCH_TIME",
        "fee": "B_FEE",
        "time": "B_TIME",
        "cost": "B_COST",
    }[feature]


def run_torch(case: MixedCase, draws: torch.Tensor, max_iter: int, device: str) -> BackendResult:
    if torch.device(device).type != "cpu":
        raise ValueError("Cross-estimator mixed-logit batteries must use CPU for TorchDCM.")
    model = MixedLogit(
        case.spec,
        [RandomCoefficient(name, sigma_init=case.sigma_init) for name in case.random_names],
        draws=draws,
        panel=False,
        device=device,
        max_iter=max_iter,
    )
    data = case.data.to(device=model.device, dtype=model.dtype)
    compiled = model.compile(data)
    internal_initial = torch.cat(
        [
            compiled.free_initial,
            model._sigma_to_internal(compiled.sigma_initial[~compiled.sigma_is_fixed]),
            compiled.chol_offdiag_initial,
        ]
    )

    # Exclude one-time tensor-kernel and autograd initialization from timing,
    # following the generated MNL and NL runtime policy.
    warmup_internal = internal_initial.clone().detach().requires_grad_(True)
    warmup_natural = model._internal_to_natural(warmup_internal, compiled)
    (-model.loglike(warmup_natural, data, compiled)).backward()

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
    estimate_s = time.perf_counter() - estimate_start
    final_internal = internal_params.detach().clone().requires_grad_(True)
    final_natural = model._internal_to_natural(final_internal, compiled)
    loglike = float(model.loglike(final_natural, data, compiled).detach().cpu())
    covariance_start = time.perf_counter()
    covariance = None
    try:
        hessian = torch.autograd.functional.hessian(
            lambda p: model.loglike(model._internal_to_natural(p, compiled), data, compiled),
            final_internal,
        )
        cov_internal = torch.linalg.pinv(-hessian.detach(), hermitian=True)
        transform = model._natural_jacobian(final_internal.detach(), compiled)
        covariance = (transform @ cov_internal @ transform.T).detach().cpu().numpy()
    except RuntimeError as exc:
        message = f"covariance unavailable: {exc}"
    else:
        message = ""
    covariance_s = time.perf_counter() - covariance_start
    return BackendResult(
        backend="torchdcm",
        available=True,
        total_s=estimate_s + covariance_s,
        estimate_s=estimate_s,
        covariance_s=covariance_s,
        loglike=loglike,
        params={name: float(value) for name, value in zip(compiled.free_names, final_natural.detach().cpu().numpy())},
        covariance=covariance,
        probabilities=model.predict_proba(data, final_natural.detach(), compiled).detach().cpu().numpy(),
        message=message,
    )


def run_biogeme(
    case: MixedCase,
    draws: np.ndarray,
    max_iter: int,
    *,
    smooth_positive_scales: bool = False,
) -> BackendResult:
    tmp_root = Path(tempfile.gettempdir()) if False else Path(os.environ.get("TMPDIR", "/tmp"))
    os.environ.setdefault("MPLCONFIGDIR", str(tmp_root / "torchdcm_matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(tmp_root / "torchdcm_cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    try:
        import biogeme.biogeme as bio
        import biogeme.database as db
        from biogeme.expressions import Beta as BioBeta
        from biogeme.expressions import Variable
        from biogeme.expressions import exp
        from biogeme.expressions import log
        from biogeme import models
        from biogeme.results_processing.variance_covariance import EstimateVarianceCovariance
    except ImportError as exc:
        return BackendResult("biogeme", False, message=f"Biogeme unavailable: {exc}")

    try:
        df = case.df.copy().reset_index(drop=True)
        code_by_alt = {alt: i + 1 for i, alt in enumerate(case.alternatives)}
        if case.choice_col == "choice":
            df["choice_code"] = df["choice"].map(code_by_alt).astype(int)
            choice_col = "choice_code"
            df = df.drop(columns=["choice"])
        else:
            choice_col = case.choice_col
        for column in df.select_dtypes(include=["bool"]).columns:
            df[column] = df[column].astype(int)
        draw_columns = {
            f"DRAW_{name}_{draw_index}": float(draws[draw_index, random_index])
            for draw_index in range(draws.shape[0])
            for random_index, name in enumerate(case.random_names)
        }
        if draw_columns:
            df = pd.concat([df, pd.DataFrame(draw_columns, index=df.index)], axis=1)

        sigma_names = [f"SIGMA_{name}" for name in case.random_names]
        names = [*case.parameter_names, *sigma_names]
        betas = {name: BioBeta(name, 0.0, None, None, 0) for name in case.parameter_names}
        if smooth_positive_scales:
            phi_names = [f"PHI_{name}" for name in sigma_names]
            optimization_names = [*case.parameter_names, *phi_names]
            phi_init = float(np.log(np.expm1(case.sigma_init)))
            for sigma_name, phi_name in zip(sigma_names, phi_names):
                phi = BioBeta(phi_name, phi_init, None, None, 0)
                # Match TorchDCM's smooth positive-scale parameterization. A
                # hard lower bound at zero creates a stationary boundary under
                # symmetric draws, while an unrestricted scale changes the
                # finite-draw objective when one draw dimension changes sign.
                betas[sigma_name] = log(1 + exp(phi))
        else:
            phi_names = []
            optimization_names = names
            for sigma_name in sigma_names:
                betas[sigma_name] = BioBeta(
                    sigma_name, case.sigma_init, 0.0, None, 0
                )

        database = db.Database(f"torchdcm_mixed_{case.case}_{len(df)}", df)
        choice = Variable(choice_col)
        availability = {code_by_alt[alt]: Variable(case.availability_columns[alt]) for alt in case.alternatives}
        chosen_probs = []
        for draw_index in range(draws.shape[0]):
            utility = {}
            for alt in case.alternatives:
                expr = 0
                for param, column in case.utility_terms[alt]:
                    coeff = betas[param]
                    if param in case.random_names:
                        coeff = coeff + betas[f"SIGMA_{param}"] * Variable(f"DRAW_{param}_{draw_index}")
                    expr += coeff if column is None else coeff * Variable(column)
                utility[code_by_alt[alt]] = expr
            chosen_probs.append(models.logit(utility, availability, choice))
        average_prob = chosen_probs[0]
        for expression in chosen_probs[1:]:
            average_prob = average_prob + expression
        logprob = log(average_prob / float(len(chosen_probs)))

        biogeme = bio.BIOGEME(database, logprob)
        biogeme.model_name = f"torchdcm_mixed_{case.case}_{len(df)}_{draws.shape[0]}"
        biogeme.biogeme_parameters.set_value("save_iterations", False)
        try:
            biogeme.biogeme_parameters.set_value("max_iterations", max_iter)
        except Exception:
            pass
        total_start = time.perf_counter()
        estimate_start = time.perf_counter()
        estimates = biogeme.estimate()
        estimate_s = time.perf_counter() - estimate_start
        covariance_start = time.perf_counter()
        covariance_obj = estimates.get_variance_covariance_matrix(EstimateVarianceCovariance.RAO_CRAMER)
        covariance_s = time.perf_counter() - covariance_start
        beta_values = estimates.get_beta_values()
        covariance_internal = covariance_to_array(covariance_obj, optimization_names)
        if smooth_positive_scales:
            transform = np.ones(len(names))
            natural_values = {
                name: float(beta_values[name]) for name in case.parameter_names
            }
            for index, (sigma_name, phi_name) in enumerate(
                zip(sigma_names, phi_names), start=len(case.parameter_names)
            ):
                phi_value = float(beta_values[phi_name])
                natural_values[sigma_name] = float(np.logaddexp(0.0, phi_value))
                transform[index] = 1.0 / (1.0 + np.exp(-phi_value))
            covariance = (
                transform[:, None] * covariance_internal * transform[None, :]
            )
        else:
            natural_values = {name: float(beta_values[name]) for name in names}
            covariance = covariance_internal
        return BackendResult(
            backend="biogeme",
            available=True,
            total_s=estimation_covariance_total(estimate_s, covariance_s),
            estimate_s=estimate_s,
            covariance_s=covariance_s,
            loglike=float(estimates.final_log_likelihood),
            params=natural_values,
            covariance=covariance,
        )
    except Exception as exc:
        return BackendResult("biogeme", False, message=f"{type(exc).__name__}: {exc}")


def run_apollo(case: MixedCase, n_draws: int, timeout: float) -> BackendResult:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return BackendResult("apollo", False, message="Rscript not found.")
    if not APOLLO_MIXED_SCRIPT.exists():
        return BackendResult("apollo", False, message=f"Missing Apollo script: {APOLLO_MIXED_SCRIPT}")
    with tempfile.TemporaryDirectory(prefix=f"torchdcm_apollo_real_mixl_{case.case}_") as tmp:
        tmp_path = Path(tmp)
        data_path = tmp_path / "data.csv"
        spec_path = tmp_path / "spec.json"
        output_path = tmp_path / "apollo_result.json"
        df, spec = apollo_inputs(case, n_draws)
        df.to_csv(data_path, index=False)
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        command = [
            rscript,
            str(APOLLO_MIXED_SCRIPT),
            "--data",
            str(data_path),
            "--spec",
            str(spec_path),
            "--output",
            str(output_path),
        ]
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                command,
                text=True,
                capture_output=True,
                env=r_env(),
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return BackendResult(
                "apollo",
                False,
                total_s=timeout,
                message=f"Apollo exceeded {timeout:.0f}s timeout.",
            )
        wall_s = time.perf_counter() - start
        if proc.returncode != 0 or not output_path.exists():
            return BackendResult(
                "apollo",
                False,
                total_s=wall_s,
                message=(proc.stderr or proc.stdout).strip(),
            )
        result = json.loads(output_path.read_text(encoding="utf-8"))
        names = [*case.parameter_names, *[f"SIGMA_{name}" for name in case.random_names]]
        estimate_s = result.get("timing", {}).get("estimate_seconds")
        covariance_s = result.get("timing", {}).get("covariance_seconds")
        total_s = estimation_covariance_total(estimate_s, covariance_s)
        convergence = result.get("convergence") or {}
        message = (
            f"apollo_version={result.get('apollo_version')}; "
            f"apollo_halton_draws={n_draws}; "
            f"convergence_status={convergence.get('status')}; "
            f"convergence_message={convergence.get('message')}"
        )
        if "unfavorable" in str(convergence.get("message")).lower():
            return BackendResult(
                "apollo",
                False,
                total_s=total_s,
                estimate_s=estimate_s,
                covariance_s=covariance_s,
                message=message,
            )
        covariance = reorder_covariance(
            result.get("covariance"),
            result.get("covariance_names"),
            names,
        )
        return BackendResult(
            "apollo",
            True,
            total_s=total_s,
            estimate_s=estimate_s,
            covariance_s=covariance_s,
            loglike=float(result["loglike"]),
            params={name: float(result["estimates"][name]) for name in names},
            covariance=covariance,
            message=message,
        )


def apollo_inputs(case: MixedCase, n_draws: int) -> tuple[pd.DataFrame, dict]:
    df = case.df.copy()
    code_by_alt = {alt: index + 1 for index, alt in enumerate(case.alternatives)}
    if case.choice_col == "choice":
        df["choice_code"] = df["choice"].map(code_by_alt).astype(int)
        df = df.drop(columns=["choice"])
        choice_col = "choice_code"
    else:
        choice_col = case.choice_col
    for column in df.select_dtypes(include=["bool"]).columns:
        df[column] = df[column].astype(int)
    parameters = {name: 0.0 for name in case.parameter_names}
    parameters.update({f"SIGMA_{name}": case.sigma_init for name in case.random_names})
    utility = {
        alt: {
            "code": code_by_alt[alt],
            "availability": case.availability_columns[alt],
            "terms": [
                {"parameter": parameter, "variable": variable}
                for parameter, variable in case.utility_terms[alt]
            ],
        }
        for alt in case.alternatives
    }
    return df, {
        "model_name": f"apollo_real_mixl_{case.case}_{case.data.n_obs}",
        "alternatives": case.alternatives,
        "choice_col": choice_col,
        "parameters": parameters,
        "utility": utility,
        "random_coefficients": case.random_names,
        "n_draws": n_draws,
    }


def reorder_covariance(covariance, source_names, target_names: list[str]) -> np.ndarray | None:
    if covariance is None or source_names is None:
        return None
    try:
        matrix = pd.DataFrame(covariance, index=source_names, columns=source_names)
        ordered = matrix.loc[target_names, target_names].to_numpy(dtype=float)
    except Exception:
        return None
    return ordered if np.isfinite(ordered).all() else None


def r_env() -> dict[str, str]:
    env = os.environ.copy()
    r_user_lib = str(Path.home() / "R" / "site-library")
    existing = env.get("R_LIBS_USER")
    env["R_LIBS_USER"] = r_user_lib if not existing else f"{r_user_lib}:{existing}"
    return env


def replay_probabilities(case: MixedCase, params: dict[str, float], draws: torch.Tensor) -> np.ndarray | None:
    try:
        model = MixedLogit(
            case.spec,
            [RandomCoefficient(name, sigma_init=case.sigma_init) for name in case.random_names],
            draws=draws,
            panel=False,
            device="cpu",
        )
        compiled = model.compile(case.data)
        vector = torch.as_tensor([params[name] for name in compiled.free_names], dtype=torch.float64)
        return model.predict_proba(case.data, vector, compiled).detach().cpu().numpy()
    except Exception:
        return None


def compare_results(case: MixedCase, results: list[BackendResult]) -> bool:
    torch_result = next((result for result in results if result.backend == "torchdcm" and result.available), None)
    if torch_result is None:
        return False
    consistent = True
    for result in results:
        if not result.available:
            continue
        if result.probabilities is None and result.params is not None:
            result.probabilities = replay_probabilities(case, result.params, make_draws_cached)
        result.ll_diff = None if result.loglike is None else result.loglike - torch_result.loglike  # type: ignore[attr-defined]
        common = sorted(set(result.params or {}) & set(torch_result.params or {}))
        result.max_param_diff = max((abs(result.params[name] - torch_result.params[name]) for name in common), default=None)  # type: ignore[attr-defined]
        if result.probabilities is not None and torch_result.probabilities is not None:
            result.max_prob_diff = float(np.max(np.abs(result.probabilities - torch_result.probabilities)))  # type: ignore[attr-defined]
        else:
            result.max_prob_diff = None  # type: ignore[attr-defined]
        if result.backend == "biogeme":
            row_ok = (
                result.ll_diff is not None
                and abs(result.ll_diff) <= max(0.25, 1e-5 * abs(torch_result.loglike or 0.0))
                and (result.max_param_diff is None or result.max_param_diff <= 0.2)
                and (result.max_prob_diff is None or result.max_prob_diff <= 0.03)
            )
            consistent = consistent and row_ok
    return consistent


make_draws_cached = torch.empty(0)


def build_cases(datasets: list[str], n_obs: int | None, sigma_init: float) -> list[MixedCase | dict]:
    cases: list[MixedCase | dict] = []
    for name in datasets:
        try:
            if name == "swissmetro":
                cases.append(swissmetro_case(n_obs, sigma_init))
            elif name in PUBLIC_CASE_BUILDERS:
                cases.append(public_case_to_mixed(name, n_obs, sigma_init))
            elif name.startswith("mlogit_"):
                cases.append(mlogit_case_to_mixed(name.removeprefix("mlogit_"), sigma_init))
            else:
                cases.append({"case": name, "status": "skipped", "message": "Unknown dataset key."})
        except Exception as exc:
            cases.append({"case": name, "status": "skipped", "message": f"{type(exc).__name__}: {exc}"})
    return cases


def run_case(case: MixedCase, args) -> dict:
    global make_draws_cached
    draws = make_draws(args.n_draws, args.seed, len(case.random_names))
    make_draws_cached = draws
    results = []
    try:
        results.append(run_torch(case, draws, args.max_iter, args.torch_device))
    except Exception as exc:
        results.append(BackendResult("torchdcm", False, message=f"{type(exc).__name__}: {exc}"))
    if not args.torch_only:
        results.append(run_biogeme(case, draws.detach().cpu().numpy(), args.max_iter))
        results.append(run_apollo(case, args.n_draws, args.backend_timeout))
    consistent = compare_results(case, results)
    return payload(case, draws, results, consistent)


def payload(case: MixedCase, draws: torch.Tensor, results: list[BackendResult], consistent: bool) -> dict:
    return {
        "case": case.case,
        "dataset_id": case.dataset_id,
        "source": case.source,
        "model_name": case.model_name,
        "n_obs": case.data.n_obs,
        "n_rows": case.data.n_rows,
        "n_alternatives": len(case.alternatives),
        "n_draws": int(draws.shape[0]),
        "device_policy": "CPU for all cross-estimator comparisons; CUDA only for torch-only profiles.",
        "runtime_policy": runtime_policy_metadata(),
        "parameter_names": case.parameter_names,
        "random_coefficients": case.random_names,
        "sigma_init": case.sigma_init,
        "specification": {
            "alternatives": case.alternatives,
            "availability": case.availability_columns,
            "utilities": {
                alt: [
                    {"parameter": param, "variable": column, "random": param in case.random_names}
                    for param, column in case.utility_terms[alt]
                ]
                for alt in case.alternatives
            },
        },
        "consistent": "Yes" if consistent else "No",
        "backends": [
            {
                "backend": result.backend,
                "available": result.available,
                "total_s": result.total_s,
                "estimate_s": result.estimate_s,
                "covariance_s": result.covariance_s,
                "loglike": result.loglike,
                "ll_diff": getattr(result, "ll_diff", None),
                "max_param_diff": getattr(result, "max_param_diff", None),
                "max_prob_diff": getattr(result, "max_prob_diff", None),
                "params": result.params,
                "message": result.message,
            }
            for result in results
        ],
    }


def long_to_wide(df: pd.DataFrame, variables: list[str]) -> tuple[pd.DataFrame, dict[object, dict[str, str | int]], str]:
    alternatives = list(pd.unique(df["alt"]))
    code_by_alt = {alt: i + 1 for i, alt in enumerate(alternatives)}
    safe_by_alt = {alt: safe_name(str(alt), i) for i, alt in enumerate(alternatives)}
    rows = []
    has_availability = "availability" in df.columns
    for obs_id, group in df.groupby("obs_id", sort=False):
        chosen = group.loc[group["choice"].astype(bool), "alt"]
        if chosen.empty:
            continue
        row = {"obs_id": obs_id, "choice_code": code_by_alt[chosen.iloc[0]]}
        by_alt = {alt: alt_group.iloc[0] for alt, alt_group in group.groupby("alt", sort=False)}
        for alt in alternatives:
            source = by_alt.get(alt)
            safe = safe_by_alt[alt]
            row[f"avail_{safe}"] = int(bool(source["availability"])) if source is not None and has_availability else int(source is not None)
            for variable in variables:
                row[f"{variable}_{safe}"] = float(source[variable]) if source is not None else 0.0
        rows.append(row)
    alt_columns = {
        alt: {
            "code": code_by_alt[alt],
            "availability": f"avail_{safe_by_alt[alt]}",
            **{variable: f"{variable}_{safe_by_alt[alt]}" for variable in variables},
        }
        for alt in alternatives
    }
    return pd.DataFrame(rows), alt_columns, "choice_code"


def safe_name(value: str, index: int) -> str:
    import re

    base = re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_").lower() or f"alt_{index + 1}"
    if base[0].isdigit():
        base = f"alt_{base}"
    return f"{base}_{index + 1}"


def covariance_to_array(covariance_obj, names: list[str]) -> np.ndarray:
    if hasattr(covariance_obj, "loc"):
        return covariance_obj.loc[names, names].to_numpy(dtype=float)
    return np.asarray(covariance_obj, dtype=float)


def render_markdown(rows: list[dict]) -> str:
    lines = [
        "# Real-data Mixed Logit Battery",
        "",
        "All cross-estimator runtimes report estimation plus covariance on one logical CPU. Each runnable model uses 2-4 independent normal random coefficients selected from observed-variable coefficients first, then ASC terms only when needed.",
        "",
        "| case | N | RC | TorchDCM s | Biogeme s | Apollo s | LL diff | Param diff | Prob diff | Consistent? |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        if row.get("status") == "skipped":
            lines.append(f"| {row['case']} | NA | skipped | NA | NA | NA | NA | NA | NA | No |")
            continue
        torch_row = backend(row, "torchdcm")
        biogeme_row = backend(row, "biogeme")
        apollo_row = backend(row, "apollo")
        lines.append(
            "| {case} | {n} | {rc} | {torch_s} | {bio_s} | {apollo_s} | {ll} | {pd} | {prob} | {ok} |".format(
                case=row["case"],
                n=row["n_obs"],
                rc=", ".join(row["random_coefficients"]),
                torch_s=fmt(torch_row.get("total_s")),
                bio_s=fmt(biogeme_row.get("total_s")),
                apollo_s=fmt(apollo_row.get("total_s")) if apollo_row.get("available") else "NA",
                ll=sci(biogeme_row.get("ll_diff")),
                pd=sci(biogeme_row.get("max_param_diff")),
                prob=sci(biogeme_row.get("max_prob_diff")),
                ok=row["consistent"],
            )
        )
    lines.extend(["", "## Specifications", ""])
    for row in rows:
        if row.get("status") == "skipped":
            lines.append(f"- `{row['case']}` skipped: {row.get('message')}")
            continue
        lines.append(f"- `{row['case']}`: random coefficients = {', '.join(row['random_coefficients'])}; parameters = {', '.join(row['parameter_names'])}.")
    return "\n".join(lines)


def backend(row: dict, name: str) -> dict:
    return next((item for item in row.get("backends", []) if item["backend"] == name), {"available": False})


def fmt(value) -> str:
    return f"{float(value):.3f}" if isinstance(value, (int, float)) else "NA"


def sci(value) -> str:
    return f"{float(value):.2e}" if isinstance(value, (int, float)) else "NA"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["swissmetro", "airline", "parking", "telephone", "lpmc", *[f"mlogit_{name}" for name in MLOGIT_DATASETS]],
    )
    parser.add_argument("--n-obs", type=int, default=None)
    parser.add_argument("--n-draws", type=int, default=32)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--backend-timeout", type=float, default=300.0)
    parser.add_argument("--torch-device", default="cpu")
    parser.add_argument("--torch-only", action="store_true")
    parser.add_argument("--profile", default="full")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--md-output", type=Path)
    args = parser.parse_args()
    if not args.torch_only and torch.device(args.torch_device).type != "cpu":
        raise ValueError("Cross-estimator mixed-logit batteries must use --torch-device cpu.")

    GENERATED.mkdir(parents=True, exist_ok=True)
    json_path = args.json_output or GENERATED / f"mixed_real_battery_{args.profile}.json"
    md_path = args.md_output or GENERATED / f"mixed_real_battery_{args.profile}.md"
    rows = []
    for case_or_skip in build_cases(args.datasets, args.n_obs, args.sigma):
        if isinstance(case_or_skip, dict):
            rows.append(case_or_skip)
            print(f"{case_or_skip['case']}: skipped {case_or_skip.get('message')}", flush=True)
            json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            md_path.write_text(render_markdown(rows), encoding="utf-8")
            continue
        print(f"{case_or_skip.case}: RC={case_or_skip.random_names}", flush=True)
        row = run_case(case_or_skip, args)
        rows.append(row)
        print(
            f"{row['case']}: torch={fmt(backend(row, 'torchdcm').get('total_s'))} "
            f"biogeme={fmt(backend(row, 'biogeme').get('total_s'))} "
            f"apollo={fmt(backend(row, 'apollo').get('total_s'))} consistent={row['consistent']}",
            flush=True,
        )
        json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        md_path.write_text(render_markdown(rows), encoding="utf-8")
    print(f"json: {json_path}", flush=True)
    print(f"markdown: {md_path}", flush=True)


if __name__ == "__main__":
    main()
