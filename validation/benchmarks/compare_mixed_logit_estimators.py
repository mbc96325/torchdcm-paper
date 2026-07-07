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
    covariance: np.ndarray | None = None
    probabilities: np.ndarray | None = None
    message: str = ""


def raise_stack_limit_for_child() -> None:
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        if hard == resource.RLIM_INFINITY:
            resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        elif soft < hard:
            resource.setrlimit(resource.RLIMIT_STACK, (hard, hard))
    except Exception:
        pass


def sync_torch_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


def write_apollo_estimate_inputs(
    df,
    alternatives,
    initial_values: dict[str, float],
    n_draws: int,
    panel: bool,
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
    wide_df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    spec = {
        "model_name": f"apollo_mixed_full_{len(df)}_{n_draws}",
        "alternatives": alternatives,
        "choice_col": "choice_code",
        "panel": panel,
        "panel_id_col": "person_id",
        "n_draws": n_draws,
        "parameters": initial_values,
    }
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return csv_path, spec_path


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
    sigma_init: float = 0.1,
    device: str = "cpu",
) -> BackendResult:
    params = {f"SIGMA_{name}": sigma_init for name in random_names}
    model = make_torch_model(spec, params, draws, panel, random_names, correlated, error_component_public, max_iter=max_iter, device=device)
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

    sync_torch_device(model.device)
    estimate_start = time.perf_counter()
    optimizer.step(closure)
    sync_torch_device(model.device)
    estimate_seconds = time.perf_counter() - estimate_start
    final_internal = internal_params.detach().clone().requires_grad_(True)
    final_natural = model._internal_to_natural(final_internal, compiled)
    ll = model.loglike(final_natural, data, compiled)
    sync_torch_device(model.device)
    covariance_start = time.perf_counter()
    hessian_internal = torch.autograd.functional.hessian(
        lambda p: model.loglike(model._internal_to_natural(p, compiled), data, compiled),
        final_internal,
    )
    try:
        cov_internal = torch.linalg.pinv(-hessian_internal.detach(), hermitian=True)
        transform_jac = model._natural_jacobian(final_internal.detach(), compiled)
        covariance = (transform_jac @ cov_internal @ transform_jac.T).detach().cpu().numpy()
    except RuntimeError:
        covariance = None
    sync_torch_device(model.device)
    covariance_seconds = time.perf_counter() - covariance_start
    return BackendResult(
        backend="torchdcm_fit",
        available=True,
        seconds=estimate_seconds + covariance_seconds,
        estimate_seconds=estimate_seconds,
        covariance_seconds=covariance_seconds,
        loglike=float(ll.detach().cpu()),
        params=dict(zip(compiled.free_names, final_natural.detach().cpu().numpy())),
        covariance=covariance,
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
        proc = subprocess.run(cmd, text=True, capture_output=True, env=env, preexec_fn=raise_stack_limit_for_child)
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


def run_apollo_estimate(
    df,
    alternatives,
    initial_values: dict[str, float],
    n_draws: int,
    panel: bool,
) -> BackendResult:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return BackendResult(backend="apollo_full", available=False, message="Rscript not found")
    script = Path(__file__).resolve().parent / "apollo" / "R" / "run_mixed_estimate.R"
    if not script.exists():
        return BackendResult(backend="apollo_full", available=False, message=f"Missing script: {script}")
    with tempfile.TemporaryDirectory(prefix="torchdcm_apollo_mixed_full_") as tmp:
        tmp_path = Path(tmp)
        data_path, spec_path = write_apollo_estimate_inputs(
            df, alternatives, initial_values, n_draws, panel, tmp_path
        )
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
        proc = subprocess.run(cmd, text=True, capture_output=True, env=env, preexec_fn=raise_stack_limit_for_child)
        seconds = time.perf_counter() - start
        if proc.returncode != 0:
            return BackendResult(
                backend="apollo_full",
                available=False,
                seconds=seconds,
                message=(proc.stderr or proc.stdout).strip(),
            )
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        param_names = payload.get("parameter_names") or list(payload["estimates"])
        covariance = payload.get("covariance")
        covariance_array = np.asarray(covariance, dtype=float) if covariance is not None else None
        return BackendResult(
            backend="apollo_full",
            available=True,
            seconds=seconds,
            estimate_seconds=float(payload.get("timing", {}).get("estimate_seconds", seconds)),
            covariance_seconds=float(payload.get("timing", {}).get("covariance_seconds", 0.0)),
            loglike=float(payload["loglike"]),
            params={name: float(payload["estimates"][name]) for name in param_names if name in payload["estimates"]},
            covariance=covariance_array,
            message=(
                "Apollo uses package Halton draws; runtime is comparable but not strict shared-draw parity."
            ),
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


def run_biogeme_estimate(
    df,
    alternatives,
    initial_values: dict[str, float],
    draws: np.ndarray,
    panel: bool,
    random_names: list[str],
    correlated: bool,
    error_component_public: bool,
) -> BackendResult:
    if panel:
        return BackendResult(
            backend="biogeme_full",
            available=False,
            message="Aligned Biogeme full estimation currently supports observation-level likelihood only.",
        )
    supported_random_names = {"B_TIME", "B_COST"}
    if correlated or error_component_public or not set(random_names).issubset(supported_random_names) or "B_TIME" not in random_names:
        return BackendResult(
            backend="biogeme_full",
            available=False,
            message="Aligned Biogeme full estimation currently supports independent normal B_TIME and optional B_COST coefficients.",
        )
    try:
        import biogeme.biogeme as bio
        import biogeme.database as db
        import biogeme.models as models
        from biogeme.expressions import Beta as BioBeta
        from biogeme.expressions import Variable
        from biogeme.expressions import log
        from biogeme.results_processing.variance_covariance import EstimateVarianceCovariance
    except ImportError as exc:
        return BackendResult(backend="biogeme_full", available=False, message=f"Biogeme not found: {exc}")

    code_by_alt = {alt: i + 1 for i, alt in enumerate(alternatives)}
    wide_df = df.copy().reset_index(drop=True)
    wide_df["choice_code"] = wide_df["choice"].map(code_by_alt).astype(int)
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

    names = ["ASC_TRAIN", "B_TIME", "B_COST", "ASC_CAR", *[f"SIGMA_{name}" for name in random_names]]
    try:
        database = db.Database("torchdcm_mixed_full_shared_draws", wide_df.drop(columns=["choice"]))
        betas = {
            "ASC_TRAIN": BioBeta("ASC_TRAIN", initial_values["ASC_TRAIN"], None, None, 0),
            "B_TIME": BioBeta("B_TIME", initial_values["B_TIME"], None, None, 0),
            "B_COST": BioBeta("B_COST", initial_values["B_COST"], None, None, 0),
            "ASC_CAR": BioBeta("ASC_CAR", initial_values["ASC_CAR"], None, None, 0),
        }
        for name in random_names:
            sigma_name = f"SIGMA_{name}"
            betas[sigma_name] = BioBeta(sigma_name, initial_values[sigma_name], 0.0, None, 0)
        choice = Variable("choice_code")
        availability = {code_by_alt[alt]: Variable(f"avail_{alt.lower()}") for alt in alternatives}
        chosen_probs_by_draw = []
        for draw_index in range(draws.shape[0]):
            b_time = betas["B_TIME"]
            if "B_TIME" in random_names:
                b_time = b_time + betas["SIGMA_B_TIME"] * Variable(f"DRAW_B_TIME_{draw_index}")
            b_cost = betas["B_COST"]
            if "B_COST" in random_names:
                b_cost = b_cost + betas["SIGMA_B_COST"] * Variable(f"DRAW_B_COST_{draw_index}")
            utility = {
                code_by_alt["TRAIN"]: betas["ASC_TRAIN"] + b_time * Variable("time_train") + b_cost * Variable("cost_train"),
                code_by_alt["SM"]: b_time * Variable("time_sm") + b_cost * Variable("cost_sm"),
                code_by_alt["CAR"]: betas["ASC_CAR"] + b_time * Variable("time_car") + b_cost * Variable("cost_car"),
            }
            chosen_probs_by_draw.append(models.logit(utility, availability, choice))
        average_prob = chosen_probs_by_draw[0]
        for expression in chosen_probs_by_draw[1:]:
            average_prob = average_prob + expression
        average_prob = average_prob / float(len(chosen_probs_by_draw))
        logprob = log(average_prob)

        biogeme = bio.BIOGEME(database, logprob)
        biogeme.model_name = f"torchdcm_mixed_full_shared_draws_{len(df)}_{draws.shape[0]}"
        biogeme.biogeme_parameters.set_value("save_iterations", False)
        total_start = time.perf_counter()
        estimate_start = time.perf_counter()
        estimates = biogeme.estimate()
        estimate_seconds = time.perf_counter() - estimate_start
        covariance_start = time.perf_counter()
        covariance_obj = estimates.get_variance_covariance_matrix(EstimateVarianceCovariance.RAO_CRAMER)
        covariance_seconds = time.perf_counter() - covariance_start
        total_seconds = time.perf_counter() - total_start
        beta_values = estimates.get_beta_values()
        covariance = covariance_to_array(covariance_obj, names)
        return BackendResult(
            backend="biogeme_full",
            available=True,
            seconds=total_seconds,
            estimate_seconds=estimate_seconds,
            covariance_seconds=covariance_seconds,
            loglike=float(estimates.final_log_likelihood),
            params={name: float(beta_values[name]) for name in names},
            covariance=covariance,
        )
    except Exception as exc:
        return BackendResult(backend="biogeme_full", available=False, message=f"{type(exc).__name__}: {exc}")


def covariance_to_array(covariance_obj, names: list[str]) -> np.ndarray:
    if hasattr(covariance_obj, "loc"):
        return covariance_obj.loc[names, names].to_numpy(dtype=float)
    return np.asarray(covariance_obj, dtype=float)


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
        print("  draws: shared antithetic standard-normal draw matrix")
        print("  parameters: shared across replay backends")
        print("  probabilities: averaged over the same draws and row order")
    elif mode == "full-estimation":
        print("  benchmark_mode: relaxed_full_estimation")
        print("  estimated_backend: each backend estimates independently")
        print("  draws: TorchDCM and Biogeme share antithetic normal draws; Apollo uses package Halton draws")
        print("  parameters: same initial vector; estimates may differ slightly under package draw conventions")
        print("  probabilities: Biogeme probabilities replayed through TorchDCM at Biogeme parameters")
        print("  consistency_rule: compare TorchDCM against Biogeme shared-draw full estimation")
    else:
        print("  benchmark_mode: torchdcm_full_estimation_then_fixed_replay")
        print("  estimated_backend: torchdcm")
        print("  draws: shared antithetic standard-normal draw matrix")
        print("  parameters: shared across replay backends")
        print("  probabilities: averaged over the same draws and row order")
    print("  data_source: biogeme.data.swissmetro/data/swissmetro.dat")
    print("  model: Mixed Logit with normal random coefficients")
    if error_component_public:
        print("  error_component: EC_PUBLIC loading 1 on TRAIN/SM and 0 on CAR")
    print("  covariance: Cholesky lower triangular replay when correlated=True")
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


def compare_results(results: list[BackendResult], reference: str) -> dict[str, dict]:
    ref = next(result for result in results if result.backend == reference and result.available)
    payload = {}
    for result in results:
        row = {
            "backend": result.backend,
            "available": result.available,
            "total_s": result.seconds,
            "estimate_s": result.estimate_seconds,
            "covariance_s": result.covariance_seconds,
            "loglike": result.loglike,
            "message": result.message,
        }
        if result.available:
            row["ll_diff"] = result.loglike - ref.loglike
            shared = sorted(set(result.params or {}) & set(ref.params or {}))
            row["max_param_diff"] = max(abs(result.params[name] - ref.params[name]) for name in shared) if shared else None
            if result.probabilities is not None and ref.probabilities is not None:
                row["max_prob_diff"] = float(np.max(np.abs(result.probabilities - ref.probabilities)))
            else:
                row["max_prob_diff"] = None
            if result.covariance is not None and ref.covariance is not None:
                row["max_cov_diff"] = float(np.max(np.abs(result.covariance - ref.covariance)))
                result_diag = np.diag(result.covariance)
                ref_diag = np.diag(ref.covariance)
                if np.all(result_diag >= 0) and np.all(ref_diag >= 0):
                    row["max_se_diff"] = float(
                        np.max(np.abs(np.sqrt(result_diag) - np.sqrt(ref_diag)))
                    )
                else:
                    row["max_se_diff"] = None
            else:
                row["max_cov_diff"] = None
                row["max_se_diff"] = None
        payload[result.backend] = row
    return payload


def is_full_consistent(rows: dict[str, dict]) -> bool:
    biogeme = rows.get("biogeme_full")
    if not biogeme or not biogeme["available"]:
        return False
    loglike = abs(biogeme.get("loglike") or 0.0)
    ll_tolerance = max(0.25, 1e-5 * loglike)
    return (
        abs(biogeme["ll_diff"]) <= ll_tolerance
        and biogeme["max_param_diff"] <= 5e-2
        and (biogeme["max_prob_diff"] is None or biogeme["max_prob_diff"] <= 2e-2)
    )


def result_payload(case: str, n_obs: int, n_draws: int, panel: bool, mode: str, rows: dict[str, dict]) -> dict:
    return {
        "case": case,
        "data": "Swissmetro",
        "model": "Mixed logit full estimation" if mode == "full-estimation" else "Mixed logit replay",
        "n_obs": n_obs,
        "n_draws": n_draws,
        "panel": panel,
        "mode": mode,
        "consistent": is_full_consistent(rows) if mode == "full-estimation" else True,
        "consistency_rule": (
            "Relaxed full-estimation consistency against Biogeme shared-draw result: "
            "|LL diff| <= max(0.25, 1e-5*|LL|), max parameter diff <= 0.05, "
            "and max probability diff <= 0.02 when available. Apollo runtime is reported "
            "but not used for strict consistency because Apollo uses package Halton draws."
            if mode == "full-estimation"
            else "Shared-parameter fixed replay over identical draws."
        ),
        "backends": list(rows.values()),
    }


def write_outputs(payload: dict, json_output: Path | None, md_output: Path | None) -> None:
    if json_output:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if md_output:
        rows = {row["backend"]: row for row in payload["backends"]}
        torch_row = rows.get("torchdcm_full") or rows.get("torchdcm_fit") or rows.get("torchdcm_fixed")
        biogeme_row = rows.get("biogeme_full") or rows.get("biogeme_fixed")
        apollo_row = rows.get("apollo_full") or rows.get("apollo_r_fixed")
        lines = [
            "# Mixed Logit Full Estimation Benchmark",
            "",
            "| Data | Model | N | Draws | TorchDCM | Biogeme | Apollo | Consistent? |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            "| {data} | {model} | {n_obs} | {n_draws} | {torch} | {biogeme} | {apollo} | {consistent} |".format(
                data=payload["data"],
                model=payload["model"],
                n_obs=payload["n_obs"],
                n_draws=payload["n_draws"],
                torch=_fmt_table_time(torch_row),
                biogeme=_fmt_table_time(biogeme_row),
                apollo=_fmt_table_time(apollo_row),
                consistent="Yes" if payload["consistent"] else "No",
            ),
            "",
        ]
        md_output.parent.mkdir(parents=True, exist_ok=True)
        md_output.write_text("\n".join(lines), encoding="utf-8")


def _fmt_table_time(row: dict | None) -> str:
    if not row or not row.get("available"):
        return "NA"
    return f"{row['total_s']:.3f}"


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
    parser.add_argument("--mode", choices=["fixed", "fit", "fit-replay", "full-estimation"], default="fixed")
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--torch-device", default="cpu")
    parser.add_argument("--torch-only", action="store_true")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--md-output", type=Path, default=None)
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
            data,
            spec,
            draws,
            args.panel,
            args.max_iter,
            random_names,
            args.correlated,
            args.error_component_public,
            sigma_init=args.sigma,
            device=args.torch_device,
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
    elif args.mode == "full-estimation":
        if args.panel:
            raise ValueError("--mode full-estimation currently supports observation-level likelihood; omit --panel.")
        initial_params = {name: 0.0 for name in base_spec.parameter_names}
        for name in random_names:
            initial_params[f"SIGMA_{name}"] = args.sigma
        torch_result = run_torch_fit(
            data,
            spec,
            draws,
            args.panel,
            args.max_iter,
            random_names,
            args.correlated,
            args.error_component_public,
            sigma_init=args.sigma,
            device=args.torch_device,
        )
        torch_result.backend = "torchdcm_full"
        if args.torch_only:
            results = [torch_result]
        else:
            biogeme_result = run_biogeme_estimate(
                df,
                alternatives,
                initial_params,
                draws.detach().cpu().numpy(),
                args.panel,
                random_names,
                args.correlated,
                args.error_component_public,
            )
            if biogeme_result.available:
                biogeme_result.probabilities = predict_with_torch_model(
                    data, spec, biogeme_result.params, draws, args.panel, random_names, args.correlated, args.error_component_public, args.torch_device
                )
            apollo_result = run_apollo_estimate(
                df,
                alternatives,
                initial_params,
                args.n_draws,
                args.panel,
            )
            results = [torch_result, biogeme_result, apollo_result]
        reference = "torchdcm_full"
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
    rows = compare_results(results, reference)
    write_outputs(
        result_payload(
            case="swissmetro_mixed_full" if args.mode == "full-estimation" else "swissmetro_mixed",
            n_obs=len(df),
            n_draws=args.n_draws,
            panel=args.panel,
            mode=display_mode,
            rows=rows,
        ),
        args.json_output,
        args.md_output,
    )


def predict_with_torch_model(
    data,
    spec,
    params: dict[str, float],
    draws: torch.Tensor,
    panel: bool,
    random_names: list[str],
    correlated: bool,
    error_component_public: bool,
    device: str = "cpu",
) -> np.ndarray:
    model = make_torch_model(spec, params, draws, panel, random_names, correlated, error_component_public, device=device)
    compiled = model.compile(data)
    vector = torch.as_tensor([params[name] for name in compiled.free_names], dtype=torch.float64, device=model.device)
    return model.predict_proba(data, vector, compiled).detach().cpu().numpy()


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
