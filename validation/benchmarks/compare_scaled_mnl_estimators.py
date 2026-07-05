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

from torchdcm import AlternativeScale, ScaledMultinomialLogit
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


def swissmetro_scales() -> dict[str, AlternativeScale]:
    return {
        "TRAIN": AlternativeScale(init=0.8),
        "SM": AlternativeScale(init=1.0, fixed=True),
        "CAR": AlternativeScale(init=1.2),
    }


def default_params(names: list[str]) -> dict[str, float]:
    values = {
        "ASC_TRAIN": 0.3,
        "B_TIME": -1.0,
        "B_COST": -1.2,
        "ASC_CAR": 0.6,
        "SCALE_TRAIN": 0.8,
        "SCALE_CAR": 1.2,
    }
    return {name: values[name] for name in names}


def run_torch_fixed(data, spec, scales, params: dict[str, float]) -> BackendResult:
    model = ScaledMultinomialLogit(spec, scales)
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


def run_torch_fit(data, spec, scales, max_iter: int) -> BackendResult:
    model = ScaledMultinomialLogit(spec, scales, max_iter=max_iter)
    data = data.to(device=model.device, dtype=model.dtype)
    compiled = model.compile(data)
    internal_initial = torch.cat(
        [
            compiled.free_initial,
            model._scale_to_internal(compiled.scale_initial[~compiled.scale_is_fixed]),
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
        "model_name": f"scaled_mnl_fixed_{len(df)}",
        "alternatives": alternatives,
        "choice_col": "choice_code",
        "parameters": params,
        "scales": {"TRAIN": "SCALE_TRAIN", "SM": 1.0, "CAR": "SCALE_CAR"},
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
    return csv_path, spec_path


def run_apollo_fixed(df, alternatives, params: dict[str, float]) -> BackendResult:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return BackendResult(backend="apollo_r_fixed", available=False, message="Rscript not found")
    script = Path(__file__).resolve().parent / "apollo" / "R" / "run_scaled_mnl_fixed.R"
    if not script.exists():
        return BackendResult(backend="apollo_r_fixed", available=False, message=f"Missing script: {script}")
    with tempfile.TemporaryDirectory(prefix="torchdcm_apollo_scaled_") as tmp:
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
        import biogeme.models as models
        from biogeme.expressions import Variable
    except ImportError as exc:
        return BackendResult(backend="biogeme_fixed", available=False, message=f"Biogeme not found: {exc}")

    code_by_alt = {alt: i + 1 for i, alt in enumerate(alternatives)}
    wide_df = df.copy().reset_index(drop=True)
    wide_df["choice_code"] = wide_df["choice"].map(code_by_alt).astype(int)
    for alt in alternatives:
        wide_df[f"avail_{alt.lower()}"] = wide_df[f"avail_{alt.lower()}"].astype(int)
    scale_values = {"TRAIN": params["SCALE_TRAIN"], "SM": 1.0, "CAR": params["SCALE_CAR"]}

    try:
        database = db.Database("torchdcm_scaled_mnl_fixed", wide_df.drop(columns=["choice"]))
        choice = Variable("choice_code")
        av = {code_by_alt[alt]: Variable(f"avail_{alt.lower()}") for alt in alternatives}
        utility = {}
        for alt in alternatives:
            raw = params.get(f"ASC_{alt.upper()}", 0.0)
            raw = raw + params["B_TIME"] * Variable(f"time_{alt.lower()}")
            raw = raw + params["B_COST"] * Variable(f"cost_{alt.lower()}")
            utility[code_by_alt[alt]] = raw / float(scale_values[alt])
        formulas = {"chosen_prob": models.logit(utility, av, choice)}
        for alt in alternatives:
            formulas[f"prob_{alt.lower()}"] = models.logit(utility, av, code_by_alt[alt])
        start = time.perf_counter()
        biogeme = bio.BIOGEME(database, formulas)
        biogeme.model_name = "torchdcm_scaled_mnl_fixed"
        simulated = biogeme.simulate({})
        seconds = time.perf_counter() - start
    except Exception as exc:
        return BackendResult(backend="biogeme_fixed", available=False, message=f"{type(exc).__name__}: {exc}")

    chosen = simulated["chosen_prob"].clip(lower=np.finfo(float).tiny).to_numpy(dtype=float)
    probabilities = np.column_stack([simulated[f"prob_{alt.lower()}"].to_numpy(dtype=float) for alt in alternatives]).reshape(-1)
    return BackendResult(
        backend="biogeme_fixed",
        available=True,
        seconds=seconds,
        loglike=float(np.log(chosen).sum()),
        params=params,
        probabilities=probabilities,
    )


def print_results(results: list[BackendResult], reference: str, n_obs: int, mode: str):
    ref = next(result for result in results if result.backend == reference and result.available)
    print("case: biogeme_swissmetro_scaled_mnl")
    print(f"mode: {mode}")
    print(f"n_obs: {n_obs}")
    print("alignment:")
    if mode == "fixed":
        print("  benchmark_mode: fixed_likelihood_replay")
        print("  estimated_backend: none")
    else:
        print("  benchmark_mode: torchdcm_full_estimation_then_fixed_replay")
        print("  estimated_backend: torchdcm")
    print("  data_source: biogeme.data.swissmetro/data/swissmetro.dat")
    print("  model: Scaled MNL with alternative-specific utility scales")
    print("  identification: SCALE_SM fixed to 1")
    print("  parameters: shared across replay backends")
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
        prob_diff = None
        if result.probabilities is not None and ref.probabilities is not None:
            prob_diff = float(np.max(np.abs(result.probabilities - ref.probabilities)))
        print(
            f"{result.backend:<18}{str(result.available):>10}"
            f"{_fmt_seconds(result.seconds):>12}"
            f"{_fmt_seconds(result.estimate_seconds):>12}"
            f"{_fmt_seconds(result.covariance_seconds):>12}"
            f"{result.loglike:>18.10f}"
            f"{result.loglike - ref.loglike:>14.3e}"
            f"{_fmt_optional(prob_diff):>14}"
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
    parser.add_argument("--mode", choices=["fixed", "fit-replay"], default="fixed")
    parser.add_argument("--max-iter", type=int, default=40)
    args = parser.parse_args()

    df, data, base_spec, alternatives = load_biogeme_swissmetro(args.n_obs)
    initial_values = make_initial_values(base_spec.parameter_names, mode="zero", seed=20260704, scale=0.1)
    spec = spec_with_initials(base_spec, initial_values)
    scales = swissmetro_scales()
    names = [*base_spec.parameter_names, "SCALE_TRAIN", "SCALE_CAR"]

    if args.mode == "fixed":
        params = default_params(names)
        torch_result = run_torch_fixed(data, spec, scales, params)
        apollo_result = run_apollo_fixed(df, alternatives, params)
        biogeme_result = run_biogeme_fixed(df, alternatives, params)
        results = [torch_result, apollo_result, biogeme_result]
        reference = "torchdcm_fixed"
    else:
        torch_result = run_torch_fit(data, spec, scales, args.max_iter)
        params = torch_result.params
        apollo_result = run_apollo_fixed(df, alternatives, params)
        biogeme_result = run_biogeme_fixed(df, alternatives, params)
        results = [torch_result, apollo_result, biogeme_result]
        reference = "torchdcm_fit"
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
