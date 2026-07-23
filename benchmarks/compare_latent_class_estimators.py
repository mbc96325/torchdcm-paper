from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from torchdcm import Beta, ChoiceDataset, LatentClassLogit, UtilitySpec
from compare_mnl_estimators import load_biogeme_swissmetro


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
    class_probabilities: np.ndarray | None = None
    message: str = ""


def make_two_class_specs(membership: str) -> tuple[list[UtilitySpec], list]:
    specs = []
    for suffix in ("C1", "C2"):
        b_time = Beta(f"B_TIME_{suffix}", init=-0.01)
        b_cost = Beta(f"B_COST_{suffix}", init=-0.1)
        asc_train = Beta(f"ASC_TRAIN_{suffix}")
        asc_car = Beta(f"ASC_CAR_{suffix}")
        spec = UtilitySpec()
        spec.utility("TRAIN", asc_train + b_time * "time" + b_cost * "cost")
        spec.utility("SM", b_time * "time" + b_cost * "cost")
        spec.utility("CAR", asc_car + b_time * "time" + b_cost * "cost")
        specs.append(spec)
    if membership == "constant":
        return specs, [Beta("CLASS_2", init=0.0)]
    if membership == "ga":
        return specs, [Beta("CLASS_2", init=0.0) + Beta("CLASS_2_GA", init=0.0) * "ga"]
    raise ValueError(f"Unknown membership specification: {membership}")


def default_params(membership: str) -> dict[str, float]:
    params = {
        "ASC_TRAIN_C1": 0.3,
        "B_TIME_C1": -1.0,
        "B_COST_C1": -1.2,
        "ASC_CAR_C1": 0.6,
        "ASC_TRAIN_C2": -0.2,
        "B_TIME_C2": -0.4,
        "B_COST_C2": -0.8,
        "ASC_CAR_C2": 0.1,
        "CLASS_2": -0.3,
    }
    if membership == "ga":
        params["CLASS_2_GA"] = 0.7
    return params


def class_probs_from_params(params: dict[str, float], df) -> np.ndarray:
    class_2 = np.full(len(df), params["CLASS_2"], dtype=float)
    if "CLASS_2_GA" in params:
        class_2 = class_2 + params["CLASS_2_GA"] * df["GA"].to_numpy(dtype=float)
    logits = np.column_stack([np.zeros(len(df), dtype=float), class_2])
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def data_with_membership_vars(df, alternatives):
    return ChoiceDataset.from_wide(
        df,
        alternatives=alternatives,
        choice="choice",
        variables={
            "time": {"TRAIN": "time_train", "SM": "time_sm", "CAR": "time_car"},
            "cost": {"TRAIN": "cost_train", "SM": "cost_sm", "CAR": "cost_car"},
            "ga": {alt: "GA" for alt in alternatives},
        },
        availability={"TRAIN": "avail_train", "SM": "avail_sm", "CAR": "avail_car"},
        obs_id="obs_id",
        individual_id="ID",
    )


def per_obs_class_probabilities(values: np.ndarray, n_obs: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1 and len(values) == 2:
        return np.tile(values, (n_obs, 1))
    return values.reshape(n_obs, 2)


def run_torch_fixed(data, specs, membership, params: dict[str, float]) -> BackendResult:
    model = LatentClassLogit(specs, class_membership=membership)
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
        class_probabilities=per_obs_class_probabilities(model.class_probabilities(vector, data).detach().cpu().numpy(), data.n_obs),
    )


def run_torch_fit(data, specs, membership, max_iter: int) -> BackendResult:
    model = LatentClassLogit(specs, class_membership=membership, max_iter=max_iter)
    data = data.to(device=model.device, dtype=model.dtype)
    compiled = model.compile(data)
    params = compiled.free_initial.clone().detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [params],
        max_iter=max_iter,
        tolerance_grad=model.tolerance_grad,
        line_search_fn=model.line_search_fn,
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = -model.loglike(params, data, compiled)
        loss.backward()
        return loss

    estimate_start = time.perf_counter()
    optimizer.step(closure)
    estimate_seconds = time.perf_counter() - estimate_start
    final_params = params.detach().clone().requires_grad_(True)
    ll = model.loglike(final_params, data, compiled)
    covariance_start = time.perf_counter()
    _ = torch.linalg.pinv(
        -torch.autograd.functional.hessian(lambda p: model.loglike(p, data, compiled), final_params).detach(),
        hermitian=True,
    )
    covariance_seconds = time.perf_counter() - covariance_start
    return BackendResult(
        backend="torchdcm_fit",
        available=True,
        seconds=estimate_seconds + covariance_seconds,
        estimate_seconds=estimate_seconds,
        covariance_seconds=covariance_seconds,
        loglike=float(ll.detach().cpu()),
        params=dict(zip(compiled.free_names, final_params.detach().cpu().numpy())),
        probabilities=model.predict_proba(data, final_params.detach(), compiled).detach().cpu().numpy(),
        class_probabilities=per_obs_class_probabilities(model.class_probabilities(final_params.detach(), data).detach().cpu().numpy(), data.n_obs),
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
        "model_name": f"apollo_latent_class_fixed_{len(df)}",
        "alternatives": alternatives,
        "choice_col": "choice_code",
        "parameters": params,
        "availability": {alt: f"avail_{alt.lower()}" for alt in alternatives},
        "classes": [
            {
                "name": "class_1",
                "membership_terms": [],
                "utility": _class_utility_spec("C1", alternatives),
            },
            {
                "name": "class_2",
                "membership_terms": _membership_terms(params),
                "utility": _class_utility_spec("C2", alternatives),
            },
        ],
    }
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return csv_path, spec_path


def _membership_terms(params: dict[str, float]) -> list[dict[str, str | None]]:
    terms = [{"param": "CLASS_2", "variable": None}]
    if "CLASS_2_GA" in params:
        terms.append({"param": "CLASS_2_GA", "variable": "GA"})
    return terms


def _class_utility_spec(suffix: str, alternatives: list[str]) -> dict:
    return {
        alt: {
            "asc": f"ASC_{alt.upper()}_{suffix}" if alt != "SM" else None,
            "time_param": f"B_TIME_{suffix}",
            "cost_param": f"B_COST_{suffix}",
            "time": f"time_{alt.lower()}",
            "cost": f"cost_{alt.lower()}",
        }
        for alt in alternatives
    }


def run_apollo_fixed(df, alternatives, params: dict[str, float]) -> BackendResult:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return BackendResult(backend="apollo_r_fixed", available=False, message="Rscript not found")
    script = Path(__file__).resolve().parent / "apollo" / "R" / "run_latent_class_fixed.R"
    if not script.exists():
        return BackendResult(backend="apollo_r_fixed", available=False, message=f"Missing script: {script}")
    with tempfile.TemporaryDirectory(prefix="torchdcm_apollo_lc_") as tmp:
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
            class_probabilities=np.asarray(payload["class_probabilities"], dtype=float).reshape(len(df), 2),
        )


def run_biogeme_fixed(df, alternatives, params: dict[str, float]) -> BackendResult:
    try:
        import biogeme.biogeme as bio
        import biogeme.database as db
        import biogeme.models as models
        from biogeme.expressions import Variable, exp
    except ImportError as exc:
        return BackendResult(backend="biogeme_fixed", available=False, message=f"Biogeme not found: {exc}")

    code_by_alt = {alt: i + 1 for i, alt in enumerate(alternatives)}
    wide_df = df.copy().reset_index(drop=True)
    wide_df["choice_code"] = wide_df["choice"].map(code_by_alt).astype(int)
    for alt in alternatives:
        wide_df[f"avail_{alt.lower()}"] = wide_df[f"avail_{alt.lower()}"].astype(int)

    try:
        database = db.Database("torchdcm_latent_class_fixed", wide_df.drop(columns=["choice"]))
        choice = Variable("choice_code")
        av = {code_by_alt[alt]: Variable(f"avail_{alt.lower()}") for alt in alternatives}
        class_2_logit = params["CLASS_2"]
        if "CLASS_2_GA" in params:
            class_2_logit = class_2_logit + params["CLASS_2_GA"] * Variable("GA")
        class_2_exp = exp(class_2_logit)
        class_prob_exprs = [1 / (1 + class_2_exp), class_2_exp / (1 + class_2_exp)]
        chosen_prob = 0
        alt_probs = {alt: 0 for alt in alternatives}
        for class_index, suffix in enumerate(("C1", "C2")):
            v = {
                code_by_alt["TRAIN"]: params[f"ASC_TRAIN_{suffix}"]
                + params[f"B_TIME_{suffix}"] * Variable("time_train")
                + params[f"B_COST_{suffix}"] * Variable("cost_train"),
                code_by_alt["SM"]: params[f"B_TIME_{suffix}"] * Variable("time_sm")
                + params[f"B_COST_{suffix}"] * Variable("cost_sm"),
                code_by_alt["CAR"]: params[f"ASC_CAR_{suffix}"]
                + params[f"B_TIME_{suffix}"] * Variable("time_car")
                + params[f"B_COST_{suffix}"] * Variable("cost_car"),
            }
            weight = class_prob_exprs[class_index]
            chosen_prob = chosen_prob + weight * models.logit(v, av, choice)
            for alt in alternatives:
                alt_probs[alt] = alt_probs[alt] + weight * models.logit(v, av, code_by_alt[alt])
        formulas = {
            "chosen_prob": chosen_prob,
            "class_prob_1": class_prob_exprs[0],
            "class_prob_2": class_prob_exprs[1],
        } | {f"prob_{alt.lower()}": expr for alt, expr in alt_probs.items()}
        start = time.perf_counter()
        biogeme = bio.BIOGEME(database, formulas)
        biogeme.model_name = "torchdcm_latent_class_fixed"
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
        class_probabilities=np.column_stack(
            [
                simulated["class_prob_1"].to_numpy(dtype=float),
                simulated["class_prob_2"].to_numpy(dtype=float),
            ]
        ),
    )


def print_results(results: list[BackendResult], reference: str, n_obs: int, mode: str):
    ref = next(result for result in results if result.backend == reference and result.available)
    print("case: biogeme_swissmetro_latent_class")
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
    print("  model: 2-class latent class logit with class-specific MNL utilities")
    print("  class_membership: class 1 reference; supports constant or GA covariate allocation")
    print("  parameters: shared across replay backends")
    print(f"  reference: {reference}")
    print()
    print(
        f"{'backend':<18}{'available':>10}{'total_s':>12}{'estimate_s':>12}{'cov_s':>12}"
        f"{'loglike':>18}{'ll_diff':>14}{'prob_diff':>14}{'class_prob_diff':>18}"
    )
    for result in results:
        if not result.available:
            print(
                f"{result.backend:<18}{str(result.available):>10}"
                f"{'':>12}{'':>12}{'':>12}{'':>18}{'':>14}{'':>14}{'':>18}  {result.message}"
            )
            continue
        prob_diff = None
        class_prob_diff = None
        if result.probabilities is not None and ref.probabilities is not None:
            prob_diff = float(np.max(np.abs(result.probabilities - ref.probabilities)))
        if result.class_probabilities is not None and ref.class_probabilities is not None:
            class_prob_diff = float(np.max(np.abs(result.class_probabilities - ref.class_probabilities)))
        print(
            f"{result.backend:<18}{str(result.available):>10}"
            f"{_fmt_seconds(result.seconds):>12}"
            f"{_fmt_seconds(result.estimate_seconds):>12}"
            f"{_fmt_seconds(result.covariance_seconds):>12}"
            f"{result.loglike:>18.10f}"
            f"{result.loglike - ref.loglike:>14.3e}"
            f"{_fmt_optional(prob_diff):>14}"
            f"{_fmt_optional(class_prob_diff):>18}"
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
    parser.add_argument("--membership", choices=["constant", "ga"], default="ga")
    parser.add_argument("--max-iter", type=int, default=40)
    args = parser.parse_args()

    df, data, _, alternatives = load_biogeme_swissmetro(args.n_obs)
    data = data_with_membership_vars(df, alternatives)
    specs, membership = make_two_class_specs(args.membership)

    if args.mode == "fixed":
        params = default_params(args.membership)
        torch_result = run_torch_fixed(data, specs, membership, params)
        apollo_result = run_apollo_fixed(df, alternatives, params)
        biogeme_result = run_biogeme_fixed(df, alternatives, params)
        results = [torch_result, apollo_result, biogeme_result]
        reference = "torchdcm_fixed"
    else:
        torch_result = run_torch_fit(data, specs, membership, args.max_iter)
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
