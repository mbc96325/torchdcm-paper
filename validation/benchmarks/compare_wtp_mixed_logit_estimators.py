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
import pandas as pd
import torch

from torchdcm import Beta, UtilitySpec, WTPCoefficient, WTPMixedLogit
from compare_mnl_estimators import load_biogeme_swissmetro


@dataclass
class BackendResult:
    backend: str
    available: bool
    seconds: float | None = None
    loglike: float | None = None
    params: dict[str, float] | None = None
    probabilities: np.ndarray | None = None
    message: str = ""


def asc_spec() -> UtilitySpec:
    spec = UtilitySpec()
    spec.utility("TRAIN", Beta("ASC_TRAIN", init=0.3))
    spec.utility("SM", Beta("ASC_SM", init=0.0, fixed=True))
    spec.utility("CAR", Beta("ASC_CAR", init=0.6))
    return spec


def default_params() -> dict[str, float]:
    return {
        "ASC_TRAIN": 0.3,
        "ASC_CAR": 0.6,
        "B_COST": -1.2,
        "WTP_TIME": 0.75,
        "SIGMA_WTP_TIME": 0.35,
    }


def make_model(draws: torch.Tensor, panel: bool) -> WTPMixedLogit:
    return WTPMixedLogit(
        asc_spec(),
        cost=Beta("B_COST", init=-1.2),
        cost_variable="cost",
        wtp_coefficients=[WTPCoefficient("WTP_TIME", "time", init=0.75, sigma_init=0.35)],
        draws=draws,
        panel=panel,
    )


def make_draws(n_draws: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    half = (n_draws + 1) // 2
    base = torch.randn((half, 1), generator=generator, dtype=torch.float64)
    return torch.cat([base, -base], dim=0)[:n_draws]


def run_torch_fixed(data, params: dict[str, float], draws: torch.Tensor, panel: bool) -> BackendResult:
    model = make_model(draws, panel)
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


def write_apollo_inputs(df, alternatives, params: dict[str, float], draws: np.ndarray, panel: bool, directory: Path):
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
    np.savetxt(draws_path, draws, delimiter=",", header="WTP_TIME", comments="")
    spec = {
        "model_name": f"apollo_wtp_mixed_fixed_{len(df)}",
        "alternatives": alternatives,
        "choice_col": "choice_code",
        "panel": panel,
        "panel_id_col": "person_id",
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
    }
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return csv_path, spec_path, draws_path


def run_apollo_fixed(df, alternatives, params: dict[str, float], draws: np.ndarray, panel: bool) -> BackendResult:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return BackendResult(backend="apollo_r_fixed", available=False, message="Rscript not found")
    script = Path(__file__).resolve().parent / "apollo" / "R" / "run_wtp_mixed_fixed.R"
    if not script.exists():
        return BackendResult(backend="apollo_r_fixed", available=False, message=f"Missing script: {script}")
    with tempfile.TemporaryDirectory(prefix="torchdcm_apollo_wtp_") as tmp:
        tmp_path = Path(tmp)
        data_path, spec_path, draws_path = write_apollo_inputs(df, alternatives, params, draws, panel, tmp_path)
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


def run_biogeme_fixed(df, alternatives, params: dict[str, float], draws: np.ndarray, panel: bool) -> BackendResult:
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
    draw_columns = {f"DRAW_WTP_TIME_{draw_index}": float(draws[draw_index, 0]) for draw_index in range(draws.shape[0])}
    if draw_columns:
        wide_df = pd.concat([wide_df, pd.DataFrame(draw_columns, index=wide_df.index)], axis=1)

    try:
        database = db.Database("torchdcm_wtp_mixed_fixed", wide_df.drop(columns=["choice"]))
        choice = Variable("choice_code")
        av = {code_by_alt[alt]: Variable(f"avail_{alt.lower()}") for alt in alternatives}
        chosen_probs_by_draw = []
        alt_probs_by_draw = {alt: [] for alt in alternatives}
        for draw_index in range(draws.shape[0]):
            wtp_time = float(params["WTP_TIME"]) + float(params["SIGMA_WTP_TIME"]) * Variable(f"DRAW_WTP_TIME_{draw_index}")
            b_cost = float(params["B_COST"])
            v = {
                code_by_alt["TRAIN"]: float(params["ASC_TRAIN"])
                + b_cost * Variable("cost_train")
                + b_cost * wtp_time * Variable("time_train"),
                code_by_alt["SM"]: b_cost * Variable("cost_sm") + b_cost * wtp_time * Variable("time_sm"),
                code_by_alt["CAR"]: float(params["ASC_CAR"])
                + b_cost * Variable("cost_car")
                + b_cost * wtp_time * Variable("time_car"),
            }
            chosen_probs_by_draw.append(models.logit(v, av, choice))
            for alt in alternatives:
                alt_probs_by_draw[alt].append(models.logit(v, av, code_by_alt[alt]))

        formulas = {"chosen_prob": _average_expr(chosen_probs_by_draw)}
        for draw_index, expression in enumerate(chosen_probs_by_draw):
            formulas[f"chosen_prob_draw_{draw_index}"] = expression
        for alt, expressions in alt_probs_by_draw.items():
            formulas[f"prob_{alt.lower()}"] = _average_expr(expressions)

        start = time.perf_counter()
        biogeme = bio.BIOGEME(database, formulas)
        biogeme.model_name = "torchdcm_wtp_mixed_fixed_shared_draws"
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


def _average_expr(expressions):
    total = expressions[0]
    for expression in expressions[1:]:
        total = total + expression
    return total / float(len(expressions))


def print_results(results: list[BackendResult], reference: str, n_obs: int, n_draws: int, panel: bool):
    ref = next(result for result in results if result.backend == reference and result.available)
    print("case: biogeme_swissmetro_wtp_mixed_logit")
    print("mode: fixed")
    print(f"n_obs: {n_obs}")
    print(f"n_draws: {n_draws}")
    print(f"panel: {panel}")
    print("alignment:")
    print("  benchmark_mode: fixed_likelihood_replay")
    print("  estimated_backend: none")
    print("  data_source: biogeme.data.swissmetro/data/swissmetro.dat")
    print("  model: WTP-space mixed logit with normal random WTP_TIME")
    print("  utility: ASC + B_COST*cost + B_COST*WTP_TIME*time")
    print("  draws: shared antithetic standard-normal draw matrix")
    print("  parameters: shared across replay backends")
    print("  probabilities: averaged over the same draws and row order")
    print(f"  reference: {reference}")
    print()
    print(f"{'backend':<18}{'available':>10}{'total_s':>12}{'loglike':>18}{'ll_diff':>14}{'prob_diff':>14}")
    for result in results:
        if not result.available:
            print(f"{result.backend:<18}{str(result.available):>10}{'':>12}{'':>18}{'':>14}{'':>14}  {result.message}")
            continue
        prob_diff = None
        if result.probabilities is not None and ref.probabilities is not None:
            prob_diff = float(np.max(np.abs(result.probabilities - ref.probabilities)))
        print(
            f"{result.backend:<18}{str(result.available):>10}"
            f"{_fmt_seconds(result.seconds):>12}"
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
    parser.add_argument("--n-draws", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--panel", action="store_true")
    args = parser.parse_args()

    df, data, _, alternatives = load_biogeme_swissmetro(args.n_obs)
    params = default_params()
    draws = make_draws(args.n_draws, args.seed)
    torch_result = run_torch_fixed(data, params, draws, args.panel)
    apollo_result = run_apollo_fixed(df, alternatives, params, draws.detach().cpu().numpy(), args.panel)
    biogeme_result = run_biogeme_fixed(df, alternatives, params, draws.detach().cpu().numpy(), args.panel)
    results = [torch_result, apollo_result, biogeme_result]
    print_results(results, "torchdcm_fixed", len(df), args.n_draws, args.panel)


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
