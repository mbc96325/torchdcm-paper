from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Callable

import numpy as np
import pandas as pd
import torch

from torchdcm import (
    Beta,
    ChoiceDataset,
    ChoiceLatentEffect,
    ContinuousIndicator,
    HybridChoiceModel,
    LatentClassLogit,
    LatentVariable,
    MixedLogit,
    RandomCoefficient,
    UtilitySpec,
)

from compare_mnl_estimators import load_biogeme_swissmetro


DATA_ROOT = Path(__file__).resolve().parents[1] / "datasets" / "raw"


@dataclass
class BackendResult:
    backend: str
    available: bool
    loglike: float | None = None
    eval_seconds: float | None = None
    message: str = ""


PARAMS_LATENT = {
    "ASC_B_C1": -0.45,
    "ASC_C_C1": 0.25,
    "B_X_C1": -1.10,
    "ASC_B_C2": 0.75,
    "ASC_C_C2": -0.35,
    "B_X_C2": -0.35,
    "CLASS_2": -0.25,
    "CLASS_2_Z": 0.90,
}

PARAMS_HYBRID = {
    "ASC_B": 0.25,
    "B_X": 0.70,
    "G_Q": 0.65,
    "SIGMA_LV": 0.80,
    "B_ATT": 0.90,
    "SIGMA_Y1": 0.70,
    "A2": 0.20,
    "L2": 0.75,
    "SIGMA_Y2": 0.85,
}

PARAMS_PANEL = {
    "ASC_B": 0.35,
    "B_X": -0.80,
    "ASC_C": -0.20,
    "ASC_D": 0.10,
    "SIGMA_B_X": 0.55,
}

STARTS_LATENT = {
    "ASC_B_C1": -0.30,
    "ASC_C_C1": 0.15,
    "B_X_C1": -0.80,
    "ASC_B_C2": 0.50,
    "ASC_C_C2": -0.20,
    "B_X_C2": -0.20,
    "CLASS_2": -0.10,
    "CLASS_2_Z": 0.60,
}

STARTS_HYBRID = {
    "ASC_B": 0.10,
    "B_X": 0.40,
    "G_Q": 0.40,
    "SIGMA_LV": 0.60,
    "B_ATT": 0.60,
    "SIGMA_Y1": 0.80,
    "A2": 0.10,
    "L2": 0.60,
    "SIGMA_Y2": 0.90,
}

STARTS_PANEL = {
    "ASC_B": 0.20,
    "B_X": -0.50,
    "ASC_C": -0.10,
    "ASC_D": 0.05,
    "SIGMA_B_X": 0.40,
}


def timed_median(function: Callable[[], float], repeats: int) -> tuple[float, float]:
    reference = float(function())
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        current = float(function())
        values.append(time.perf_counter() - start)
        if not math.isclose(current, reference, rel_tol=1e-11, abs_tol=1e-9):
            raise RuntimeError("Likelihood replay changed across repeated evaluations")
    return reference, median(values)


def antithetic_draws(n_draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    half = (n_draws + 1) // 2
    base = rng.standard_normal(half)
    return np.concatenate([base, -base])[:n_draws]


def make_latent_model(
    frame: pd.DataFrame,
) -> tuple[ChoiceDataset, LatentClassLogit, torch.Tensor]:
    alternatives = ["A", "B", "C"]
    data = ChoiceDataset.from_wide(
        frame,
        alternatives=alternatives,
        choice="choice",
        variables={
            "x": {alt: f"x_{alt}" for alt in alternatives},
            "z": {alt: "z" for alt in alternatives},
        },
        availability={alt: f"av_{alt}" for alt in alternatives},
        obs_id="id",
    )
    specs = []
    for suffix in ("C1", "C2"):
        spec = UtilitySpec()
        spec.utility("A", Beta(f"B_X_{suffix}", init=STARTS_LATENT[f"B_X_{suffix}"]) * "x")
        spec.utility(
            "B",
            Beta(f"ASC_B_{suffix}", init=STARTS_LATENT[f"ASC_B_{suffix}"])
            + Beta(f"B_X_{suffix}", init=STARTS_LATENT[f"B_X_{suffix}"]) * "x",
        )
        spec.utility(
            "C",
            Beta(f"ASC_C_{suffix}", init=STARTS_LATENT[f"ASC_C_{suffix}"])
            + Beta(f"B_X_{suffix}", init=STARTS_LATENT[f"B_X_{suffix}"]) * "x",
        )
        specs.append(spec)
    membership = [
        Beta("CLASS_2", init=STARTS_LATENT["CLASS_2"])
        + Beta("CLASS_2_Z", init=STARTS_LATENT["CLASS_2_Z"]) * "z"
    ]
    model = LatentClassLogit(specs, class_membership=membership)
    compiled = model.compile(data)
    params = torch.as_tensor([PARAMS_LATENT[name] for name in compiled.free_names], dtype=torch.float64)
    return data, model, params


def make_latent_class_actual(
    n_obs: int,
) -> tuple[pd.DataFrame, ChoiceDataset, LatentClassLogit, torch.Tensor]:
    raw, _, _, _ = load_biogeme_swissmetro(n_obs)
    frame = pd.DataFrame(
        {
            "id": np.arange(len(raw)),
            "choice": raw["choice"].map({"TRAIN": "A", "SM": "B", "CAR": "C"}),
            "choice_code": raw["choice"].map({"TRAIN": 1, "SM": 2, "CAR": 3}).astype(int),
            "x_A": raw["time_train"].astype(float),
            "x_B": raw["time_sm"].astype(float),
            "x_C": raw["time_car"].astype(float),
            "z": raw["GA"].astype(float),
            "av_A": raw["avail_train"].astype(int),
            "av_B": raw["avail_sm"].astype(int),
            "av_C": raw["avail_car"].astype(int),
        }
    )
    data, model, params = make_latent_model(frame)
    return frame, data, model, params


def make_latent_class_synthetic(
    n_obs: int,
    seed: int,
) -> tuple[pd.DataFrame, ChoiceDataset, LatentClassLogit, torch.Tensor]:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_obs)
    x = rng.standard_normal((n_obs, 3))
    class_2_probability = 1.0 / (
        1.0 + np.exp(-(PARAMS_LATENT["CLASS_2"] + PARAMS_LATENT["CLASS_2_Z"] * z))
    )
    class_index = (rng.random(n_obs) < class_2_probability).astype(int)
    utility = np.empty((n_obs, 3), dtype=float)
    for index, suffix in enumerate(("C1", "C2")):
        selected = class_index == index
        utility[selected, 0] = PARAMS_LATENT[f"B_X_{suffix}"] * x[selected, 0]
        utility[selected, 1] = (
            PARAMS_LATENT[f"ASC_B_{suffix}"]
            + PARAMS_LATENT[f"B_X_{suffix}"] * x[selected, 1]
        )
        utility[selected, 2] = (
            PARAMS_LATENT[f"ASC_C_{suffix}"]
            + PARAMS_LATENT[f"B_X_{suffix}"] * x[selected, 2]
        )
    gumbel = -np.log(-np.log(rng.random((n_obs, 3))))
    chosen = np.argmax(utility + gumbel, axis=1)
    labels = np.asarray(["A", "B", "C"])
    frame = pd.DataFrame(
        {
            "id": np.arange(n_obs),
            "choice": labels[chosen],
            "choice_code": chosen + 1,
            "x_A": x[:, 0],
            "x_B": x[:, 1],
            "x_C": x[:, 2],
            "z": z,
            "av_A": np.ones(n_obs, dtype=int),
            "av_B": np.ones(n_obs, dtype=int),
            "av_C": np.ones(n_obs, dtype=int),
        }
    )
    data, model, params = make_latent_model(frame)
    return frame, data, model, params


def make_hybrid_model(
    frame: pd.DataFrame,
    draws: np.ndarray,
) -> tuple[ChoiceDataset, HybridChoiceModel, torch.Tensor]:
    data = ChoiceDataset.from_wide(
        frame,
        alternatives=["A", "B"],
        choice="choice",
        variables={"x": {"A": "x_A", "B": "x_B"}},
        obs_variables={"q": "q", "y1": "y1", "y2": "y2"},
        obs_id="id",
    )
    spec = UtilitySpec()
    spec.utility("A", Beta("ASC_A", init=0.0, fixed=True))
    spec.utility(
        "B",
        Beta("ASC_B", init=STARTS_HYBRID["ASC_B"])
        + Beta("B_X", init=STARTS_HYBRID["B_X"]) * "x",
    )
    model = HybridChoiceModel(
        spec,
        latent_variables=[
            LatentVariable(
                "ATT",
                intercept=0.0,
                coefficients={"q": Beta("G_Q", init=STARTS_HYBRID["G_Q"])},
                sigma_name="SIGMA_LV",
                sigma_init=STARTS_HYBRID["SIGMA_LV"],
                sigma_fixed=False,
            )
        ],
        choice_effects=[
            ChoiceLatentEffect(
                "B", "ATT", Beta("B_ATT", init=STARTS_HYBRID["B_ATT"])
            )
        ],
        indicators=[
            ContinuousIndicator(
                "y1",
                "ATT",
                intercept=0.0,
                loading=1.0,
                sigma_name="SIGMA_Y1",
                sigma_init=STARTS_HYBRID["SIGMA_Y1"],
                sigma_fixed=False,
            ),
            ContinuousIndicator(
                "y2",
                "ATT",
                intercept=Beta("A2", init=STARTS_HYBRID["A2"]),
                loading=Beta("L2", init=STARTS_HYBRID["L2"]),
                sigma_name="SIGMA_Y2",
                sigma_init=STARTS_HYBRID["SIGMA_Y2"],
                sigma_fixed=False,
            ),
        ],
        draws=torch.as_tensor(draws[:, None], dtype=torch.float64),
        panel=False,
    )
    compiled = model.compile(data)
    params = torch.as_tensor([PARAMS_HYBRID[name] for name in compiled.free_names], dtype=torch.float64)
    return data, model, params


def make_hybrid_synthetic(
    n_obs: int,
    n_draws: int,
    seed: int,
) -> tuple[pd.DataFrame, ChoiceDataset, HybridChoiceModel, torch.Tensor, np.ndarray]:
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(n_obs)
    x = rng.standard_normal(n_obs)
    latent = PARAMS_HYBRID["G_Q"] * q + PARAMS_HYBRID["SIGMA_LV"] * rng.standard_normal(n_obs)
    utility_b = (
        PARAMS_HYBRID["ASC_B"]
        + PARAMS_HYBRID["B_X"] * x
        + PARAMS_HYBRID["B_ATT"] * latent
    )
    probability_b = 1.0 / (1.0 + np.exp(-utility_b))
    choice_b = rng.random(n_obs) < probability_b
    y1 = latent + PARAMS_HYBRID["SIGMA_Y1"] * rng.standard_normal(n_obs)
    y2 = (
        PARAMS_HYBRID["A2"]
        + PARAMS_HYBRID["L2"] * latent
        + PARAMS_HYBRID["SIGMA_Y2"] * rng.standard_normal(n_obs)
    )
    frame = pd.DataFrame(
        {
            "id": np.arange(n_obs),
            "choice": np.where(choice_b, "B", "A"),
            "choice_code": np.where(choice_b, 2, 1),
            "x_A": np.zeros(n_obs),
            "x_B": x,
            "x": x,
            "q": q,
            "y1": y1,
            "y2": y2,
        }
    )
    draws = antithetic_draws(n_draws, seed + 1000)
    data, model, params = make_hybrid_model(frame, draws)
    return frame, data, model, params, draws


def standardized(values: pd.Series) -> np.ndarray:
    array = values.to_numpy(dtype=float)
    return (array - array.mean()) / array.std(ddof=0)


def make_hybrid_actual(
    n_obs: int,
    n_draws: int,
    seed: int,
) -> tuple[pd.DataFrame, ChoiceDataset, HybridChoiceModel, torch.Tensor, np.ndarray]:
    raw = pd.read_csv(DATA_ROOT / "biogeme_optima" / "data.csv")
    valid = (
        raw["Choice"].isin([1, 2])
        & (raw["Envir01"] > 0)
        & (raw["Envir02"] > 0)
        & np.isfinite(
            raw[["TimePT_scaled", "TimeCar_scaled", "ScaledIncome"]].to_numpy(dtype=float)
        ).all(axis=1)
    )
    raw = raw.loc[valid].sample(frac=1.0, random_state=7321).reset_index(drop=True)
    q = standardized(raw["ScaledIncome"])
    x = standardized(raw["TimePT_scaled"] - raw["TimeCar_scaled"])
    y1 = standardized(raw["Envir01"])
    y2 = standardized(raw["Envir02"])
    raw = raw.iloc[:n_obs].copy()
    frame = pd.DataFrame(
        {
            "id": np.arange(len(raw)),
            "choice": np.where(raw["Choice"].to_numpy(dtype=int) == 2, "B", "A"),
            "choice_code": raw["Choice"].to_numpy(dtype=int),
            "x_A": np.zeros(len(raw)),
            "x_B": x[: len(raw)],
            "x": x[: len(raw)],
            "q": q[: len(raw)],
            "y1": y1[: len(raw)],
            "y2": y2[: len(raw)],
        }
    )
    draws = antithetic_draws(n_draws, seed + 1000)
    data, model, params = make_hybrid_model(frame, draws)
    return frame, data, model, params, draws


def make_panel_model(
    frame: pd.DataFrame,
    alternatives: list[str],
    draws: np.ndarray,
) -> tuple[ChoiceDataset, MixedLogit, torch.Tensor]:
    data = ChoiceDataset.from_wide(
        frame,
        alternatives=alternatives,
        choice="choice",
        variables={"x": {alt: f"x_{alt}" for alt in alternatives}},
        obs_id="obs_id",
        individual_id="person_id",
    )
    spec = UtilitySpec()
    spec.utility("A", Beta("B_X", init=STARTS_PANEL["B_X"]) * "x")
    for alt in alternatives[1:]:
        spec.utility(
            alt,
            Beta(f"ASC_{alt}", init=STARTS_PANEL[f"ASC_{alt}"])
            + Beta("B_X", init=STARTS_PANEL["B_X"]) * "x",
        )
    model = MixedLogit(
        spec,
        [RandomCoefficient("B_X", sigma_init=STARTS_PANEL["SIGMA_B_X"])],
        draws=torch.as_tensor(draws[:, None], dtype=torch.float64),
        panel=True,
    )
    compiled = model.compile(data)
    params = torch.as_tensor([PARAMS_PANEL[name] for name in compiled.free_names], dtype=torch.float64)
    return data, model, params


def make_panel_synthetic(
    n_units: int,
    choices_per_unit: int,
    n_draws: int,
    seed: int,
) -> tuple[pd.DataFrame, ChoiceDataset, MixedLogit, torch.Tensor, np.ndarray]:
    rng = np.random.default_rng(seed)
    person_id = np.repeat(np.arange(n_units), choices_per_unit)
    n_obs = len(person_id)
    alternatives = ["A", "B", "C", "D"]
    x = rng.standard_normal((n_obs, len(alternatives)))
    individual_beta = (
        PARAMS_PANEL["B_X"]
        + PARAMS_PANEL["SIGMA_B_X"] * rng.standard_normal(n_units)
    )[person_id]
    utilities = individual_beta[:, None] * x
    utilities[:, 1] += PARAMS_PANEL["ASC_B"]
    utilities[:, 2] += PARAMS_PANEL["ASC_C"]
    utilities[:, 3] += PARAMS_PANEL["ASC_D"]
    gumbel = -np.log(-np.log(rng.random((n_obs, len(alternatives)))))
    chosen = np.argmax(utilities + gumbel, axis=1)
    labels = np.asarray(alternatives)
    frame = pd.DataFrame(
        {
            "obs_id": np.arange(n_obs),
            "person_id": person_id,
            "choice": labels[chosen],
            "choice_code": chosen + 1,
            "x_A": x[:, 0],
            "x_B": x[:, 1],
            "x_C": x[:, 2],
            "x_D": x[:, 3],
        }
    )
    draws = antithetic_draws(n_draws, seed + 1000)
    data, model, params = make_panel_model(frame, alternatives, draws)
    return frame, data, model, params, draws


def make_panel_actual(
    n_units: int,
    n_draws: int,
    seed: int,
) -> tuple[pd.DataFrame, ChoiceDataset, MixedLogit, torch.Tensor, np.ndarray]:
    raw = pd.read_csv(DATA_ROOT / "mlogit_electricity" / "data.csv")
    counts = raw.groupby("id").size()
    complete_ids = counts.index[counts == 12].to_numpy()
    selected_ids = np.sort(complete_ids)[:n_units]
    raw = raw.loc[raw["id"].isin(selected_ids)].reset_index(drop=True)
    alternatives = ["A", "B", "C", "D"]
    price = raw[["pf1", "pf2", "pf3", "pf4"]].to_numpy(dtype=float)
    price = (price - price.mean()) / price.std(ddof=0)
    choice_code = raw["choice"].to_numpy(dtype=int)
    labels = np.asarray(alternatives)
    frame = pd.DataFrame(
        {
            "obs_id": np.arange(len(raw)),
            "person_id": raw["id"].to_numpy(dtype=int),
            "choice": labels[choice_code - 1],
            "choice_code": choice_code,
            "x_A": price[:, 0],
            "x_B": price[:, 1],
            "x_C": price[:, 2],
            "x_D": price[:, 3],
        }
    )
    draws = antithetic_draws(n_draws, seed + 1000)
    data, model, params = make_panel_model(frame, alternatives, draws)
    return frame, data, model, params, draws


def run_torch(model, data: ChoiceDataset, params: torch.Tensor, repeats: int) -> BackendResult:
    compiled = model.compile(data)

    def evaluate() -> float:
        return float(model.loglike(params, data, compiled).detach().cpu())

    loglike, seconds = timed_median(evaluate, repeats)
    return BackendResult("torchdcm", True, loglike, seconds)


def run_biogeme_latent(frame: pd.DataFrame, repeats: int) -> BackendResult:
    try:
        import biogeme.biogeme as bio
        import biogeme.database as db
        import biogeme.models as models
        from biogeme.expressions import Variable, exp, log
    except ImportError as exc:
        return BackendResult("biogeme", False, message=str(exc))
    database = db.Database("torchdcm_advanced_latent", frame.drop(columns=["choice"]))
    choice = Variable("choice_code")
    availability = {1: Variable("av_A"), 2: Variable("av_B"), 3: Variable("av_C")}
    class_2_exp = exp(PARAMS_LATENT["CLASS_2"] + PARAMS_LATENT["CLASS_2_Z"] * Variable("z"))
    class_probs = [1.0 / (1.0 + class_2_exp), class_2_exp / (1.0 + class_2_exp)]
    chosen = 0
    for index, suffix in enumerate(("C1", "C2")):
        v = {
            1: PARAMS_LATENT[f"B_X_{suffix}"] * Variable("x_A"),
            2: PARAMS_LATENT[f"ASC_B_{suffix}"] + PARAMS_LATENT[f"B_X_{suffix}"] * Variable("x_B"),
            3: PARAMS_LATENT[f"ASC_C_{suffix}"] + PARAMS_LATENT[f"B_X_{suffix}"] * Variable("x_C"),
        }
        chosen = chosen + class_probs[index] * models.logit(v, availability, choice)
    engine = bio.BIOGEME(database, log(chosen))
    engine.model_name = "torchdcm_advanced_latent_fixed"
    loglike, seconds = timed_median(engine.calculate_init_likelihood, repeats)
    return BackendResult("biogeme", True, loglike, seconds)


def normal_density(value, mean, sigma, exp_function):
    return exp_function(-0.5 * ((value - mean) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))


def run_biogeme_hybrid(frame: pd.DataFrame, draws: np.ndarray, repeats: int) -> BackendResult:
    try:
        import biogeme.biogeme as bio
        import biogeme.database as db
        import biogeme.models as models
        from biogeme.expressions import Variable, exp, log
    except ImportError as exc:
        return BackendResult("biogeme", False, message=str(exc))
    database = db.Database("torchdcm_advanced_hybrid", frame.drop(columns=["choice"]))
    choice = Variable("choice_code")
    joint = 0
    for draw in draws:
        latent = PARAMS_HYBRID["G_Q"] * Variable("q") + PARAMS_HYBRID["SIGMA_LV"] * float(draw)
        v = {
            1: 0,
            2: PARAMS_HYBRID["ASC_B"]
            + PARAMS_HYBRID["B_X"] * Variable("x")
            + PARAMS_HYBRID["B_ATT"] * latent,
        }
        choice_prob = models.logit(v, None, choice)
        density_1 = normal_density(Variable("y1"), latent, PARAMS_HYBRID["SIGMA_Y1"], exp)
        density_2 = normal_density(
            Variable("y2"),
            PARAMS_HYBRID["A2"] + PARAMS_HYBRID["L2"] * latent,
            PARAMS_HYBRID["SIGMA_Y2"],
            exp,
        )
        joint = joint + choice_prob * density_1 * density_2
    engine = bio.BIOGEME(database, log(joint / len(draws)))
    engine.model_name = "torchdcm_advanced_hybrid_fixed"
    loglike, seconds = timed_median(engine.calculate_init_likelihood, repeats)
    return BackendResult("biogeme", True, loglike, seconds)


def run_biogeme_panel(frame: pd.DataFrame, draws: np.ndarray, repeats: int) -> BackendResult:
    try:
        import biogeme.biogeme as bio
        import biogeme.database as db
        import biogeme.models as models
        from biogeme.expressions import PanelLikelihoodTrajectory, Variable, log
    except ImportError as exc:
        return BackendResult("biogeme", False, message=str(exc))
    database = db.Database("torchdcm_advanced_panel", frame.drop(columns=["choice"]))
    database.panel("person_id")
    choice = Variable("choice_code")
    integrated = 0
    for draw in draws:
        b_x = PARAMS_PANEL["B_X"] + PARAMS_PANEL["SIGMA_B_X"] * float(draw)
        v = {
            1: b_x * Variable("x_A"),
            2: PARAMS_PANEL["ASC_B"] + b_x * Variable("x_B"),
            3: PARAMS_PANEL["ASC_C"] + b_x * Variable("x_C"),
        }
        if "x_D" in frame.columns:
            v[4] = PARAMS_PANEL["ASC_D"] + b_x * Variable("x_D")
        integrated = integrated + PanelLikelihoodTrajectory(models.logit(v, None, choice))
    engine = bio.BIOGEME(database, log(integrated / len(draws)))
    engine.model_name = "torchdcm_advanced_panel_fixed"
    # Fixed-likelihood replay bypasses ``estimate()``, which is where Biogeme
    # normally activates its flat panel adapter.
    engine.use_flatten_database = True
    loglike, seconds = timed_median(engine.calculate_init_likelihood, repeats)
    return BackendResult("biogeme", True, loglike, seconds)


def run_apollo(
    kind: str,
    frame: pd.DataFrame,
    params: dict[str, float],
    draws: np.ndarray,
    repeats: int,
) -> BackendResult:
    rscript = os.environ.get("TORCHDCM_RSCRIPT") or shutil.which("Rscript")
    if not rscript:
        return BackendResult("apollo", False, message="Rscript not found")
    script = Path(__file__).resolve().parent / "apollo" / "R" / "run_advanced_fixed.R"
    with tempfile.TemporaryDirectory(prefix="torchdcm_advanced_apollo_") as tmp:
        directory = Path(tmp)
        data_path = directory / "data.csv"
        spec_path = directory / "spec.json"
        output_path = directory / "result.json"
        export = frame.drop(columns=["choice"], errors="ignore").copy()
        export.to_csv(data_path, index=False)
        id_col = "person_id" if kind == "panel_likelihood" else "id"
        payload = {
            "kind": kind,
            "model_name": f"torchdcm_advanced_{kind}",
            "id_col": id_col,
            "parameters": params,
            "draws": draws.tolist(),
            "n_repeats": repeats,
        }
        spec_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        env = os.environ.copy()
        env.setdefault("R_LIBS_USER", str(Path.home() / "R" / "site-library"))
        proc = subprocess.run(
            [
                rscript,
                str(script),
                "--data",
                str(data_path),
                "--spec",
                str(spec_path),
                "--output",
                str(output_path),
            ],
            text=True,
            capture_output=True,
            env=env,
        )
        if proc.returncode != 0:
            return BackendResult("apollo", False, message=(proc.stderr or proc.stdout).strip())
        result = json.loads(output_path.read_text(encoding="utf-8"))
        return BackendResult(
            "apollo",
            True,
            float(result["loglike"]),
            float(result["eval_seconds"]),
        )


def summarize_case(
    name: str,
    kind: str,
    data_type: str,
    data_source: str,
    n_obs: int,
    n_units: int,
    n_draws: int,
    results: list[BackendResult],
    extra: dict | None = None,
) -> dict:
    torch_result = next(result for result in results if result.backend == "torchdcm")
    available = [result for result in results if result.available]
    differences = {
        result.backend: abs(float(result.loglike) - float(torch_result.loglike))
        for result in available
        if result.backend != "torchdcm"
    }
    tolerance = max(1e-6, 1e-10 * abs(float(torch_result.loglike)))
    return {
        "case": name,
        "kind": kind,
        "data_type": data_type,
        "data_source": data_source,
        "n_obs": n_obs,
        "n_units": n_units,
        "n_draws": n_draws,
        "extra": extra or {},
        "results": {result.backend: asdict(result) for result in results},
        "absolute_loglike_differences": differences,
        "tolerance": tolerance,
        "consistent": len(available) == 3 and all(value <= tolerance for value in differences.values()),
    }


def run_suite(repeats: int, kinds: set[str] | None = None) -> dict:
    kinds = kinds or {"latent_class", "hybrid_choice", "panel_likelihood"}
    cases = []
    if "latent_class" in kinds:
        for n_obs in (500, 2_000, 10_000):
            frame, data, model, params = make_latent_class_synthetic(n_obs, seed=100 + n_obs)
            results = [
                run_torch(model, data, params, repeats),
                run_biogeme_latent(frame, repeats),
                run_apollo("latent_class", frame, PARAMS_LATENT, np.asarray([], dtype=float), repeats),
            ]
            cases.append(
                summarize_case(
                    f"Synthetic {n_obs:,}",
                    "latent_class",
                    "Synthetic",
                    "Controlled latent-class DGP",
                    len(frame),
                    len(frame),
                    0,
                    results,
                    {"classes": 2, "membership_covariates": 1},
                )
            )
        for n_obs in (500, 2_000, 10_719):
            frame, data, model, params = make_latent_class_actual(n_obs)
            results = [
                run_torch(model, data, params, repeats),
                run_biogeme_latent(frame, repeats),
                run_apollo("latent_class", frame, PARAMS_LATENT, np.asarray([], dtype=float), repeats),
            ]
            cases.append(
                summarize_case(
                    f"Swissmetro {len(frame):,}",
                    "latent_class",
                    "Actual",
                    "Swissmetro",
                    len(frame),
                    len(frame),
                    0,
                    results,
                    {"classes": 2, "membership_covariates": 1},
                )
            )

    if "hybrid_choice" in kinds:
        for n_obs, n_draws in ((500, 32), (2_000, 64), (10_000, 128)):
            frame, data, model, params, draws = make_hybrid_synthetic(n_obs, n_draws, seed=200 + n_obs)
            results = [
                run_torch(model, data, params, repeats),
                run_biogeme_hybrid(frame, draws, repeats),
                run_apollo("hybrid_choice", frame, PARAMS_HYBRID, draws, repeats),
            ]
            cases.append(
                summarize_case(
                    f"Hybrid {n_obs:,}",
                    "hybrid_choice",
                    "Synthetic",
                    "Controlled hybrid-choice DGP",
                    n_obs,
                    n_obs,
                    n_draws,
                    results,
                    {"latent_variables": 1, "continuous_indicators": 2},
                )
            )
        for n_obs, n_draws in ((500, 32), (1_000, 64), (1_298, 128)):
            frame, data, model, params, draws = make_hybrid_actual(
                n_obs,
                n_draws,
                seed=400 + n_obs,
            )
            results = [
                run_torch(model, data, params, repeats),
                run_biogeme_hybrid(frame, draws, repeats),
                run_apollo("hybrid_choice", frame, PARAMS_HYBRID, draws, repeats),
            ]
            cases.append(
                summarize_case(
                    f"Optima {len(frame):,}",
                    "hybrid_choice",
                    "Actual",
                    "Optima",
                    len(frame),
                    len(frame),
                    n_draws,
                    results,
                    {"latent_variables": 1, "continuous_indicators": 2},
                )
            )

    if "panel_likelihood" in kinds:
        for n_units, choices_per_unit, n_draws in ((250, 2, 32), (500, 4, 64), (1_250, 8, 128)):
            frame, data, model, params, draws = make_panel_synthetic(
                n_units,
                choices_per_unit,
                n_draws,
                seed=500 + n_units + choices_per_unit,
            )
            results = [
                run_torch(model, data, params, repeats),
                run_biogeme_panel(frame, draws, repeats),
                run_apollo("panel_likelihood", frame, PARAMS_PANEL, draws, repeats),
            ]
            cases.append(
                summarize_case(
                    f"Panel {n_units:,}x{choices_per_unit}",
                    "panel_likelihood",
                    "Synthetic",
                    "Controlled panel DGP",
                    len(frame),
                    n_units,
                    n_draws,
                    results,
                    {
                        "alternatives": 4,
                        "choices_per_unit": choices_per_unit,
                        "random_coefficients": 1,
                    },
                )
            )
        for n_units, n_draws in ((100, 32), (250, 64), (348, 128)):
            frame, data, model, params, draws = make_panel_actual(
                n_units,
                n_draws,
                seed=800 + n_units,
            )
            results = [
                run_torch(model, data, params, repeats),
                run_biogeme_panel(frame, draws, repeats),
                run_apollo("panel_likelihood", frame, PARAMS_PANEL, draws, repeats),
            ]
            cases.append(
                summarize_case(
                    f"Electricity {n_units:,}",
                    "panel_likelihood",
                    "Actual",
                    "Electricity",
                    len(frame),
                    n_units,
                    n_draws,
                    results,
                    {
                        "alternatives": 4,
                        "choices_per_unit": 12,
                        "random_coefficients": 1,
                    },
                )
            )

    return {
        "benchmark": "advanced_fixed_likelihood_validation",
        "timing_scope": "median likelihood evaluation after construction and warm-up",
        "cpu_threads": 1,
        "repeats": repeats,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--kinds",
        nargs="+",
        choices=["latent_class", "hybrid_choice", "panel_likelihood"],
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    payload = run_suite(args.repeats, None if args.kinds is None else set(args.kinds))
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
