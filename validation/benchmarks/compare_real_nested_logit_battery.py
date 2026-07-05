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

import numpy as np
import pandas as pd
import torch

from torchdcm import Beta, ChoiceDataset, Nest, NestedLogit, UtilitySpec

import compare_biogeme_public_mnl as public_mnl
import compare_nhts_mnl as nhts_mnl
from compare_nested_logit_estimators import load_biogeme_swissmetro


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "generated"
APOLLO_SCRIPT = ROOT / "benchmarks" / "apollo" / "R" / "run_nl.R"


@dataclass(frozen=True)
class NestSpec:
    alternatives: list[str]
    init: float = 0.8
    fixed: bool = False


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


@dataclass
class NestedCase:
    case: str
    data_label: str
    model_label: str
    source: str
    wide_df: pd.DataFrame
    data: ChoiceDataset
    spec: UtilitySpec
    alternatives: list[str]
    beta_names: list[str]
    nests: dict[str, NestSpec]
    initial_values: dict[str, float]

    @property
    def lambda_names(self) -> list[str]:
        return [lambda_name(name) for name, nest in self.nests.items() if not nest.fixed]

    @property
    def parameter_names(self) -> list[str]:
        return [*self.beta_names, *self.lambda_names]


def safe_alt(alt: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in alt)


def lambda_name(nest_name: str) -> str:
    return f"LAMBDA_{nest_name.upper()}"


def raw_lambda_name(nest_name: str) -> str:
    return f"RAW_{lambda_name(nest_name)}"


def nested_case_from_design_long(
    *,
    case: str,
    data_label: str,
    model_label: str,
    source: str,
    long_df: pd.DataFrame,
    alternatives: list[str],
    beta_names: list[str],
    nests: dict[str, NestSpec],
) -> NestedCase:
    chosen_rows = long_df.loc[long_df["choice"].astype(bool), ["obs_id", "alt"]].copy()
    if chosen_rows["obs_id"].duplicated().any():
        raise ValueError(f"More than one chosen alternative per observation in {case}.")
    ordered_obs = long_df[["obs_id"]].drop_duplicates("obs_id").sort_values("obs_id")
    wide_df = ordered_obs.merge(chosen_rows.rename(columns={"alt": "choice"}), on="obs_id", how="left")
    if wide_df["choice"].isna().any():
        raise ValueError(f"Missing chosen alternative for at least one observation in {case}.")
    wide_df = wide_df.reset_index(drop=True)
    feature_columns: dict[str, dict[str, str]] = {name: {} for name in beta_names}
    availability_columns: dict[str, str] = {}
    for alt in alternatives:
        alt_key = safe_alt(alt)
        sub = long_df.loc[long_df["alt"] == alt].sort_values("obs_id").reset_index(drop=True)
        if len(sub) != len(wide_df):
            raise ValueError(f"Alternative {alt} is not present for every observation in {case}.")
        if not np.array_equal(sub["obs_id"].to_numpy(), wide_df["obs_id"].to_numpy()):
            raise ValueError(f"Observation ordering mismatch for alternative {alt} in {case}.")
        for name in beta_names:
            col = f"x__{name}__{alt_key}"
            wide_df[col] = sub[name].to_numpy(dtype=float)
            feature_columns[name][alt] = col
        avail_col = f"avail__{alt_key}"
        wide_df[avail_col] = sub["availability"].astype(bool).to_numpy()
        availability_columns[alt] = avail_col

    data = ChoiceDataset.from_wide(
        wide_df,
        alternatives=alternatives,
        choice="choice",
        variables=feature_columns,
        availability=availability_columns,
        obs_id="obs_id",
    )
    spec = UtilitySpec()
    for alt in alternatives:
        expr = None
        for name in beta_names:
            term = Beta(name) * name
            expr = term if expr is None else expr + term
        spec.utility(alt, expr if expr is not None else 0)

    initial_values = {name: 0.0 for name in beta_names}
    for nest_name, nest in nests.items():
        if not nest.fixed:
            initial_values[lambda_name(nest_name)] = nest.init

    return NestedCase(
        case=case,
        data_label=data_label,
        model_label=model_label,
        source=source,
        wide_df=wide_df,
        data=data,
        spec=spec,
        alternatives=alternatives,
        beta_names=beta_names,
        nests=nests,
        initial_values=initial_values,
    )


def make_swissmetro_case(n_obs: int | None) -> NestedCase:
    df, _, _, alternatives = load_biogeme_swissmetro(n_obs)
    utility_columns = {
        "TRAIN": {"ASC_TRAIN": 1.0, "B_TIME": "time_train", "B_COST": "cost_train"},
        "SM": {"B_TIME": "time_sm", "B_COST": "cost_sm"},
        "CAR": {"ASC_CAR": 1.0, "B_TIME": "time_car", "B_COST": "cost_car"},
    }
    long_df = public_mnl.make_design_long(
        df,
        alternatives,
        "choice",
        utility_columns,
        {"TRAIN": "avail_train", "SM": "avail_sm", "CAR": "avail_car"},
        ["ASC_TRAIN", "B_TIME", "B_COST", "ASC_CAR"],
    )
    return nested_case_from_design_long(
        case="swissmetro_nested",
        data_label="Swissmetro",
        model_label="Nested logit",
        source="biogeme.data.swissmetro/data/swissmetro.dat",
        long_df=long_df,
        alternatives=alternatives,
        beta_names=["ASC_TRAIN", "B_TIME", "B_COST", "ASC_CAR"],
        nests={
            "PUBLIC": NestSpec(["TRAIN", "SM"], init=0.8),
            "PRIVATE": NestSpec(["CAR"], init=1.0, fixed=True),
        },
    )


def make_lpmc_case(_: int | None) -> NestedCase:
    base = public_mnl.make_lpmc(None)
    return nested_case_from_design_long(
        case="lpmc_nested",
        data_label="LPMC London",
        model_label="Nested logit",
        source=base.source,
        long_df=public_mnl.make_case_design_long(base),
        alternatives=base.alternatives,
        beta_names=base.parameter_names,
        nests={
            "ACTIVE": NestSpec(["walk", "cycle"], init=0.8),
            "MOTORIZED": NestSpec(["pt", "drive"], init=0.8),
        },
    )


def make_nhts_case(_: int | None) -> NestedCase:
    base = nhts_mnl.make_nhts_2022_case(None)
    return nested_case_from_design_long(
        case="nhts_2022_nested",
        data_label="NHTS 2022",
        model_label="Nested logit",
        source=base.source,
        long_df=nhts_mnl.make_case_design_long(base),
        alternatives=base.alternatives,
        beta_names=base.parameter_names,
        nests={
            "ACTIVE": NestSpec(["WALK", "BIKE"], init=0.8),
            "MOTORIZED": NestSpec(["AUTO", "TRANSIT", "OTHER"], init=0.8),
        },
    )


def make_parking_case(_: int | None) -> NestedCase:
    base = public_mnl.make_parking(None)
    return nested_case_from_design_long(
        case="parking_nested",
        data_label="Parking Spain",
        model_label="Nested logit",
        source=base.source,
        long_df=public_mnl.make_case_design_long(base),
        alternatives=base.alternatives,
        beta_names=base.parameter_names,
        nests={
            "FACILITY": NestSpec(["FSP", "PSP"], init=0.8),
            "PUP_NEST": NestSpec(["PUP"], init=1.0, fixed=True),
        },
    )


CASE_BUILDERS = {
    "swissmetro": make_swissmetro_case,
    "lpmc": make_lpmc_case,
    "nhts": make_nhts_case,
    "parking": make_parking_case,
}


def torch_nests(case: NestedCase) -> dict[str, Nest]:
    return {
        name: Nest(nest.alternatives, init=nest.init, fixed=nest.fixed)
        for name, nest in case.nests.items()
    }


def run_torch(case: NestedCase, max_iter: int) -> BackendResult:
    model = NestedLogit(case.spec, torch_nests(case), max_iter=max_iter)
    data = case.data.to(device=model.device, dtype=model.dtype)
    compiled = model.compile(data)
    initial = torch.cat(
        [
            compiled.free_initial,
            model._lambda_to_internal(compiled.lambda_initial[~compiled.lambda_is_fixed]),
        ]
    )
    params = initial.clone().detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [params],
        max_iter=model.max_iter,
        tolerance_grad=model.tolerance_grad,
        line_search_fn=model.line_search_fn,
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        natural = model._internal_to_natural(params, compiled)
        loss = -model.loglike(natural, data, compiled)
        loss.backward()
        return loss

    estimate_start = time.perf_counter()
    optimizer.step(closure)
    estimate_s = time.perf_counter() - estimate_start
    final_internal = params.detach().clone().requires_grad_(True)
    final_natural = model._internal_to_natural(final_internal, compiled)
    loglike = float(model.loglike(final_natural, data, compiled).detach().cpu())
    covariance_start = time.perf_counter()
    hessian = torch.autograd.functional.hessian(
        lambda p: model.loglike(model._internal_to_natural(p, compiled), data, compiled),
        final_internal,
    )
    cov_internal = torch.linalg.pinv(-hessian.detach(), hermitian=True)
    transform = model._natural_jacobian(final_internal.detach(), compiled)
    covariance = (transform @ cov_internal @ transform.T).detach().cpu().numpy()
    covariance_s = time.perf_counter() - covariance_start
    names = compiled.free_names
    probabilities = model.predict_proba(data, final_natural, compiled).detach().cpu().numpy()
    return BackendResult(
        backend="torchdcm",
        available=True,
        total_s=estimate_s + covariance_s,
        estimate_s=estimate_s,
        covariance_s=covariance_s,
        loglike=loglike,
        params={name: float(final_natural[i].detach().cpu()) for i, name in enumerate(names)},
        covariance=covariance,
        probabilities=probabilities,
    )


def run_biogeme(case: NestedCase, lambda_min: float) -> BackendResult:
    try:
        import biogeme.biogeme as bio
        import biogeme.database as db
        from biogeme import models
        from biogeme.expressions import Beta as BioBeta
        from biogeme.expressions import Variable
        from biogeme.nests import NestsForNestedLogit, OneNestForNestedLogit
        from biogeme.results_processing.variance_covariance import EstimateVarianceCovariance
    except ImportError as exc:
        return BackendResult(backend="biogeme", available=False, message=f"Biogeme unavailable: {exc}")

    df = case.wide_df.copy()
    code_by_alt = {alt: i + 1 for i, alt in enumerate(case.alternatives)}
    df["choice_code"] = df["choice"].map(code_by_alt).astype(int)
    df = df.drop(columns=["choice"])
    for column in df.select_dtypes(include=["bool"]).columns:
        df[column] = df[column].astype(int)
    database = db.Database(f"torchdcm_{case.case}", df)
    betas = {name: BioBeta(name, case.initial_values.get(name, 0.0), None, None, 0) for name in case.beta_names}
    lambda_betas = {
        lambda_name(nest_name): BioBeta(lambda_name(nest_name), nest.init, lambda_min, 1.0, 0)
        for nest_name, nest in case.nests.items()
        if not nest.fixed
    }
    utility = {}
    availability = {}
    for alt, code in code_by_alt.items():
        alt_key = safe_alt(alt)
        expr = 0
        for name in case.beta_names:
            expr += betas[name] * Variable(f"x__{name}__{alt_key}")
        utility[code] = expr
        availability[code] = Variable(f"avail__{alt_key}")
    biogeme_nests = []
    for nest_name, nest in case.nests.items():
        nest_codes = [code_by_alt[alt] for alt in nest.alternatives]
        nest_parameter = 1.0 if nest.fixed else 1.0 / lambda_betas[lambda_name(nest_name)]
        biogeme_nests.append(OneNestForNestedLogit(nest_parameter, nest_codes, name=nest_name))
    nests = NestsForNestedLogit(choice_set=list(code_by_alt.values()), tuple_of_nests=tuple(biogeme_nests))
    logprob = models.lognested(utility, availability, nests, Variable("choice_code"))
    biogeme = bio.BIOGEME(database, logprob)
    biogeme.model_name = f"torchdcm_{case.case}_{case.data.n_obs}"
    biogeme.biogeme_parameters.set_value("save_iterations", False)
    total_start = time.perf_counter()
    estimate_start = time.perf_counter()
    estimates = biogeme.estimate()
    estimate_s = time.perf_counter() - estimate_start
    covariance_start = time.perf_counter()
    covariance_obj = estimates.get_variance_covariance_matrix(EstimateVarianceCovariance.RAO_CRAMER)
    covariance_s = time.perf_counter() - covariance_start
    total_s = time.perf_counter() - total_start
    names = case.parameter_names
    covariance = covariance_to_array(covariance_obj, names)
    beta_values = estimates.get_beta_values()
    return BackendResult(
        backend="biogeme",
        available=True,
        total_s=total_s,
        estimate_s=estimate_s,
        covariance_s=covariance_s,
        loglike=float(estimates.final_log_likelihood),
        params={name: float(beta_values[name]) for name in names},
        covariance=covariance,
    )


def run_apollo(case: NestedCase, lambda_min: float) -> BackendResult:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return BackendResult(backend="apollo", available=False, message="Rscript not found.")
    if not APOLLO_SCRIPT.exists():
        return BackendResult(backend="apollo", available=False, message=f"Missing Apollo script: {APOLLO_SCRIPT}")
    with tempfile.TemporaryDirectory(prefix=f"torchdcm_apollo_{case.case}_") as tmp:
        tmp_path = Path(tmp)
        data_path = tmp_path / "data.csv"
        spec_path = tmp_path / "spec.json"
        output_path = tmp_path / "apollo_result.json"
        df, spec = make_apollo_inputs(case, lambda_min)
        df.to_csv(data_path, index=False)
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        env = os.environ.copy()
        r_user_lib = str(Path.home() / "R" / "site-library")
        existing = env.get("R_LIBS_USER")
        env["R_LIBS_USER"] = r_user_lib if not existing else f"{r_user_lib}:{existing}"
        command = [rscript, str(APOLLO_SCRIPT), "--data", str(data_path), "--spec", str(spec_path), "--output", str(output_path)]
        total_start = time.perf_counter()
        proc = subprocess.run(command, text=True, capture_output=True, env=env)
        total_s = time.perf_counter() - total_start
        if proc.returncode != 0:
            return BackendResult(backend="apollo", available=False, total_s=total_s, message=(proc.stderr or proc.stdout).strip())
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        covariance_names = payload.get("covariance_names") or case.parameter_names
        covariance = reorder_covariance_or_none(payload.get("covariance"), covariance_names, case.parameter_names)
        return BackendResult(
            backend="apollo",
            available=True,
            total_s=total_s,
            estimate_s=payload.get("timing", {}).get("estimate_seconds"),
            covariance_s=payload.get("timing", {}).get("covariance_seconds"),
            loglike=float(payload["loglike"]),
            params={name: float(payload["estimates"][name]) for name in case.parameter_names},
            covariance=covariance,
            message=f"apollo_version={payload.get('apollo_version')}",
        )


def make_apollo_inputs(case: NestedCase, lambda_min: float) -> tuple[pd.DataFrame, dict]:
    code_by_alt = {alt: i + 1 for i, alt in enumerate(case.alternatives)}
    df = case.wide_df.copy()
    df["choice_code"] = df["choice"].map(code_by_alt).astype(int)
    df = df.drop(columns=["choice"])
    for column in df.select_dtypes(include=["bool"]).columns:
        df[column] = df[column].astype(int)
    parameters = {name: case.initial_values.get(name, 0.0) for name in case.beta_names}
    nests = {}
    for nest_name, nest in case.nests.items():
        if nest.fixed:
            nests[nest_name] = {"alternatives": nest.alternatives, "lambda_param": None, "lambda_value": nest.init}
        else:
            lam = np.clip((nest.init - lambda_min) / (1.0 - lambda_min), 1e-12, 1.0 - 1e-12)
            parameters[raw_lambda_name(nest_name)] = float(np.log(lam / (1.0 - lam)))
            nests[nest_name] = {
                "alternatives": nest.alternatives,
                "lambda_param": lambda_name(nest_name),
                "lambda_raw_param": raw_lambda_name(nest_name),
                "lambda_min": lambda_min,
            }
    utility = {}
    for alt in case.alternatives:
        alt_key = safe_alt(alt)
        utility[alt] = {
            "code": code_by_alt[alt],
            "asc": None,
            "availability": f"avail__{alt_key}",
            "variables": {name: f"x__{name}__{alt_key}" for name in case.beta_names},
        }
    spec = {
        "model_name": f"apollo_{case.case}_{case.data.n_obs}",
        "alternatives": case.alternatives,
        "choice_col": "choice_code",
        "parameters": parameters,
        "utility": utility,
        "nests": nests,
    }
    return df, spec


def covariance_to_array(covariance_obj, names: list[str]) -> np.ndarray:
    if hasattr(covariance_obj, "loc"):
        return covariance_obj.loc[names, names].to_numpy(dtype=float)
    return np.asarray(covariance_obj, dtype=float)


def reorder_covariance_or_none(covariance, source_names: list[str], target_names: list[str]) -> np.ndarray | None:
    if covariance is None:
        return None
    matrix = pd.DataFrame(covariance, index=source_names, columns=source_names)
    matrix = matrix.loc[target_names, target_names].apply(pd.to_numeric, errors="coerce")
    array = matrix.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        return None
    return array


def predict_probabilities(case: NestedCase, params: dict[str, float]) -> np.ndarray:
    model = NestedLogit(case.spec, torch_nests(case))
    vector = torch.as_tensor([params[name] for name in case.parameter_names], dtype=torch.float64)
    return model.predict_proba(case.data, vector).detach().cpu().numpy()


def compare(results: list[BackendResult], case: NestedCase) -> None:
    for result in results:
        if result.available and result.probabilities is None:
            result.probabilities = predict_probabilities(case, result.params or {})
    ref = next(result for result in results if result.backend == "torchdcm" and result.available)
    for result in results:
        if not result.available:
            continue
        result.ll_diff = result.loglike - ref.loglike  # type: ignore[attr-defined]
        result.max_param_diff = max(abs(result.params[name] - ref.params[name]) for name in case.parameter_names)  # type: ignore[attr-defined]
        result.max_prob_diff = float(np.max(np.abs(result.probabilities - ref.probabilities)))  # type: ignore[attr-defined]
        if result.covariance is not None and ref.covariance is not None:
            result.max_cov_diff = float(np.max(np.abs(result.covariance - ref.covariance)))  # type: ignore[attr-defined]
            result.max_se_diff = float(
                np.max(np.abs(np.sqrt(np.diag(result.covariance)) - np.sqrt(np.diag(ref.covariance))))
            )  # type: ignore[attr-defined]
        else:
            result.max_cov_diff = None  # type: ignore[attr-defined]
            result.max_se_diff = None  # type: ignore[attr-defined]


def is_consistent(results: list[BackendResult]) -> bool:
    biogeme = next((result for result in results if result.backend == "biogeme"), None)
    if biogeme is None or not biogeme.available:
        return False
    if abs(getattr(biogeme, "ll_diff")) > 1e-4:
        return False
    if getattr(biogeme, "max_param_diff") > 5e-4:
        return False
    if getattr(biogeme, "max_prob_diff") > 2e-4:
        return False
    return True


def result_payload(case: NestedCase, results: list[BackendResult]) -> dict:
    compare(results, case)
    return {
        "case": case.case,
        "data": case.data_label,
        "model": case.model_label,
        "n_obs": case.data.n_obs,
        "n_alternatives": len(case.alternatives),
        "n_parameters": len(case.parameter_names),
        "source": case.source,
        "nests": {name: {"alternatives": nest.alternatives, "fixed": nest.fixed, "init": nest.init} for name, nest in case.nests.items()},
        "parameters": case.parameter_names,
        "consistent": is_consistent(results),
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
                "max_cov_diff": getattr(result, "max_cov_diff", None),
                "max_se_diff": getattr(result, "max_se_diff", None),
                "message": result.message,
            }
            for result in results
        ],
    }


def run_case(case: NestedCase, max_iter: int, lambda_min: float) -> dict:
    results = [
        run_torch(case, max_iter=max_iter),
        run_biogeme(case, lambda_min=lambda_min),
        run_apollo(case, lambda_min=lambda_min),
    ]
    return result_payload(case, results)


def render_markdown(payloads: list[dict]) -> str:
    lines = [
        "# Real-data Nested Logit Benchmark",
        "",
        "| Data | Model | N | TorchDCM | Biogeme | Apollo | Consistent? |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for payload in payloads:
        rows = {row["backend"]: row for row in payload["backends"]}
        lines.append(
            "| {data} | {model} | {n_obs} | {torchdcm} | {biogeme} | {apollo} | {consistent} |".format(
                data=payload["data"],
                model=payload["model"],
                n_obs=payload["n_obs"],
                torchdcm=fmt_time(rows.get("torchdcm")),
                biogeme=fmt_time(rows.get("biogeme")),
                apollo=fmt_time(rows.get("apollo")),
                consistent="Yes" if payload["consistent"] else "No",
            )
        )
    lines.extend(["", "## Nest specifications", ""])
    for payload in payloads:
        nest_text = "; ".join(f"{name}={','.join(spec['alternatives'])}" for name, spec in payload["nests"].items())
        lines.append(f"- `{payload['case']}`: {nest_text}")
    return "\n".join(lines) + "\n"


def fmt_time(row: dict | None) -> str:
    if not row:
        return "NA"
    if not row["available"]:
        return "Fail"
    return f"{row['total_s']:.3f}"


def print_payload(payload: dict) -> None:
    print(f"case: {payload['case']}")
    print(f"data: {payload['data']}")
    print(f"model: {payload['model']}")
    print(f"n_obs: {payload['n_obs']}")
    print(f"n_parameters: {payload['n_parameters']}")
    print(f"consistent: {payload['consistent']}")
    print("nests:")
    for name, spec in payload["nests"].items():
        fixed = " fixed" if spec["fixed"] else ""
        print(f"  {name}: {', '.join(spec['alternatives'])}{fixed}")
    print()
    print(
        f"{'backend':<12}{'available':>10}{'total_s':>12}{'estimate_s':>12}{'cov_s':>12}"
        f"{'loglike':>18}{'ll_diff':>14}{'param_diff':>14}{'prob_diff':>14}{'cov_diff':>14}{'se_diff':>14}"
    )
    for row in payload["backends"]:
        if row["available"]:
            print(
                f"{row['backend']:<12}{str(row['available']):>10}"
                f"{fmt_optional(row['total_s']):>12}{fmt_optional(row['estimate_s']):>12}{fmt_optional(row['covariance_s']):>12}"
                f"{row['loglike']:>18.10f}{row['ll_diff']:>14.3e}{row['max_param_diff']:>14.3e}"
                f"{row['max_prob_diff']:>14.3e}{fmt_optional(row['max_cov_diff']):>14}{fmt_optional(row['max_se_diff']):>14}"
            )
        else:
            print(f"{row['backend']:<12}{str(row['available']):>10}  {row['message']}")


def fmt_optional(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=[*sorted(CASE_BUILDERS), "all"], default="all")
    parser.add_argument("--n-obs", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--lambda-min", type=float, default=0.0001)
    parser.add_argument("--json-output", type=Path, default=GENERATED / "nested_real_battery_full.json")
    parser.add_argument("--md-output", type=Path, default=GENERATED / "nested_real_battery_full.md")
    args = parser.parse_args()

    selected = list(CASE_BUILDERS) if args.case == "all" else [args.case]
    payloads = []
    for key in selected:
        case = CASE_BUILDERS[key](args.n_obs)
        payload = run_case(case, max_iter=args.max_iter, lambda_min=args.lambda_min)
        payloads.append(payload)
        print_payload(payload)
        print()

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payloads, indent=2), encoding="utf-8")
    if args.md_output:
        args.md_output.parent.mkdir(parents=True, exist_ok=True)
        args.md_output.write_text(render_markdown(payloads), encoding="utf-8")


if __name__ == "__main__":
    main()
