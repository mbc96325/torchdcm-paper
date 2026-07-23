from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from torchdcm import Beta, ChoiceDataset, MultinomialLogit, UtilitySpec


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "generated"


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    n_obs: int
    n_alternatives: int
    n_features: int
    feature_corr: float
    signal_scale: float = 1.0
    seed: int = 20260704


@dataclass
class SyntheticResult:
    case: dict
    n_rows: int
    n_parameters: int
    torch_total_s: float
    torch_estimate_s: float
    torch_cov_s: float
    loglike_at_true: float
    loglike_at_mle: float
    ll_gain_over_true: float
    max_beta_error: float
    rmse_beta_error: float
    max_prob_error_vs_true: float
    mean_prob_error_vs_true: float
    max_classic_se: float
    gradient_norm: float
    closure_evaluations: int


def profile_cases(profile: str) -> list[SyntheticCase]:
    if profile == "smoke":
        return [
            SyntheticCase("smoke_base", 2_000, 4, 6, 0.0, 1.0, 11),
            SyntheticCase("smoke_corr", 2_000, 4, 6, 0.8, 1.0, 12),
        ]

    cases: list[SyntheticCase] = []
    for n_obs in [1_000, 10_000, 100_000]:
        cases.append(SyntheticCase(f"N_{n_obs}", n_obs, 4, 6, 0.3, 1.0, 100 + n_obs))
    for n_alternatives in [3, 5, 10, 20]:
        cases.append(SyntheticCase(f"J_{n_alternatives}", 20_000, n_alternatives, 6, 0.3, 1.0, 200 + n_alternatives))
    for n_features in [4, 8, 16, 32]:
        cases.append(SyntheticCase(f"K_{n_features}", 20_000, 5, n_features, 0.3, 1.0, 300 + n_features))
    for feature_corr in [0.0, 0.5, 0.9, 0.98]:
        name = f"rho_{str(feature_corr).replace('.', 'p')}"
        cases.append(SyntheticCase(name, 20_000, 5, 12, feature_corr, 1.0, 400 + int(feature_corr * 100)))
    for signal_scale in [0.5, 1.0, 1.5]:
        name = f"signal_{str(signal_scale).replace('.', 'p')}"
        cases.append(SyntheticCase(name, 20_000, 5, 12, 0.5, signal_scale, 500 + int(signal_scale * 10)))
    return cases


def make_synthetic_mnl(case: SyntheticCase) -> tuple[ChoiceDataset, UtilitySpec, dict[str, float], np.ndarray]:
    rng = np.random.default_rng(case.seed)
    n, j, k = case.n_obs, case.n_alternatives, case.n_features
    corr = np.full((k, k), case.feature_corr, dtype=float)
    np.fill_diagonal(corr, 1.0)
    min_eigen = np.linalg.eigvalsh(corr).min()
    if min_eigen <= 1e-10:
        corr = corr + np.eye(k) * (1e-8 - min_eigen)
    chol = np.linalg.cholesky(corr)

    raw = rng.standard_normal((n, j, k))
    x = raw @ chol.T
    x = (x - x.mean(axis=(0, 1), keepdims=True)) / x.std(axis=(0, 1), keepdims=True)

    beta_generic = np.linspace(-0.8, 0.8, k)
    beta_generic += 0.15 * rng.standard_normal(k)
    beta_generic = case.signal_scale * beta_generic / max(np.linalg.norm(beta_generic), 1e-12) * math.sqrt(k) * 0.35
    asc = np.zeros(j)
    if j > 1:
        asc[1:] = case.signal_scale * np.linspace(-0.4, 0.4, j - 1)
    true_params = {f"ASC_ALT{alt_idx}": float(asc[alt_idx]) for alt_idx in range(1, j)}
    true_params.update({f"B_X{feature_idx}": float(beta_generic[feature_idx]) for feature_idx in range(k)})

    utilities = x @ beta_generic + asc.reshape(1, j)
    probabilities = softmax_np(utilities, axis=1)
    choices = np.asarray([rng.choice(j, p=probabilities[row]) for row in range(n)], dtype=np.int64)

    obs_ptr = torch.arange(0, (n + 1) * j, j, dtype=torch.long)
    alt_id = torch.arange(j, dtype=torch.long).repeat(n)
    chosen_row = torch.arange(n, dtype=torch.long) * j + torch.as_tensor(choices, dtype=torch.long)
    x_alt = {
        f"x{idx}": torch.as_tensor(x[:, :, idx].reshape(-1), dtype=torch.float64)
        for idx in range(k)
    }
    data = ChoiceDataset(
        obs_ptr=obs_ptr,
        alt_id=alt_id,
        chosen_row=chosen_row,
        x_alt=x_alt,
        weights=torch.ones(n, dtype=torch.float64),
        availability=torch.ones(n * j, dtype=torch.bool),
        obs_ids=list(range(n)),
        alt_names=[f"ALT{idx}" for idx in range(j)],
    )

    spec = UtilitySpec()
    for alt_idx in range(j):
        expr = None
        if alt_idx > 0:
            expr = Beta(f"ASC_ALT{alt_idx}", init=0.0)
        for feature_idx in range(k):
            term = Beta(f"B_X{feature_idx}", init=0.0) * f"x{feature_idx}"
            expr = term if expr is None else expr + term
        if expr is None:
            raise RuntimeError("Synthetic MNL specification unexpectedly empty.")
        spec.utility(f"ALT{alt_idx}", expr)

    return data, spec, true_params, probabilities.reshape(-1)


def fit_torch_timed(data: ChoiceDataset, spec: UtilitySpec, max_iter: int) -> dict:
    model = MultinomialLogit(spec, max_iter=max_iter, tolerance_grad=1e-8)
    compiled = model.compile(data)
    params = compiled.free_initial.clone().detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [params],
        max_iter=max_iter,
        tolerance_grad=1e-8,
        line_search_fn="strong_wolfe",
    )
    iterations = {"count": 0}

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = -model.loglike(params, data, compiled)
        loss.backward()
        iterations["count"] += 1
        return loss

    start = time.perf_counter()
    optimizer.step(closure)
    estimate_s = time.perf_counter() - start

    final_params = params.detach().clone()
    final_params.requires_grad_(True)
    ll = model.loglike(final_params, data, compiled)
    grad = torch.autograd.grad(ll, final_params, create_graph=False)[0].detach()

    cov_start = time.perf_counter()
    hessian_ll = torch.autograd.functional.hessian(lambda p: model.loglike(p, data, compiled), final_params)
    information = -hessian_ll.detach()
    covariance = torch.linalg.pinv(information)
    cov_s = time.perf_counter() - cov_start

    probabilities = model.predict_proba(data, final_params.detach(), compiled).detach().cpu().numpy()
    return {
        "model": model,
        "compiled": compiled,
        "params": final_params.detach().cpu().numpy(),
        "probabilities": probabilities,
        "loglike": float(ll.detach().cpu()),
        "covariance": covariance.detach().cpu().numpy(),
        "gradient_norm": float(torch.linalg.vector_norm(grad).detach().cpu()),
        "closure_evaluations": iterations["count"],
        "estimate_s": estimate_s,
        "cov_s": cov_s,
        "total_s": estimate_s + cov_s,
    }


def evaluate_case(case: SyntheticCase, max_iter: int) -> SyntheticResult:
    data, spec, true_param_map, true_probabilities = make_synthetic_mnl(case)
    true_model = MultinomialLogit(spec)
    compiled = true_model.compile(data)
    true_params = np.asarray([true_param_map[name] for name in compiled.free_names], dtype=float)
    true_tensor = torch.as_tensor(true_params, dtype=torch.float64)
    ll_true = float(true_model.loglike(true_tensor, data, compiled).detach().cpu())

    fit = fit_torch_timed(data, spec, max_iter)
    beta_error = fit["params"] - true_params
    prob_error = fit["probabilities"] - true_probabilities
    se = np.sqrt(np.maximum(np.diag(fit["covariance"]), 0.0))
    return SyntheticResult(
        case=asdict(case),
        n_rows=data.n_rows,
        n_parameters=len(true_params),
        torch_total_s=fit["total_s"],
        torch_estimate_s=fit["estimate_s"],
        torch_cov_s=fit["cov_s"],
        loglike_at_true=ll_true,
        loglike_at_mle=fit["loglike"],
        ll_gain_over_true=fit["loglike"] - ll_true,
        max_beta_error=float(np.max(np.abs(beta_error))),
        rmse_beta_error=float(np.sqrt(np.mean(beta_error**2))),
        max_prob_error_vs_true=float(np.max(np.abs(prob_error))),
        mean_prob_error_vs_true=float(np.mean(np.abs(prob_error))),
        max_classic_se=float(np.max(se)),
        gradient_norm=fit["gradient_norm"],
        closure_evaluations=fit["closure_evaluations"],
    )


def write_outputs(results: list[SyntheticResult], profile: str) -> tuple[Path, Path]:
    GENERATED.mkdir(parents=True, exist_ok=True)
    json_path = GENERATED / f"synthetic_controlled_mnl_{profile}.json"
    md_path = GENERATED / f"synthetic_controlled_mnl_{profile}.md"
    payload = [asdict(result) for result in results]
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(results, profile), encoding="utf-8")
    return json_path, md_path


def render_markdown(results: list[SyntheticResult], profile: str) -> str:
    lines = [
        f"# Synthetic Controlled MNL Benchmark ({profile})",
        "",
        "Synthetic data are generated from a known MNL data-generating process. The controlled factors are sample size, number of alternatives, number of generic parameters, feature correlation, and utility signal scale.",
        "",
        "| case | N | J | K | rho | signal | params | rows | total_s | est_s | cov_s | beta_rmse | beta_max | prob_mean | prob_max | max_se | grad_norm |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        case = result.case
        lines.append(
            f"| {case['name']} | {case['n_obs']} | {case['n_alternatives']} | {case['n_features']} | "
            f"{case['feature_corr']:.2f} | {case['signal_scale']:.2f} | {result.n_parameters} | {result.n_rows} | "
            f"{result.torch_total_s:.4f} | {result.torch_estimate_s:.4f} | {result.torch_cov_s:.4f} | "
            f"{result.rmse_beta_error:.3e} | {result.max_beta_error:.3e} | "
            f"{result.mean_prob_error_vs_true:.3e} | {result.max_prob_error_vs_true:.3e} | "
            f"{result.max_classic_se:.3e} | {result.gradient_norm:.3e} |"
        )

    lines.extend(
        [
            "",
            "Interpretation notes:",
            "",
            "- `beta_rmse` and `beta_max` compare the fitted MLE with the known data-generating parameters; finite-sample sampling error is expected.",
            "- `prob_mean` and `prob_max` compare fitted probabilities with true data-generating probabilities on the same realized synthetic design matrix.",
            "- `est_s` times only LBFGS parameter estimation; `cov_s` times Hessian/inverse-information covariance.",
            "- Increasing feature correlation stresses Hessian conditioning and parameter recovery without changing the true model class.",
            "",
        ]
    )
    return "\n".join(lines)


def softmax_np(values: np.ndarray, axis: int) -> np.ndarray:
    centered = values - values.max(axis=axis, keepdims=True)
    exp = np.exp(centered)
    return exp / exp.sum(axis=axis, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--max-iter", type=int, default=120)
    args = parser.parse_args()

    results: list[SyntheticResult] = []
    for case in profile_cases(args.profile):
        print(f"[synthetic] running {case.name}", flush=True)
        start = time.perf_counter()
        result = evaluate_case(case, args.max_iter)
        results.append(result)
        print(
            f"[synthetic] {case.name}: total={result.torch_total_s:.4f}s "
            f"beta_rmse={result.rmse_beta_error:.3e} wall={time.perf_counter() - start:.3f}s",
            flush=True,
        )
    json_path, md_path = write_outputs(results, args.profile)
    print(f"json: {json_path}")
    print(f"markdown: {md_path}")


if __name__ == "__main__":
    main()
