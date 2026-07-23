from __future__ import annotations

import argparse
import time

import numpy as np

from torchdcm import Beta, MultinomialLogit, UtilitySpec
from torchdcm.spec.expressions import Expression, Term
from compare_biogeme import build_case, run_biogeme


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
        new_terms = [
            Term(
                parameter=convert_beta(term.parameter),
                variable=term.variable,
                multiplier=term.multiplier,
            )
            for term in expr.terms
        ]
        new_spec.utility(alt, Expression(new_terms))
    return new_spec


def make_initial_values(names: list[str], mode: str, seed: int, scale: float) -> dict[str, float]:
    if mode == "zero":
        return {name: 0.0 for name in names}
    if mode == "random":
        rng = np.random.default_rng(seed)
        return {name: float(value) for name, value in zip(names, rng.normal(0.0, scale, len(names)))}
    raise ValueError(f"Unknown initialization mode: {mode}")


def run_torchdcm_timed(data, spec):
    start = time.perf_counter()
    result = MultinomialLogit(spec).fit(data, cov_type="classic")
    elapsed = time.perf_counter() - start
    return result, elapsed


def run_case(case: str, n_obs: int, data_seed: int, initial: str, init_seed: int, init_scale: float):
    df, data, base_spec, alternatives = build_case(case, n_obs, data_seed)
    names = base_spec.parameter_names
    initial_values = make_initial_values(names, initial, init_seed, init_scale)
    spec = spec_with_initials(base_spec, initial_values)

    torch_result, torch_seconds = run_torchdcm_timed(data, spec)

    bio_start = time.perf_counter()
    biogeme_result = run_biogeme(df, alternatives, names, initial_values=initial_values)
    biogeme_seconds = time.perf_counter() - bio_start

    torch_params = dict(zip(torch_result.param_names, torch_result.values))
    diffs = {name: torch_params[name] - biogeme_result["params"][name] for name in names}
    max_abs_param_diff = max(abs(value) for value in diffs.values())
    ll_diff = torch_result.loglike - biogeme_result["loglike"]

    return {
        "case": case,
        "n_obs": n_obs,
        "initial": initial,
        "initial_values": initial_values,
        "torch_seconds": torch_seconds,
        "biogeme_seconds": biogeme_seconds,
        "speed_ratio_biogeme_over_torch": biogeme_seconds / torch_seconds,
        "torch_loglike": torch_result.loglike,
        "biogeme_loglike": biogeme_result["loglike"],
        "loglike_diff": ll_diff,
        "max_abs_param_diff": max_abs_param_diff,
        "param_diffs": diffs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["swissmetro"], default="swissmetro")
    parser.add_argument("--n-obs", type=int, default=500)
    parser.add_argument("--data-seed", type=int, default=3)
    parser.add_argument("--initial", choices=["zero", "random"], default="zero")
    parser.add_argument("--init-seed", type=int, default=20260704)
    parser.add_argument("--init-scale", type=float, default=0.1)
    args = parser.parse_args()

    result = run_case(
        case=args.case,
        n_obs=args.n_obs,
        data_seed=args.data_seed,
        initial=args.initial,
        init_seed=args.init_seed,
        init_scale=args.init_scale,
    )

    print(f"case: {result['case']}")
    print(f"n_obs: {result['n_obs']}")
    print("alignment:")
    print("  benchmark_mode: full_estimation_runtime")
    print("  data_source: biogeme.data.swissmetro/data/swissmetro.dat")
    print("  model: MNL with shared initial values")
    print("  timing: total estimation call time; use compare_mnl_estimators.py for estimate/covariance split")
    print(f"initial: {result['initial']}")
    print("initial_values:")
    for name, value in result["initial_values"].items():
        print(f"  {name}: {value:.12g}")
    print(f"torch_seconds: {result['torch_seconds']:.6f}")
    print(f"biogeme_seconds: {result['biogeme_seconds']:.6f}")
    print(f"speed_ratio_biogeme_over_torch: {result['speed_ratio_biogeme_over_torch']:.3f}")
    print(f"torch_loglike: {result['torch_loglike']:.12f}")
    print(f"biogeme_loglike: {result['biogeme_loglike']:.12f}")
    print(f"loglike_diff: {result['loglike_diff']:.3e}")
    print(f"max_abs_param_diff: {result['max_abs_param_diff']:.3e}")
    print("parameter_diffs:")
    for name, value in result["param_diffs"].items():
        print(f"  {name}: {value:.3e}")


if __name__ == "__main__":
    main()
