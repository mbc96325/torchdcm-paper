from __future__ import annotations

import argparse
import sys

import torch

from torchdcm import MixedLogit, MultinomialLogit, RandomCoefficient
from compare_mnl_estimators import load_biogeme_swissmetro, spec_with_initials, make_initial_values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-obs", type=int, default=500)
    parser.add_argument("--n-draws", type=int, default=64)
    args = parser.parse_args()

    _, data, base_spec, _ = load_biogeme_swissmetro(args.n_obs)
    names = base_spec.parameter_names
    initial_values = make_initial_values(names, mode="zero", seed=20260704, scale=0.1)
    spec = spec_with_initials(base_spec, initial_values)

    mnl = MultinomialLogit(spec)
    mnl_compiled = mnl.compile(data)
    mnl_params = mnl_compiled.free_initial
    mnl_ll = mnl.loglike(mnl_params, data, mnl_compiled)
    mnl_probs = mnl.predict_proba(data, mnl_params, mnl_compiled)

    mixed = MixedLogit(
        spec,
        [RandomCoefficient("B_TIME", sigma_init=0.0, fixed=True)],
        n_draws=args.n_draws,
        seed=11,
        panel=False,
    )
    mixed_compiled = mixed.compile(data)
    mixed_params = mixed_compiled.free_initial
    mixed_ll = mixed.loglike(mixed_params, data, mixed_compiled)
    mixed_probs = mixed.predict_proba(data, mixed_params, mixed_compiled)

    ll_diff = float((mixed_ll - mnl_ll).detach().cpu())
    prob_diff = float(torch.max(torch.abs(mixed_probs - mnl_probs)).detach().cpu())
    print(f"case: biogeme_swissmetro_mixed_logit_sigma0")
    print(f"n_obs: {args.n_obs}")
    print(f"n_draws: {args.n_draws}")
    print(f"mnl_loglike: {float(mnl_ll.detach().cpu()):.12f}")
    print(f"mixed_loglike: {float(mixed_ll.detach().cpu()):.12f}")
    print(f"loglike_diff: {ll_diff:.3e}")
    print(f"probability_max_abs_diff: {prob_diff:.3e}")

    if abs(ll_diff) > 1e-9 or prob_diff > 1e-12:
        sys.exit(1)


if __name__ == "__main__":
    main()
