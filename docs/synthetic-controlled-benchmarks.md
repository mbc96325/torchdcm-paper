# Synthetic Controlled Benchmarks

Synthetic benchmarks complement the public-data estimator comparisons. Public
data validate parity against external software; synthetic data provide a pure
controlled runtime benchmark that is not based on Swissmetro or any other public
empirical dataset.

The current controlled benchmark generates MNL data with known utility
parameters and varies:

- sample size `N`;
- number of alternatives `J`;
- number of generic utility parameters `K`;
- feature correlation `rho`, which stresses Hessian conditioning;
- utility signal scale.

Each row reports parameter-estimation time, covariance/Hessian time, total
runtime, and the controlled data dimensions. Error columns are retained as
sanity checks for the generated MNL problem, but the paper-facing comparison
emphasizes runtime scaling across `N`, `J`, `K`, and `rho`.

Source result files:

- `validation/generated/synthetic_controlled_mnl_full.json`
- `validation/generated/synthetic_controlled_mnl_full.md`

## Full Grid Summary

| case | N | J | K | rho | signal | params | rows | est_s | cov_s | total_s | Consistent? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| N_1000 | 1,000 | 4 | 6 | 0.30 | 1.00 | 9 | 4,000 | 0.019 | 0.001 | 0.021 | Yes |
| N_10000 | 10,000 | 4 | 6 | 0.30 | 1.00 | 9 | 40,000 | 0.009 | 0.003 | 0.012 | Yes |
| N_100000 | 100,000 | 4 | 6 | 0.30 | 1.00 | 9 | 400,000 | 0.046 | 0.022 | 0.068 | Yes |
| J_3 | 20,000 | 3 | 6 | 0.30 | 1.00 | 8 | 60,000 | 0.010 | 0.004 | 0.014 | Yes |
| J_5 | 20,000 | 5 | 6 | 0.30 | 1.00 | 10 | 100,000 | 0.017 | 0.006 | 0.023 | Yes |
| J_10 | 20,000 | 10 | 6 | 0.30 | 1.00 | 15 | 200,000 | 0.067 | 0.020 | 0.087 | Yes |
| J_20 | 20,000 | 20 | 6 | 0.30 | 1.00 | 25 | 400,000 | 0.258 | 0.152 | 0.409 | Yes |
| K_4 | 20,000 | 5 | 4 | 0.30 | 1.00 | 8 | 100,000 | 0.017 | 0.005 | 0.021 | Yes |
| K_8 | 20,000 | 5 | 8 | 0.30 | 1.00 | 12 | 100,000 | 0.018 | 0.007 | 0.025 | Yes |
| K_16 | 20,000 | 5 | 16 | 0.30 | 1.00 | 20 | 100,000 | 0.024 | 0.015 | 0.038 | Yes |
| K_32 | 20,000 | 5 | 32 | 0.30 | 1.00 | 36 | 100,000 | 0.052 | 0.061 | 0.113 | Yes |
| rho_0p0 | 20,000 | 5 | 12 | 0.00 | 1.00 | 16 | 100,000 | 0.021 | 0.011 | 0.031 | Yes |
| rho_0p5 | 20,000 | 5 | 12 | 0.50 | 1.00 | 16 | 100,000 | 0.020 | 0.010 | 0.030 | Yes |
| rho_0p9 | 20,000 | 5 | 12 | 0.90 | 1.00 | 16 | 100,000 | 0.018 | 0.011 | 0.029 | Yes |
| rho_0p98 | 20,000 | 5 | 12 | 0.98 | 1.00 | 16 | 100,000 | 0.043 | 0.011 | 0.054 | Yes |
| signal_0p5 | 20,000 | 5 | 12 | 0.50 | 0.50 | 16 | 100,000 | 0.016 | 0.011 | 0.027 | Yes |
| signal_1p0 | 20,000 | 5 | 12 | 0.50 | 1.00 | 16 | 100,000 | 0.018 | 0.011 | 0.029 | Yes |
| signal_1p5 | 20,000 | 5 | 12 | 0.50 | 1.50 | 16 | 100,000 | 0.018 | 0.011 | 0.029 | Yes |

## Takeaways

- Runtime scales smoothly with sample size, alternatives, and variables:
  `N=100,000` and 400,000 long rows takes `0.068s` total including covariance;
  `J=20` and 400,000
  long rows takes `0.409s` total including covariance; `K=32` and 36 parameters
  takes `0.113s`.
- High feature correlation creates the expected identification stress:
  total runtime rises from `0.031s` at `rho=0` to `0.054s` at `rho=0.98`.
- All full-grid cases complete in less than half a second including covariance
  on the remote benchmark machine.

## Current Scope

This first controlled benchmark is MNL-only. It is intended to support the
controlled runtime-scaling argument in the paper. The next synthetic extensions
should add:

1. nested logit with known nest dissimilarity parameters;
2. mixed logit with controlled random-coefficient variance/correlation;
3. panel length and draw-count sweeps for simulated likelihood;
4. controlled misspecification tests where the fitted model omits a relevant
   variable or uses an incorrect nest structure.
