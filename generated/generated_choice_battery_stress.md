# Generated Choice Benchmark Battery (stress)

All cross-estimator runtimes report estimation plus covariance on one logical CPU. Stress rows apply a 300-second worker-wall-clock limit to every external backend; timeout is not treated as numerical disagreement.

| case | family | N | J | K | rho | TorchDCM | SciPy | Biogeme | Apollo | mlogit | gmnl | xlogit | Consistent? |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| stress_mnl_large | MNL | 50000 | 35 | 20 | 0.5 | 6.671 | 155.897 | Timeout | Timeout | 41.971 | Timeout | 135.719 | Yes |
| stress_nl_NJK | Nested logit | 50000 | 20 | 12 | 0.5 | 4.293 | NA | Timeout | 27.826 | NA | NA | NA | No |
| stress_mixl_NJK | Mixed logit | 40000 | 20 | 12 | 0.5 | 100.510 | NA | Timeout | Timeout | NA | NA | NA | -- |

## Objective Diagnostics

- `stress_mnl_large`: reference loglike=-1.98e+04; scipy_bfgs ll_diff=2.75e-09, biogeme ll_diff=NA, apollo ll_diff=NA, mlogit ll_diff=2.56e-09, gmnl ll_diff=NA, xlogit ll_diff=-1.86e-09
- `stress_nl_NJK`: reference loglike=-2.95e+04; biogeme ll_diff=NA, apollo ll_diff=-2.19e+04
- `stress_mixl_NJK`: reference loglike=-2.38e+04; biogeme ll_diff=NA, apollo ll_diff=NA
