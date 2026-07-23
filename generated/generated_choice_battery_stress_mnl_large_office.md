# Generated Choice Benchmark Battery (stress_mnl_large_office)

All cross-estimator runtimes report estimation plus covariance on one logical CPU. Stress rows apply a 300-second worker-wall-clock limit to every external backend; timeout is not treated as numerical disagreement.

| case | family | N | J | K | rho | TorchDCM | SciPy | Biogeme | Apollo | mlogit | gmnl | xlogit | Consistent? |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| stress_mnl_large | MNL | 50000 | 35 | 20 | 0.5 | 5.203 | 101.333 | Timeout | Timeout | 25.269 | Timeout | 127.409 | Yes |

## Objective Diagnostics

- `stress_mnl_large`: reference loglike=-2.00e+04; scipy_bfgs ll_diff=1.25e-09, biogeme ll_diff=NA, apollo ll_diff=NA, mlogit ll_diff=1.30e-09, gmnl ll_diff=NA, xlogit ll_diff=-3.57e-09
