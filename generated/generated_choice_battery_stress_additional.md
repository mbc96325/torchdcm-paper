# Generated Choice Benchmark Battery (stress_additional)

All cross-estimator runtimes report estimation plus covariance on one logical CPU. Stress rows apply a 300-second worker-wall-clock limit to every external backend; timeout is not treated as numerical disagreement.

| case | family | N | J | K | rho | TorchDCM | SciPy | Biogeme | Apollo | mlogit | gmnl | xlogit | Consistent? |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| stress_mnl_small | MNL | 30000 | 20 | 12 | 0.5 | 1.123 | 50.930 | 84.292 | 71.288 | 7.948 | Timeout | 11.657 | Yes |
| stress_mnl_medium | MNL | 40000 | 28 | 16 | 0.5 | 3.183 | 76.779 | Timeout | Timeout | 20.800 | Timeout | 49.511 | Yes |

## Objective Diagnostics

- `stress_mnl_small`: reference loglike=-1.80e+04; scipy_bfgs ll_diff=1.53e-10, biogeme ll_diff=-3.76e-09, apollo ll_diff=-8.55e-10, mlogit ll_diff=8.00e-11, gmnl ll_diff=NA, xlogit ll_diff=-5.71e-08
- `stress_mnl_medium`: reference loglike=-1.92e+04; scipy_bfgs ll_diff=3.35e-10, biogeme ll_diff=NA, apollo ll_diff=NA, mlogit ll_diff=3.31e-10, gmnl ll_diff=NA, xlogit ll_diff=-4.68e-09
