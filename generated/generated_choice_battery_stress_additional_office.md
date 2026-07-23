# Generated Choice Benchmark Battery (stress_additional_office)

All cross-estimator runtimes report estimation plus covariance on one logical CPU. Stress rows apply a 300-second worker-wall-clock limit to every external backend; timeout is not treated as numerical disagreement.

| case | family | N | J | K | rho | TorchDCM | SciPy | Biogeme | Apollo | mlogit | gmnl | xlogit | Consistent? |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| stress_mnl_small | MNL | 30000 | 20 | 12 | 0.5 | 0.723 | 43.010 | 75.347 | 61.989 | 6.013 | 201.058 | 10.694 | Yes |
| stress_mnl_medium | MNL | 40000 | 28 | 16 | 0.5 | 2.480 | 73.944 | Timeout | 236.970 | 13.672 | Timeout | 43.686 | Yes |

## Objective Diagnostics

- `stress_mnl_small`: reference loglike=-1.80e+04; scipy_bfgs ll_diff=1.93e-10, biogeme ll_diff=-2.81e-09, apollo ll_diff=-2.07e-09, mlogit ll_diff=8.00e-11, gmnl ll_diff=8.00e-11, xlogit ll_diff=-4.79e-08
- `stress_mnl_medium`: reference loglike=-1.92e+04; scipy_bfgs ll_diff=3.35e-10, biogeme ll_diff=NA, apollo ll_diff=-2.15e-09, mlogit ll_diff=3.31e-10, gmnl ll_diff=NA, xlogit ll_diff=-4.68e-09
