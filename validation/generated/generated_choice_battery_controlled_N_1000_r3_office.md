# Generated Choice Benchmark Battery (controlled_N_1000_r3_office)

All cross-estimator runtimes report estimation plus covariance on one logical CPU. Synthetic cases vary sample size (N), number of alternatives (J), number of observed variables (K), and equicorrelation (rho). MNL rows compare TorchDCM against SciPy, Biogeme, Apollo, mlogit, gmnl, and xlogit where available. Nested-logit and mixed-logit rows compare TorchDCM against Biogeme and Apollo. Repeated rows report the median of 3 independent backend workers. TorchDCM performs one untimed likelihood-and-gradient warm-up per worker and records LBFGS closure evaluations.

| case | family | N | J | K | rho | TorchDCM | SciPy | Biogeme | Apollo | mlogit | gmnl | xlogit | Consistent? |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| N_1000 | MNL | 1000 | 4 | 6 | 0.3 | 0.005 | 0.761 | 2.231 | 0.353 | 0.032 | 0.118 | 0.010 | Yes |

## Objective Diagnostics

- `N_1000`: reference loglike=-6.09e+02; scipy_bfgs ll_diff=3.56e-11, biogeme ll_diff=-6.69e-07, apollo ll_diff=-2.57e-10, mlogit ll_diff=3.48e-11, gmnl ll_diff=3.47e-11, xlogit ll_diff=-1.34e-08
