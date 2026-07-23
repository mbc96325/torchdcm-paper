# Generated Choice Benchmark Battery (controlled_N_10000_r3_office)

All cross-estimator runtimes report estimation plus covariance on one logical CPU. Synthetic cases vary sample size (N), number of alternatives (J), number of observed variables (K), and equicorrelation (rho). MNL rows compare TorchDCM against SciPy, Biogeme, Apollo, mlogit, gmnl, and xlogit where available. Nested-logit and mixed-logit rows compare TorchDCM against Biogeme and Apollo. Repeated rows report the median of 3 independent backend workers. TorchDCM performs one untimed likelihood-and-gradient warm-up per worker and records LBFGS closure evaluations.

| case | family | N | J | K | rho | TorchDCM | SciPy | Biogeme | Apollo | mlogit | gmnl | xlogit | Consistent? |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| N_10000 | MNL | 10000 | 4 | 6 | 0.3 | 0.015 | 8.954 | 2.214 | 1.029 | 0.334 | 1.833 | 0.074 | Yes |

## Objective Diagnostics

- `N_10000`: reference loglike=-6.05e+03; scipy_bfgs ll_diff=1.36e-11, biogeme ll_diff=5.46e-12, apollo ll_diff=2.55e-11, mlogit ll_diff=5.46e-12, gmnl ll_diff=5.46e-12, xlogit ll_diff=-1.39e-08
