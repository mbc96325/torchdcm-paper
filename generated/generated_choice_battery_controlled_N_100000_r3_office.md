# Generated Choice Benchmark Battery (controlled_N_100000_r3_office)

All cross-estimator runtimes report estimation plus covariance on one logical CPU. Synthetic cases vary sample size (N), number of alternatives (J), number of observed variables (K), and equicorrelation (rho). MNL rows compare TorchDCM against SciPy, Biogeme, Apollo, mlogit, gmnl, and xlogit where available. Nested-logit and mixed-logit rows compare TorchDCM against Biogeme and Apollo. Repeated rows report the median of 3 independent backend workers. TorchDCM performs one untimed likelihood-and-gradient warm-up per worker and records LBFGS closure evaluations.

| case | family | N | J | K | rho | TorchDCM | SciPy | Biogeme | Apollo | mlogit | gmnl | xlogit | Consistent? |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| N_100000 | MNL | 100000 | 4 | 6 | 0.3 | 0.107 | 106.261 | 3.520 | 20.587 | 2.805 | 13.466 | 0.734 | Yes |

## Objective Diagnostics

- `N_100000`: reference loglike=-6.10e+04; scipy_bfgs ll_diff=2.47e-10, biogeme ll_diff=0.00e+00, apollo ll_diff=5.38e-10, mlogit ll_diff=0.00e+00, gmnl ll_diff=0.00e+00, xlogit ll_diff=-4.66e-10
