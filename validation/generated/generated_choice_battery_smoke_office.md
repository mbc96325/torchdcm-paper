# Generated Choice Benchmark Battery (smoke_office)

All cross-estimator runtimes report estimation plus covariance on one logical CPU. Synthetic cases vary sample size (N), number of alternatives (J), number of observed variables (K), and equicorrelation (rho). MNL rows compare TorchDCM against SciPy, Biogeme, Apollo, mlogit, gmnl, and xlogit where available. Nested-logit and mixed-logit rows compare TorchDCM against Biogeme and Apollo.

| case | family | N | J | K | rho | TorchDCM | SciPy | Biogeme | Apollo | mlogit | gmnl | xlogit | Consistent? |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gen_mnl_base | MNL | 400 | 3 | 3 | 0.0 | 0.004 | 0.202 | Fail | 0.242 | 0.018 | 0.024 | 0.003 | Yes |
| gen_nl_base | Nested logit | 400 | 4 | 3 | 0.0 | 0.018 | NA | Fail | 0.404 | NA | NA | NA | Yes |
| gen_mixl_base | Mixed logit | 300 | 3 | 3 | 0.0 | 0.023 | NA | Fail | 0.356 | NA | NA | NA | -- |

## Objective Diagnostics

- `gen_mnl_base`: reference loglike=-2.93e+02; scipy_bfgs ll_diff=1.80e-11, biogeme ll_diff=NA, apollo ll_diff=1.66e-11, mlogit ll_diff=1.79e-11, gmnl ll_diff=1.78e-11, xlogit ll_diff=-6.77e-10
- `gen_nl_base`: reference loglike=-3.79e+02; biogeme ll_diff=NA, apollo ll_diff=4.89e-11
- `gen_mixl_base`: reference loglike=-2.29e+02; biogeme ll_diff=NA, apollo ll_diff=NA
