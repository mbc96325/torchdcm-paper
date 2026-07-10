# Generated Choice Benchmark Battery (full)

Synthetic cases vary sample size (N), number of alternatives (J), number of observed variables (K), and equicorrelation (rho). MNL rows compare TorchDCM against SciPy, Biogeme, Apollo, mlogit, gmnl, and xlogit where available. Nested-logit and mixed-logit rows compare TorchDCM against Biogeme and Apollo.

| case | family | N | J | K | rho | TorchDCM | SciPy | Biogeme | Apollo | mlogit | gmnl | xlogit | Consistent? |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gen_mnl_base | MNL | 1000 | 3 | 4 | 0.0 | 0.140 | 0.260 | 1.147 | 1.084 | 0.522 | 0.665 | 0.005 | Yes |
| gen_mnl_N | MNL | 10000 | 3 | 4 | 0.0 | 0.012 | 9.685 | 0.808 | 1.769 | 0.733 | 1.381 | 0.035 | Yes |
| gen_mnl_J | MNL | 3000 | 8 | 4 | 0.0 | 0.012 | 1.012 | 2.044 | 1.678 | 0.705 | 1.888 | 0.051 | Yes |
| gen_mnl_K | MNL | 3000 | 4 | 12 | 0.0 | 0.009 | 2.364 | 2.219 | 1.946 | 0.655 | 2.048 | 0.053 | Yes |
| gen_mnl_rho | MNL | 3000 | 4 | 6 | 0.8 | 0.007 | 1.061 | 1.531 | 1.326 | 0.601 | 1.218 | 0.023 | Yes |
| gen_nl_base | Nested logit | 1000 | 4 | 4 | 0.0 | 0.022 | NA | 15.227 | 1.317 | NA | NA | NA | Yes |
| gen_nl_N | Nested logit | 5000 | 4 | 4 | 0.0 | 0.056 | NA | 15.245 | 1.930 | NA | NA | NA | Yes |
| gen_nl_J | Nested logit | 2500 | 8 | 4 | 0.0 | 0.078 | NA | 1031.681 | 1.468 | NA | NA | NA | Yes |
| gen_nl_K | Nested logit | 2500 | 4 | 8 | 0.0 | 0.037 | NA | 35.250 | 1.966 | NA | NA | NA | Yes |
| gen_nl_rho | Nested logit | 2500 | 4 | 6 | 0.8 | 0.040 | NA | 29.793 | 1.782 | NA | NA | NA | Yes |
| gen_mixl_base | Mixed logit | 1000 | 3 | 4 | 0.0 | 0.062 | NA | 20.893 | 1.987 | NA | NA | NA | Yes |
| gen_mixl_N | Mixed logit | 3000 | 3 | 4 | 0.0 | 0.092 | NA | 19.721 | 4.318 | NA | NA | NA | Yes |
| gen_mixl_J | Mixed logit | 1500 | 5 | 4 | 0.0 | 0.077 | NA | 48.593 | 4.736 | NA | NA | NA | Yes |
| gen_mixl_K | Mixed logit | 1500 | 4 | 8 | 0.0 | 0.170 | NA | 60.486 | 8.612 | NA | NA | NA | Yes |
| gen_mixl_rho | Mixed logit | 1500 | 4 | 6 | 0.8 | 0.084 | NA | 54.087 | 5.606 | NA | NA | NA | Yes |

## Objective Diagnostics

- `gen_mnl_base`: reference loglike=-6.87e+02; scipy_bfgs ll_diff=3.25e-10, biogeme ll_diff=3.24e-10, apollo ll_diff=3.20e-10, mlogit ll_diff=3.24e-10, gmnl ll_diff=3.24e-10, xlogit ll_diff=1.30e-10
- `gen_mnl_N`: reference loglike=-7.16e+03; scipy_bfgs ll_diff=9.55e-11, biogeme ll_diff=6.18e-11, apollo ll_diff=4.37e-11, mlogit ll_diff=6.09e-11, gmnl ll_diff=6.09e-11, xlogit ll_diff=-9.09e-13
- `gen_mnl_J`: reference loglike=-4.35e+03; scipy_bfgs ll_diff=6.73e-11, biogeme ll_diff=-2.39e-10, apollo ll_diff=4.09e-11, mlogit ll_diff=6.82e-11, gmnl ll_diff=6.73e-11, xlogit ll_diff=6.37e-12
- `gen_mnl_K`: reference loglike=-1.98e+03; scipy_bfgs ll_diff=5.25e-10, biogeme ll_diff=-1.90e-07, apollo ll_diff=5.13e-10, mlogit ll_diff=5.20e-10, gmnl ll_diff=5.20e-10, xlogit ll_diff=-8.28e-09
- `gen_mnl_rho`: reference loglike=-1.32e+03; scipy_bfgs ll_diff=2.59e-10, biogeme ll_diff=-3.61e-07, apollo ll_diff=-3.68e-11, mlogit ll_diff=2.59e-10, gmnl ll_diff=2.59e-10, xlogit ll_diff=-3.34e-08
- `gen_nl_base`: reference loglike=-9.03e+02; biogeme ll_diff=-1.70e-07, apollo ll_diff=3.94e-05
- `gen_nl_N`: reference loglike=-4.57e+03; biogeme ll_diff=-3.05e-06, apollo ll_diff=-3.34e-05
- `gen_nl_J`: reference loglike=-3.61e+03; biogeme ll_diff=-4.83e-07, apollo ll_diff=-4.42e-05
- `gen_nl_K`: reference loglike=-1.90e+03; biogeme ll_diff=-7.60e-07, apollo ll_diff=1.47e-05
- `gen_nl_rho`: reference loglike=-1.08e+03; biogeme ll_diff=-1.34e-06, apollo ll_diff=-3.97e-05
- `gen_mixl_base`: reference loglike=-7.10e+02; biogeme ll_diff=8.50e-10, apollo ll_diff=5.43e-01
- `gen_mixl_N`: reference loglike=-2.17e+03; biogeme ll_diff=1.28e-09, apollo ll_diff=1.06e-01
- `gen_mixl_J`: reference loglike=-1.62e+03; biogeme ll_diff=-1.49e-07, apollo ll_diff=7.62e-03
- `gen_mixl_K`: reference loglike=-1.16e+03; biogeme ll_diff=-3.83e-08, apollo ll_diff=7.72e-02
- `gen_mixl_rho`: reference loglike=-6.76e+02; biogeme ll_diff=-2.51e-07, apollo ll_diff=1.19e-02
