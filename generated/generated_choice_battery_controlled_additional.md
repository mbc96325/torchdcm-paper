# Generated Choice Benchmark Battery (controlled_additional)

Synthetic cases vary sample size (N), number of alternatives (J), number of observed variables (K), and equicorrelation (rho). MNL rows compare TorchDCM against SciPy, Biogeme, Apollo, mlogit, gmnl, and xlogit where available. Nested-logit and mixed-logit rows compare TorchDCM against Biogeme and Apollo.

| case | family | N | J | K | rho | TorchDCM | SciPy | Biogeme | Apollo | mlogit | gmnl | xlogit | Consistent? |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| J_10 | MNL | 20000 | 10 | 6 | 0.3 | 0.228 | 36.908 | 4.170 | 10.684 | 4.518 | 28.171 | 0.741 | Yes |
| K_12 | MNL | 20000 | 5 | 12 | 0.3 | 0.030 | 40.723 | 4.409 | 9.274 | 3.271 | 20.021 | 0.447 | Yes |
| rho_0p5 | MNL | 20000 | 5 | 12 | 0.5 | 0.027 | 57.087 | 3.965 | 9.287 | 3.273 | 19.887 | 0.451 | Yes |

## Objective Diagnostics

- `J_10`: reference loglike=-2.09e+04; scipy_bfgs ll_diff=4.73e-10, biogeme ll_diff=-8.25e-08, apollo ll_diff=7.28e-11, mlogit ll_diff=2.66e-10, gmnl ll_diff=2.66e-10, xlogit ll_diff=6.91e-11
- `K_12`: reference loglike=-8.08e+03; scipy_bfgs ll_diff=6.64e-11, biogeme ll_diff=-4.22e-10, apollo ll_diff=-6.46e-11, mlogit ll_diff=2.82e-11, gmnl ll_diff=2.82e-11, xlogit ll_diff=-1.58e-07
- `rho_0p5`: reference loglike=-6.66e+03; scipy_bfgs ll_diff=2.10e-10, biogeme ll_diff=-1.16e-06, apollo ll_diff=-1.59e-10, mlogit ll_diff=1.76e-10, gmnl ll_diff=1.76e-10, xlogit ll_diff=-7.99e-09
