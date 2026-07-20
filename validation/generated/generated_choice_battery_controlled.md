# Generated Choice Benchmark Battery (controlled)

Synthetic cases vary sample size (N), number of alternatives (J), number of observed variables (K), and equicorrelation (rho). MNL rows compare TorchDCM against SciPy, Biogeme, Apollo, mlogit, gmnl, and xlogit where available. Nested-logit and mixed-logit rows compare TorchDCM against Biogeme and Apollo.

The three sample-size rows report the median of three independent worker runs. TorchDCM performs one untimed likelihood-and-gradient warm-up in each worker; its LBFGS closure counts are 16, 17, and 17 as N increases.

| case | family | N | J | K | rho | TorchDCM | SciPy | Biogeme | Apollo | mlogit | gmnl | xlogit | Consistent? |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| N_1000 | MNL | 1000 | 4 | 6 | 0.3 | 0.006 | 0.767 | 2.391 | 0.450 | 0.034 | 0.147 | 0.010 | Yes |
| N_10000 | MNL | 10000 | 4 | 6 | 0.3 | 0.015 | 8.946 | 2.377 | 1.218 | 0.406 | 1.939 | 0.072 | Yes |
| N_100000 | MNL | 100000 | 4 | 6 | 0.3 | 0.132 | 105.606 | 3.946 | 20.080 | 3.324 | 15.361 | 0.771 | Yes |
| J_3 | MNL | 20000 | 3 | 6 | 0.3 | 0.022 | 23.649 | 1.825 | 1.987 | 0.658 | 2.835 | 0.093 | Yes |
| J_10 | MNL | 20000 | 10 | 6 | 0.3 | 0.106 | 25.440 | 7.134 | 7.840 | 1.862 | 25.229 | 0.736 | Yes |
| J_20 | MNL | 20000 | 20 | 6 | 0.3 | 0.478 | 22.153 | 24.138 | 30.662 | 4.276 | 113.837 | 4.379 | Yes |
| K_4 | MNL | 20000 | 5 | 4 | 0.3 | 0.026 | 10.643 | 2.019 | 2.091 | 0.853 | 4.141 | 0.136 | Yes |
| K_12 | MNL | 20000 | 5 | 12 | 0.3 | 0.039 | 32.521 | 6.407 | 6.382 | 1.219 | 17.757 | 0.445 | Yes |
| K_32 | MNL | 20000 | 5 | 32 | 0.3 | 0.128 | 39.878 | 42.517 | 34.958 | 2.241 | 125.070 | 3.208 | Yes |
| rho_0p0 | MNL | 20000 | 5 | 12 | 0.0 | 0.039 | 27.617 | 5.681 | 6.323 | 1.187 | 14.029 | 0.438 | Yes |
| rho_0p5 | MNL | 20000 | 5 | 12 | 0.5 | 0.037 | 44.141 | 6.146 | 6.469 | 1.254 | 17.618 | 0.448 | Yes |
| rho_0p98 | MNL | 20000 | 5 | 12 | 0.98 | 0.048 | 20.520 | 6.439 | 6.293 | 1.253 | 19.432 | 0.451 | Yes |

## Objective Diagnostics

- `N_1000`: reference loglike=-6.09e+02; scipy_bfgs ll_diff=3.56e-11, biogeme ll_diff=-6.69e-07, apollo ll_diff=-2.57e-10, mlogit ll_diff=3.47e-11, gmnl ll_diff=3.47e-11, xlogit ll_diff=-1.34e-08
- `N_10000`: reference loglike=-6.05e+03; scipy_bfgs ll_diff=1.36e-11, biogeme ll_diff=5.46e-12, apollo ll_diff=2.36e-11, mlogit ll_diff=5.46e-12, gmnl ll_diff=5.46e-12, xlogit ll_diff=-1.39e-08
- `N_100000`: reference loglike=-6.10e+04; scipy_bfgs ll_diff=2.47e-10, biogeme ll_diff=0.00e+00, apollo ll_diff=5.38e-10, mlogit ll_diff=0.00e+00, gmnl ll_diff=0.00e+00, xlogit ll_diff=-4.66e-10
- `J_3`: reference loglike=-9.74e+03; scipy_bfgs ll_diff=5.64e-11, biogeme ll_diff=3.64e-12, apollo ll_diff=1.46e-11, mlogit ll_diff=3.64e-12, gmnl ll_diff=1.82e-12, xlogit ll_diff=-1.53e-09
- `J_10`: reference loglike=-2.09e+04; scipy_bfgs ll_diff=4.15e-10, biogeme ll_diff=-8.25e-08, apollo ll_diff=7.28e-11, mlogit ll_diff=2.66e-10, gmnl ll_diff=2.66e-10, xlogit ll_diff=6.91e-11
- `J_20`: reference loglike=-2.78e+04; scipy_bfgs ll_diff=6.91e-11, biogeme ll_diff=-4.98e-10, apollo ll_diff=-7.64e-11, mlogit ll_diff=2.91e-11, gmnl ll_diff=2.91e-11, xlogit ll_diff=-1.03e-09
- `K_4`: reference loglike=-1.83e+04; scipy_bfgs ll_diff=1.49e-10, biogeme ll_diff=9.46e-11, apollo ll_diff=-6.55e-11, mlogit ll_diff=1.13e-10, gmnl ll_diff=1.13e-10, xlogit ll_diff=-1.24e-10
- `K_12`: reference loglike=-8.08e+03; scipy_bfgs ll_diff=6.91e-11, biogeme ll_diff=-4.20e-10, apollo ll_diff=-6.46e-11, mlogit ll_diff=2.82e-11, gmnl ll_diff=2.82e-11, xlogit ll_diff=-1.58e-07
- `K_32`: reference loglike=-3.15e+03; scipy_bfgs ll_diff=1.61e-10, biogeme ll_diff=-6.23e-08, apollo ll_diff=-8.25e-10, mlogit ll_diff=1.57e-10, gmnl ll_diff=1.57e-10, xlogit ll_diff=-4.45e-07
- `rho_0p0`: reference loglike=-1.51e+04; scipy_bfgs ll_diff=4.18e-11, biogeme ll_diff=-5.91e-07, apollo ll_diff=-1.27e-11, mlogit ll_diff=3.64e-12, gmnl ll_diff=3.64e-12, xlogit ll_diff=-6.48e-10
- `rho_0p5`: reference loglike=-6.66e+03; scipy_bfgs ll_diff=2.08e-10, biogeme ll_diff=-1.16e-06, apollo ll_diff=-1.59e-10, mlogit ll_diff=1.76e-10, gmnl ll_diff=1.76e-10, xlogit ll_diff=-7.99e-09
- `rho_0p98`: reference loglike=-5.04e+03; scipy_bfgs ll_diff=1.36e-11, biogeme ll_diff=-1.16e-08, apollo ll_diff=-9.91e-11, mlogit ll_diff=1.00e-11, gmnl ll_diff=1.00e-11, xlogit ll_diff=-2.06e-08
