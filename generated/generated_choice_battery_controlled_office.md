# Generated Choice Benchmark Battery (controlled_office)

All cross-estimator runtimes report estimation plus covariance on one logical CPU. Synthetic cases vary sample size (N), number of alternatives (J), number of observed variables (K), and equicorrelation (rho). MNL rows compare TorchDCM against SciPy, Biogeme, Apollo, mlogit, gmnl, and xlogit where available. Nested-logit and mixed-logit rows compare TorchDCM against Biogeme and Apollo.

| case | family | N | J | K | rho | TorchDCM | SciPy | Biogeme | Apollo | mlogit | gmnl | xlogit | Consistent? |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| N_1000 | MNL | 1000 | 4 | 6 | 0.3 | 0.006 | 0.768 | 2.239 | 0.354 | 0.033 | 0.117 | 0.010 | Yes |
| N_10000 | MNL | 10000 | 4 | 6 | 0.3 | 0.015 | 9.047 | 2.215 | 1.033 | 0.331 | 1.846 | 0.074 | Yes |
| N_100000 | MNL | 100000 | 4 | 6 | 0.3 | 0.107 | 107.744 | 3.462 | 20.676 | 2.831 | 14.080 | 0.759 | Yes |
| J_3 | MNL | 20000 | 3 | 6 | 0.3 | 0.024 | 23.971 | 1.762 | 1.855 | 0.495 | 2.039 | 0.098 | Yes |
| J_10 | MNL | 20000 | 10 | 6 | 0.3 | 0.086 | 25.729 | 6.883 | 5.965 | 1.624 | 17.134 | 0.722 | Yes |
| J_20 | MNL | 20000 | 20 | 6 | 0.3 | 0.337 | 21.923 | 22.980 | 23.805 | 3.415 | 78.993 | 4.060 | Yes |
| K_4 | MNL | 20000 | 5 | 4 | 0.3 | 0.029 | 10.722 | 2.060 | 1.914 | 0.768 | 3.018 | 0.137 | Yes |
| K_12 | MNL | 20000 | 5 | 12 | 0.3 | 0.044 | 32.483 | 5.904 | 5.163 | 0.944 | 14.772 | 0.460 | Yes |
| K_32 | MNL | 20000 | 5 | 32 | 0.3 | 0.087 | 40.246 | 41.282 | 27.004 | 1.563 | 80.933 | 2.880 | Yes |
| rho_0p0 | MNL | 20000 | 5 | 12 | 0.0 | 0.043 | 27.920 | 5.418 | 4.808 | 0.923 | 11.751 | 0.455 | Yes |
| rho_0p5 | MNL | 20000 | 5 | 12 | 0.5 | 0.041 | 44.127 | 5.880 | 5.140 | 1.023 | 14.412 | 0.478 | Yes |
| rho_0p98 | MNL | 20000 | 5 | 12 | 0.98 | 0.053 | 20.545 | 6.109 | 4.897 | 1.030 | 15.981 | 0.474 | Yes |

## Objective Diagnostics

- `N_1000`: reference loglike=-6.09e+02; scipy_bfgs ll_diff=3.56e-11, biogeme ll_diff=-6.69e-07, apollo ll_diff=-2.57e-10, mlogit ll_diff=3.48e-11, gmnl ll_diff=3.47e-11, xlogit ll_diff=-1.34e-08
- `N_10000`: reference loglike=-6.05e+03; scipy_bfgs ll_diff=1.36e-11, biogeme ll_diff=5.46e-12, apollo ll_diff=2.55e-11, mlogit ll_diff=5.46e-12, gmnl ll_diff=5.46e-12, xlogit ll_diff=-1.39e-08
- `N_100000`: reference loglike=-6.10e+04; scipy_bfgs ll_diff=2.47e-10, biogeme ll_diff=0.00e+00, apollo ll_diff=5.38e-10, mlogit ll_diff=0.00e+00, gmnl ll_diff=0.00e+00, xlogit ll_diff=-4.66e-10
- `J_3`: reference loglike=-9.74e+03; scipy_bfgs ll_diff=5.64e-11, biogeme ll_diff=3.64e-12, apollo ll_diff=1.64e-11, mlogit ll_diff=3.64e-12, gmnl ll_diff=1.82e-12, xlogit ll_diff=-1.53e-09
- `J_10`: reference loglike=-2.09e+04; scipy_bfgs ll_diff=4.15e-10, biogeme ll_diff=-8.25e-08, apollo ll_diff=7.28e-11, mlogit ll_diff=2.66e-10, gmnl ll_diff=2.66e-10, xlogit ll_diff=6.91e-11
- `J_20`: reference loglike=-2.78e+04; scipy_bfgs ll_diff=6.91e-11, biogeme ll_diff=-4.98e-10, apollo ll_diff=-7.64e-11, mlogit ll_diff=2.91e-11, gmnl ll_diff=2.91e-11, xlogit ll_diff=-1.03e-09
- `K_4`: reference loglike=-1.83e+04; scipy_bfgs ll_diff=1.49e-10, biogeme ll_diff=9.46e-11, apollo ll_diff=-6.55e-11, mlogit ll_diff=1.13e-10, gmnl ll_diff=1.13e-10, xlogit ll_diff=-1.24e-10
- `K_12`: reference loglike=-8.08e+03; scipy_bfgs ll_diff=6.91e-11, biogeme ll_diff=-4.20e-10, apollo ll_diff=-6.28e-11, mlogit ll_diff=2.82e-11, gmnl ll_diff=2.82e-11, xlogit ll_diff=-1.58e-07
- `K_32`: reference loglike=-3.15e+03; scipy_bfgs ll_diff=1.61e-10, biogeme ll_diff=-6.23e-08, apollo ll_diff=-8.24e-10, mlogit ll_diff=1.57e-10, gmnl ll_diff=1.57e-10, xlogit ll_diff=-4.45e-07
- `rho_0p0`: reference loglike=-1.51e+04; scipy_bfgs ll_diff=4.18e-11, biogeme ll_diff=-5.91e-07, apollo ll_diff=-1.27e-11, mlogit ll_diff=3.64e-12, gmnl ll_diff=3.64e-12, xlogit ll_diff=-6.48e-10
- `rho_0p5`: reference loglike=-6.66e+03; scipy_bfgs ll_diff=2.08e-10, biogeme ll_diff=-1.16e-06, apollo ll_diff=-1.59e-10, mlogit ll_diff=1.76e-10, gmnl ll_diff=1.76e-10, xlogit ll_diff=-7.99e-09
- `rho_0p98`: reference loglike=-5.04e+03; scipy_bfgs ll_diff=1.36e-11, biogeme ll_diff=-1.16e-08, apollo ll_diff=-9.91e-11, mlogit ll_diff=1.00e-11, gmnl ll_diff=1.00e-11, xlogit ll_diff=-2.06e-08
