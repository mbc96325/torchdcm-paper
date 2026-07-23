# Generated Choice Benchmark Battery (table4_mixl_true_dgp_softplus)

All cross-estimator runtimes report estimation plus covariance on one logical CPU. Synthetic cases vary sample size (N), number of alternatives (J), number of observed variables (K), and equicorrelation (rho). MNL rows compare TorchDCM against SciPy, Biogeme, Apollo, mlogit, gmnl, and xlogit where available. Nested-logit and mixed-logit rows compare TorchDCM against Biogeme and Apollo.

| case | family | N | J | K | rho | TorchDCM | SciPy | Biogeme | Apollo | mlogit | gmnl | xlogit | Consistent? |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| mixl_N_1000 | Mixed logit | 1000 | 4 | 6 | 0.3 | 0.134 | NA | 67.315 | 2.990 | NA | NA | NA | Yes |
| mixl_N_10000 | Mixed logit | 10000 | 4 | 6 | 0.3 | 1.117 | NA | 76.765 | 39.406 | NA | NA | NA | Yes |
| mixl_N_100000 | Mixed logit | 100000 | 4 | 6 | 0.3 | 17.840 | NA | 201.229 | Timeout | NA | NA | NA | Yes |
| mixl_C_3 | Mixed logit | 20000 | 3 | 6 | 0.3 | 2.751 | NA | 68.418 | 52.979 | NA | NA | NA | Yes |
| mixl_C_10 | Mixed logit | 20000 | 10 | 6 | 0.3 | 9.035 | NA | Timeout | Timeout | NA | NA | NA | -- |
| mixl_C_20 | Mixed logit | 20000 | 20 | 6 | 0.3 | 20.921 | NA | Timeout | Timeout | NA | NA | NA | -- |
| mixl_K_4 | Mixed logit | 20000 | 5 | 4 | 0.3 | 2.409 | NA | 76.207 | 74.521 | NA | NA | NA | Yes |
| mixl_K_12 | Mixed logit | 20000 | 5 | 12 | 0.3 | 5.700 | NA | Timeout | Timeout | NA | NA | NA | -- |
| mixl_K_32 | Mixed logit | 20000 | 5 | 32 | 0.3 | 10.294 | NA | Timeout | Timeout | NA | NA | NA | -- |
| mixl_rho_0p0 | Mixed logit | 20000 | 5 | 12 | 0.0 | 4.779 | NA | Timeout | Timeout | NA | NA | NA | -- |
| mixl_rho_0p5 | Mixed logit | 20000 | 5 | 12 | 0.5 | 6.120 | NA | Timeout | Timeout | NA | NA | NA | -- |
| mixl_rho_0p98 | Mixed logit | 20000 | 5 | 12 | 0.98 | 6.477 | NA | Timeout | Timeout | NA | NA | NA | -- |
| mixl_stress_small | Mixed logit | 20000 | 12 | 8 | 0.5 | 20.324 | NA | Timeout | Timeout | NA | NA | NA | -- |
| mixl_stress_medium | Mixed logit | 30000 | 16 | 10 | 0.5 | 34.252 | NA | Timeout | Timeout | NA | NA | NA | -- |
| stress_mixl_NJK | Mixed logit | 40000 | 20 | 12 | 0.5 | 86.521 | NA | Timeout | Timeout | NA | NA | NA | -- |

## Objective Diagnostics

- `mixl_N_1000`: reference loglike=-6.63e+02; biogeme ll_diff=-8.35e-04, apollo ll_diff=1.21e+00
- `mixl_N_10000`: reference loglike=-6.40e+03; biogeme ll_diff=-1.33e-05, apollo ll_diff=-2.41e+00
- `mixl_N_100000`: reference loglike=-6.42e+04; biogeme ll_diff=-1.37e-03, apollo ll_diff=NA
- `mixl_C_3`: reference loglike=-9.99e+03; biogeme ll_diff=-3.71e-03, apollo ll_diff=1.76e+00
- `mixl_C_10`: reference loglike=-2.17e+04; biogeme ll_diff=NA, apollo ll_diff=NA
- `mixl_C_20`: reference loglike=-2.88e+04; biogeme ll_diff=NA, apollo ll_diff=NA
- `mixl_K_4`: reference loglike=-1.89e+04; biogeme ll_diff=-4.72e-08, apollo ll_diff=-1.72e+00
- `mixl_K_12`: reference loglike=-8.83e+03; biogeme ll_diff=NA, apollo ll_diff=NA
- `mixl_K_32`: reference loglike=-4.06e+03; biogeme ll_diff=NA, apollo ll_diff=NA
- `mixl_rho_0p0`: reference loglike=-1.66e+04; biogeme ll_diff=NA, apollo ll_diff=NA
- `mixl_rho_0p5`: reference loglike=-7.05e+03; biogeme ll_diff=NA, apollo ll_diff=NA
- `mixl_rho_0p98`: reference loglike=-4.94e+03; biogeme ll_diff=NA, apollo ll_diff=NA
- `mixl_stress_small`: reference loglike=-1.60e+04; biogeme ll_diff=NA, apollo ll_diff=NA
- `mixl_stress_medium`: reference loglike=-2.11e+04; biogeme ll_diff=NA, apollo ll_diff=NA
- `stress_mixl_NJK`: reference loglike=-2.57e+04; biogeme ll_diff=NA, apollo ll_diff=NA
