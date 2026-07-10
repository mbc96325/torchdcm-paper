# Generated Choice Benchmark Battery (stress)

Stress rows compare TorchDCM against Biogeme and Apollo under a 300-second per-backend timeout. Timeout means the backend did not finish the aligned full-estimation run within the wall-clock budget; it is not treated as a numerical inconsistency.

| case | family | N | J | K | rho | TorchDCM | SciPy | Biogeme | Apollo | mlogit | gmnl | xlogit | Consistent? |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| stress_mnl_NJK | MNL | 50000 | 35 | 20 | 0.5 | 6.224 | NA | Timeout | Timeout | NA | NA | NA | No |
| stress_nl_NJK | Nested logit | 50000 | 20 | 12 | 0.5 | 2.678 | NA | Timeout | 57.066 | NA | NA | NA | Timeout |
| stress_mixl_NJK | Mixed logit | 20000 | 8 | 8 | 0.5 | 2.560 | NA | 134.011 | Timeout | NA | NA | NA | Timeout |

## Objective Diagnostics

- `stress_mnl_NJK`: reference loglike=-1.98e+04; biogeme ll_diff=NA, apollo ll_diff=NA
- `stress_nl_NJK`: reference loglike=-2.95e+04; biogeme ll_diff=NA, apollo ll_diff=-2.19e+04
- `stress_mixl_NJK`: reference loglike=-1.28e+04; biogeme ll_diff=-2.54e-05, apollo ll_diff=NA
