# Real-data Mixed Logit Battery

All cross-estimator runtimes report estimation plus covariance on one logical CPU. Each runnable model uses 2-4 independent normal random coefficients selected from observed-variable coefficients first, then ASC terms only when needed.

| case | N | RC | TorchDCM s | Biogeme s | LL diff | Param diff | Prob diff | Consistent? |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| mlogit_mode | 453 | B_COST, B_TIME | 0.233 | 25.504 | 3.18e-10 | 6.30e-07 | 9.11e-07 | Yes |
| mlogit_modecanada | 4324 | B_COST, B_IVT, B_OVT, B_FREQ | 77.299 | 64.637 | -3.80e+02 | 1.84e-01 | 3.84e-01 | No |
| mlogit_nox | 632 | B_POST, B_VCOST, B_KCOST | 0.313 | 209.003 | -6.26e-08 | 1.17e-04 | 8.66e-06 | Yes |
| mlogit_risky_transport | 1793 | B_COST, B_RISK, B_SEATS, B_NOISE | 14.594 | 59.597 | -2.50e+00 | 2.91e+00 | 7.48e-02 | No |
| mlogit_train | 2929 | B_PRICE, B_TIME, B_CHANGE, B_COMFORT | 0.660 | 34.256 | 1.52e+02 | 9.40e+01 | 5.00e-01 | No |

## Specifications

- `mlogit_mode`: random coefficients = B_COST, B_TIME; parameters = B_COST, B_TIME.
- `mlogit_modecanada`: random coefficients = B_COST, B_IVT, B_OVT, B_FREQ; parameters = B_COST, B_IVT, B_OVT, B_FREQ.
- `mlogit_nox`: random coefficients = B_POST, B_VCOST, B_KCOST; parameters = B_POST, B_VCOST, B_KCOST.
- `mlogit_risky_transport`: random coefficients = B_COST, B_RISK, B_SEATS, B_NOISE; parameters = B_COST, B_RISK, B_SEATS, B_NOISE, B_CROWDNESS, B_CONVLOC, B_CLIENTELE.
- `mlogit_train`: random coefficients = B_PRICE, B_TIME, B_CHANGE, B_COMFORT; parameters = B_PRICE, B_TIME, B_CHANGE, B_COMFORT.