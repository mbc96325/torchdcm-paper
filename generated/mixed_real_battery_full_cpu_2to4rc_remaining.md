# Real-data Mixed Logit Battery

All cross-estimator rows use CPU for TorchDCM and Biogeme. Each runnable model uses 2-4 independent normal random coefficients selected from observed-variable coefficients first, then ASC terms only when needed.

| case | N | RC | TorchDCM s | Biogeme s | LL diff | Param diff | Prob diff | Consistent? |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| mlogit_mode | 453 | B_COST, B_TIME | 0.237 | 18.436 | 3.18e-10 | 6.30e-07 | 9.11e-07 | Yes |
| mlogit_modecanada | 4324 | B_COST, B_IVT, B_OVT, B_FREQ | 38.682 | 54.785 | 1.06e-09 | 4.81e-10 | 5.20e-09 | Yes |
| mlogit_nox | 632 | B_POST, B_VCOST, B_KCOST | 0.058 | 156.158 | -6.26e-08 | 1.17e-04 | 8.66e-06 | Yes |
| mlogit_risky_transport | 1793 | B_COST, B_RISK, B_SEATS, B_NOISE | 11.101 | 60.486 | -2.50e+00 | 2.91e+00 | 7.48e-02 | No |
| mlogit_train | 2929 | B_PRICE, B_TIME, B_CHANGE, B_COMFORT | 0.332 | 30.663 | 1.34e+02 | 5.79e+02 | 5.00e-01 | No |

## Specifications

- `mlogit_mode`: random coefficients = B_COST, B_TIME; parameters = B_COST, B_TIME.
- `mlogit_modecanada`: random coefficients = B_COST, B_IVT, B_OVT, B_FREQ; parameters = B_COST, B_IVT, B_OVT, B_FREQ.
- `mlogit_nox`: random coefficients = B_POST, B_VCOST, B_KCOST; parameters = B_POST, B_VCOST, B_KCOST.
- `mlogit_risky_transport`: random coefficients = B_COST, B_RISK, B_SEATS, B_NOISE; parameters = B_COST, B_RISK, B_SEATS, B_NOISE, B_CROWDNESS, B_CONVLOC, B_CLIENTELE.
- `mlogit_train`: random coefficients = B_PRICE, B_TIME, B_CHANGE, B_COMFORT; parameters = B_PRICE, B_TIME, B_CHANGE, B_COMFORT.