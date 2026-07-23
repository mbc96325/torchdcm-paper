# Real-data Mixed Logit Battery

All cross-estimator runtimes report estimation plus covariance on one logical CPU. Each runnable model uses 2-4 independent normal random coefficients selected from observed-variable coefficients first, then ASC terms only when needed.

† The solver's final log likelihood is below the row best by more than the stated tolerance; its runtime is retained but its estimate is excluded from consistency.

| case | N | RC | TorchDCM s | Biogeme s | Apollo s | LL diff | Param diff | Prob diff | Consistent? |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| swissmetro | 10719 | B_TIME, B_COST | 0.411 | 24.729 | 8.587 | 3.64e-12 | 1.75e-07 | 3.04e-08 | Yes |
| airline | 3609 | B_TRIP_TIME, B_FARE, B_LEGROOM | 0.183 | 33.840 | 3.954 | -3.15e-08 | 5.13e-05 | 4.71e-06 | Yes |
| parking | 1576 | B_ACCESS_TIME, B_SEARCH_TIME, B_FEE | 0.102 | 33.537 | 1.874 | -3.60e-07 | 3.04e-03 | 2.04e-05 | Yes |
| telephone | 434 | B_COST, ASC_A2 | 0.035 | 28.633 | 0.780 | 1.28e-06 | 1.19e-03 | 4.13e-05 | Yes |
| lpmc | 81086 | B_TIME, B_COST | 17.478 | 73.769 | Fail | -2.10e-02 | 2.23e-05 | 4.91e-06 | Yes |
| mlogit_catsup | 2798 | B_DISP, B_FEAT, B_PRICE | 0.231 | 43.311 | 2.465 | -1.16e-03 | 3.95e-04 | 1.53e-05 | Yes |
| mlogit_cracker | 3292 | B_DISP, B_FEAT, B_PRICE | 0.298 | 43.987 | 3.194 | -7.35e-05 | 3.14e-02 | 7.10e-04 | Yes |
| mlogit_electricity | 4308 | B_PF, B_CL, B_LOC, B_WK | 0.565 | 93.019 | 10.002 | -3.42e-03 | 8.26e-04 | 7.91e-05 | Yes |
| mlogit_fishing | 1182 | B_PRICE, B_CATCH | 0.071 | 28.844 | 0.676 | 1.51e-10 | 3.39e-06 | 4.46e-07 | Yes |
| mlogit_hc | 250 | B_ICH, B_OCH | 0.045 | 103.683 | 0.366† | 1.23e-02 | 5.55e-05 | 3.38e-04 | Yes |
| mlogit_heating | 900 | B_IC, B_OC | 0.109 | 77.211 | 0.715 | -3.97e-06 | 2.12e-08 | 1.09e-06 | Yes |
| mlogit_mode | 453 | B_COST, B_TIME | 0.033 | 28.581 | 0.367 | -5.92e-11 | 8.89e-07 | 2.06e-07 | Yes |
| mlogit_modecanada | 4324 | B_COST, B_IVT, B_OVT, B_FREQ | 65.396 | 170.166 | 5.444 | -1.48e-03 | 2.76e-06 | 1.42e-05 | Yes |
| mlogit_nox | 632 | B_POST, B_VCOST, B_KCOST | 0.151 | 282.477 | 2.137† | -5.37e-08 | 4.94e-05 | 4.71e-05 | Yes |
| mlogit_risky_transport | 1793 | B_COST, B_RISK, B_SEATS, B_NOISE | 17.061 | 84.348 | 7.132 | -9.23e-04 | 3.67e-03 | 2.61e-05 | Yes |
| mlogit_train | 2929 | B_PRICE, B_TIME, B_CHANGE, B_COMFORT | 0.474 | 40.851 | 1.276 | -8.86e-07 | 8.93e-03 | 6.78e-05 | Yes |

## Specifications

- `swissmetro`: random coefficients = B_TIME, B_COST; parameters = ASC_TRAIN, B_TIME, B_COST, ASC_CAR.
- `airline`: random coefficients = B_TRIP_TIME, B_FARE, B_LEGROOM; parameters = B_TRIP_TIME, B_FARE, B_LEGROOM, ASC_ALT2, ASC_ALT3.
- `parking`: random coefficients = B_ACCESS_TIME, B_SEARCH_TIME, B_FEE; parameters = B_ACCESS_TIME, B_SEARCH_TIME, B_FEE, ASC_PSP, ASC_PUP.
- `telephone`: random coefficients = B_COST, ASC_A2; parameters = B_COST, ASC_A2, ASC_A3, ASC_A4, ASC_A5.
- `lpmc`: random coefficients = B_TIME, B_COST; parameters = B_TIME, B_COST, ASC_CYCLE, ASC_PT, ASC_DRIVE.
- `mlogit_catsup`: random coefficients = B_DISP, B_FEAT, B_PRICE; parameters = B_DISP, B_FEAT, B_PRICE.
- `mlogit_cracker`: random coefficients = B_DISP, B_FEAT, B_PRICE; parameters = B_DISP, B_FEAT, B_PRICE.
- `mlogit_electricity`: random coefficients = B_PF, B_CL, B_LOC, B_WK; parameters = B_PF, B_CL, B_LOC, B_WK, B_TOD, B_SEAS.
- `mlogit_fishing`: random coefficients = B_PRICE, B_CATCH; parameters = B_PRICE, B_CATCH.
- `mlogit_hc`: random coefficients = B_ICH, B_OCH; parameters = B_ICH, B_OCH.
- `mlogit_heating`: random coefficients = B_IC, B_OC; parameters = B_IC, B_OC.
- `mlogit_mode`: random coefficients = B_COST, B_TIME; parameters = B_COST, B_TIME.
- `mlogit_modecanada`: random coefficients = B_COST, B_IVT, B_OVT, B_FREQ; parameters = B_COST, B_IVT, B_OVT, B_FREQ.
- `mlogit_nox`: random coefficients = B_POST, B_VCOST, B_KCOST; parameters = B_POST, B_VCOST, B_KCOST.
- `mlogit_risky_transport`: random coefficients = B_COST, B_RISK, B_SEATS, B_NOISE; parameters = B_COST, B_RISK, B_SEATS, B_NOISE, B_CROWDNESS, B_CONVLOC, B_CLIENTELE.
- `mlogit_train`: random coefficients = B_PRICE, B_TIME, B_CHANGE, B_COMFORT; parameters = B_PRICE, B_TIME, B_CHANGE, B_COMFORT.
