# Real-data Mixed Logit Battery

All cross-estimator runtimes report estimation plus covariance on one logical CPU. Each runnable model uses 2-4 independent normal random coefficients selected from observed-variable coefficients first, then ASC terms only when needed.

| case | N | RC | TorchDCM s | Biogeme s | LL diff | Param diff | Prob diff | Consistent? |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| swissmetro | 10719 | B_TIME, B_COST | 0.383 | 19.915 | -2.90e-06 | 2.83e-04 | 3.11e-05 | Yes |
| airline | 3609 | B_TRIP_TIME, B_FARE, B_LEGROOM | 0.185 | 23.786 | 1.96e-10 | 3.99e-06 | 4.70e-07 | Yes |
| parking | 1576 | B_ACCESS_TIME, B_SEARCH_TIME, B_FEE | 0.170 | 23.909 | -4.98e-03 | 3.81e+00 | 2.60e-02 | No |
| telephone | 434 | B_COST, ASC_A2 | 0.036 | 23.035 | -1.37e-01 | 2.29e+00 | 4.48e-02 | No |
| lpmc | 81086 | B_TIME, B_COST | 14.908 | 39.069 | -1.14e+02 | 1.72e+00 | 6.94e-02 | No |
| mlogit_car | NA | skipped | NA | NA | NA | NA | NA | No |
| mlogit_catsup | 2798 | B_DISP, B_FEAT, B_PRICE | 0.266 | 30.344 | -7.40e-06 | 5.98e-03 | 2.10e-04 | Yes |
| mlogit_cracker | 3292 | B_DISP, B_FEAT, B_PRICE | 0.368 | 33.894 | -2.20e+01 | 1.09e+00 | 1.30e-01 | No |
| mlogit_electricity | 4308 | B_PF, B_CL, B_LOC, B_WK | 1.118 | 58.016 | nan | inf | nan | No |
| mlogit_fishing | 1182 | B_PRICE, B_CATCH | 0.146 | 22.336 | -5.62e+00 | 7.81e-01 | 6.03e-02 | No |
| mlogit_game | NA | skipped | NA | NA | NA | NA | NA | No |
| mlogit_game2 | NA | skipped | NA | NA | NA | NA | NA | No |
| mlogit_hc | 250 | B_ICH, B_OCH | 0.139 | 39.376 | -2.74e+01 | 6.35e-03 | 3.80e-01 | No |
| mlogit_heating | 900 | B_IC, B_OC | 0.283 | 27.642 | -3.53e+00 | 4.34e-03 | 8.72e-02 | No |
| mlogit_japanese_fdi | NA | skipped | NA | NA | NA | NA | NA | No |
| mlogit_mode | 453 | B_COST, B_TIME | 0.043 | 22.933 | 3.18e-10 | 6.30e-07 | 9.11e-07 | Yes |
| mlogit_modecanada | 4324 | B_COST, B_IVT, B_OVT, B_FREQ | 76.503 | 56.206 | -3.80e+02 | 1.84e-01 | 3.83e-01 | No |
| mlogit_nox | 632 | B_POST, B_VCOST, B_KCOST | 0.189 | 187.263 | -6.26e-08 | 1.17e-04 | 8.66e-06 | Yes |
| mlogit_risky_transport | 1793 | B_COST, B_RISK, B_SEATS, B_NOISE | 14.740 | 51.971 | -2.50e+00 | 2.91e+00 | 7.48e-02 | No |
| mlogit_train | 2929 | B_PRICE, B_TIME, B_CHANGE, B_COMFORT | 0.554 | 29.860 | 1.01e+02 | 4.98e+02 | 4.69e-01 | No |

## Specifications

- `swissmetro`: random coefficients = B_TIME, B_COST; parameters = ASC_TRAIN, B_TIME, B_COST, ASC_CAR.
- `airline`: random coefficients = B_TRIP_TIME, B_FARE, B_LEGROOM; parameters = B_TRIP_TIME, B_FARE, B_LEGROOM, ASC_ALT2, ASC_ALT3.
- `parking`: random coefficients = B_ACCESS_TIME, B_SEARCH_TIME, B_FEE; parameters = B_ACCESS_TIME, B_SEARCH_TIME, B_FEE, ASC_PSP, ASC_PUP.
- `telephone`: random coefficients = B_COST, ASC_A2; parameters = B_COST, ASC_A2, ASC_A3, ASC_A4, ASC_A5.
- `lpmc`: random coefficients = B_TIME, B_COST; parameters = B_TIME, B_COST, ASC_CYCLE, ASC_PT, ASC_DRIVE.
- `mlogit_car` skipped: RuntimeError: Error in if (abs(x - oldx) < ftol) { :
  missing value where TRUE/FALSE needed
Calls: fit_long -> mlogit -> eval -> eval -> mlogit.optim
Execution halted
- `mlogit_catsup`: random coefficients = B_DISP, B_FEAT, B_PRICE; parameters = B_DISP, B_FEAT, B_PRICE.
- `mlogit_cracker`: random coefficients = B_DISP, B_FEAT, B_PRICE; parameters = B_DISP, B_FEAT, B_PRICE.
- `mlogit_electricity`: random coefficients = B_PF, B_CL, B_LOC, B_WK; parameters = B_PF, B_CL, B_LOC, B_WK, B_TOD, B_SEAS.
- `mlogit_fishing`: random coefficients = B_PRICE, B_CATCH; parameters = B_PRICE, B_CATCH.
- `mlogit_game` skipped: RuntimeError: Need at least two observed variables for 2+ random coefficients; found ['own'].
- `mlogit_game2` skipped: RuntimeError: Need at least two observed variables for 2+ random coefficients; found ['own'].
- `mlogit_hc`: random coefficients = B_ICH, B_OCH; parameters = B_ICH, B_OCH.
- `mlogit_heating`: random coefficients = B_IC, B_OC; parameters = B_IC, B_OC.
- `mlogit_japanese_fdi` skipped: case exceeded 300s worker limit
- `mlogit_mode`: random coefficients = B_COST, B_TIME; parameters = B_COST, B_TIME.
- `mlogit_modecanada`: random coefficients = B_COST, B_IVT, B_OVT, B_FREQ; parameters = B_COST, B_IVT, B_OVT, B_FREQ.
- `mlogit_nox`: random coefficients = B_POST, B_VCOST, B_KCOST; parameters = B_POST, B_VCOST, B_KCOST.
- `mlogit_risky_transport`: random coefficients = B_COST, B_RISK, B_SEATS, B_NOISE; parameters = B_COST, B_RISK, B_SEATS, B_NOISE, B_CROWDNESS, B_CONVLOC, B_CLIENTELE.
- `mlogit_train`: random coefficients = B_PRICE, B_TIME, B_CHANGE, B_COMFORT; parameters = B_PRICE, B_TIME, B_CHANGE, B_COMFORT.
