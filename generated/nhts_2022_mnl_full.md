# NHTS 2022 MNL Full-Estimation Benchmark

Remote run on `baichuan-mo` using the official NHTS 2022 public-use CSV zip.
The benchmark constructs a five-alternative trip-mode choice set from
`TRIPMODE`: `AUTO`, `WALK`, `BIKE`, `TRANSIT`, and `OTHER`.

The Table 1 specification uses alternative-specific coefficients on
standardized `log_miles`, `veh_per_adult`, and `urban` covariates, with shared
zero initial values and classic inverse observed-information covariance.

| Case | Model | N | K | TorchDCM | Biogeme | Consistent? |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| NHTS 2022 | MNL | 27375 | 16 | 0.121 | 1.735 | Yes |

Raw differences from `nhts_2022_mnl_full.json`: log-likelihood difference
`8.731e-11`, max parameter difference `3.923e-06`, max probability difference
`3.781e-07`, max covariance difference `1.393e-07`, and max standard-error
difference `2.891e-07`.

