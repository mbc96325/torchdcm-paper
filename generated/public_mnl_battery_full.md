# Public MNL Full-Estimation Battery

This table follows the IJOC software-paper benchmark style: public data, aligned model specification, shared zero starts, runtime split, and numerical parity metrics.

| Case | Dataset | n | k | Torch est. (s) | Torch cov. (s) | Biogeme est. (s) | Biogeme cov. (s) | LL diff | beta diff | prob diff | cov diff | SE diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| airline | biogeme_airline_itinerary | 3609 | 5 | 0.024 | 0.002 | 1.182 | 0.000 | 1.96e-11 | 1.46e-07 | 1.21e-07 | 9.92e-10 | 3.95e-09 |
| parking | biogeme_parking_spain | 1576 | 5 | 0.025 | 0.001 | 1.169 | 0.000 | 2.65e-10 | 3.88e-06 | 4.84e-07 | 1.90e-08 | 3.73e-08 |
| telephone | biogeme_telephone | 434 | 5 | 0.023 | 0.001 | 1.197 | 0.000 | -6.35e-09 | 3.54e-05 | 7.51e-06 | 1.77e-06 | 2.52e-06 |
| lpmc | lpmc_london | 81086 | 5 | 0.062 | 0.010 | 1.233 | 0.000 | -4.14e-08 | 1.21e-05 | 1.49e-06 | 6.53e-09 | 7.38e-08 |

## Per-Case Logs

### airline

```text
case: airline
dataset_id: biogeme_airline_itinerary
model: Airline itinerary MNL
n_obs: 3609
n_alternatives: 3
n_parameters: 5
alignment:
  benchmark_mode: full_estimation
  data_source: Biogeme data page airline.dat
  initial_values: zeros shared across TorchDCM and Biogeme
  covariance: classic inverse observed information / Rao-Cramer
  reference: torchdcm

backend      available     total_s  estimate_s       cov_s           loglike       ll_diff    param_diff     prob_diff      cov_diff       se_diff
torchdcm          True    0.025511    0.023910    0.001601  -2386.4184135815     0.000e+00     0.000e+00     0.000e+00     0.000e+00     0.000e+00
biogeme           True    1.182210    1.182146    0.000063  -2386.4184135815     1.955e-11     1.459e-07     1.210e-07     9.924e-10     3.948e-09
```

stderr:
```text
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
```

### parking

```text
case: parking
dataset_id: biogeme_parking_spain
model: Parking Spain MNL
n_obs: 1576
n_alternatives: 3
n_parameters: 5
alignment:
  benchmark_mode: full_estimation
  data_source: Biogeme data page parking.dat
  initial_values: zeros shared across TorchDCM and Biogeme
  covariance: classic inverse observed information / Rao-Cramer
  reference: torchdcm

backend      available     total_s  estimate_s       cov_s           loglike       ll_diff    param_diff     prob_diff      cov_diff       se_diff
torchdcm          True    0.025688    0.024532    0.001156  -1121.3392676025     0.000e+00     0.000e+00     0.000e+00     0.000e+00     0.000e+00
biogeme           True    1.168583    1.168513    0.000069  -1121.3392676022     2.649e-10     3.876e-06     4.839e-07     1.901e-08     3.725e-08
```

stderr:
```text
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
```

### telephone

```text
case: telephone
dataset_id: biogeme_telephone
model: Telephone service MNL
n_obs: 434
n_alternatives: 5
n_parameters: 5
alignment:
  benchmark_mode: full_estimation
  data_source: Biogeme data page telephone.dat
  initial_values: zeros shared across TorchDCM and Biogeme
  covariance: classic inverse observed information / Rao-Cramer
  reference: torchdcm

backend      available     total_s  estimate_s       cov_s           loglike       ll_diff    param_diff     prob_diff      cov_diff       se_diff
torchdcm          True    0.023637    0.022736    0.000901   -482.7191245741     0.000e+00     0.000e+00     0.000e+00     0.000e+00     0.000e+00
biogeme           True    1.196635    1.196570    0.000064   -482.7191245805    -6.354e-09     3.538e-05     7.509e-06     1.768e-06     2.519e-06
```

stderr:
```text
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
```

### lpmc

```text
case: lpmc
dataset_id: lpmc_london
model: London Passenger Mode Choice MNL
n_obs: 81086
n_alternatives: 4
n_parameters: 5
alignment:
  benchmark_mode: full_estimation
  data_source: Biogeme data page lpmc.dat
  initial_values: zeros shared across TorchDCM and Biogeme
  covariance: classic inverse observed information / Rao-Cramer
  reference: torchdcm

backend      available     total_s  estimate_s       cov_s           loglike       ll_diff    param_diff     prob_diff      cov_diff       se_diff
torchdcm          True    0.071624    0.061745    0.009879 -74975.9760746140     0.000e+00     0.000e+00     0.000e+00     0.000e+00     0.000e+00
biogeme           True    1.232903    1.232828    0.000074 -74975.9760746554    -4.137e-08     1.207e-05     1.487e-06     6.529e-09     7.379e-08
```

stderr:
```text
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
```
