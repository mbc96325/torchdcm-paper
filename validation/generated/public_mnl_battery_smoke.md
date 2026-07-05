# Public MNL Full-Estimation Battery

This table follows the IJOC software-paper benchmark style: public data, aligned model specification, shared zero starts, runtime split, and numerical parity metrics.

| Case | Dataset | n | k | Torch est. (s) | Torch cov. (s) | Biogeme est. (s) | Biogeme cov. (s) | LL diff | beta diff | prob diff | cov diff | SE diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| airline | biogeme_airline_itinerary | 500 | 5 | 0.024 | 0.001 | 1.173 | 0.000 | -1.47e-08 | 2.53e-05 | 7.43e-06 | 1.13e-06 | 1.68e-06 |
| parking | biogeme_parking_spain | 500 | 5 | 0.019 | 0.001 | 1.161 | 0.000 | 3.54e-11 | 1.41e-06 | 2.11e-07 | 7.93e-08 | 1.23e-07 |
| telephone | biogeme_telephone | 434 | 5 | 0.021 | 0.001 | 1.167 | 0.000 | -6.35e-09 | 3.54e-05 | 7.51e-06 | 1.77e-06 | 2.52e-06 |
| lpmc | lpmc_london | 500 | 5 | 0.017 | 0.001 | 1.156 | 0.000 | -4.49e-10 | 1.47e-05 | 4.58e-07 | 2.77e-06 | 2.84e-06 |

## Per-Case Logs

### airline

```text
case: airline
dataset_id: biogeme_airline_itinerary
model: Airline itinerary MNL
n_obs: 500
n_alternatives: 3
n_parameters: 5
alignment:
  benchmark_mode: full_estimation
  data_source: Biogeme data page airline.dat
  initial_values: zeros shared across TorchDCM and Biogeme
  covariance: classic inverse observed information / Rao-Cramer
  reference: torchdcm

backend      available     total_s  estimate_s       cov_s           loglike       ll_diff    param_diff     prob_diff      cov_diff       se_diff
torchdcm          True    0.024487    0.023766    0.000721   -331.5531014330     0.000e+00     0.000e+00     0.000e+00     0.000e+00     0.000e+00
biogeme           True    1.173212    1.173147    0.000064   -331.5531014476    -1.469e-08     2.530e-05     7.429e-06     1.131e-06     1.676e-06
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
n_obs: 500
n_alternatives: 3
n_parameters: 5
alignment:
  benchmark_mode: full_estimation
  data_source: Biogeme data page parking.dat
  initial_values: zeros shared across TorchDCM and Biogeme
  covariance: classic inverse observed information / Rao-Cramer
  reference: torchdcm

backend      available     total_s  estimate_s       cov_s           loglike       ll_diff    param_diff     prob_diff      cov_diff       se_diff
torchdcm          True    0.019344    0.018652    0.000692   -344.3038148046     0.000e+00     0.000e+00     0.000e+00     0.000e+00     0.000e+00
biogeme           True    1.161358    1.161282    0.000075   -344.3038148046     3.536e-11     1.407e-06     2.112e-07     7.931e-08     1.230e-07
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
torchdcm          True    0.022150    0.021237    0.000913   -482.7191245741     0.000e+00     0.000e+00     0.000e+00     0.000e+00     0.000e+00
biogeme           True    1.167237    1.167172    0.000064   -482.7191245805    -6.354e-09     3.538e-05     7.509e-06     1.768e-06     2.519e-06
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
n_obs: 500
n_alternatives: 4
n_parameters: 5
alignment:
  benchmark_mode: full_estimation
  data_source: Biogeme data page lpmc.dat
  initial_values: zeros shared across TorchDCM and Biogeme
  covariance: classic inverse observed information / Rao-Cramer
  reference: torchdcm

backend      available     total_s  estimate_s       cov_s           loglike       ll_diff    param_diff     prob_diff      cov_diff       se_diff
torchdcm          True    0.017472    0.016772    0.000700   -449.0899214559     0.000e+00     0.000e+00     0.000e+00     0.000e+00     0.000e+00
biogeme           True    1.156171    1.156106    0.000064   -449.0899214564    -4.486e-10     1.467e-05     4.578e-07     2.766e-06     2.844e-06
```

stderr:
```text
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
```
