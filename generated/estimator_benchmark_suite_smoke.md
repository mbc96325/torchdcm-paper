# TorchDCM Estimator Benchmark Suite (smoke)

All commands were run on the remote benchmark machine. Timing columns reported by the underlying scripts split parameter estimation and covariance calculation where the backend exposes both.

| case | dataset | model | mode | status | wall_s | backends |
|---|---|---|---|---:|---:|---|
| swissmetro_mnl_estimate | Biogeme Swissmetro | MNL | full_estimation | ok | 5.302 | torchdcm, scipy_bfgs, biogeme, apollo |
| swissmetro_nested_estimate | Biogeme Swissmetro | Nested Logit | full_estimation | ok | 7.027 | torchdcm, biogeme, apollo |
| swissmetro_cross_nested_estimate | Biogeme Swissmetro | Cross-Nested Logit | full_estimation | ok | 7.984 | torchdcm_fit, biogeme |
| swissmetro_mixed_panel_fixed | Biogeme Swissmetro | Mixed Logit | fixed_replay_shared_draws | ok | 8.303 | torchdcm_fixed, apollo_r_fixed, biogeme_fixed |
| swissmetro_wtp_mixed_panel_fixed | Biogeme Swissmetro | WTP Mixed Logit | fixed_replay_shared_draws | ok | 8.514 | torchdcm_fixed, apollo_r_fixed, biogeme_fixed |
| swissmetro_latent_class_fit_replay | Biogeme Swissmetro | Latent Class Logit | torch_fit_then_replay | ok | 3.507 | torchdcm_fit, apollo_r_fixed, biogeme_fixed |
| optima_ordered_logit_estimate | Biogeme Optima | Ordered Logit | full_estimation | ok | 6.917 | torchdcm_fit, biogeme |
| optima_ordered_probit_estimate | Biogeme Optima | Ordered Probit | full_estimation | ok | 7.498 | torchdcm_fit, biogeme |
| mlogit_fishing_mnl_estimate | R mlogit Fishing | MNL | full_estimation | ok | 2.697 | torchdcm, mlogit |
| mlogit_modecanada_mnl_estimate | R mlogit ModeCanada | MNL | full_estimation | ok | 2.175 | torchdcm, mlogit |

## swissmetro_mnl_estimate

```text
case: swissmetro
n_obs: 500
alignment:
  benchmark_mode: full_estimation
  data_source: biogeme.data.swissmetro/data/swissmetro.dat
  model: MNL with TRAIN/SM/CAR, ASC_TRAIN, ASC_CAR, B_TIME, B_COST
  scaling: time/100, cost/100, GA discount and official availability filters
  initial_values: shared across TorchDCM, SciPy, Biogeme, Apollo
  covariance: classic inverse observed information where available
  reference: torchdcm
initial: zero
initial_values:
  ASC_TRAIN: 0
  B_TIME: 0
  B_COST: 0
  ASC_CAR: 0

backend      available     total_s  estimate_s       cov_s           loglike       ll_diff    max_param_diff
torchdcm          True    0.020080    0.019427    0.000653   -281.2738197113     0.000e+00         0.000e+00
scipy_bfgs        True    0.096161    0.095251    0.000910   -281.2738197112     5.764e-11         2.205e-06
biogeme           True    1.102820    1.068993    0.000064   -281.2738197112     5.718e-11         2.242e-06
apollo            True    1.068503    0.328000    0.000000   -281.2738000000     1.971e-05         4.344e-05

backend          prob_diff      cov_diff       se_diff           wtp      wtp_diff        wtp_se   wtp_se_diff     elas_time     elas_cost
torchdcm         0.000e+00     0.000e+00     0.000e+00     9.096e-02     0.000e+00     5.925e-02     0.000e+00     0.000e+00     0.000e+00
scipy_bfgs       1.098e-06     1.823e-07     2.765e-07     9.096e-02    -2.089e-07     5.925e-02     1.150e-07     3.674e-06     9.346e-06
biogeme          9.675e-07     1.855e-07     2.813e-07     9.096e-02    -1.724e-07     5.925e-02     1.145e-07     3.361e-06     9.511e-06
apollo           5.491e-05     4.594e-05     2.982e-04     9.098e-02     1.842e-05     5.907e-02    -1.776e-04     1.464e-04     2.006e-04
```

stderr:

```text
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
```

## swissmetro_nested_estimate

```text
case: biogeme_swissmetro_nested
n_obs: 500
alignment:
  benchmark_mode: full_estimation
  data_source: biogeme.data.swissmetro/data/swissmetro.dat
  model: Nested Logit with PUBLIC={TRAIN, SM}, PRIVATE={CAR}
  lambda_constraints: LAMBDA_PUBLIC in [0.0001, 1], PRIVATE fixed to 1
  initial_values: shared across TorchDCM, Biogeme, Apollo
  covariance: classic inverse observed information where available
  reference: torchdcm
initial: zero
nests:
  PUBLIC: TRAIN, SM
  PRIVATE: CAR (lambda fixed to 1)
initial_values:
  ASC_TRAIN: 0
  B_TIME: 0
  B_COST: 0
  ASC_CAR: 0
  LAMBDA_PUBLIC: 0.8

backend      available     total_s  estimate_s       cov_s           loglike       ll_diff    max_param_diff
torchdcm          True    0.032861    0.030075    0.002786   -280.1719304379     0.000e+00         0.000e+00
biogeme           True    2.749395    2.749328    0.000065   -280.1719304434    -5.504e-09         1.110e-05
apollo            True    1.176423    0.419000    0.008000   -280.1719000000     3.044e-05         6.882e-05

backend          prob_diff      cov_diff       se_diff           wtp      wtp_diff        wtp_se   wtp_se_diff     elas_time     elas_cost
torchdcm         0.000e+00     0.000e+00     0.000e+00     5.329e-02     0.000e+00     7.356e-02     0.000e+00     0.000e+00     0.000e+00
biogeme          8.159e-06     1.076e-01     5.910e-02     5.329e-02    -1.639e-06     7.356e-02    -5.584e-07     1.676e-05     4.448e-05
apollo           7.979e-05     4.342e-05     1.464e-04     5.325e-02    -3.540e-05     7.346e-02    -1.085e-04     4.631e-04     1.407e-04
```

stderr:

```text
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
```

## swissmetro_cross_nested_estimate

```text
case: biogeme_swissmetro_cross_nested
mode: full-estimation
n_obs: 500
alignment:
  benchmark_mode: full_estimation
  estimated_backend: each backend estimates independently
  data_source: biogeme.data.swissmetro/data/swissmetro.dat
  model: Cross-Nested Logit with fixed allocation weights
  allocations: PUBLIC={TRAIN:.7, SM:.8}, PRIVATE={TRAIN:.3, SM:.2, CAR:1}
  lambda_constraints: lambdas in [0.0001, 1]
  parameters: shared across replay backends
  reference: torchdcm_fit

backend            available     total_s  estimate_s       cov_s           loglike       ll_diff    param_diff     prob_diff      cov_diff       se_diff        t_diff
torchdcm_fit            True    0.896173          NA          NA   -265.8044577289     0.000e+00     0.000e+00     0.000e+00     0.000e+00            NA            NA
biogeme                 True    4.525132    4.525057    0.000074   -265.7748721470     2.959e-02     5.489e-03     5.393e-03     3.515e-01            NA            NA

torchdcm_fit params:
  ASC_TRAIN: -0.00296916387829
  B_TIME: 8.62027640445e-05
  B_COST: -0.00667295276156
  ASC_CAR: -0.00231744594639
  LAMBDA_PUBLIC: 0.00357949189616
  LAMBDA_PRIVATE: 0.000566196078496
biogeme params:
  ASC_TRAIN: -0.000525983629784
  B_TIME: 1.54735503918e-05
  B_COST: -0.00118398413946
  ASC_CAR: -0.000408286439715
  LAMBDA_PUBLIC: 0.000634143282623
  LAMBDA_PRIVATE: 0.0001
```

stderr:

```text
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
/home/baichuan-mo/torchdcm/validation/benchmarks/compare_cross_nested_logit_estimators.py:377: RuntimeWarning: invalid value encountered in sqrt
  se = np.sqrt(np.diag(result.covariance))
/home/baichuan-mo/torchdcm/validation/benchmarks/compare_cross_nested_logit_estimators.py:378: RuntimeWarning: invalid value encountered in sqrt
  ref_se = np.sqrt(np.diag(ref.covariance))
```

## swissmetro_mixed_panel_fixed

```text
case: biogeme_swissmetro_mixed_logit
mode: fixed
n_obs: 500
n_draws: 32
panel: True
random_coefficients: ['B_TIME']
correlated: False
error_component_public: False
alignment:
  benchmark_mode: fixed_likelihood_replay
  estimated_backend: none
  data_source: biogeme.data.swissmetro/data/swissmetro.dat
  model: Mixed Logit with normal random coefficients
  draws: shared antithetic standard-normal draw matrix
  covariance: Cholesky lower triangular replay when correlated=True
  parameters: shared across replay backends
  probabilities: averaged over the same draws and row order
  reference: torchdcm_fixed

backend            available     total_s  estimate_s       cov_s           loglike       ll_diff     prob_diff
torchdcm_fixed          True    0.001131          NA          NA   -411.2055971514     0.000e+00     0.000e+00
apollo_r_fixed          True    0.113239          NA          NA   -411.2055971514     4.547e-13     3.331e-16
biogeme_fixed           True    5.854124          NA          NA   -411.2055971514     0.000e+00     3.331e-16

torchdcm_fixed params:
  ASC_TRAIN: 0.3
  B_TIME: -1
  B_COST: -1.2
  ASC_CAR: 0.6
  SIGMA_B_TIME: 0.5
apollo_r_fixed params:
  ASC_TRAIN: 0.3
  B_TIME: -1
  B_COST: -1.2
  ASC_CAR: 0.6
  SIGMA_B_TIME: 0.5
biogeme_fixed params:
  ASC_TRAIN: 0.3
  B_TIME: -1
  B_COST: -1.2
  ASC_CAR: 0.6
  SIGMA_B_TIME: 0.5
```

stderr:

```text
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
```

## swissmetro_wtp_mixed_panel_fixed

```text
case: biogeme_swissmetro_wtp_mixed_logit
mode: fixed
n_obs: 500
n_draws: 32
panel: True
alignment:
  benchmark_mode: fixed_likelihood_replay
  estimated_backend: none
  data_source: biogeme.data.swissmetro/data/swissmetro.dat
  model: WTP-space mixed logit with normal random WTP_TIME
  utility: ASC + B_COST*cost + B_COST*WTP_TIME*time
  draws: shared antithetic standard-normal draw matrix
  parameters: shared across replay backends
  probabilities: averaged over the same draws and row order
  reference: torchdcm_fixed

backend            available     total_s           loglike       ll_diff     prob_diff
torchdcm_fixed          True    0.001107   -422.4739498712     0.000e+00     0.000e+00
apollo_r_fixed          True    0.105293   -422.4739498712    -1.705e-13     2.776e-16
biogeme_fixed           True    6.031634   -422.4739498712     0.000e+00     2.220e-16

torchdcm_fixed params:
  ASC_TRAIN: 0.3
  ASC_CAR: 0.6
  B_COST: -1.2
  WTP_TIME: 0.75
  SIGMA_WTP_TIME: 0.35
apollo_r_fixed params:
  ASC_TRAIN: 0.3
  ASC_CAR: 0.6
  B_COST: -1.2
  WTP_TIME: 0.75
  SIGMA_WTP_TIME: 0.35
biogeme_fixed params:
  ASC_TRAIN: 0.3
  ASC_CAR: 0.6
  B_COST: -1.2
  WTP_TIME: 0.75
  SIGMA_WTP_TIME: 0.35
```

stderr:

```text
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
```

## swissmetro_latent_class_fit_replay

```text
case: biogeme_swissmetro_latent_class
mode: fit-replay
n_obs: 500
alignment:
  benchmark_mode: torchdcm_full_estimation_then_fixed_replay
  estimated_backend: torchdcm
  data_source: biogeme.data.swissmetro/data/swissmetro.dat
  model: 2-class latent class logit with class-specific MNL utilities
  class_membership: class 1 reference; supports constant or GA covariate allocation
  parameters: shared across replay backends
  reference: torchdcm_fit

backend            available     total_s  estimate_s       cov_s           loglike       ll_diff     prob_diff   class_prob_diff
torchdcm_fit            True    0.026606    0.023665    0.002940   -281.2738197112     0.000e+00     0.000e+00         0.000e+00
apollo_r_fixed          True    0.095291          NA          NA   -281.2738197112     0.000e+00     2.220e-16         3.429e-12
biogeme_fixed           True    0.308082          NA          NA   -281.2738197112     0.000e+00     2.220e-16         3.429e-12

torchdcm_fit params:
  ASC_TRAIN_C1: -1.72731366436
  B_TIME_C1: 0.119179327918
  B_COST_C1: -1.3102413702
  ASC_CAR_C1: -2.60828339673
  ASC_TRAIN_C2: -1.72731366422
  B_TIME_C2: 0.119179327552
  B_COST_C2: -1.31024137017
  ASC_CAR_C2: -2.60828339678
  CLASS_2: 1.15448220023e-14
  CLASS_2_GA: -1.37143856356e-11
apollo_r_fixed params:
  ASC_TRAIN_C1: -1.72731366436
  B_TIME_C1: 0.119179327918
  B_COST_C1: -1.3102413702
  ASC_CAR_C1: -2.60828339673
  ASC_TRAIN_C2: -1.72731366422
  B_TIME_C2: 0.119179327552
  B_COST_C2: -1.31024137017
  ASC_CAR_C2: -2.60828339678
  CLASS_2: 1.15448220023e-14
  CLASS_2_GA: -1.37143856356e-11
biogeme_fixed params:
  ASC_TRAIN_C1: -1.72731366436
  B_TIME_C1: 0.119179327918
  B_COST_C1: -1.3102413702
  ASC_CAR_C1: -2.60828339673
  ASC_TRAIN_C2: -1.72731366422
  B_TIME_C2: 0.119179327552
  B_COST_C2: -1.31024137017
  ASC_CAR_C2: -2.60828339678
  CLASS_2: 1.15448220023e-14
  CLASS_2_GA: -1.37143856356e-11
```

stderr:

```text
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
```

## optima_ordered_logit_estimate

```text
case: biogeme_optima_ordered_envir01_logit
mode: full-estimation
n_obs: 500
alignment:
  benchmark_mode: full_estimation
  estimated_backend: each backend estimates independently
  data_source: biogeme.data.optima/data/optima.dat
  source_loader: biogeme.data.optima.read_data
  outcome: Envir01 Likert ordered categories [1, 2, 3, 4, 5, 6]
  latent_index: B_MALE*male + B_HIGH_EDUCATION*highEducation + B_GA*haveGA + B_INCOME*ScaledIncome
  thresholds: shared explicit thresholds
  weights: normalized_weight
  biogeme_model_function: ordered_logit_from_thresholds / ordered_probit_from_thresholds
  example_alignment: Biogeme Optima case-study data and Biogeme ordered model API
  reference: torchdcm_fit

backend            available     total_s  estimate_s       cov_s           loglike       ll_diff    param_diff     prob_diff      cov_diff       se_diff        t_diff
torchdcm_fit            True    0.032001    0.028652    0.003349   -732.2833979739     0.000e+00     0.000e+00     0.000e+00     0.000e+00     0.000e+00     0.000e+00
biogeme                 True    2.678802    2.573099    0.000075   -732.2833984717    -4.978e-07     2.086e-04     5.724e-05     2.844e-06     4.483e-06     7.053e-04

torchdcm_fit params:
  B_MALE: 0.186323404388
  B_HIGH_EDUCATION: 0.844877725958
  B_GA: 0.4207561517
  B_INCOME: -0.0283346097801
  TH_1: -0.849589904072
  TH_2: 0.639723168227
  TH_3: 1.32689772148
  TH_4: 2.5683594598
  TH_5: 3.31284344124
biogeme params:
  B_MALE: 0.186258525082
  B_HIGH_EDUCATION: 0.844842386502
  B_GA: 0.420554793278
  B_INCOME: -0.0283465720443
  TH_1: -0.849740101431
  TH_2: 0.639558185214
  TH_3: 1.32672105287
  TH_4: 2.568159076
  TH_5: 3.31263485141
```

stderr:

```text
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
```

## optima_ordered_probit_estimate

```text
case: biogeme_optima_ordered_envir01_probit
mode: full-estimation
n_obs: 500
alignment:
  benchmark_mode: full_estimation
  estimated_backend: each backend estimates independently
  data_source: biogeme.data.optima/data/optima.dat
  source_loader: biogeme.data.optima.read_data
  outcome: Envir01 Likert ordered categories [1, 2, 3, 4, 5, 6]
  latent_index: B_MALE*male + B_HIGH_EDUCATION*highEducation + B_GA*haveGA + B_INCOME*ScaledIncome
  thresholds: shared explicit thresholds
  weights: normalized_weight
  biogeme_model_function: ordered_logit_from_thresholds / ordered_probit_from_thresholds
  example_alignment: Biogeme Optima case-study data and Biogeme ordered model API
  reference: torchdcm_fit

backend            available     total_s  estimate_s       cov_s           loglike       ll_diff    param_diff     prob_diff      cov_diff       se_diff        t_diff
torchdcm_fit            True    0.040864    0.036560    0.004304   -733.7871019931     0.000e+00     0.000e+00     0.000e+00     0.000e+00     0.000e+00     0.000e+00
biogeme                 True    3.183317    3.068091    0.000076   -733.7871020108    -1.768e-08     3.023e-05     9.849e-06     2.365e-07     7.297e-07     1.195e-04

torchdcm_fit params:
  B_MALE: 0.107929063601
  B_HIGH_EDUCATION: 0.449972876793
  B_GA: 0.220464446542
  B_INCOME: -0.018111713917
  TH_1: -0.538377512361
  TH_2: 0.378075776765
  TH_3: 0.786508481513
  TH_4: 1.45240947907
  TH_5: 1.7979240455
biogeme params:
  B_MALE: 0.107927925061
  B_HIGH_EDUCATION: 0.449969555838
  B_GA: 0.220494680955
  B_INCOME: -0.0181119948868
  TH_1: -0.538379290619
  TH_2: 0.378074622905
  TH_3: 0.786507349808
  TH_4: 1.45240879195
  TH_5: 1.79791188924
```

stderr:

```text
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
```

## mlogit_fishing_mnl_estimate

```text
case: mlogit_fishing_mnl
n_obs: 1182
alignment:
  benchmark_mode: full_estimation
  data_source: R package mlogit built-in data
  estimator_reference: R mlogit::mlogit
  initial_values: zeros for TorchDCM; mlogit default start
  covariance: classic inverse observed information / vcov
  reference: torchdcm

backend      available     total_s  estimate_s       cov_s           loglike       ll_diff    param_diff      cov_diff       se_diff        t_diff
torchdcm          True    0.030752    0.029703    0.001049  -1230.7838304155     0.000e+00     0.000e+00     0.000e+00     0.000e+00     0.000e+00
mlogit            True    1.148443    0.031000    0.001000  -1230.7838304155     0.000e+00     1.506e-08     4.088e-10     1.538e-09     4.219e-07

parameter             torch_beta   mlogit_beta     beta_diff      torch_se     mlogit_se       torch_t      mlogit_t
ASC_BOAT                0.871375      0.871375    -1.506e-08      0.114043      0.114043       7.64077       7.64077
ASC_CHARTER              1.49889       1.49889    -1.290e-08      0.132933      0.132933       11.2755       11.2755
ASC_PIER                0.307055      0.307055     1.204e-08      0.114574      0.114574       2.67998       2.67998
B_PRICE               -0.0247896    -0.0247896    -1.283e-10     0.0017044     0.0017044      -14.5444      -14.5444
B_CATCH                 0.377169      0.377169    -4.176e-11      0.109971      0.109971       3.42972       3.42972
```

## mlogit_modecanada_mnl_estimate

```text
case: mlogit_modecanada_mnl
n_obs: 4324
alignment:
  benchmark_mode: full_estimation
  data_source: R package mlogit built-in data
  estimator_reference: R mlogit::mlogit
  initial_values: zeros for TorchDCM; mlogit default start
  covariance: classic inverse observed information / vcov
  reference: torchdcm

backend      available     total_s  estimate_s       cov_s           loglike       ll_diff    param_diff      cov_diff       se_diff        t_diff
torchdcm          True    0.023957    0.022349    0.001608  -3404.3011945472     0.000e+00     0.000e+00     0.000e+00     0.000e+00     0.000e+00
mlogit            True    0.614833    0.064000    0.001000  -3404.3011945472     0.000e+00     4.841e-08     2.394e-15     1.996e-12     8.717e-05

parameter             torch_beta   mlogit_beta     beta_diff      torch_se     mlogit_se       torch_t      mlogit_t
B_COST               -0.00864882   -0.00864883     8.982e-09   0.000912586   0.000912586      -9.47727      -9.47728
B_IVT                 -0.0163008    -0.0163007    -2.223e-08   0.000439881   0.000439881      -37.0572      -37.0572
B_OVT                 -0.0256264    -0.0256264    -4.841e-08   0.000556428   0.000556428      -46.0553      -46.0552
```
