# TorchDCM Estimator Benchmark Suite (full)

All commands were run on the remote benchmark machine. Timing columns reported by the underlying scripts split parameter estimation and covariance calculation where the backend exposes both.

| case | dataset | model | mode | status | wall_s | backends |
|---|---|---|---|---:|---:|---|
| swissmetro_mnl_estimate | Biogeme Swissmetro | MNL | full_estimation | ok | 8.809 | torchdcm, scipy_bfgs, biogeme, apollo |
| swissmetro_nested_estimate | Biogeme Swissmetro | Nested Logit | full_estimation | ok | 8.629 | torchdcm, biogeme, apollo |
| swissmetro_cross_nested_estimate | Biogeme Swissmetro | Cross-Nested Logit | full_estimation | ok | 8.127 | torchdcm_fit, biogeme |
| swissmetro_mixed_panel_fixed | Biogeme Swissmetro | Mixed Logit | fixed_replay_shared_draws | ok | 16.465 | torchdcm_fixed, apollo_r_fixed, biogeme_fixed |
| swissmetro_wtp_mixed_panel_fixed | Biogeme Swissmetro | WTP Mixed Logit | fixed_replay_shared_draws | ok | 17.016 | torchdcm_fixed, apollo_r_fixed, biogeme_fixed |
| swissmetro_latent_class_fit_replay | Biogeme Swissmetro | Latent Class Logit | torch_fit_then_replay | ok | 4.642 | torchdcm_fit, apollo_r_fixed, biogeme_fixed |
| optima_ordered_logit_estimate | Biogeme Optima | Ordered Logit | full_estimation | ok | 7.184 | torchdcm_fit, biogeme |
| optima_ordered_probit_estimate | Biogeme Optima | Ordered Probit | full_estimation | ok | 7.748 | torchdcm_fit, biogeme |
| mlogit_fishing_mnl_estimate | R mlogit Fishing | MNL | full_estimation | ok | 3.768 | torchdcm, mlogit, gmnl, xlogit |
| mlogit_modecanada_mnl_estimate | R mlogit ModeCanada | MNL | full_estimation | ok | 3.231 | torchdcm, mlogit, gmnl, xlogit |

## swissmetro_mnl_estimate

```text
case: swissmetro
n_obs: 100000
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
torchdcm          True    0.028400    0.025684    0.002716  -8670.1631185238     0.000e+00         0.000e+00
scipy_bfgs        True    2.309038    2.305812    0.003226  -8670.1631185237     7.276e-11         2.187e-07
biogeme           True    1.192852    1.155851    0.000058  -8670.1631185238     1.819e-12         2.668e-07
apollo            True    1.818776    0.851000    0.000000  -8670.1631000000     1.852e-05         4.154e-05

backend          prob_diff      cov_diff       se_diff           wtp      wtp_diff        wtp_se   wtp_se_diff     elas_time     elas_cost
torchdcm         0.000e+00     0.000e+00     0.000e+00    -1.619e+00     0.000e+00     8.521e-02     0.000e+00     0.000e+00     0.000e+00
scipy_bfgs       6.183e-08     2.690e-10     3.156e-09    -1.619e+00     7.839e-09     8.521e-02    -1.162e-08     3.412e-06     1.136e-06
biogeme          8.700e-08     1.183e-10     1.388e-09    -1.619e+00     2.541e-07     8.521e-02    -1.043e-08     3.054e-06     2.842e-07
apollo           2.523e-05     4.823e-05     5.808e-04    -1.619e+00     7.216e-05     8.551e-02     2.989e-04     6.480e-04     1.337e-04
```

stderr:

```text
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
```

## swissmetro_nested_estimate

```text
case: biogeme_swissmetro_nested
n_obs: 100000
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
torchdcm          True    0.079110    0.065387    0.013723  -8669.4965822884     0.000e+00         0.000e+00
biogeme           True    2.926978    2.926908    0.000069  -8669.4965925525    -1.026e-05         3.249e-04
apollo            True    2.097240    1.107000    0.009000  -8669.4966000000    -1.771e-05         5.419e-05

backend          prob_diff      cov_diff       se_diff           wtp      wtp_diff        wtp_se   wtp_se_diff     elas_time     elas_cost
torchdcm         0.000e+00     0.000e+00     0.000e+00    -1.603e+00     0.000e+00     8.569e-02     0.000e+00     0.000e+00     0.000e+00
biogeme          1.057e-04     3.398e-03     2.378e-02    -1.603e+00    -2.445e-05     8.569e-02    -5.250e-08     1.445e-03     5.476e-04
apollo           2.971e-05     4.996e-05     6.669e-04    -1.602e+00     9.098e-05     8.688e-02     1.185e-03     6.959e-04     1.663e-04
```

stderr:

```text
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
```

## swissmetro_cross_nested_estimate

```text
case: biogeme_swissmetro_cross_nested
mode: full-estimation
n_obs: 100000
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
torchdcm_fit            True    1.241074          NA          NA  -8620.9451323125     0.000e+00     0.000e+00     0.000e+00     0.000e+00     0.000e+00     0.000e+00
biogeme                 True    3.870117    3.870035    0.000082  -8620.9451323306    -1.814e-08     6.276e-06     4.616e-06     1.734e-03     2.314e-02     1.221e+01

torchdcm_fit params:
  ASC_TRAIN: -0.138834043943
  B_TIME: -0.484683051612
  B_COST: -0.413713045137
  ASC_CAR: 0.0252956273938
  LAMBDA_PUBLIC: 0.2502778232
  LAMBDA_PRIVATE: 0.30695977382
biogeme params:
  ASC_TRAIN: -0.138829956991
  B_TIME: -0.484676775669
  B_COST: -0.413708927555
  ASC_CAR: 0.0252957177042
  LAMBDA_PUBLIC: 0.250271574235
  LAMBDA_PRIVATE: 0.30695565803
```

stderr:

```text
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
```

## swissmetro_mixed_panel_fixed

```text
case: biogeme_swissmetro_mixed_logit
mode: fixed
n_obs: 100000
n_draws: 64
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
torchdcm_fixed          True    0.018182          NA          NA  -9246.2175642458     0.000e+00     0.000e+00
apollo_r_fixed          True    0.601198          NA          NA  -9246.2175642458     1.091e-11     6.661e-16
biogeme_fixed           True   12.956559          NA          NA  -9246.2175642458     0.000e+00     6.661e-16

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
n_obs: 100000
n_draws: 64
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
torchdcm_fixed          True    0.011026  -9511.3257921248     0.000e+00     0.000e+00
apollo_r_fixed          True    0.579467  -9511.3257921248     0.000e+00     6.661e-16
biogeme_fixed           True   13.577981  -9511.3257921248     1.819e-11     6.661e-16

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
n_obs: 100000
alignment:
  benchmark_mode: torchdcm_full_estimation_then_fixed_replay
  estimated_backend: torchdcm
  data_source: biogeme.data.swissmetro/data/swissmetro.dat
  model: 2-class latent class logit with class-specific MNL utilities
  class_membership: class 1 reference; supports constant or GA covariate allocation
  parameters: shared across replay backends
  reference: torchdcm_fit

backend            available     total_s  estimate_s       cov_s           loglike       ll_diff     prob_diff   class_prob_diff
torchdcm_fit            True    0.044091    0.032556    0.011535  -8670.1631185237     0.000e+00     0.000e+00         0.000e+00
apollo_r_fixed          True    0.150989          NA          NA  -8670.1631185237    -1.819e-12     2.220e-16         7.438e-15
biogeme_fixed           True    0.503074          NA          NA  -8670.1631185237    -1.819e-12     5.551e-16         7.383e-15

torchdcm_fit params:
  ASC_TRAIN_C1: -0.652238697078
  B_TIME_C1: -1.27894164914
  B_COST_C1: -0.789790713638
  ASC_CAR_C1: 0.0162280339258
  ASC_TRAIN_C2: -0.652238697078
  B_TIME_C2: -1.27894164913
  B_COST_C2: -0.789790713639
  ASC_CAR_C2: 0.0162280339259
  CLASS_2: 5.94088439923e-16
  CLASS_2_GA: 2.94486877447e-14
apollo_r_fixed params:
  ASC_TRAIN_C1: -0.652238697078
  B_TIME_C1: -1.27894164914
  B_COST_C1: -0.789790713638
  ASC_CAR_C1: 0.0162280339258
  ASC_TRAIN_C2: -0.652238697078
  B_TIME_C2: -1.27894164913
  B_COST_C2: -0.789790713639
  ASC_CAR_C2: 0.0162280339259
  CLASS_2: 5.94088439923e-16
  CLASS_2_GA: 2.94486877447e-14
biogeme_fixed params:
  ASC_TRAIN_C1: -0.652238697078
  B_TIME_C1: -1.27894164914
  B_COST_C1: -0.789790713638
  ASC_CAR_C1: 0.0162280339258
  ASC_TRAIN_C2: -0.652238697078
  B_TIME_C2: -1.27894164913
  B_COST_C2: -0.789790713639
  ASC_CAR_C2: 0.0162280339259
  CLASS_2: 5.94088439923e-16
  CLASS_2_GA: 2.94486877447e-14
```

stderr:

```text
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
```

## optima_ordered_logit_estimate

```text
case: biogeme_optima_ordered_envir01_logit
mode: full-estimation
n_obs: 1822
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
torchdcm_fit            True    0.041778    0.038099    0.003680  -2895.9309175959     0.000e+00     0.000e+00     0.000e+00     0.000e+00     0.000e+00     0.000e+00
biogeme                 True    2.898195    2.779437    0.000077  -2895.9309175960    -1.055e-10     1.593e-06     2.813e-07     3.671e-09     1.163e-08     1.163e-05

torchdcm_fit params:
  B_MALE: -0.143493800972
  B_HIGH_EDUCATION: 0.800273656697
  B_GA: 0.640008048167
  B_INCOME: -0.0105254116861
  TH_1: -0.903617089178
  TH_2: 0.346834836055
  TH_3: 1.07957335375
  TH_4: 2.03058484565
  TH_5: 3.33332991396
biogeme params:
  B_MALE: -0.143493718975
  B_HIGH_EDUCATION: 0.800272063605
  B_GA: 0.640008875438
  B_INCOME: -0.0105254615065
  TH_1: -0.903617571922
  TH_2: 0.346834277386
  TH_3: 1.07957269084
  TH_4: 2.03058414103
  TH_5: 3.33332915912
```

stderr:

```text
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
```

## optima_ordered_probit_estimate

```text
case: biogeme_optima_ordered_envir01_probit
mode: full-estimation
n_obs: 1822
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
torchdcm_fit            True    0.054414    0.049689    0.004725  -2900.7663669527     0.000e+00     0.000e+00     0.000e+00     0.000e+00     0.000e+00     0.000e+00
biogeme                 True    3.474948    3.360366    0.000079  -2900.7663669528    -1.114e-10     1.387e-06     4.026e-07     2.609e-10     1.506e-09     1.608e-05

torchdcm_fit params:
  B_MALE: -0.0709878178484
  B_HIGH_EDUCATION: 0.424918600536
  B_GA: 0.343003595224
  B_INCOME: -0.00738379217148
  TH_1: -0.564756415056
  TH_2: 0.205821494864
  TH_3: 0.648986962059
  TH_4: 1.18043119509
  TH_5: 1.80676590114
biogeme params:
  B_MALE: -0.0709877365084
  B_HIGH_EDUCATION: 0.424918441046
  B_GA: 0.343002208388
  B_INCOME: -0.0073838031305
  TH_1: -0.564756579776
  TH_2: 0.20582132633
  TH_3: 0.64898679967
  TH_4: 1.18043095688
  TH_5: 1.80676571245
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
torchdcm          True    0.031488    0.030445    0.001043  -1230.7838304155     0.000e+00     0.000e+00     0.000e+00     0.000e+00     0.000e+00
mlogit            True    1.166950    0.032000    0.000000  -1230.7838304155     0.000e+00     1.506e-08     4.088e-10     1.538e-09     4.219e-07
gmnl              True    0.719375    0.062000    0.001000  -1230.7838304155     0.000e+00     1.506e-08     1.065e-10     4.647e-10     1.273e-07
xlogit            True    0.006006    0.006005    0.000000  -1230.7838304194    -3.949e-09     6.182e-06     6.722e-08     2.529e-07     5.322e-05

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
torchdcm          True    0.028502    0.026475    0.002027  -3404.3011945472     0.000e+00     0.000e+00     0.000e+00     0.000e+00     0.000e+00
mlogit            True    0.643757    0.075000    0.001000  -3404.3011945472     0.000e+00     4.841e-08     2.394e-15     1.996e-12     8.717e-05
gmnl             False          NA          NA          NA                NA            NA            NA            NA            NA            NA  Error in s + x[[i]] : non-conformable arrays
Calls: gmnl ... eval -> eval -> maxNRCompute -> fn -> fnOrig -> suml
Execution halted
xlogit           False          NA          NA          NA                NA            NA            NA            NA            NA            NA  xlogit MultinomialLogit requires consistent alternatives in long format; ModeCanada has ragged choice sets.

parameter             torch_beta   mlogit_beta     beta_diff      torch_se     mlogit_se       torch_t      mlogit_t
B_COST               -0.00864882   -0.00864883     8.982e-09   0.000912586   0.000912586      -9.47727      -9.47728
B_IVT                 -0.0163008    -0.0163007    -2.223e-08   0.000439881   0.000439881      -37.0572      -37.0572
B_OVT                 -0.0256264    -0.0256264    -4.841e-08   0.000556428   0.000556428      -46.0553      -46.0552
```
