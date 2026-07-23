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
torchdcm          True    0.026630    0.024571    0.002059  -3404.3011945472     0.000e+00     0.000e+00     0.000e+00     0.000e+00     0.000e+00
mlogit            True    0.609791    0.064000    0.001000  -3404.3011945472     0.000e+00     4.841e-08     2.394e-15     1.996e-12     8.717e-05
gmnl             False          NA          NA          NA                NA            NA            NA            NA            NA            NA  Error in s + x[[i]] : non-conformable arrays
Calls: gmnl ... eval -> eval -> maxNRCompute -> fn -> fnOrig -> suml
Execution halted
xlogit           False          NA          NA          NA                NA            NA            NA            NA            NA            NA  xlogit MultinomialLogit requires consistent alternatives in long format; ModeCanada has ragged choice sets.

parameter             torch_beta   mlogit_beta     beta_diff      torch_se     mlogit_se       torch_t      mlogit_t
B_COST               -0.00864882   -0.00864883     8.982e-09   0.000912586   0.000912586      -9.47727      -9.47728
B_IVT                 -0.0163008    -0.0163007    -2.223e-08   0.000439881   0.000439881      -37.0572      -37.0572
B_OVT                 -0.0256264    -0.0256264    -4.841e-08   0.000556428   0.000556428      -46.0553      -46.0552
