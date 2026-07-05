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
torchdcm          True    0.030957    0.029920    0.001037  -1230.7838304155     0.000e+00     0.000e+00     0.000e+00     0.000e+00     0.000e+00
mlogit            True    1.164740    0.032000    0.000000  -1230.7838304155     0.000e+00     1.506e-08     4.088e-10     1.538e-09     4.219e-07
gmnl              True    0.706304    0.060000    0.000000  -1230.7838304155     0.000e+00     1.506e-08     1.065e-10     4.647e-10     1.273e-07
xlogit            True    0.006352    0.006352    0.000000  -1230.7838304194    -3.949e-09     6.182e-06     6.722e-08     2.529e-07     5.322e-05

parameter             torch_beta   mlogit_beta     beta_diff      torch_se     mlogit_se       torch_t      mlogit_t
ASC_BOAT                0.871375      0.871375    -1.506e-08      0.114043      0.114043       7.64077       7.64077
ASC_CHARTER              1.49889       1.49889    -1.290e-08      0.132933      0.132933       11.2755       11.2755
ASC_PIER                0.307055      0.307055     1.204e-08      0.114574      0.114574       2.67998       2.67998
B_PRICE               -0.0247896    -0.0247896    -1.283e-10     0.0017044     0.0017044      -14.5444      -14.5444
B_CATCH                 0.377169      0.377169    -4.176e-11      0.109971      0.109971       3.42972       3.42972
