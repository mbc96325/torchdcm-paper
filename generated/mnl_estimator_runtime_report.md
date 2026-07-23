# MNL Estimator Benchmark Report

Date: 2026-07-04

This report follows the Apollo benchmark plan principle that all estimators
must use the same data, utility specification, availability, ASC normalization,
and initial parameter vector.

Backends:

- TorchDCM: PyTorch MNL with classical covariance only.
- SciPy BFGS: NumPy/SciPy reference MLE with analytic gradient.
- Biogeme: Biogeme 3.3.3.
- Apollo: backend script added, skipped on this machine because `Rscript` is
  not installed.

Post-estimation comparisons:

- `prob_diff`: max absolute difference in long-row predicted probabilities.
- `cov_diff`: max absolute difference in Hessian/classical covariance matrix.
- `se_diff`: max absolute difference in classical standard errors.
- `wtp`: `-B_TIME / B_COST`.
- `wtp_se`: delta-method standard error using classical covariance.
- `elas_time`, `elas_cost`: max absolute difference in direct own elasticity,
  computed as `beta * x * (1 - P)` for each long row.

## N = 500, Zero Initial

### Swissmetro-like

| backend | seconds | loglike | LL diff | max parameter diff |
|---|---:|---:|---:|---:|
| TorchDCM | 1.403734 | -484.3279087285 | 0.000e+00 | 0.000e+00 |
| SciPy BFGS | 0.117624 | -484.3279087284 | 2.473e-11 | 1.938e-07 |
| Biogeme | 2.639214 | -484.3279087284 | 2.427e-11 | 1.635e-07 |
| Apollo | skipped | | | |

| backend | prob diff | cov diff | SE diff | WTP | WTP diff | WTP SE | WTP SE diff | time elasticity diff | cost elasticity diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TorchDCM | 0.000e+00 | 0.000e+00 | 0.000e+00 | -2.752e-01 | 0.000e+00 | 6.600e-02 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| SciPy BFGS | 4.402e-07 | 2.043e-09 | 5.081e-09 | -2.752e-01 | 2.390e-07 | 6.600e-02 | -7.289e-08 | 7.412e-06 | 4.857e-06 |
| Biogeme | 4.537e-07 | 2.439e-09 | 6.064e-09 | -2.752e-01 | 2.336e-07 | 6.600e-02 | -6.631e-08 | 8.288e-06 | 4.490e-06 |

### London-like

| backend | seconds | loglike | LL diff | max parameter diff |
|---|---:|---:|---:|---:|
| TorchDCM | 1.634216 | -458.4621149648 | 0.000e+00 | 0.000e+00 |
| SciPy BFGS | 0.182788 | -458.4621149647 | 2.046e-11 | 1.288e-06 |
| Biogeme | 2.888840 | -458.4621149649 | -1.544e-10 | 2.884e-06 |
| Apollo | skipped | | | |

| backend | prob diff | cov diff | SE diff | WTP | WTP diff | WTP SE | WTP SE diff | time elasticity diff | cost elasticity diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TorchDCM | 0.000e+00 | 0.000e+00 | 0.000e+00 | -9.713e-02 | 0.000e+00 | 1.712e-02 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| SciPy BFGS | 2.848e-07 | 5.970e-08 | 7.990e-08 | -9.713e-02 | 4.426e-08 | 1.712e-02 | -8.989e-09 | 2.757e-06 | 7.418e-06 |
| Biogeme | 4.877e-07 | 1.127e-07 | 1.853e-07 | -9.713e-02 | -3.969e-09 | 1.712e-02 | 2.867e-08 | 2.292e-05 | 2.672e-05 |

## N = 500, Shared Random Initial

Random seed: `20260704`, normal scale: `0.1`.

### Swissmetro-like

| backend | seconds | loglike | LL diff | max parameter diff |
|---|---:|---:|---:|---:|
| TorchDCM | 1.520624 | -484.3279087287 | 0.000e+00 | 0.000e+00 |
| SciPy BFGS | 0.663102 | -484.3279087284 | 2.177e-10 | 2.139e-06 |
| Biogeme | 2.646581 | -484.3279087284 | 2.173e-10 | 2.127e-06 |
| Apollo | skipped | | | |

| backend | prob diff | cov diff | SE diff | WTP | WTP diff | WTP SE | WTP SE diff | time elasticity diff | cost elasticity diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TorchDCM | 0.000e+00 | 0.000e+00 | 0.000e+00 | -2.752e-01 | 0.000e+00 | 6.600e-02 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| SciPy BFGS | 1.654e-06 | 2.418e-08 | 6.014e-08 | -2.752e-01 | 8.638e-07 | 6.600e-02 | -1.798e-07 | 4.206e-05 | 1.402e-05 |
| Biogeme | 1.655e-06 | 2.421e-08 | 6.020e-08 | -2.752e-01 | 8.573e-07 | 6.600e-02 | -1.765e-07 | 4.221e-05 | 1.395e-05 |

### London-like

| backend | seconds | loglike | LL diff | max parameter diff |
|---|---:|---:|---:|---:|
| TorchDCM | 1.487475 | -458.4621149648 | 0.000e+00 | 0.000e+00 |
| SciPy BFGS | 0.146541 | -458.4621149647 | 1.108e-11 | 5.596e-07 |
| Biogeme | 2.839338 | -458.4621149648 | -3.752e-12 | 1.113e-06 |
| Apollo | skipped | | | |

| backend | prob diff | cov diff | SE diff | WTP | WTP diff | WTP SE | WTP SE diff | time elasticity diff | cost elasticity diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TorchDCM | 0.000e+00 | 0.000e+00 | 0.000e+00 | -9.713e-02 | 0.000e+00 | 1.712e-02 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| SciPy BFGS | 1.983e-07 | 1.270e-08 | 1.700e-08 | -9.713e-02 | 1.222e-08 | 1.712e-02 | -2.873e-09 | 1.937e-06 | 2.242e-06 |
| Biogeme | 2.527e-07 | 6.136e-08 | 8.212e-08 | -9.713e-02 | 1.263e-09 | 1.712e-02 | 8.023e-09 | 6.941e-06 | 7.712e-06 |

## Notes

- All available estimators satisfy the Apollo plan's MNL numerical tolerance:
  log-likelihood difference below `1e-5`, maximum parameter difference below
  `1e-4`, and probability difference below `1e-6` for zero initial. Shared
  random initial remains close, with the largest probability difference around
  `1.7e-6` on Swissmetro-like data.
- Classical covariance and standard errors are also aligned. The largest
  Biogeme covariance difference in these runs is `1.127e-07`, and the largest
  Biogeme SE difference is `1.853e-07`.
- WTP/VOT and delta-method WTP SE align to around `1e-7` or better.
- Direct own elasticity differences are small; the largest reported difference
  is `4.221e-05` for time elasticity in the Swissmetro-like random-initial run.
- SciPy is fastest in this small CPU benchmark because it uses a compact
  NumPy likelihood and analytic gradient.
- TorchDCM is faster than Biogeme in this benchmark, but it is still a reference
  PyTorch implementation with Python loops. The planned ragged fused logsumexp
  backend should be the meaningful scalability benchmark.
- Apollo support is implemented through `benchmarks/apollo/R/run_mnl.R`.
  Install R, Apollo, and jsonlite to run it.
