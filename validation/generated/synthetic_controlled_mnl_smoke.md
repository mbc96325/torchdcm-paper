# Synthetic Controlled MNL Benchmark (smoke)

Synthetic data are generated from a known MNL data-generating process. The controlled factors are sample size, number of alternatives, number of generic parameters, feature correlation, and utility signal scale.

| case | N | J | K | rho | signal | params | rows | total_s | est_s | cov_s | beta_rmse | beta_max | prob_mean | prob_max | max_se | grad_norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke_base | 2000 | 4 | 6 | 0.00 | 1.00 | 9 | 8000 | 0.0221 | 0.0204 | 0.0018 | 1.728e-02 | 3.860e-02 | 4.061e-03 | 2.313e-02 | 7.516e-02 | 2.703e-04 |
| smoke_corr | 2000 | 4 | 6 | 0.80 | 1.00 | 9 | 8000 | 0.0064 | 0.0048 | 0.0016 | 5.644e-02 | 1.114e-01 | 1.096e-02 | 5.064e-02 | 7.345e-02 | 2.102e-04 |

Interpretation notes:

- `beta_rmse` and `beta_max` compare the fitted MLE with the known data-generating parameters; finite-sample sampling error is expected.
- `prob_mean` and `prob_max` compare fitted probabilities with true data-generating probabilities on the same realized synthetic design matrix.
- `est_s` times only LBFGS parameter estimation; `cov_s` times Hessian/inverse-information covariance.
- Increasing feature correlation stresses Hessian conditioning and parameter recovery without changing the true model class.
