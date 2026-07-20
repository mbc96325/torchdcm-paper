# Real-data Mixed Logit Battery

All cross-estimator runtimes report estimation plus covariance on one logical CPU. Each runnable model uses 2-4 independent normal random coefficients selected from observed-variable coefficients first, then ASC terms only when needed.

| case | N | RC | TorchDCM s | Biogeme s | LL diff | Param diff | Prob diff | Consistent? |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| mlogit_japanese_fdi | NA | skipped | NA | NA | NA | NA | NA | No |

## Specifications

- `mlogit_japanese_fdi` skipped: case exceeded 300s worker limit