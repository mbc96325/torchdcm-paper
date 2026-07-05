# R mlogit Dataset Battery

Baseline MNL with generic coefficients and no alternative-specific constants.

| case | N | K | TorchDCM total_s | mlogit total_s | LL diff | Consistent? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| car | NA | NA | NA | 4.455 | NA | No |
| game | NA | NA | NA | 0.011 | NA | No |
| game2 | NA | NA | NA | 0.016 | NA | No |
| japanese_fdi | 452 | 12 | 0.186 | 0.258 | -8.43e+00 | No |
| nox | 632 | 3 | 0.006 | 0.040 | -3.41e-10 | Yes |

## Failures

- `car`: Error in if (abs(x - oldx) < ftol) { : 
  missing value where TRUE/FALSE needed
Calls: fit_long -> mlogit -> eval -> eval -> mlogit.optim
Execution halted
- `game`: Each observation must have exactly one chosen row.
- `game2`: Each observation must have exactly one chosen row.