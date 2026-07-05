# R mlogit Dataset Battery

Baseline MNL with generic coefficients and no alternative-specific constants.

| case | N | K | TorchDCM total_s | mlogit total_s | LL diff | Consistent? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| car | NA | NA | NA | 8.743 | NA | No |
| catsup | 2798 | 3 | 0.021 | 0.052 | -1.36e-12 | Yes |
| cracker | 3292 | 3 | 0.009 | 0.057 | -1.44e-10 | Yes |
| electricity | 4308 | 6 | 0.011 | 0.192 | -3.81e-10 | Yes |
| fishing | 1182 | 2 | 0.008 | 0.026 | -1.75e-11 | Yes |
| game | NA | NA | NA | 0.012 | NA | No |
| game2 | NA | NA | NA | 0.016 | NA | No |
| hc | 250 | 2 | 0.004 | 0.016 | -2.83e-10 | Yes |
| heating | 900 | 2 | 0.004 | 0.025 | -2.73e-12 | Yes |
| japanese_fdi | NA | NA | NA | 0.569 | NA | No |
| mode | 453 | 2 | 0.003 | 0.016 | -4.20e-10 | Yes |
| modecanada | 4324 | 4 | 0.014 | 0.064 | -1.86e-10 | Yes |
| nox | NA | NA | NA | 0.584 | NA | No |
| risky_transport | 1793 | 7 | 0.024 | 0.045 | -5.68e-12 | Yes |
| train | 2929 | 4 | 0.016 | 0.033 | -3.22e-10 | Yes |

## Failures

- `car`: Error in if (abs(x - oldx) < ftol) { : 
  missing value where TRUE/FALSE needed
Calls: fit_long -> mlogit -> eval -> eval -> mlogit.optim
Execution halted
- `game`: Missing columns in long data: ['o', 'w', 'n']
- `game2`: Missing columns in long data: ['o', 'w', 'n']
- `japanese_fdi`: Error in dfidx(data = data, dfa$idx, drop.index = dfa$drop.index, as.factor = dfa$as.factor,  : 
  the two indexes don't define unique observations
Calls: fit_long -> mlogit.data -> dfidx
Execution halted
- `nox`: Error in solve.default(H, g[!fixed]) : 
  system is computationally singular: reciprocal condition number = 4.52818e-34
Calls: fit_long ... mlogit.optim -> as.vector -> solve -> solve.default
Execution halted