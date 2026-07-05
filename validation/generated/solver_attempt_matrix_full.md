# Solver Attempt Matrix

Each benchmark case is attempted with every configured solver where an aligned wrapper exists.
`ok` means the solver completed the aligned case, `failed` means the wrapper was attempted but failed, and `unsupported` means no aligned wrapper exists for that solver/model case.

| case | dataset | model | torchdcm | scipy_bfgs | biogeme | apollo | mlogit | gmnl | xlogit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| swissmetro_mnl | Swissmetro | MNL | ok (0.028s) | ok (2.328s) | ok (1.187s) | ok (1.825s) | ok (0.692s) | failed | ok (0.029s) |
| swissmetro_nested | Swissmetro | Nested logit | ok (0.080s) | unsupported | ok (2.952s) | ok (2.091s) | unsupported | unsupported | unsupported |
| swissmetro_cross_nested | Swissmetro | Cross-nested logit | ok (1.118s) | unsupported | ok (3.834s) | unsupported | unsupported | unsupported | unsupported |
| swissmetro_mixed_replay | Swissmetro | Mixed logit replay | ok (0.012s) | unsupported | ok (13.101s) | ok (0.593s) | unsupported | unsupported | unsupported |
| swissmetro_wtp_mixed_replay | Swissmetro | WTP mixed replay | ok (0.014s) | unsupported | ok (13.506s) | ok (0.594s) | unsupported | unsupported | unsupported |
| optima_ordered_logit | Optima | Ordered logit | ok (0.042s) | unsupported | ok (2.875s) | unsupported | unsupported | unsupported | unsupported |
| optima_ordered_probit | Optima | Ordered probit | ok (0.053s) | unsupported | ok (3.459s) | unsupported | unsupported | unsupported | unsupported |
| nhts_2022_mnl | NHTS 2022 | MNL | ok (0.099s) | ok (44.345s) | ok (1.752s) | ok (9.843s) | ok (2.898s) | ok (31.955s) | ok (0.711s) |
| biogeme_public_airline | airline | MNL | ok (0.025s) | ok (3.103s) | ok (1.195s) | ok (1.199s) | ok (0.585s) | ok (0.787s) | ok (0.011s) |
| biogeme_public_parking | parking | MNL | ok (0.023s) | ok (1.873s) | ok (1.174s) | ok (1.129s) | ok (0.557s) | ok (0.698s) | ok (0.006s) |
| biogeme_public_telephone | telephone | MNL | ok (0.022s) | ok (0.241s) | ok (1.181s) | ok (1.094s) | ok (0.544s) | failed | ok (0.004s) |
| biogeme_public_lpmc | lpmc | MNL | ok (0.074s) | ok (59.196s) | ok (1.398s) | ok (21.171s) | ok (4.000s) | ok (7.769s) | ok (0.325s) |
| mlogit_catsup | catsup | MNL | ok (0.021s) | ok (0.375s) | ok (1.158s) | ok (1.120s) | ok (0.053s) | ok (0.705s) | ok (0.008s) |
| mlogit_cracker | cracker | MNL | ok (0.028s) | ok (0.544s) | ok (1.185s) | ok (1.138s) | ok (0.057s) | ok (0.716s) | ok (0.009s) |
| mlogit_electricity | electricity | MNL | ok (0.032s) | ok (2.821s) | ok (1.592s) | ok (1.282s) | ok (0.197s) | ok (0.970s) | ok (0.013s) |
| mlogit_hc | hc | MNL | ok (0.020s) | ok (0.048s) | ok (1.365s) | ok (1.045s) | ok (0.017s) | ok (0.667s) | ok (0.002s) |
| mlogit_heating | heating | MNL | ok (0.019s) | ok (0.140s) | ok (1.016s) | ok (1.055s) | ok (0.025s) | ok (0.667s) | ok (0.003s) |
| mlogit_mode | mode | MNL | ok (0.017s) | ok (0.201s) | ok (0.952s) | ok (1.026s) | ok (0.016s) | ok (0.663s) | ok (0.002s) |
| mlogit_nox | nox | MNL | ok (0.020s) | ok (0.238s) | ok (2.426s) | ok (1.132s) | ok (0.041s) | failed | ok (0.004s) |
| mlogit_risky_transport | risky_transport | MNL | ok (0.038s) | ok (1.147s) | ok (1.815s) | ok (1.214s) | ok (0.045s) | failed | failed |
| mlogit_train | train | MNL | ok (0.032s) | ok (1.771s) | ok (0.886s) | ok (1.137s) | ok (0.033s) | ok (0.705s) | ok (0.005s) |
| mlogit_fishing | fishing | MNL | ok (0.033s) | ok (0.813s) | ok (1.176s) | ok (1.101s) | ok (1.176s) | ok (0.707s) | ok (0.006s) |
| mlogit_modecanada | modecanada | MNL | ok (0.028s) | ok (1.960s) | ok (1.155s) | ok (1.200s) | ok (0.613s) | failed | failed |

## Failures

- `swissmetro_mnl` / `gmnl`: Error in s + x[[i]] : non-conformable arrays
- `biogeme_public_telephone` / `gmnl`: Error in s + x[[i]] : non-conformable arrays
- `mlogit_nox` / `gmnl`: Error in s + x[[i]] : non-conformable arrays
Calls: gmnl ... eval -> eval -> maxNRCompute -> fn -> fnOrig -> suml
Execution halted
- `mlogit_risky_transport` / `gmnl`: Error in s + x[[i]] : non-conformable arrays
Calls: gmnl ... eval -> eval -> maxNRCompute -> fn -> fnOrig -> suml
Execution halted
- `mlogit_risky_transport` / `xlogit`: ValueError: inconsistent alts values in long format
- `mlogit_modecanada` / `gmnl`: NA NA NA NA NA NA NA NA NA Error in s + x[[i]] : non-conformable arrays
- `mlogit_modecanada` / `xlogit`: NA NA NA NA NA NA NA NA NA xlogit MultinomialLogit requires consistent alternatives in long format; ModeCanada has ragged choice sets.
