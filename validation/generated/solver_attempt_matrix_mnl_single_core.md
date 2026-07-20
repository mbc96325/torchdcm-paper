# Solver Attempt Matrix

Each benchmark case is attempted with every configured solver where an aligned wrapper exists. Runtimes report estimation plus covariance on one logical CPU.
`ok` means the solver completed the aligned case, `failed` means the wrapper was attempted but failed, and `unsupported` means no aligned wrapper exists for that solver/model case.

| case | dataset | model | torchdcm | scipy_bfgs | biogeme | apollo | mlogit | gmnl | xlogit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| swissmetro_mnl | Swissmetro | MNL | ok (0.154s) | ok (2.313s) | ok (1.529s) | ok (0.827s) | ok (0.135s) | failed | ok (0.025s) |
| nhts_2022_mnl | NHTS 2022 | MNL | ok (0.246s) | ok (38.608s) | ok (2.946s) | ok (7.850s) | ok (1.717s) | ok (29.599s) | ok (0.691s) |
| biogeme_public_airline | airline | MNL | ok (0.148s) | ok (3.081s) | ok (1.737s) | ok (0.417s) | ok (0.053s) | ok (0.127s) | ok (0.011s) |
| biogeme_public_parking | parking | MNL | ok (0.150s) | ok (1.903s) | ok (1.677s) | ok (0.360s) | ok (0.033s) | ok (0.058s) | ok (0.006s) |
| biogeme_public_telephone | telephone | MNL | ok (0.139s) | ok (0.242s) | ok (1.844s) | ok (0.342s) | ok (0.019s) | failed | ok (0.004s) |
| biogeme_public_lpmc | lpmc | MNL | ok (0.257s) | ok (56.524s) | ok (1.714s) | ok (12.830s) | ok (2.434s) | ok (5.946s) | ok (0.324s) |
| mlogit_catsup | catsup | MNL | ok (0.150s) | ok (0.371s) | ok (1.755s) | ok (0.345s) | ok (0.051s) | ok (0.051s) | ok (0.008s) |
| mlogit_cracker | cracker | MNL | ok (0.156s) | ok (0.541s) | ok (1.730s) | ok (0.369s) | ok (0.055s) | ok (0.056s) | ok (0.009s) |
| mlogit_electricity | electricity | MNL | ok (0.156s) | ok (2.779s) | ok (2.212s) | ok (0.492s) | ok (0.186s) | ok (0.293s) | ok (0.013s) |
| mlogit_hc | hc | MNL | ok (0.159s) | ok (0.047s) | ok (2.316s) | ok (0.294s) | ok (0.016s) | ok (0.020s) | ok (0.002s) |
| mlogit_heating | heating | MNL | ok (0.137s) | ok (0.140s) | ok (1.664s) | ok (0.303s) | ok (0.024s) | ok (0.025s) | ok (0.004s) |
| mlogit_mode | mode | MNL | ok (0.157s) | ok (0.197s) | ok (1.608s) | ok (0.292s) | ok (0.016s) | ok (0.017s) | ok (0.002s) |
| mlogit_nox | nox | MNL | ok (0.137s) | ok (0.231s) | ok (5.267s) | ok (0.368s) | ok (0.039s) | failed | ok (0.004s) |
| mlogit_risky_transport | risky_transport | MNL | ok (0.187s) | ok (1.124s) | ok (2.512s) | ok (0.448s) | ok (0.045s) | failed | failed |
| mlogit_train | train | MNL | ok (0.148s) | ok (1.757s) | ok (1.370s) | ok (0.367s) | ok (0.032s) | ok (0.048s) | ok (0.005s) |
| mlogit_fishing | fishing | MNL | ok (0.161s) | ok (0.802s) | ok (1.707s) | ok (0.355s) | ok (1.163s) | ok (0.707s) | ok (0.006s) |
| mlogit_modecanada | modecanada | MNL | ok (0.157s) | ok (1.937s) | ok (1.635s) | ok (0.408s) | ok (0.586s) | failed | failed |

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
