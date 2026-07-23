# Swissmetro Actual-Data Full-Estimation Rerun

This note records the remote rerun used for the paper-facing Table 1 after
removing the erroneous `N=100000` label from the Swissmetro rows. The Biogeme
Swissmetro source has 10,728 raw rows and 10,719 rows after the shared
`CHOICE != 0` filter used by the TorchDCM benchmark loaders.

All rows use the actual filtered Swissmetro data, shared zero initial values,
and aligned model specifications.

| Case | Model | N | K | TorchDCM | SciPy | Biogeme | Apollo | Consistent? |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Swissmetro | MNL | 10719 | 4 | 0.030 | 2.338 | 1.188 | 1.807 | Yes |
| Swissmetro | Nested logit | 10719 | 5 | 0.069 | NA | 2.925 | 2.100 | Yes |
| Swissmetro | Cross-nested logit | 10719 | 6 | 1.108 | NA | 3.870 | NA | Yes |

