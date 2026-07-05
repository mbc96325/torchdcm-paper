# DCM Package Benchmark Landscape

This note tracks discrete choice model packages considered for TorchDCM
benchmarking. The paper-facing goal is to compare runtime splits and report a
single consistency flag on public datasets. Raw numerical-difference audits
remain in generated validation artifacts.

## Package Matrix

| Package | Language | Remote status | Benchmark role | Current integration |
| --- | --- | --- | --- | --- |
| TorchDCM | Python/PyTorch | installed | focal package | full |
| Biogeme | Python | installed | econometric reference | MNL, NL, CNL, ordered, mixed/fixed replay |
| Apollo | R | installed | flexible R reference | MNL/NL, fixed replay for advanced models |
| mlogit | R | installed | classic MNL reference | Fishing, ModeCanada full estimation |
| gmnl | R | installed | generalized MNL/mixed reference | Fishing MNL full estimation; ModeCanada ragged case fails in current wrapper |
| xlogit | Python | installed | Python MNL/mixed reference | Fishing MNL full estimation; ModeCanada ragged case unsupported by current API |
| pylogit | Python | installed but import blocked on Python 3.12 | legacy Python DCM | blocked by `collections.Iterable` import error |
| logitr | R | install blocked | modern R MNL/mixed reference | blocked by `nloptr`/`cmake` dependency on remote |
| choice-learn | Python | not installed | ML-oriented choice modeling | planned |
| torch-choice | Python/PyTorch | not installed | PyTorch choice modeling | planned |
| mixl | R | not installed | R mixed logit | planned |
| mnlogit | R | not installed | fast multinomial logit | planned |

## Biogeme Public MNL

All cases use shared zero initial values and compare classic inverse observed
information / Rao-Cramer covariance.

| Case | Dataset | n | k | Torch cov. (s) | Biogeme cov. (s) | Consistent? |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| airline | biogeme_airline_itinerary | 3,609 | 5 | 0.002 | 0.000 | Yes |
| parking | biogeme_parking_spain | 1,576 | 5 | 0.001 | 0.000 | Yes |
| telephone | biogeme_telephone | 434 | 5 | 0.001 | 0.000 | Yes |
| lpmc | lpmc_london | 81,086 | 5 | 0.010 | 0.000 | Yes |

Source: `validation/generated/public_mnl_battery_full.md`.

## Covariance Comparison: R/Python Community Packages

### Fishing MNL

| Backend | Estimate (s) | Cov. (s) | Consistent? |
| --- | ---: | ---: | --- |
| TorchDCM | 0.030 | 0.001 | Reference |
| mlogit | 0.032 | 0.000 | Yes |
| gmnl | 0.062 | 0.001 | Yes |
| xlogit | 0.006 | 0.000 | Yes |

### ModeCanada MNL

| Backend | Estimate (s) | Cov. (s) | Consistent? | Status |
| --- | ---: | ---: | --- | --- |
| TorchDCM | 0.026 | 0.002 | Reference | available |
| mlogit | 0.075 | 0.001 | Yes | available |
| gmnl | NA | NA | NA | failed: non-conformable arrays in current wrapper |
| xlogit | NA | NA | NA | unsupported: requires consistent alternatives in long format |

Sources:

- `validation/generated/mlogit_fishing_package_comparison.md`
- `validation/generated/mlogit_modecanada_package_comparison.md`

## Next Integration Targets

1. Install/enable `logitr` after adding remote `cmake` or a binary dependency path.
2. Add `xlogit` mixed-logit comparison on Electricity/Fishing with shared draws where possible.
3. Add `gmnl` mixed-logit comparison on Electricity/Train.
4. Add Apollo mode-choice full estimation on Apollo's built-in examples.
5. Investigate Python 3.12 compatibility or pinned environment for `pylogit`.
