# TorchDCM IJOC Software Paper Plan

Target: INFORMS Journal on Computing, software/tools contribution.

## Positioning

TorchDCM is a PyTorch-first discrete choice modeling package with econometric
inference and reproducible estimator benchmarks. The paper should follow the
software-paper pattern in the two reference manuscripts:

1. identify a software gap;
2. introduce an open, modular package;
3. explain implementation and API design;
4. compare with existing software;
5. report public benchmark results;
6. make code and data reproducible.

## Core Claim

Existing discrete-choice software such as Biogeme, Apollo, and R `mlogit`
provides mature econometric modeling interfaces, while modern tensor frameworks
provide efficient differentiable computation. TorchDCM connects these two
worlds: it exposes familiar discrete-choice model abstractions while using
PyTorch for vectorized likelihoods, automatic differentiation, and scalable
estimation.

## Proposed Structure

1. Introduction
2. Related Software
3. TorchDCM Package Design
4. Implementation Details
5. Public Benchmark Data System
6. Computational Experiments
7. Conclusion

## Experiment Families

| Family | Datasets | Models | Backends | Status |
| --- | --- | --- | --- | --- |
| Biogeme public MNL | Airline, Parking, Telephone, LPMC | MNL | TorchDCM, Biogeme | first full battery complete |
| Swissmetro parity | Swissmetro | MNL, NL, CNL, Mixed, WTP, LC | TorchDCM, Biogeme, Apollo, SciPy | MNL/NL/CNL full; mixed/WTP fixed replay; LC fit replay |
| Apollo examples | Mode choice, drug choice, Swiss route | MNL, NL, mixed, hybrid | TorchDCM, Apollo | next |
| R community data | Fishing, ModeCanada, Electricity, Train, Car | MNL, mixed, WTP | TorchDCM, mlogit/logitr/gmnl | partial |
| Large processed data | LPMC, NHTS-like surveys | MNL, NL, mixed | TorchDCM, Biogeme/xlogit where feasible | data workflow started |
| Synthetic controlled | Generated known-truth data | MNL now; NL/mixed next | TorchDCM, known DGP | MNL grid complete |

## Package Landscape

Current remote package status is documented in
`docs/dcm-package-benchmark-landscape.md`.

Integrated or partially integrated packages:

- Biogeme
- Apollo
- R `mlogit`
- R `gmnl`
- Python `xlogit`

Blocked or planned packages:

- `pylogit`: installed but import is blocked on Python 3.12 by legacy
  `collections.Iterable` usage.
- `logitr`: install blocked by remote `cmake`/`nloptr` dependency.
- `choice-learn`, `torch-choice`, `mixl`, `mnlogit`: planned.

## First Full-Estimation Table

Public Biogeme MNL battery, full data, shared zero initial values. Times are
remote `baichuan-mo` wall-clock seconds split into parameter estimation and
covariance/Hessian calculation.

| Case | Dataset | n | k | Torch est. | Torch cov. | Biogeme est. | Biogeme cov. | Consistent? |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| airline | biogeme_airline_itinerary | 3,609 | 5 | 0.024 | 0.002 | 1.182 | 0.000 | Yes |
| parking | biogeme_parking_spain | 1,576 | 5 | 0.025 | 0.001 | 1.169 | 0.000 | Yes |
| telephone | biogeme_telephone | 434 | 5 | 0.023 | 0.001 | 1.197 | 0.000 | Yes |
| lpmc | lpmc_london | 81,086 | 5 | 0.062 | 0.010 | 1.233 | 0.000 | Yes |

Result files:

- `validation/generated/public_mnl_battery_full.md`
- `validation/generated/public_mnl_battery_full.json`

## Cross-Model Benchmark Snapshot

The current model-family comparison is rendered in
`docs/model-family-benchmark-comparison.md` from the remote full-suite JSON.
It separates MNL, nested logit, cross-nested logit, mixed logit, WTP-space
mixed logit, latent class logit, ordered logit, and ordered probit.
The controlled synthetic MNL benchmark is rendered in
`docs/synthetic-controlled-benchmarks.md`.

Key status:

- Full-estimation parity is available for Swissmetro MNL/NL/CNL, Optima
  ordered logit/probit, Fishing MNL, and ModeCanada MNL.
- Shared-draw fixed replay is available for Swissmetro mixed logit and
  WTP-space mixed logit against Biogeme and Apollo.
- Torch-fit plus reference replay is available for Swissmetro latent class
  logit.
- The next manuscript-critical gap is full mixed-logit estimation, including
  simulated covariance, against Biogeme/Apollo and R/Python mixed-logit
  packages where their draw conventions can be aligned.
- Known-truth synthetic MNL experiments now cover controlled variation in
  sample size, alternatives, parameters, feature correlation, and signal scale.

## Immediate Next Steps

1. Add canonical specs for Netherlands and Switzerland/Optima mode-choice data.
2. Expand R `mlogit` full-estimation battery beyond Fishing and ModeCanada.
3. Add Apollo mode-choice full-estimation benchmark.
4. Add xlogit/gmnl mixed-logit comparisons on Electricity/Fishing.
5. Convert current benchmark outputs into manuscript-ready tables with grouped
   software comparisons.
6. Start a LaTeX manuscript skeleton after the main benchmark matrix stabilizes.
