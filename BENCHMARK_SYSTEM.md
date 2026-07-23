# TorchDCM Public Benchmark System

TorchDCM benchmarks are organized around public, reproducible datasets and
aligned estimator comparisons.

## Principles

1. Use public data only.
2. Prefer datasets shipped by Biogeme, Apollo, R `mlogit`, or other estimator
   packages because they come with known model examples.
3. Separate package code from validation code.
4. Split parameter-estimation time and covariance-calculation time whenever the
   backend exposes both.
5. Report parameter, significance, covariance, probability, WTP, and elasticity
   differences where the model defines them.
6. If TorchDCM is slower on an aligned case, optimize the TorchDCM path before
   declaring the benchmark complete.

## Dataset Registry

Dataset metadata lives in:

- `datasets/open_choice_benchmark_registry.csv`
- `datasets/VALIDATION.md`

Statuses:

- `implemented`: included in a runnable comparison script.
- `partial`: data source is used, but not all target models are covered.
- `planned`: source/spec identified, benchmark wrapper not yet implemented.
- `scoping`: public data exists, but choice-set and cleaning protocol still need
  definition.

## Current Implemented Suite

The unified runner is:

```bash
python benchmarks/run_estimator_benchmark_suite.py --profile smoke
python benchmarks/run_estimator_benchmark_suite.py --profile full
```

The advanced-model full-estimation runner is:

```bash
python benchmarks/run_advanced_full_suite.py \
  --output generated/advanced_full_estimation_office.json
```

Current implemented datasets:

- Biogeme Swissmetro:
  - MNL full estimation;
  - Nested Logit full estimation;
  - Cross-Nested Logit Torch fit + Biogeme/Apollo replay;
  - Mixed Logit fixed replay with shared draws;
  - WTP Mixed Logit fixed replay with shared draws;
  - Latent Class full estimation in TorchDCM, Biogeme, and Apollo.
- Biogeme Optima:
  - Ordered Logit and Ordered Probit full estimation;
  - Hybrid Choice full estimation in TorchDCM, Biogeme, and Apollo.
- R `mlogit` Electricity:
  - Panel Mixed Logit full estimation in TorchDCM, Biogeme, and Apollo.
- R `mlogit`:
  - Fishing MNL full estimation;
  - ModeCanada ragged MNL full estimation.

## Adding A Dataset

1. Add a row to `open_choice_benchmark_registry.csv`.
2. Create a loader that exports a canonical TorchDCM dataset:
   - long rows;
   - stable alternative ordering;
   - explicit availability;
   - observation and panel IDs;
   - documented scaling.
3. Implement an upstream backend wrapper:
   - Biogeme for Biogeme examples;
   - Apollo for Apollo examples;
   - R `mlogit`/`gmnl`/`logitr` for CRAN examples;
   - xlogit for Python estimator comparisons.
4. Align parameter names and ordering.
5. Report:
   - runtime split;
   - beta diff;
   - standard error diff;
   - t-value diff;
   - covariance diff;
   - probability diff;
   - WTP/elasticity diff when applicable.
6. Add the case to `run_estimator_benchmark_suite.py`.
7. Run remotely on `ssh office`; do not rely on local runtime results.

## Priority Queue

1. Apollo examples:
   - mode choice for MNL/NL/CNL;
   - Swiss route choice for Mixed Logit/WTP/LC.
2. R community datasets:
   - `mlogit::Train` for WTP and binary choice;
   - `mlogit::Car` for high-dimensional categorical attributes.
3. Large public surveys:
   - NHTS 2017 after preprocessing protocol is defined.
