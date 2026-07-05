# Open Discrete Choice Benchmark Data Registry

This folder tracks public datasets that can become TorchDCM validation and
runtime benchmarks. The goal is to build a reproducible benchmark system, not a
one-off collection of examples.

## Registry

- `open_choice_benchmark_registry.csv`: machine-readable dataset registry.
- `download_public_datasets.py`: reproducible downloader/exporter for the
  registry.
- `export_r_package_datasets.R`: helper used by the downloader to export data
  objects from R packages such as `apollo` and `mlogit`.
- `raw/<dataset_id>/`: immutable raw dataset exports plus per-dataset
  `metadata.json`.
- `dataset_manifest.json`: generated summary with status, rows, columns, file
  sizes, and SHA-256 checksums.

Each dataset entry records:

- source family and upstream URL;
- access method and license/terms note;
- size and choice structure;
- model coverage;
- benchmark role;
- current implementation status.

## Tiers

### T1 Core Aligned

Datasets with an upstream estimator example or package API that lets us align
model specification, data cleaning, likelihood, parameters, covariance, and
runtime.

Current T1 anchors:

- Biogeme Swissmetro: MNL, NL, CNL, Mixed Logit, WTP, latent class, panel.
- Biogeme Optima: ordered logit/probit and hybrid choice/measurement models.
- Apollo mode choice, drug choice, and Swiss route choice: Apollo-first
  benchmarks for MNL/NL/CNL, hybrid choice, mixed logit, WTP, and latent class.
- R `mlogit` Fishing and ModeCanada: community MNL benchmarks with beta,
  standard error, t-value, covariance, and runtime comparison.

### T2 Community Classic

Datasets that are widely used in choice-model packages but may require custom
specification alignment.

High-priority T2 datasets:

- `mlogit::Electricity`: panel mixed logit and random coefficients.
- `mlogit::Train`: binary/panel WTP and random-coefficient examples.
- `mlogit::Car`: high-dimensional categorical vehicle-choice benchmark.
- xlogit Electricity/Fishing: independent Python estimator comparison.

### T3 Large Public Survey

Large public surveys useful for stress testing and predictive validation. These
are not parameter-parity datasets until we define a clean choice set, LOS
features, filtering, and weighting protocol.

Current T3 candidate:

- U.S. NHTS 2017: large public travel survey for mode-choice and heterogeneity
  stress tests.

## Benchmark Types

Each dataset can support one or more benchmark types:

- `fixed_replay`: all estimators evaluate the same fixed parameters.
- `full_estimation`: each estimator estimates from the same initial values.
- `torch_fit_then_replay`: TorchDCM estimates; Biogeme/Apollo/R replay the same
  parameters to validate likelihood/probabilities.
- `runtime_only`: used when a backend cannot expose comparable covariance or
  optimizer internals.

## Metrics

For estimator benchmarks, record:

- `estimate_s`;
- `covariance_s`;
- `total_s`;
- final log likelihood;
- beta difference;
- standard error difference;
- t-value/significance difference;
- covariance matrix max absolute difference;
- probability difference;
- WTP and elasticity differences when defined.

## Implementation Order

1. Finish T1:
   - Add Biogeme/Apollo-aligned hybrid choice benchmarks on Optima/drug choice.
   - Add Apollo mode choice and Swiss route choice wrappers.
2. Expand T2:
   - Add `mlogit::Electricity` mixed-logit benchmark.
   - Add `mlogit::Train` WTP/random-coefficient benchmark.
   - Add `mlogit::Car` high-dimensional categorical MNL/mixed benchmark.
3. Scope T3:
   - Define NHTS preprocessing, choice-set construction, and weighting protocol.

No dataset should become part of the formal suite unless the upstream source,
cleaning protocol, model specification, and comparison metrics are documented.

## Downloading

Run the downloader on the remote benchmark machine:

```bash
cd /home/baichuan-mo/torchdcm
.venv/bin/python validation/datasets/download_public_datasets.py
```

The downloader exports all small package/example datasets that can be fetched
reproducibly from installed Python/R packages. Large survey sources such as NHTS
are documented in `raw/nhts_2017/MANUAL_DOWNLOAD.md` until their benchmark
choice-set and preprocessing protocol is fixed.
