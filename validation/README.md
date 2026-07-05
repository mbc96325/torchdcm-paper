# TorchDCM Validation Workspace

This directory is intentionally outside the Git-tracked Python package.

- `benchmarks/`: local comparison scripts, including Biogeme parity checks.
  - `compare_mnl_estimators.py`: TorchDCM vs SciPy BFGS vs Biogeme vs optional Apollo.
  - `compare_ordered_estimators.py`: ordered logit/probit parity on Biogeme Optima
    Likert indicators (`Envir*`, `Mobil*`) using `biogeme.data.optima.read_data`
    and Biogeme's `ordered_*_from_thresholds` model functions.
  - `apollo/R/run_mnl.R`: Apollo Rscript backend used when R and Apollo are installed.
- `datasets/`: public benchmark dataset registry and benchmark-tier definitions.
- `BENCHMARK_SYSTEM.md`: benchmark-system principles, metrics, and extension workflow.
- `tests/`: optional validation tests that require external packages.
- `docs/`: planning documents.
- `references/`: external reference PDFs and notes.
- `generated/`: Biogeme reports, YAML files, and other generated outputs.

The package repository itself should remain a clean `torchdcm` Python package:
source code, packaging metadata, examples, and lightweight tests only.

Benchmark cases should use real data and should be aligned with Biogeme or
Apollo examples whenever those examples are available. If an upstream package
ships the data and model API but not a standalone example script, the benchmark
must say so explicitly in its printed `alignment` block.

The public dataset registry is maintained in
`datasets/open_choice_benchmark_registry.csv`. Add a dataset there before adding
a new validation script.
