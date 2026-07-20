# TorchDCM Validation Workspace

This directory contains the public benchmark and estimator-validation system
used by the TorchDCM software paper. It is separate from the lightweight
installable package so that Biogeme, Apollo, R packages, public datasets, and
generated artifacts do not become runtime dependencies of `torchdcm`.

## Canonical Pipelines

| Benchmark | Primary runner | Committed result |
| --- | --- | --- |
| Controlled synthetic MNL, NL, and MixL | [`compare_generated_choice_battery.py`](benchmarks/compare_generated_choice_battery.py) | [MNL](generated/generated_choice_battery_controlled_office.md), [NL](generated/generated_choice_battery_table4_nl_office.md), [MixL](generated/generated_choice_battery_table4_mixl_office.md) |
| Real-data MNL solver matrix | [`run_solver_attempt_matrix.py`](benchmarks/run_solver_attempt_matrix.py) | [`solver_attempt_matrix_mnl_single_core_office.md`](generated/solver_attempt_matrix_mnl_single_core_office.md) |
| Real-data NL | [`compare_real_nested_logit_battery.py`](benchmarks/compare_real_nested_logit_battery.py) | [`nested_real_battery_single_core_office.md`](generated/nested_real_battery_single_core_office.md) |
| Real-data MixL | [`compare_real_mixed_logit_battery.py`](benchmarks/compare_real_mixed_logit_battery.py) | [`mixed_real_battery_apollo_office.md`](generated/mixed_real_battery_apollo_office.md) |
| CPU--GPU scaling | [`compare_torch_device_stress.py`](benchmarks/compare_torch_device_stress.py) | [`torch_device_stress_battery.md`](generated/torch_device_stress_battery.md) |
| Ordered logit and probit | [`compare_ordered_estimators.py`](benchmarks/compare_ordered_estimators.py) | [Logit JSON](generated/ordered_logit_single_core_office.json), [probit JSON](generated/ordered_probit_single_core_office.json) |

Shared single-core timing controls are implemented in
[`benchmark_runtime.py`](benchmarks/benchmark_runtime.py). The `apollo/R/` and
`mlogit/R/` directories contain the external R backends used by these runners.

## Directory Layout

- `benchmarks/`: Python orchestration and external-estimator wrappers.
- `datasets/`: validation manifests, downloaded data, and preparation notes.
- `generated/`: machine-readable JSON and paper-facing Markdown results.
- `tests/`: optional validation tests requiring external software.
- `docs/`: benchmark design and execution notes.
- `references/`: external reference papers and notes.
- `BENCHMARK_SYSTEM.md`: metrics, tolerances, and extension principles.

The package repository itself should remain a clean `torchdcm` Python package:
source code, packaging metadata, examples, and lightweight tests only.

Benchmark cases should use real data and should be aligned with Biogeme or
Apollo examples whenever those examples are available. If an upstream package
ships the data and model API but not a standalone example script, the benchmark
must say so explicitly in its printed `alignment` block.

The public dataset registry is maintained in
`datasets/open_choice_benchmark_registry.csv`. Add a dataset there before adding
a new validation script.
