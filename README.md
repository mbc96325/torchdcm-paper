# TorchDCM Paper Benchmarks

This repository contains the empirical validation, estimator comparisons,
synthetic benchmark scripts, figures, tables, and LaTeX source for the
TorchDCM software paper.

The TorchDCM package itself lives in a separate repository:

- PyPI: https://pypi.org/project/torchdcm/
- Package repo: https://github.com/mbc96325/torchdcm
- Paper repo: https://github.com/mbc96325/torchdcm-paper

## Repository Layout

| Path | Purpose |
| --- | --- |
| `paper/` | LaTeX manuscript, bibliography, tables, and compiled local PDF. |
| `validation/` | Full-estimation comparisons against Biogeme, Apollo, SciPy, `mlogit`, `gmnl`, and `xlogit`. |
| `datasets/` | Public benchmark dataset index, small processed datasets, and large-data release links. |
| `scripts/` | Benchmark rendering, synthetic benchmark, and dataset materialization utilities. |
| `docs/` | Benchmark notes, data documentation, and paper planning notes. |

## Environment

Install TorchDCM from PyPI, then install the paper dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install torchdcm
python -m pip install -e ".[bench]"
```

Some benchmark backends are external to Python:

- Biogeme: installed through the Python extra when available.
- Apollo, `mlogit`, and `gmnl`: installed in R.
- `xlogit`: installed through the Python benchmark extra.

## Benchmark Matrix

The paper tables are generated from the following committed validation
artifacts. The linked runner and output use the same model specification and
benchmark conventions.

| Benchmark | Runner | Paper-facing output |
| --- | --- | --- |
| Controlled synthetic MNL, NL, and MixL | [`compare_generated_choice_battery.py`](validation/benchmarks/compare_generated_choice_battery.py) | [MNL](validation/generated/generated_choice_battery_controlled_office.md), [NL](validation/generated/generated_choice_battery_table4_nl_office.md), [MixL](validation/generated/generated_choice_battery_table4_mixl_office.md) |
| Real-data MNL | [`run_solver_attempt_matrix.py`](validation/benchmarks/run_solver_attempt_matrix.py) | [`solver_attempt_matrix_mnl_single_core_office.md`](validation/generated/solver_attempt_matrix_mnl_single_core_office.md) |
| Real-data NL | [`compare_real_nested_logit_battery.py`](validation/benchmarks/compare_real_nested_logit_battery.py) | [`nested_real_battery_single_core_office.md`](validation/generated/nested_real_battery_single_core_office.md) |
| Real-data MixL | [`compare_real_mixed_logit_battery.py`](validation/benchmarks/compare_real_mixed_logit_battery.py) | [`mixed_real_battery_apollo_office.md`](validation/generated/mixed_real_battery_apollo_office.md) |
| CPU--GPU scaling | [`compare_torch_device_stress.py`](validation/benchmarks/compare_torch_device_stress.py) | [`torch_device_stress_battery.md`](validation/generated/torch_device_stress_battery.md) |
| Ordered logit and probit | [`compare_ordered_estimators.py`](validation/benchmarks/compare_ordered_estimators.py) | [Logit JSON](validation/generated/ordered_logit_single_core_office.json), [probit JSON](validation/generated/ordered_probit_single_core_office.json) |

See the [validation guide](validation/README.md) and
[dataset catalog](datasets/dataset_index.csv) for setup, provenance, and the
complete collection of generated artifacts.

## Manuscript

Build the local PDF from `paper/`:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode main.tex
```

The local manuscript PDF is `paper/main.pdf`.
