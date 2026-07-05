# TorchDCM Paper Benchmarks

This repository contains the empirical validation, estimator comparisons,
synthetic benchmark scripts, figures, tables, and LaTeX source for the
TorchDCM software paper.

The TorchDCM package itself lives in a separate repository:

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

Install the package repo first, then install the paper dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e "git+https://github.com/mbc96325/torchdcm.git#egg=torchdcm"
python -m pip install -e ".[bench]"
```

Some benchmark backends are external to Python:

- Biogeme: installed through the Python extra when available.
- Apollo, `mlogit`, and `gmnl`: installed in R.
- `xlogit`: installed through the Python benchmark extra.

## Main Benchmark Tables

The paper-facing benchmark tables are generated from the validation artifacts:

- Table 1: real-data MNL comparisons.
- Table 2: real-data nested-logit-family comparisons.
- Table 3: real-data mixed-logit replay comparisons.
- Synthetic controlled MNL benchmarks vary sample size, number of variables,
  number of alternatives, and feature correlation.

The current full solver matrix is stored in:

- `validation/generated/solver_attempt_matrix_full.json`
- `validation/generated/solver_attempt_matrix_full.md`

## Manuscript

Build the local PDF from `paper/`:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode main.tex
```

The local manuscript PDF is `paper/main.pdf`.

