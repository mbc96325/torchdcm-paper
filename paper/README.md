# TorchDCM IJOC Software Paper Draft

This directory contains the LaTeX manuscript draft for the IJOC software-paper
target.

The draft follows the structure of the local reference software papers and the
writing workflow from `Master-cai/Research-Paper-Writing-Skills`:

1. define the paper story before sentence polishing;
2. keep one paragraph to one message;
3. align all major claims with benchmark evidence;
4. use clean, minimal-ink tables;
5. keep reviewer-risk notes in the appendix while the draft is still evolving.

Build:

```bash
make
```

Main result sources:

- `validation/generated/estimator_benchmark_suite_full.json`
- `validation/generated/public_mnl_battery_full.json`
- `validation/generated/synthetic_controlled_mnl_full.json`
- `docs/model-family-benchmark-comparison.md`
- `docs/synthetic-controlled-benchmarks.md`
