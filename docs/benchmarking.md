# Benchmarking

Benchmarks compare TorchDCM against Biogeme, Apollo, and R estimators with
aligned data, specification, starting values, and metrics.

Paper-facing metrics:

- parameter-estimation time;
- covariance/Hessian time;
- total runtime where measurable;
- `Consistent?` yes/no flag under the prespecified tolerance rule.

Raw audit metrics, including likelihood, parameter, probability, covariance,
standard-error, WTP, and elasticity differences, remain in generated validation
artifacts but are not displayed in the main paper tables.

Current benchmark reports:

- [Model-family benchmark comparison](model-family-benchmark-comparison.md)
- [Package benchmark landscape](dcm-package-benchmark-landscape.md)
- [Synthetic controlled benchmarks](synthetic-controlled-benchmarks.md)
- [IJOC software paper plan](ijoc-software-paper-plan.md)

## Interpreting Fast `xlogit` MNL Runs

`xlogit` is extremely fast in several small Table 1 MNL rows because the
benchmark gives it a narrow, highly optimized task. The generic wrapper passes
a prebuilt long-format numeric design matrix directly to
`xlogit.MultinomialLogit.fit`, with `fit_intercept=False` and already aligned
coefficient columns. This avoids symbolic utility parsing, wide-to-long
conversion, repeated package-level model compilation, R subprocess startup, and
Biogeme report/file-output overhead.

The speed should therefore be interpreted as strong performance for standard
long-format MNL rather than as a general statement about all DCM workloads. On
small datasets, fixed framework overhead dominates measured runtime, which is
why `xlogit` can finish in a few milliseconds. On larger real-data rows such as
NHTS 2022 and LPMC London, TorchDCM's tensorized likelihood path is faster in
the current measurements. `xlogit` also requires consistent alternatives in the
long table, so ragged choice-set cases such as ModeCanada and RiskyTransport are
recorded as attempted failures rather than consistency failures.

Remote execution remains the source of truth:

```bash
cd /home/baichuan-mo/torchdcm
.venv/bin/python validation/benchmarks/run_estimator_benchmark_suite.py --profile full
```
