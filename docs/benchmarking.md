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

Remote execution remains the source of truth:

```bash
cd /home/baichuan-mo/torchdcm
.venv/bin/python validation/benchmarks/run_estimator_benchmark_suite.py --profile full
```
