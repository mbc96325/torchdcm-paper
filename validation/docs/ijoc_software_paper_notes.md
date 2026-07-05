# TorchDCM IJOC Software Paper Notes

Target: INFORMS Journal on Computing, software/tools contribution.

## Reference Paper Pattern

The two reference papers follow a similar structure:

1. Motivation: a widely used computational task lacks an open, robust, extensible, or benchmarked implementation.
2. Software contribution: introduce the package, user-facing API, and design principles.
3. Implementation details: clarify ambiguous theoretical/modeling steps and explain engineering choices.
4. Comparative software landscape: position against current packages.
5. Computational results: public benchmarks, transparent setup, runtime tables, accuracy/quality metrics.
6. Reproducibility: source code and data are public.

## TorchDCM Positioning

TorchDCM fills a gap between econometric discrete-choice packages and modern tensor computation:

- Biogeme and Apollo are mature econometric references but not PyTorch-native.
- ML frameworks are fast and differentiable but do not expose econometric model APIs, inference, WTP, and estimator parity checks.
- TorchDCM provides a compact PyTorch-first package plus a public benchmark suite against Biogeme, Apollo, and R estimators.

## Proposed Paper Structure

1. Introduction
   - discrete choice models are central in transportation, marketing, operations, and revenue management;
   - modern research needs differentiable, GPU-ready, reproducible estimators;
   - existing packages are powerful but fragmented across R/Python and not tensor-native.
2. Related Software
   - Biogeme, Apollo, mlogit, gmnl/logitr, xlogit, PyLogit/choice-learn.
3. Package Design
   - data containers, utility specification, model zoo, inference, post-estimation metrics.
4. Implementation
   - ragged choice sets, vectorized likelihoods, Hessian/covariance, panel likelihood, draws.
5. Benchmark Data System
   - public datasets from Biogeme, Apollo, R mlogit, and LPMC;
   - GitHub-small vs Google-Drive processed large data policy.
6. Computational Experiments
   - full estimation parity: beta, SE, covariance, probability, runtime split;
   - fixed replay for simulated-likelihood models with shared draws;
   - scalability on LPMC and large public surveys once processed.
7. Conclusion
   - TorchDCM as open infrastructure for discrete-choice software research.

## First Full-Estimation Battery

The first expanded battery focuses on public Biogeme MNL datasets:

- airline itinerary choice;
- parking choice in Spain;
- telephone service choice;
- London Passenger Mode Choice (LPMC).

Metrics:

- parameter-estimation time;
- covariance/Hessian time;
- final log likelihood;
- max beta difference;
- max probability difference;
- max covariance and standard-error difference.
