# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-08-31

### Added
- **Core Algorithm (`BMOPSO_CDRCE`)**:
  - Implemented Binary Multi-Objective Particle Swarm Optimization using Crowding Distance Roulette and Catfish Effect (originally published as `CatfishBMOPSO-CDR` in Souza et al., WTF 2014, standardized to `BMOPSO-CDRCE`):
    - **Crowding Distance Roulette (CDR)** (*Santana et al., 2009; Souza et al., ICTAI 2011, DOI: [10.1109/ICTAI.2011.45](https://doi.org/10.1109/ICTAI.2011.45)*):
      - Social leader (`gbest`) selection via Crowding Distance Roulette wheel weighted by normalized crowding distance.
      - External archive capacity pruning based on descending crowding distance to maintain maximum Pareto front spread.
      - Archive-guided cognitive leader (`pbest`) replacement comparing Euclidean nearest neighbor crowding distance for mutually non-dominated solutions.
    - **Catfish Effect Operator** (*Chuang et al., 2011; Souza et al., WTF 2014, DOI: [10.5753/wtf.2014.22943](https://doi.org/10.5753/wtf.2014.22943)*):
      - Stagnation detection on non-dominated Pareto archive updates across consecutive iterations (`catfish_threshold = 24`).
      - Probabilistic replacement of randomly selected particles (`catfish_rate = 0.10`) with extreme binary vectors (`[1, 1, ...]` if $r > 0.5$, else `[0, 0, ...]`).
      - Velocities preserved and natural evolutionary cycle evaluation without redundant function evaluations.
    - **BPSO & Constraint Handling Foundations**:
      - Continuous velocity to binary position mapping via logistic sigmoid activation (*Kennedy & Eberhart, 1997*).
      - Constrained-Dominance Principle handling inequality constraints natively without penalty tuning (*Deb, 2002*).
      - Dynamic linear inertia weight decay ($w_{\text{max}} = 0.9 \to w_{\text{min}} = 0.4$) and non-linear mutation/turbulence.
  - Direct inheritance from `pymoo.core.algorithm.Algorithm`.

- **Operators Subpackage (`bmopso_cdrce.operators`)**:
  - `catfish`: Stagnation perturbation, random particle replacement, and extreme binary position generation.
  - `velocity`: Clamped continuous velocity update with inertia, cognitive ($c_1=1.49$), and social ($c_2=1.49$) forces.
  - `sampling`: Logistic sigmoid activation and boolean sampling.
  - `mutation`: Non-linear decaying mutation / turbulence operator.
  - `pbest`: Archive-guided personal best replacement using crowding distance.

- **Utilities Subpackage (`bmopso_cdrce.util`)**:
  - `dominance`: Kalyanmoy Deb's constrained Pareto dominance checks and filtering.
  - `diversity`: Crowding distance computation and roulette wheel probability normalization.
  - `archive`: Non-dominated archive with crowding distance capacity pruning and change tracking for stagnation monitoring.

- **Benchmark Integration & Examples**:
  - Full native integration with [`pymoo-binary-problems`](https://github.com/luciano-professor/pymoo-binary-problems) (`MKP`, `MUBQP`, `MSTSP`, `MOSCP`, `MOFS`).
  - Standalone executable examples for all 5 benchmarks in `examples/`.
  - Comprehensive 25-test suite in `tests/`.
