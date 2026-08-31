# RULE: BMOPSO-CDRCE CODE STYLE AND PYMOO INTEGRATION STANDARD

All code, files, modules, classes, functions, tests, documentation, and project structure generated for this workspace MUST follow the standards below:

## 1. PACKAGE ARCHITECTURE
- The project name is strictly: `bmopso_cdrce` (distribution `bmopso_cdrce`)
- Repository URL: `https://github.com/luciano-professor/bmopso_cdrce`
- The package root MUST be: `src/bmopso_cdrce/` (module `src/bmopso_cdrce/`)
- The package MUST expose:
    - `BMOPSO_CDRCE` (custom pymoo Algorithm combining Crowding Distance Roulette with Catfish Effect)
    - `NonDominatedArchive` (archive utility)
    - `apply_catfish_effect`, `generate_extreme_binary_positions`, `select_random_particles` (catfish operators)
- Benchmark problems are provided via the companion package `pymoo-binary-problems`.
- No Optuna integration is allowed in this project.
- All integration MUST be designed for `pymoo`.

## 2. PYMOO INTEGRATION RULES
- Any optimization algorithm MUST inherit from `pymoo.core.algorithm.Algorithm`.
- Any problem MUST inherit from `pymoo.core.problem.Problem` (or `pymoo_binary_problems.BinaryProblem`).
- All algorithms MUST implement at least:
    - `_initialize(self)`
    - `_next(self)`
- All problems MUST implement:
    - `_evaluate(self, x, out, *args, **kwargs)`
- All algorithms MUST follow pymoo's expected data flow:
    - Use `self.pop`, `self.problem`, `self.evaluator`, and `out["F"]`.
- All problems MUST define:
    - `type_var=np.bool_`
    - `xl=0`, `xu=1`
    - `n_var`, `n_obj`, `n_constr` (if needed)

## 3. BINARY PSO, CDR & CATFISH EFFECT RULES
- Particle positions MUST be binary (0/1 / `np.bool_`).
- Velocities MUST be real-valued arrays clamped to `[-v_max, v_max]`.
- Position update MUST use a sigmoid activation (Kennedy & Eberhart, 1997).
- Archive of non-dominated solutions MUST be maintained using Crowding Distance pruning (Santana et al., 2009; Deb, 2002).
- Social leader selection MUST use Crowding Distance Roulette (CDR) wheel selection (Santana et al., 2009; Souza et al., ICTAI 2011).
- Personal best (`pbest`) update MUST use archive-guided Euclidean nearest neighbor crowding distance comparison when solutions are mutually non-dominated (Santana et al., 2009).
- Dominance MUST follow pymoo's standard utilities and Deb's Constrained-Dominance Principle (Deb, 2002).
- **Catfish Effect Operator** (*Chuang et al., 2011; Souza et al., WTF 2014, DOI: 10.5753/wtf.2014.22943*):
  - Stagnation detection is monitored on non-dominated Pareto archive changes.
  - When stagnation persists for `catfish_threshold` consecutive generations, replace `catfish_rate` (default 10%) randomly selected particles from the current swarm (as all non-dominated solutions are preserved in the External Archive).
  - ONLY particle positions MUST be updated to extreme binary points: for each catfish particle, if random r > 0.5 set all dimensions to 1, otherwise set all dimensions to 0; velocities are preserved.
  - Particles MUST NOT be re-evaluated upon catfish introduction; they are evaluated naturally in the subsequent evolutionary cycle.

## 4. PYTHON CODE STYLE
- All functions and methods MUST use Python type hints.
- All classes MUST include docstrings explaining their purpose.
- Code MUST follow PEP8.
- No unused imports.
- No dead code.
- No global mutable state.

## 5. TESTING RULES
- All tests MUST be placed in `tests/`.
- Tests MUST use `pytest`.
- Tests MUST validate:
    - Algorithm runs inside `pymoo.optimize.minimize()`
    - Solutions are binary
    - Objectives are valid
    - Catfish effect correctly perturbs stagnated swarms
    - Crowding distance roulette and archive pruning work correctly
    - No runtime errors occur

## 6. FILE GENERATION RULE
- ANY new file generated MUST automatically follow all rules above.
- ANY new class MUST automatically follow pymoo integration patterns.
- ANY new function MUST automatically include type hints and docstrings.
- ANY new test MUST automatically use pytest and `pymoo.optimize.minimize()`.

## 7. PROHIBITED CONTENT
- No Optuna imports, classes, or references.
- No continuous-variable PSO.
- No unrelated optimization frameworks.
- No deviation from pymoo's architecture.

## 8. MATHEMATICAL FORMULAS AND FORMATTING
- Do NOT use raw LaTeX math delimiters ($$ ... $$ or $ ... $) for equations in chat.
- Always present mathematical formulas and equations using clean, highly-legible ASCII/text diagrams inside code blocks (` ```text `) or Python snippets (` ```python `).

## 9. LANGUAGE AND GIT CONFIGURATION
- All code, docstrings, comments, commit messages, documentation, and agent responses MUST be written in English.
- Git commit author email MUST strictly be: `43418124+luciano-professor@users.noreply.github.com`.
