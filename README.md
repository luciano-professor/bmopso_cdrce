# bmopso_cdrce — Binary Multi-Objective Particle Swarm Optimization using Crowding Distance, Roulette Wheel, and Catfish Effect for pymoo

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![pymoo](https://img.shields.io/badge/pymoo-%3E%3D0.6.0-orange.svg)](https://pymoo.org/)
[![Tests](https://img.shields.io/badge/pytest-passing-brightgreen.svg)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**BMOPSO-CDRCE (`bmopso_cdrce`)** is an official, domain-agnostic **Binary Multi-Objective Particle Swarm Optimization using Crowding Distance, Roulette Wheel, and Catfish Effect** library built from the ground up for the [`pymoo`](https://pymoo.org/) multi-objective optimization framework.

> [!NOTE]
> **Nomenclature Note:**
> In the original publication (*Souza et al., WTF 2014*), this technique was originally designated as **`CatfishBMOPSO-CDR`** (or *Catfish BMOPSO-CDR*). For this library release, the name was officially updated and standardized to **`BMOPSO-CDRCE`** (Binary Multi-Objective Particle Swarm Optimization using Crowding Distance, Roulette Wheel, and Catfish Effect) to align with standardized algorithmic naming conventions.

It provides a native `pymoo.core.algorithm.Algorithm` implementation of **BMOPSO-CDRCE** (`BMOPSO_CDRCE`), designed to solve complex binary and combinatorial multi-objective optimization problems while escaping local optima and preventing swarm stagnation. It seamlessly integrates with the [`pymoo-binary-problems`](https://github.com/luciano-professor/pymoo-binary-problems) companion benchmark suite.

---

## 🚀 Seamless `pymoo` Integration

`bmopso_cdrce` is engineered strictly as a first-class citizen of the `pymoo` ecosystem:

```text
+-----------------------------------------------------------------------------+
|                               pymoo Ecosystem                               |
|                                                                             |
|   +---------------------------------------------------------------------+   |
|   |                        pymoo.optimize.minimize                      |   |
|   |                                                                     |   |
|   |   +----------------------+               +----------------------+   |   |
|   |   |     BMOPSO_CDRCE     |               |    BinaryProblem     |   |   |
|   |   |  (pymoo.Algorithm)   | ------------> |   (pymoo.Problem)    |   |   |
|   |   +----------------------+               +----------------------+   |   |
|   |              |                                      |               |   |
|   |              v                                      v               |   |
|   |       Population & Archive                  out["F"] & out["G"]     |   |
|   +---------------------------------------------------------------------+   |
|                                                                             |
|   +------------------------+  +---------------------+  +----------------+   |
|   |   pymoo.indicators     |  | pymoo.visualization |  |  pymoo.core    |   |
|   |   (Hypervolume, IGD)   |  | (Scatter, Petal)    |  |  (Callback)    |   |
|   +------------------------+  +---------------------+  +----------------+   |
+-----------------------------------------------------------------------------+
```

### Key Compatibility Highlights with `pymoo`:

- **Drop-in `minimize()` Execution**: Use standard `pymoo.optimize.minimize(problem, algorithm, termination)` syntax.
- **Native Problem Architecture**: Compatible with `pymoo.core.problem.Problem` and `pymoo_binary_problems.BinaryProblem` setting `type_var=np.bool_`, `xl=0`, `xu=1`, and utilizing standard `out["F"]`, `out["G"]`, and `out["H"]` dictionaries.
- **`pymoo` Termination Criteria**: Fully compatible with all `pymoo` termination formats (`("n_gen", 100)`, `("n_eval", 20000)`, `get_termination("time", "00:05:00")`, `RobustTermination`).
- **`pymoo` Callbacks & Logging**: Integrate custom `pymoo.core.callback.Callback` classes and real-time displays.
- **`pymoo` Performance Indicators**: Calculate convergence metrics using `pymoo.indicators.hv.Hypervolume`, `IGD`, and `IGDPlus`.
- **`pymoo` Visualizations**: Instantly plot resulting non-dominated Pareto Fronts using `pymoo.visualization.scatter.Scatter`.

---

## 📦 Companion Benchmark Suite (`pymoo-binary-problems`)

Combinatorial and ML benchmark problems are provided by the companion package [`pymoo-binary-problems`](https://github.com/luciano-professor/pymoo-binary-problems):

| Problem | Class | Variables | Description | Constraints |
| :--- | :--- | :---: | :--- | :---: |
| **Multiple Knapsack** | `MKP` | N x M bits | Multi-item allocation across multiple capacity-constrained knapsacks | M + N inequalities |
| **Unconstrained Quadratic** | `MUBQP` | n bits | Multi-objective unconstrained binary quadratic interaction matrices | Unconstrained |
| **Traveling Salesman** | `MSTSP` / `MOTSP` | N^2 bits | Binary position-city assignment matrix routing over N cities | 2N + 1 inequalities |
| **Set Covering** | `MOSCP` / `MSCP` | n bits | Minimum-cost subset selection covering m universe elements | m inequalities |
| **Feature Selection** | `MOFS` / `MOBFS` | D bits | Multi-objective classification error vs. dimensionality reduction | >= k_min features |

---

## 🧬 Algorithm Background & Key Pillars

The **BMOPSO-CDRCE** algorithm synthesizes the foundational pillars of swarm intelligence, evolutionary multi-objective optimization, and bio-inspired search diversification:

1. **BMOPSO-CDRCE Formulation (originally published as `CatfishBMOPSO-CDR`)**:
   * Proposed by **Luciano S. de Souza, Ricardo B. C. Prudêncio, and Flávia de A. Barros** in:
     > **"Multi-Objective Test Case Selection: A study of the influence of the Catfish effect on PSO based strategies"**, published in the *XV Workshop de Testes e Tolerância a Falhas (WTF 2014)*. DOI: [10.5753/wtf.2014.22943](https://doi.org/10.5753/wtf.2014.22943). *(Note: Designated as CatfishBMOPSO-CDR in the publication and renamed to BMOPSO-CDRCE in this software package).*
2. **Original BMOPSO Framework**:
   * Proposed by **Luciano S. de Souza, Péricles B. C. de Miranda, Ricardo B. C. Prudêncio, and Flávia de A. Barros** in:
     > **"A Multi-Objective Particle Swarm Optimization for Test Case Selection Based on Functional Requirements Coverage and Execution Effort"**, published in the *2011 23rd IEEE International Conference on Tools with Artificial Intelligence (ICTAI 2011)*. DOI: [10.1109/ICTAI.2011.45](https://doi.org/10.1109/ICTAI.2011.45).
3. **Catfish Effect in Binary PSO**:
   * Proposed by **Li-Yeh Chuang, Sheng-Wei Tsai, and Cheng-Hong Yang** in:
     > **"Improved binary particle swarm optimization using catfish effect for feature selection"**, published in *Expert Systems with Applications*, 38(10), pp. 12699-12707, 2011. DOI: [10.1016/j.eswa.2011.04.057](https://doi.org/10.1016/j.eswa.2011.04.057).
4. **Crowding Distance Roulette (CDR)**:
   * Proposed by **Roberto A. Santana, Marcos R. Pontes, and Carmelo J. A. Bastos-Filho** in:
     > **"A Multiple Objective Particle Swarm Optimization Approach Using Crowding Distance and Roulette Wheel"**, published in *2009 Fifth International Conference on Natural Computation (ICNC 2009)*. DOI: [10.1109/ICNC.2009.610](https://doi.org/10.1109/ICNC.2009.610).
5. **MOPSO with Non-Linear Mutation**:
   * Proposed by **Carlos A. Coello Coello, Gregorio Toscano Pulido, and Maximino Salazar Lechuga** in:
     > **"Handling multiple objectives with particle swarm optimization"**, published in *IEEE Transactions on Evolutionary Computation*, 8(3), pp. 256-279, 2004. DOI: [10.1109/TEVC.2004.826067](https://doi.org/10.1109/TEVC.2004.826067).
6. **Binary Particle Swarm Optimization (BPSO)**:
   * Proposed by **James Kennedy and Russell C. Eberhart** (1997) mapping continuous velocities to binary decisions via logistic sigmoid activation.
7. **Constrained-Dominance Principle**:
   * Proposed by **Kalyanmoy Deb** (2002) for inequality constraint handling without penalty parameter tuning.

---

## 🐟 The Catfish Effect Mechanism

In standard Binary PSO, particles may rapidly cluster around a suboptimal region (local optimum), leading to swarm stagnation where no new non-dominated solutions are discovered.

```text
+-----------------------------------------------------------------------------+
|                          CATFISH EFFECT MECHANISM                           |
|                                                                             |
|  1. Stagnation Detection:                                                   |
|     Pareto Archive did not change for consecutive generations >= threshold? |
|                                                                             |
|                         [ Stagnation Detected ]                             |
|                                    |                                        |
|                                    v                                        |
|  2. Random Particle Replacement:                                            |
|     Select a fraction (catfish_rate = 10%) of swarm particles randomly.     |
|     (External archive preserves all best non-dominated solutions).          |
|                                                                             |
|                                    v                                        |
|  3. Extreme Binary Placement:                                               |
|     For each catfish particle, generate random r in [0, 1]:                 |
|     - If r > 0.5:  Set all bits to 1 ([1, 1, 1, ..., 1])                    |
|     - If r <= 0.5: Set all bits to 0 ([0, 0, 0, ..., 0])                    |
|                                                                             |
|                                    v                                        |
|  4. Inertial Continuity:                                                    |
|     Velocities are PRESERVED. NO re-evaluation is executed.                 |
|     Particles are naturally evaluated in the next iteration cycle.          |
+-----------------------------------------------------------------------------+
```

---

## ⚙️ Hyperparameters

| Parameter | Default Value | Type | Description |
| :--- | :---: | :---: | :--- |
| `n_particles` | `20` | `int` | Swarm population size. |
| `w_max` | `0.9` | `float` | Initial inertia weight at generation 0 (global exploration). |
| `w_min` | `0.4` | `float` | Final inertia weight at final generation (local exploitation). |
| `w` | `None` | `float \| None` | If provided, fixes constant inertia weight (`w_max = w_min = w`). |
| `c1` | `1.49` | `float` | Cognitive acceleration coefficient (attraction to `pbest`). |
| `c2` | `1.49` | `float` | Social acceleration coefficient (attraction to `gbest`). |
| `v_max` | `4.0` | `float` | Velocity clamping bound in `[-v_max, v_max]`. |
| `mutation_rate` | `0.5` | `float \| None` | Mutation rate parameter. If `None`, defaults to adaptive `1 / n_var`. |
| `max_archive_size` | `200` | `int \| None` | Maximum capacity of the non-dominated Pareto archive. |
| `catfish_threshold` | `24` | `int` | Consecutive stagnation generations before triggering Catfish perturbation. |
| `catfish_rate` | `0.10` | `float` | Fraction of swarm particles randomly replaced by catfish particles (10%). |
| `return_least_infeasible` | `True` | `bool` | Return least infeasible solutions when no 100% feasible point is found. |

---

## 📥 Installation

Install directly from GitHub:
```powershell
pip install git+https://github.com/luciano-professor/bmopso_cdrce.git
```

Or for local development:
```powershell
git clone https://github.com/luciano-professor/bmopso_cdrce.git
cd bmopso_cdrce
pip install -e .
```

---

## 💻 Quickstart Example

```python
import numpy as np
from pymoo.optimize import minimize
from pymoo_binary_problems import MKP
from bmopso_cdrce import BMOPSO_CDRCE

# 1. Define or instantiate a multiobjective binary problem
profits = np.array([
    [25, 40, 15, 30, 50, 20, 35, 45, 10, 28],
    [15, 30, 45, 20, 35, 50, 25, 40, 30, 18],
], dtype=float)

weights = np.array([
    [10, 25, 8, 15, 30, 12, 20, 28, 5, 18],
    [8, 18, 28, 12, 22, 32, 16, 26, 20, 10],
], dtype=float)

capacities = np.array([80.0, 75.0], dtype=float)

problem = MKP(profits=profits, weights=weights, capacities=capacities, n_obj=2)

# 2. Instantiate BMOPSO-CDRCE
algorithm = BMOPSO_CDRCE(
    n_particles=30,
    catfish_threshold=15,
    catfish_rate=0.10,
)

# 3. Optimize via pymoo
res = minimize(problem, algorithm, termination=("n_gen", 50), seed=42, verbose=True)

print(f"Non-dominated solutions discovered: {len(res.X)}")
print(f"Catfish Effect triggers: {algorithm.n_catfish_triggers}")
print("Objective Values (F):\n", res.F)
```

---

## 🧪 Testing

Run the full test suite with `pytest`:

```powershell
python -m pytest tests/ -v
```

---

## 📄 License

This project is licensed under the terms of the [MIT License](LICENSE).
