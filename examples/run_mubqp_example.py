"""Example script demonstrating BMOPSO-CDRCE on the Multiobjective Unconstrained Binary Quadratic Problem (MUBQP).

This script provides a practical walkthrough to:
1. Generate and configure a Multiobjective Unconstrained Binary Quadratic Problem (MUBQP) benchmark instance.
2. Configure the BMOPSO_CDRCE algorithm hyperparameters (including Catfish Effect).
3. Execute multiobjective optimization using pymoo.optimize.minimize.
4. Analyze the resulting non-dominated Pareto Front and Catfish triggers.

To execute:
    python examples/run_mubqp_example.py
"""

from typing import Any
import numpy as np
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from pymoo_binary_problems import MUBQP
from bmopso_cdrce import BMOPSO_CDRCE


def create_sample_mubqp_instance() -> MUBQP:
    """Create a sample instance of the Multiobjective Unconstrained Binary Quadratic Problem.

    Scenario:
    - 50 binary decision variables.
    - 2 conflicting quadratic objectives.
    - Density = 0.8 (80% non-zero pairwise interaction terms).
    - Interaction values sampled in range [-100.0, 100.0].

    Returns
    -------
    MUBQP
        Configured MUBQP benchmark problem instance.
    """
    return MUBQP.from_random(
        n_var=50,
        n_obj=2,
        density=0.8,
        val_range=(-100.0, 100.0),
        symmetric=True,
        maximize=True,
        seed=42,
    )


def main() -> None:
    """Execute complete MUBQP multiobjective optimization workflow using BMOPSO-CDRCE."""
    print("=" * 75)
    print(" MULTIOBJECTIVE UNCONSTRAINED BINARY QUADRATIC PROBLEM (MUBQP) WITH BMOPSO-CDRCE")
    print("=" * 75)

    problem = create_sample_mubqp_instance()
    print("\nProblem initialized successfully:")
    print(f"  - Decision Variables (n_var) : {problem.n_var}")
    print(f"  - Objective Count (F)        : {problem.n_obj}")
    print(f"  - Optimization Direction     : Maximize (pymoo minimizes negative)")

    # Configure BMOPSO_CDRCE algorithm
    algorithm = BMOPSO_CDRCE(
        n_particles=40,
        w_max=0.9,
        w_min=0.4,
        c1=1.49,
        c2=1.49,
        v_max=4.0,
        mutation_rate=0.5,
        catfish_threshold=10,
        catfish_rate=0.10,
    )

    n_gen = 30
    print(f"\nStarting optimization with termination criterion of {n_gen} generations...")

    res = minimize(
        problem,
        algorithm,
        termination=("n_gen", n_gen),
        seed=42,
        verbose=False,
    )

    print("\nOptimization finished!")
    print(f"Catfish Effect Triggers: {algorithm.n_catfish_triggers}")
    print(f"Total Non-Dominated Solutions Discovered: {len(res.X)}")

    print("\n" + "=" * 75)
    print(" SUMMARY OF TOP PARETO FRONT SOLUTIONS (ORIGINAL MAXIMIZED VALUES)")
    print("=" * 75)
    print(f" {'#':<3} | {'Objective 1 (Value)':<24} | {'Objective 2 (Value)':<24} | {'Active Bits'}")
    print("-" * 75)

    # Sort solutions by Objective 1 in descending order
    sorted_indices = np.argsort(res.F[:, 0])

    for rank, idx in enumerate(sorted_indices[:8], 1):
        real_f1 = -res.F[idx, 0]
        real_f2 = -res.F[idx, 1]
        active_bits = int(np.sum(res.X[idx]))
        print(f" {rank:<3} | {real_f1:<24.2f} | {real_f2:<24.2f} | {active_bits}/{problem.n_var} bits")


if __name__ == "__main__":
    main()
