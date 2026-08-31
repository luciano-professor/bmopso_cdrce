"""Example script demonstrating BMOPSO-CDRCE on Multiobjective Feature Selection (MOFS).

This script provides a practical walkthrough to:
1. Generate and configure a Multiobjective Feature Selection (MOFS) benchmark instance.
2. Configure BMOPSO_CDRCE algorithm hyperparameters (including Catfish Effect).
3. Execute multiobjective optimization using pymoo.optimize.minimize.
4. Analyze the resulting non-dominated Pareto Front and Catfish triggers.

To execute:
    python examples/run_mofs_example.py
"""

from typing import Any
import numpy as np
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from pymoo_binary_problems import MOFS
from bmopso_cdrce import BMOPSO_CDRCE


def create_sample_mofs_instance() -> MOFS:
    """Create a sample instance of the Multiobjective Feature Selection (MOFS) problem.

    Scenario:
    - 200 patient / sample records.
    - 30 total features:
      * 8 truly informative features (indices 0 to 7)
      * 4 redundant features (indices 8 to 11)
      * 18 noisy / irrelevant features (indices 12 to 29)
    - 2 conflicting objectives:
      * Objective 1: Classification Error Rate (1 - Accuracy)
      * Objective 2: Feature Selection Ratio (||x||_1 / 30)

    Returns
    -------
    MOFS
        Configured MOFS benchmark problem instance.
    """
    return MOFS.from_synthetic(
        n_samples=200,
        n_features=30,
        n_informative=8,
        n_redundant=4,
        n_classes=2,
        cv=3,
        min_features=1,
        seed=42,
    )


def print_solution_details(
    sol_idx: int,
    x_flat: np.ndarray,
    f_val: np.ndarray,
    problem: MOFS,
) -> None:
    """Print selected feature details for a candidate Pareto solution."""
    selected_indices = np.where(x_flat)[0].tolist()
    error_rate = float(f_val[0])
    ratio = float(f_val[1])
    n_selected = len(selected_indices)

    print(f"\n--- Solution #{sol_idx + 1} ---")
    print(f"  Classification Error : {error_rate:.4f} (Accuracy: {(1.0 - error_rate) * 100:.2f}%)")
    print(f"  Selected Features    : {n_selected} / {problem.n_features} ({ratio * 100:.1f}%)")
    print(f"  Feature Indices      : {selected_indices}")


def main() -> None:
    """Execute complete MOFS multiobjective optimization workflow using BMOPSO-CDRCE."""
    print("=" * 75)
    print(" MULTIOBJECTIVE FEATURE SELECTION (MOFS) WITH BMOPSO-CDRCE")
    print("=" * 75)

    problem = create_sample_mofs_instance()
    print("\nProblem initialized successfully:")
    print(f"  - Sample Count              : {problem.n_samples}")
    print(f"  - Total Feature Count (n_var): {problem.n_features}")
    print(f"  - Objective Count (F)       : {problem.n_obj}")

    # Configure BMOPSO_CDRCE algorithm
    algorithm = BMOPSO_CDRCE(
        n_particles=30,
        w_max=0.9,
        w_min=0.4,
        c1=1.49,
        c2=1.49,
        v_max=4.0,
        mutation_rate=0.5,
        catfish_threshold=10,
        catfish_rate=0.10,
    )

    n_gen = 20
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
    print(" SUMMARY OF PARETO FRONT SOLUTIONS")
    print("=" * 75)
    print(f" {'#':<3} | {'Error Rate':<14} | {'Accuracy (%)':<14} | {'Features Selected':<18}")
    print("-" * 75)

    for rank, (x, f) in enumerate(zip(res.X, res.F), 1):
        err = f[0]
        acc = (1.0 - err) * 100
        n_sel = np.sum(x)
        print(f" {rank:<3} | {err:<14.4f} | {acc:<13.2f}% | {n_sel}/{problem.n_features} ({f[1]*100:.1f}%)")

    if len(res.X) > 0:
        print_solution_details(0, res.X[0], res.F[0], problem)


if __name__ == "__main__":
    main()
