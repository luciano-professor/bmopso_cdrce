"""Example script demonstrating BMOPSO-CDRCE on the Multiple Knapsack Problem (MKP).

This script provides a practical walkthrough to:
1. Construct and configure a Multiple Knapsack Problem (MKP) with capacity constraints.
2. Configure BMOPSO_CDRCE algorithm hyperparameters (including Catfish Effect).
3. Execute multiobjective optimization using pymoo.optimize.minimize.
4. Analyze the resulting non-dominated Pareto Front and Catfish triggers.

To execute:
    python examples/run_mkp_example.py
"""

from typing import Any
import numpy as np
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from pymoo_binary_problems import MKP
from bmopso_cdrce import BMOPSO_CDRCE


def create_sample_mkp_instance() -> MKP:
    """Create a sample instance of the Multiple Knapsack Problem (MKP).

    Scenario:
    - 12 items available for transportation.
    - 3 knapsacks / cargo compartments with individual capacities.
    - Total binary decision variables: 12 items * 3 knapsacks = 36 bits.

    Returns
    -------
    MKP
        Configured MKP problem instance for multiobjective optimization.
    """
    profits = np.array(
        [18.0, 26.0, 32.0, 15.0, 45.0, 22.0, 38.0, 50.0, 12.0, 29.0, 35.0, 42.0],
        dtype=float,
    )
    weights = np.array(
        [6.0, 9.0, 11.0, 5.0, 16.0, 8.0, 13.0, 18.0, 4.0, 10.0, 12.0, 15.0],
        dtype=float,
    )
    capacities = np.array([35.0, 40.0, 30.0], dtype=float)

    return MKP(
        profits=profits,
        weights=weights,
        capacities=capacities,
        n_obj=2,
        maximize_profit=True,
        minimize_weight=True,
    )


def print_solution_details(
    sol_idx: int,
    x_flat: np.ndarray,
    f_val: np.ndarray,
    problem: MKP,
) -> None:
    """Print item allocation details and feasibility status for a candidate solution."""
    x_matrix = x_flat.reshape((problem.n_items, problem.n_knapsacks))
    real_profit = -f_val[0] if problem.maximize_profit else f_val[0]
    total_weight = f_val[1] if problem.minimize_weight else -f_val[1]

    print(f"\n--- Solution #{sol_idx + 1} ---")
    print(f"  Total Profit : $ {real_profit:6.2f}")
    print(f"  Total Weight :   {total_weight:6.2f} kg")

    is_feasible = True
    for k in range(problem.n_knapsacks):
        items_in_k = np.where(x_matrix[:, k])[0]
        weight_k = float(np.sum(problem.weights[items_in_k]))
        cap_k = float(problem.capacities[k])
        status = "OK" if weight_k <= cap_k else "VIOLATED!"
        if weight_k > cap_k:
            is_feasible = False

        items_str = ", ".join(f"Item {i} ({problem.weights[i]}kg)" for i in items_in_k) or "Empty"
        print(f"    Knapsack {k}: {weight_k:4.1f} / {cap_k:4.1f} kg [{status}] -> Items: [{items_str}]")

    allocated_counts = np.sum(x_matrix, axis=1)
    multiple_allocations = np.where(allocated_counts > 1)[0]
    if len(multiple_allocations) > 0:
        is_feasible = False
        print(f"    Single-knapsack constraint violated for items: {multiple_allocations.tolist()}")

    status_str = "VALID (Feasible)" if is_feasible else "INVALID (Constraints Violated)"
    print(f"  Overall Status: {status_str}")


def main() -> None:
    """Execute complete MKP multiobjective optimization workflow using BMOPSO-CDRCE."""
    print("=" * 75)
    print(" MULTIOBJECTIVE MULTIPLE KNAPSACK PROBLEM (MKP) WITH BMOPSO-CDRCE")
    print("=" * 75)

    problem = create_sample_mkp_instance()
    print("\nProblem initialized successfully:")
    print(f"  - Item Count                : {problem.n_items}")
    print(f"  - Knapsack Count            : {problem.n_knapsacks}")
    print(f"  - Decision Variables (X)    : {problem.n_var} bits")
    print(f"  - Objective Count (F)       : {problem.n_obj}")
    print(f"  - Knapsack Capacities       : {problem.capacities.tolist()}")

    # Configure BMOPSO_CDRCE algorithm
    algorithm = BMOPSO_CDRCE(
        n_particles=60,
        w_max=0.9,
        w_min=0.4,
        c1=1.49,
        c2=1.49,
        v_max=4.0,
        mutation_rate=0.5,
        catfish_threshold=15,
        catfish_rate=0.10,
    )

    n_evals = 10000
    print(f"\nStarting optimization with termination criterion of {n_evals} evaluations...")

    res = minimize(
        problem,
        algorithm,
        termination=("n_eval", n_evals),
        seed=42,
        verbose=False,
    )

    print("\nOptimization finished!")
    print(f"Catfish Effect Triggers: {algorithm.n_catfish_triggers}")

    # Feasibility analysis
    viable_indices = []
    infeasible_indices = []
    for i in range(len(res.X)):
        x_m = res.X[i].reshape((problem.n_items, problem.n_knapsacks))
        knapsack_weights = [np.dot(x_m[:, k], problem.weights) for k in range(problem.n_knapsacks)]
        weight_ok = all(w_k <= cap for w_k, cap in zip(knapsack_weights, problem.capacities))
        uniqueness_ok = np.all(np.sum(x_m, axis=1) <= 1)
        if weight_ok and uniqueness_ok:
            viable_indices.append(i)
        else:
            infeasible_indices.append(i)

    n_total = len(res.X)
    n_viable = len(viable_indices)
    n_infeasible = len(infeasible_indices)

    print(f"Total Solutions on Pareto Front : {n_total}")
    print(f"  |-- FEASIBLE Solutions (100% Valid)  : {n_viable} ({n_viable / n_total * 100:.1f}%)")
    print(f"  \\-- INFEASIBLE Solutions (Violations): {n_infeasible} ({n_infeasible / n_total * 100:.1f}%)")

    viable_sorted = sorted(viable_indices, key=lambda idx: -res.F[idx, 0], reverse=True)

    print("\n" + "=" * 75)
    print(" SUMMARY OF TOP FEASIBLE SOLUTIONS FOUND")
    print("=" * 75)
    print(f" {'#':<3} | {'Total Profit ($)':<18} | {'Total Weight (kg)':<18} | {'Status'}")
    print("-" * 75)

    for rank, idx in enumerate(viable_sorted[:8], 1):
        profit = -res.F[idx, 0]
        weight = res.F[idx, 1]
        print(f" {rank:<3} | $ {profit:<16.2f} | {weight:<15.2f} kg | Feasible")

    if not viable_sorted:
        print("  No 100% feasible solution found in this run.")

    print("\n" + "=" * 75)
    print(" DETAILED ALLOCATION OF BEST FEASIBLE SOLUTION")
    print("=" * 75)
    if viable_sorted:
        best_viable_idx = viable_sorted[0]
        print_solution_details(best_viable_idx, res.X[best_viable_idx], res.F[best_viable_idx], problem)
    elif len(res.X) > 0:
        print_solution_details(0, res.X[0], res.F[0], problem)


if __name__ == "__main__":
    main()
