"""Example script demonstrating BMOPSO-CDRCE on the Multiobjective Set Covering Problem (MOSCP).

This script provides a practical walkthrough to:
1. Generate and configure a Multiobjective Set Covering Problem (MOSCP) instance.
2. Configure BMOPSO_CDRCE algorithm hyperparameters (including Catfish Effect).
3. Execute multiobjective optimization using pymoo.optimize.minimize.
4. Analyze the resulting Pareto Front, verifying 100% coverage feasibility across all elements.

To execute:
    python examples/run_moscp_example.py
"""

from typing import Any
import numpy as np
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from pymoo_binary_problems import MOSCP
from bmopso_cdrce import BMOPSO_CDRCE


def create_sample_moscp_instance() -> MOSCP:
    """Create a sample instance of the Multiobjective Set Covering Problem (MOSCP).

    Scenario: Emergency Facility Location & Sensor Network Coverage
    - 25 municipal zones / regions requiring emergency coverage (m = 25).
    - 35 candidate facility locations (subsets / variables, n = 35).
    - 2 conflicting cost objectives:
      * Objective 1: Capital Installation & Infrastructure Cost ($k)
      * Objective 2: Operational Maintenance & Response Risk ($k/yr)
    - Average coverage density: 25% (each facility covers ~6-7 zones).

    Returns
    -------
    MOSCP
        Configured MOSCP benchmark problem instance.
    """
    return MOSCP.from_random(
        n_elements=25,
        n_subsets=35,
        n_obj=2,
        density=0.25,
        cost_range=(15.0, 95.0),
        seed=42,
    )


def print_solution_details(
    sol_idx: int,
    x_flat: np.ndarray,
    f_val: np.ndarray,
    problem: MOSCP,
) -> None:
    """Print detailed facility selection and coverage metrics for a candidate solution."""
    selected_subsets = np.where(x_flat)[0].tolist()
    covered_mask = np.any(problem.incidence_matrix[:, x_flat.astype(bool)], axis=1)
    n_covered = int(np.sum(covered_mask))
    is_fully_covered = n_covered == problem.n_elements

    print(f"\n--- Solution #{sol_idx + 1} ---")
    print(f"  Installation Cost ($k) : $ {f_val[0]:6.2f}")
    print(f"  Maintenance Cost ($k)  : $ {f_val[1]:6.2f}")
    print(f"  Facilities Selected    : {len(selected_subsets)} / {problem.n_subsets}")
    print(f"  Facility Indices       : {selected_subsets}")
    print(f"  Coverage Status        : {n_covered}/{problem.n_elements} zones ({'100% Valid' if is_fully_covered else 'UNCOVERED ZONES!'})")


def main() -> None:
    """Execute complete MOSCP multiobjective optimization workflow using BMOPSO-CDRCE."""
    print("=" * 75)
    print(" MULTIOBJECTIVE SET COVERING PROBLEM (MOSCP) WITH BMOPSO-CDRCE")
    print("=" * 75)

    problem = create_sample_moscp_instance()
    print("\nProblem initialized successfully:")
    print(f"  - Universe Elements to Cover (m): {problem.n_elements}")
    print(f"  - Candidate Subsets/Facilities (n_var): {problem.n_subsets}")
    print(f"  - Objective Count (F)           : {problem.n_obj}")

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

    # Check feasibility
    feasible_idx = []
    for i, x in enumerate(res.X):
        covered = np.any(problem.incidence_matrix[:, x.astype(bool)], axis=1)
        if np.all(covered):
            feasible_idx.append(i)

    print(f"Total Non-Dominated Solutions Found: {len(res.X)}")
    print(f"  |-- FEASIBLE Solutions (100% Coverage): {len(feasible_idx)}")
    print(f"  \\-- INFEASIBLE Solutions               : {len(res.X) - len(feasible_idx)}")

    print("\n" + "=" * 75)
    print(" SUMMARY OF TOP FEASIBLE SOLUTIONS")
    print("=" * 75)
    print(f" {'#':<3} | {'Installation ($k)':<18} | {'Maintenance ($k)':<18} | {'Facilities'}")
    print("-" * 75)

    for rank, idx in enumerate(feasible_idx[:8], 1):
        f = res.F[idx]
        n_fac = int(np.sum(res.X[idx]))
        print(f" {rank:<3} | $ {f[0]:<16.2f} | $ {f[1]:<16.2f} | {n_fac} facilities")

    if feasible_idx:
        print_solution_details(feasible_idx[0], res.X[feasible_idx[0]], res.F[feasible_idx[0]], problem)
    elif len(res.X) > 0:
        print_solution_details(0, res.X[0], res.F[0], problem)


if __name__ == "__main__":
    main()
