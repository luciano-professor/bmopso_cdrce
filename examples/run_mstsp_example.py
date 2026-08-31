"""Example script demonstrating BMOPSO-CDRCE on the Multiobjective Traveling Salesman Problem (MSTSP).

This script provides a practical walkthrough to:
1. Generate and configure a Multiobjective Traveling Salesman Problem (MSTSP) benchmark instance.
2. Configure BMOPSO_CDRCE algorithm hyperparameters (including Catfish Effect).
3. Execute multiobjective optimization using pymoo.optimize.minimize.
4. Analyze the resulting Pareto Front.

To execute:
    python examples/run_mstsp_example.py
"""

from typing import Any
import numpy as np
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from pymoo_binary_problems import MSTSP
from bmopso_cdrce import BMOPSO_CDRCE


def create_sample_mstsp_instance() -> MSTSP:
    """Create a sample instance of the Multiobjective Traveling Salesman Problem (MSTSP).

    Scenario:
    - 6 delivery locations / cities.
    - 2 conflicting criteria:
      * Objective 1: Travel Distance (km)
      * Objective 2: Toll & Transit Cost ($)
    - Binary decision variables: 6 * 6 = 36 bits.

    Returns
    -------
    MSTSP
        Configured MSTSP benchmark problem instance.
    """
    coords_distance = np.array([
        [10.0, 20.0],
        [25.0, 80.0],
        [45.0, 60.0],
        [70.0, 90.0],
        [85.0, 30.0],
        [60.0, 10.0],
    ], dtype=float)

    coords_cost = np.array([
        [80.0, 15.0],
        [10.0, 90.0],
        [30.0, 20.0],
        [95.0, 60.0],
        [40.0, 85.0],
        [15.0, 30.0],
    ], dtype=float)

    return MSTSP.from_coordinates(
        coordinates_list=[coords_distance, coords_cost],
    )


def main() -> None:
    """Execute complete MSTSP multiobjective optimization workflow using BMOPSO-CDRCE."""
    print("=" * 75)
    print(" MULTIOBJECTIVE TRAVELING SALESMAN PROBLEM (MSTSP) WITH BMOPSO-CDRCE")
    print("=" * 75)

    problem = create_sample_mstsp_instance()
    print("\nProblem initialized successfully:")
    print(f"  - City Count                : {problem.n_cities}")
    print(f"  - Decision Variables (X)    : {problem.n_var} bits")
    print(f"  - Objective Count (F)       : {problem.n_obj}")

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

    n_evals = 8000
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
    print(f"Total Non-Dominated Solutions Found: {len(res.X)}")

    print("\n" + "=" * 75)
    print(" SUMMARY OF PARETO FRONT TOURS")
    print("=" * 75)
    print(f" {'#':<3} | {'Distance Metric (km)':<22} | {'Cost Metric ($)':<22}")
    print("-" * 75)

    for rank, f in enumerate(res.F[:8], 1):
        print(f" {rank:<3} | {f[0]:<22.2f} | {f[1]:<22.2f}")


if __name__ == "__main__":
    main()
