"""Unit tests for modular components: dominance, diversity, operators, archive, and catfish."""

import numpy as np
import pytest
from bmopso_cdrce.operators import (
    apply_catfish_effect,
    apply_mutation,
    generate_extreme_binary_positions,
    sample_binary_positions,
    select_random_particles,
    sigmoid,
    update_personal_bests,
    update_velocity,
)
from bmopso_cdrce.util import (
    NonDominatedArchive,
    calc_crowding_distance,
    calc_crowding_roulette_probabilities,
    dominates,
    find_non_dominated_constrained,
)


def test_dominance_rules_isolated() -> None:
    """Test Deb's constrained dominance rules in isolation."""
    # Feasible vs Infeasible
    assert dominates(np.array([10.0, 10.0]), np.array([1.0, 1.0]), cv1=0.0, cv2=0.5) is True
    assert dominates(np.array([1.0, 1.0]), np.array([10.0, 10.0]), cv1=0.5, cv2=0.0) is False

    # Infeasible vs Infeasible (lower violation wins)
    assert dominates(np.array([10.0, 10.0]), np.array([1.0, 1.0]), cv1=0.2, cv2=0.8) is True
    assert dominates(np.array([1.0, 1.0]), np.array([10.0, 10.0]), cv1=0.9, cv2=0.1) is False

    # Feasible vs Feasible (standard Pareto dominance)
    assert dominates(np.array([1.0, 2.0]), np.array([2.0, 3.0]), cv1=0.0, cv2=0.0) is True
    assert dominates(np.array([2.0, 3.0]), np.array([1.0, 2.0]), cv1=0.0, cv2=0.0) is False
    assert dominates(np.array([1.0, 3.0]), np.array([2.0, 2.0]), cv1=0.0, cv2=0.0) is False


def test_diversity_metrics_isolated() -> None:
    """Test crowding distance and roulette wheel probability calculations."""
    # Empty or small points
    assert len(calc_crowding_roulette_probabilities(np.array([]))) == 0
    single_p = calc_crowding_roulette_probabilities(np.array([5.0]))
    assert np.allclose(single_p, [1.0])

    f = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])
    cd = calc_crowding_distance(f)
    assert np.isinf(cd[0]) and np.isinf(cd[-1])
    assert cd[1] > 0 and cd[2] > 0

    probs = calc_crowding_roulette_probabilities(cd)
    assert np.isclose(np.sum(probs), 1.0)
    assert probs[0] > probs[1]


def test_operators_isolated() -> None:
    """Test sigmoid, binary sampling, velocity update, and mutation operators."""
    # Sigmoid
    v = np.array([0.0, 100.0, -100.0])
    s = sigmoid(v)
    assert np.isclose(s[0], 0.5)
    assert np.isclose(s[1], 1.0)
    assert np.isclose(s[2], 0.0)

    # Binary sampling
    positions = sample_binary_positions(np.array([[10.0, -10.0], [-10.0, 10.0]]))
    assert positions.dtype == bool
    assert positions.shape == (2, 2)

    # Velocity update with clamping
    x = np.array([[True, False]])
    pbest_x = np.array([[True, True]])
    gbest = np.array([[False, False]])
    v_init = np.array([[10.0, -10.0]])
    v_updated = update_velocity(v_init, x, pbest_x, gbest, w=0.5, v_max=4.0)
    assert np.all(v_updated <= 4.0) and np.all(v_updated >= -4.0)

    # Mutation operator
    x_mut = apply_mutation(x, progress=0.0, mutation_rate=0.5)
    assert x_mut.shape == x.shape


def test_catfish_isolated() -> None:
    """Test Catfish Effect operator components in isolation."""
    # Extreme binary positions
    ext_pos = generate_extreme_binary_positions(n_positions=6, n_var=10)
    assert ext_pos.shape == (6, 10)
    assert ext_pos.dtype == bool
    for row in ext_pos:
        assert np.all(row == True) or np.all(row == False)

    # Random particle selection
    selected = select_random_particles(n_particles=30, n_to_replace=5)
    assert len(selected) == 5
    assert len(set(selected)) == 5
    assert np.all(selected < 30)

    # Apply catfish effect
    x = np.zeros((20, 10), dtype=bool)
    x_perturbed = apply_catfish_effect(x, catfish_rate=0.10)
    assert x_perturbed.shape == (20, 10)
    assert x_perturbed.dtype == bool


def test_archive_class_lifecycle_and_stagnation_detection() -> None:
    """Test NonDominatedArchive lifecycle, change tracking, and leader selection."""
    archive = NonDominatedArchive(max_size=3)
    assert archive.is_empty()
    assert len(archive) == 0

    # 1. First update: must return True (archive changed)
    x1 = np.array([[True, False], [False, True], [True, True]])
    f1 = np.array([[1.0, 5.0], [3.0, 3.0], [5.0, 1.0]])
    changed = archive.update(x1, f1)

    assert changed is True
    assert len(archive) == 3
    assert not archive.is_empty()
    assert archive.x is not None
    assert archive.f is not None

    # 2. Inserting dominated / identical solutions: must return False (no change)
    x_dominated = np.array([[False, False]])
    f_dominated = np.array([[10.0, 10.0]])
    changed_dominated = archive.update(x_dominated, f_dominated)
    assert changed_dominated is False
    assert len(archive) == 3

    # 3. Leader selection
    leaders = archive.select_leaders(n_particles=5)
    assert leaders.shape == (5, 2)

    # 4. Capacity pruning with improved solution: must return True
    x_better = np.array([[True, True]])
    f_better = np.array([[0.5, 0.5]])
    changed_better = archive.update(x_better, f_better)
    assert changed_better is True
    assert len(archive) <= 3

    # 5. Population conversion
    pop = archive.to_population()
    assert len(pop) == len(archive)
