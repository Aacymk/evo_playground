"""
tests/test_determinism.py

Seeded reproducibility tests. Two runs with the same numpy seed must produce
identical results. Uses fast_config so generations complete quickly.

Both numpy.random AND Python's built-in random module are seeded everywhere
to guarantee full determinism regardless of which RNG any module uses internally.
"""

import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.conftest import make_world, fast_config, restore_config
import numpy as np


def _seed_all(n):
    """Seed both numpy.random and Python's built-in random module."""
    np.random.seed(n)
    random.seed(n)


def _snapshot(world):
    agents = sorted(world.agents, key=lambda a: a.id)
    fits   = [round(a.fitness(), 4) for a in agents]
    return {
        "generation":   world.evo_mgr.generation,
        "total_born":   world.evo_mgr.total_born,
        "total_deaths": world.evo_mgr.total_deaths,
        "alive_count":  len(agents),
        "positions":    [(round(a.x, 3), round(a.y, 3)) for a in agents],
        "energies":     [round(a.energy, 3) for a in agents],
        "ages":         [a.age for a in agents],
        "fitnesses":    fits,
        "avg_fitness":  round(float(np.mean(fits)), 4) if fits else 0.0,
    }


def _run(frames, rng_seed):
    orig = fast_config()
    _seed_all(rng_seed)
    world = make_world()
    for _ in range(frames):
        world.update()
    snap = _snapshot(world)
    restore_config(orig)
    return snap


def test_identical_seeds_identical_output():
    """Same seed → identical world state after N frames."""
    snap_a = _run(400, rng_seed=7)
    snap_b = _run(400, rng_seed=7)
    for key in snap_a:
        assert snap_a[key] == snap_b[key], (
            f"Divergence at '{key}':\n  A: {snap_a[key]}\n  B: {snap_b[key]}"
        )
    print(f"  Identical after 400 frames: gen={snap_a['generation']}, born={snap_a['total_born']} ✓")


def test_different_seeds_different_output():
    """
    Different seeds must produce meaningfully different simulations —
    not just floating-point noise. We check multiple population-level
    metrics so the test can't pass trivially.
    """
    snap_0 = _run(300, rng_seed=0)
    snap_1 = _run(300, rng_seed=999)

    # Count how many top-level metrics diverge
    diverged = []
    for key in ["generation", "total_born", "total_deaths", "positions",
                "energies", "avg_fitness"]:
        if snap_0[key] != snap_1[key]:
            diverged.append(key)

    assert len(diverged) >= 2, (
        f"Expected at least 2 metrics to diverge between seeds, only got: {diverged}.\n"
        f"Seed 0: gen={snap_0['generation']} born={snap_0['total_born']} "
        f"avg_fit={snap_0['avg_fitness']}\n"
        f"Seed 999: gen={snap_1['generation']} born={snap_1['total_born']} "
        f"avg_fit={snap_1['avg_fitness']}"
    )
    print(f"  Different seeds diverge on {len(diverged)} metrics: {diverged} ✓")


def test_simulation_uses_only_numpy_random():
    """
    Guard test: verify that no simulation module imports Python's built-in
    'random' module. All randomness should come from numpy.random so that
    seeding np.random is sufficient to control all stochasticity.
    """
    import ast
    import os

    sim_dirs = ["simulation", "evolog", "utils"]
    violations = []

    for dirname in sim_dirs:
        dirpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               dirname)
        if not os.path.isdir(dirpath):
            continue
        for fname in os.listdir(dirpath):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, fname)
            tree  = ast.parse(open(fpath).read())
            for node in ast.walk(tree):
                # Flag: import random  OR  from random import ...
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "random":
                            violations.append(f"{dirname}/{fname}: import random")
                elif isinstance(node, ast.ImportFrom):
                    if node.module == "random":
                        violations.append(f"{dirname}/{fname}: from random import ...")

    assert not violations, (
        "Simulation modules use Python's built-in random module — "
        "seed np.random AND random.seed() to guarantee determinism, "
        "or switch to np.random exclusively.\n"
        "Violations:\n  " + "\n  ".join(violations)
    )
    print("  No simulation modules import built-in random ✓")


def test_mutation_deterministic_with_seed():
    """Two brains mutated with same seed produce identical weights."""
    from simulation.brain import Brain
    _seed_all(1)
    brain = Brain()

    _seed_all(99)
    ma = brain.mutate()
    _seed_all(99)
    mb = brain.mutate()

    Ws_a, bs_a = ma.get_weights()
    Ws_b, bs_b = mb.get_weights()
    for W_a, W_b in zip(Ws_a, Ws_b):
        np.testing.assert_array_equal(W_a, W_b, err_msg="Mutated weights differ despite same seed")
    print("  Mutation deterministic with seed ✓")


def test_multiple_generations_reproducible():
    """
    Run until 3 generation transitions occur. Verify total_born matches
    on re-run with same seed.
    """
    def collect_gen_milestones(rng_seed):
        orig = fast_config()
        _seed_all(rng_seed)
        world = make_world()
        milestones = []
        last_gen = world.evo_mgr.generation
        for _ in range(600):
            world.update()
            if world.evo_mgr.generation != last_gen:
                milestones.append((world.evo_mgr.generation, world.evo_mgr.total_born))
                last_gen = world.evo_mgr.generation
                if len(milestones) >= 3:
                    break
        restore_config(orig)
        return milestones

    m_a = collect_gen_milestones(42)
    m_b = collect_gen_milestones(42)
    assert len(m_a) >= 2, f"Only {len(m_a)} generation transitions — try more frames"
    assert m_a == m_b, f"Generation milestones differ:\n  A: {m_a}\n  B: {m_b}"
    print(f"  {len(m_a)} generation transitions match ✓")


def test_sensor_output_deterministic():
    """Same agent/food layout with same seed → same sensor values."""
    from simulation.sensors import compute_sensors
    from simulation.food import Food
    from tests.conftest import make_agent

    _seed_all(5)
    agent = make_agent(x=400, y=300)
    agent.angle = 0.0
    food_list = [Food(x=500, y=300)]

    _seed_all(10)
    s1 = compute_sensors(agent, food_list, [agent])
    _seed_all(10)
    s2 = compute_sensors(agent, food_list, [agent])

    np.testing.assert_array_almost_equal(s1, s2, decimal=6)
    print("  Sensor output deterministic with seed ✓")


if __name__ == "__main__":
    print("Running determinism tests...")
    test_identical_seeds_identical_output()
    test_different_seeds_different_output()
    test_simulation_uses_only_numpy_random()
    test_mutation_deterministic_with_seed()
    test_multiple_generations_reproducible()
    test_sensor_output_deterministic()
    print("All determinism tests passed.")
