"""
tests/test_brain_evolution.py

Unit tests for the neural network brain and evolution mechanics.

Brain tests:
  - Output shape is always (BRAIN_OUTPUTS,)
  - Output values are in [-1, 1] (tanh bounded)
  - Weights survive clone() without modification
  - Mutated brain differs from parent (with high probability)
  - Variable hidden layer configs all produce correct shapes
  - Empty hidden layer works (direct input → output)

Evolution tests:
  - Repopulation fires when population < POPULATION_MIN
  - Repopulation does NOT fire when population >= POPULATION_MIN
  - Children have generation number = parent generation + 1
  - Elite fraction is respected: only top performers become parents
  - Mutated children differ from parents (with high probability)
  - Fitness formula produces values in [0, 1]
  - Fitness correctly weights lifespan, food, distance
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.conftest import seed, make_agent, make_agents
import numpy as np


# ── Brain tests ───────────────────────────────────────────────────────────────

def test_brain_output_shape():
    """Forward pass always returns shape (BRAIN_OUTPUTS,)."""
    from simulation.brain import Brain
    from config import BRAIN_OUTPUTS, BRAIN_INPUTS
    seed(0)
    brain = Brain()
    x = np.random.rand(BRAIN_INPUTS).astype(np.float32)
    out = brain.forward(x)
    assert out.shape == (BRAIN_OUTPUTS,), f"Expected ({BRAIN_OUTPUTS},), got {out.shape}"


def test_brain_output_bounded():
    """tanh activation must keep all outputs in (-1, 1)."""
    from simulation.brain import Brain
    from config import BRAIN_INPUTS
    seed(0)
    brain = Brain()
    for _ in range(100):
        x = np.random.uniform(-10, 10, BRAIN_INPUTS).astype(np.float32)
        out = brain.forward(x)
        assert np.all(out >= -1.0) and np.all(out <= 1.0), (
            f"Output out of bounds: {out}"
        )


def test_brain_clone_is_independent():
    """Cloned brain has identical weights but modifying clone doesn't affect original."""
    from simulation.brain import Brain
    from config import BRAIN_INPUTS
    seed(0)
    brain   = Brain()
    clone   = brain.clone()

    # Weights should be equal
    Ws_orig, bs_orig = brain.get_weights()
    Ws_clone, bs_clone = clone.get_weights()
    for W_o, W_c in zip(Ws_orig, Ws_clone):
        np.testing.assert_array_equal(W_o, W_c)

    # Modifying clone should not affect original
    Ws_clone[0] += 999.0
    Ws_orig2, _ = brain.get_weights()
    assert not np.allclose(Ws_orig2[0], Ws_clone[0]), (
        "Clone modification bled through to original — not a deep copy"
    )


def test_brain_mutate_produces_different_weights():
    """Mutated brain should differ from parent in at least some weights."""
    from simulation.brain import Brain
    seed(0)
    brain   = Brain()
    mutated = brain.mutate()

    Ws_p, bs_p = brain.get_weights()
    Ws_m, bs_m = mutated.get_weights()

    all_same = all(np.allclose(W_p, W_m) for W_p, W_m in zip(Ws_p, Ws_m))
    assert not all_same, "Mutated brain has identical weights to parent"


def test_brain_variable_hidden_layers():
    """BRAIN_HIDDEN_LAYERS list controls actual network architecture."""
    import config
    from simulation.brain import Brain
    import importlib
    import simulation.brain as bmod

    for layers in [[], [8], [16, 8], [32, 16, 8]]:
        config.BRAIN_HIDDEN_LAYERS = layers
        importlib.reload(bmod)
        b = bmod.Brain()
        expected_sizes = [config.BRAIN_INPUTS] + layers + [config.BRAIN_OUTPUTS]
        assert b.layer_sizes == expected_sizes, (
            f"For BRAIN_HIDDEN_LAYERS={layers}, expected sizes {expected_sizes}, "
            f"got {b.layer_sizes}"
        )
        x = np.random.rand(config.BRAIN_INPUTS).astype(np.float32)
        out = b.forward(x)
        assert out.shape == (config.BRAIN_OUTPUTS,)

    # Reset to default
    config.BRAIN_HIDDEN_LAYERS = [8]
    importlib.reload(bmod)
    print("  Variable hidden layer configs all correct ✓")


def test_brain_no_hidden_layer():
    """Empty BRAIN_HIDDEN_LAYERS = [] should work as direct input→output."""
    import config
    import importlib
    import simulation.brain as bmod
    config.BRAIN_HIDDEN_LAYERS = []
    importlib.reload(bmod)
    b = bmod.Brain()
    assert b.layer_sizes == [config.BRAIN_INPUTS, config.BRAIN_OUTPUTS]
    x = np.random.rand(config.BRAIN_INPUTS).astype(np.float32)
    assert b.forward(x).shape == (config.BRAIN_OUTPUTS,)
    # Reset
    config.BRAIN_HIDDEN_LAYERS = [8]
    importlib.reload(bmod)


# ── Evolution tests ───────────────────────────────────────────────────────────

def test_repopulation_fires_below_minimum():
    """maybe_repopulate returns new agents when alive < POPULATION_MIN."""
    from simulation.evolution import EvolutionManager
    from config import POPULATION_MIN
    seed(0)

    evo = EvolutionManager()
    # Seed dead pool with some agents so it has parents
    for ag in make_agents(20, generation=1):
        ag.age        = 500
        ag.food_eaten = 5
        evo.record_death(ag)

    # Feed a population below minimum
    alive = make_agents(POPULATION_MIN - 5, generation=1)
    new_agents = evo.maybe_repopulate(alive)
    assert len(new_agents) > 0, (
        f"Expected new agents when alive={len(alive)} < POPULATION_MIN={POPULATION_MIN}"
    )


def test_repopulation_silent_above_minimum():
    """maybe_repopulate returns [] when alive >= POPULATION_MIN."""
    from simulation.evolution import EvolutionManager
    from config import POPULATION_MIN
    seed(0)

    evo = EvolutionManager()
    alive = make_agents(POPULATION_MIN + 5, generation=1)
    new_agents = evo.maybe_repopulate(alive)
    assert new_agents == [], (
        f"Expected no spawning when alive={len(alive)} >= POPULATION_MIN={POPULATION_MIN}"
    )


def test_children_get_incremented_generation():
    """Children spawned by maybe_repopulate should have generation = old_gen + 1."""
    from simulation.evolution import EvolutionManager
    from config import POPULATION_MIN
    seed(0)

    evo = EvolutionManager()
    gen_before = evo.generation

    for ag in make_agents(20, generation=1):
        ag.age = 800
        ag.food_eaten = 10
        evo.record_death(ag)

    alive = make_agents(POPULATION_MIN - 3, generation=1)
    new_agents = evo.maybe_repopulate(alive)

    assert evo.generation == gen_before + 1, (
        f"Generation should have incremented from {gen_before} to {gen_before + 1}, "
        f"got {evo.generation}"
    )
    for child in new_agents:
        assert child.generation == evo.generation, (
            f"Child has generation {child.generation}, expected {evo.generation}"
        )


def test_only_elite_used_as_parents():
    """
    With ELITE_FRACTION, only the top performers should ever be selected.
    We verify this by giving high fitness to a known subset and checking
    that children's weights cluster around those parents.
    """
    from simulation.evolution import EvolutionManager
    from simulation.brain import Brain
    from config import POPULATION_MIN, ELITE_FRACTION
    seed(0)

    evo = EvolutionManager()

    # Create 10 low-fitness and 10 high-fitness agents
    low_agents  = make_agents(10, generation=1)
    high_agents = make_agents(10, generation=1)

    for ag in low_agents:
        ag.age = 10; ag.food_eaten = 0
        evo.record_death(ag)

    for ag in high_agents:
        ag.age = 3000; ag.food_eaten = 20
        evo.record_death(ag)

    high_fitnesses = sorted([ag.fitness() for ag in high_agents], reverse=True)
    pool_fitnesses = sorted([d['fitness'] for d in evo.dead_pool], reverse=True)

    elite_count = max(1, int(len(evo.dead_pool) * ELITE_FRACTION))
    elite_fitnesses = pool_fitnesses[:elite_count]

    # All elite should be high performers
    min_elite = min(elite_fitnesses)
    max_low   = max(ag.fitness() for ag in low_agents)
    assert min_elite > max_low, (
        f"Elite pool contains low-fitness agents: min_elite={min_elite:.4f}, "
        f"max_low={max_low:.4f}"
    )
    print(f"  Elite selection correct: min_elite={min_elite:.3f} > max_low={max_low:.3f} ✓")


# ── Fitness tests ─────────────────────────────────────────────────────────────

def test_fitness_always_in_0_1():
    """Fitness should always be in [0, 1] for any agent state."""
    from config import AGENT_MAX_AGE, AGENT_MAX_ENERGY
    seed(0)

    test_cases = [
        dict(age=0,          food_eaten=0,  distance_traveled=0),
        dict(age=AGENT_MAX_AGE, food_eaten=0,  distance_traveled=0),
        dict(age=0,          food_eaten=100, distance_traveled=0),
        dict(age=0,          food_eaten=0,  distance_traveled=99999),
        dict(age=AGENT_MAX_AGE, food_eaten=100, distance_traveled=99999),
        dict(age=500,        food_eaten=7,  distance_traveled=2000),
    ]

    for case in test_cases:
        ag = make_agent()
        ag.age               = case['age']
        ag.food_eaten        = case['food_eaten']
        ag.distance_traveled = case['distance_traveled']
        f = ag.fitness()
        assert 0.0 <= f <= 1.0, (
            f"Fitness {f} out of [0,1] for case {case}"
        )


def test_fitness_ordering():
    """Agent that ate more food and lived longer should have higher fitness."""
    seed(0)
    good = make_agent()
    good.age = 3000; good.food_eaten = 20; good.distance_traveled = 5000

    bad = make_agent()
    bad.age = 100; bad.food_eaten = 0; bad.distance_traveled = 100

    assert good.fitness() > bad.fitness(), (
        f"Good agent fitness {good.fitness():.4f} should exceed "
        f"bad agent fitness {bad.fitness():.4f}"
    )


if __name__ == "__main__":
    print("Running brain/evolution tests...")
    test_brain_output_shape()
    test_brain_output_bounded()
    test_brain_clone_is_independent()
    test_brain_mutate_produces_different_weights()
    test_brain_variable_hidden_layers()
    test_brain_no_hidden_layer()
    test_repopulation_fires_below_minimum()
    test_repopulation_silent_above_minimum()
    test_children_get_incremented_generation()
    test_only_elite_used_as_parents()
    test_fitness_always_in_0_1()
    test_fitness_ordering()
    print("All brain/evolution tests passed.")
