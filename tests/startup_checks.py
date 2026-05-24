"""
tests/startup_checks.py

Lightweight sanity checks that run in ~1 second before the sim window opens
when RUN_TESTS_ON_STARTUP = True in config.py.

These are NOT a replacement for the full pytest suite — they're a quick
smoke-test that the core invariants hold with the CURRENT config values.
Things like mutation_rate=0 or spike_count=0 won't blow up the sim, but
they would cause the full tests to behave unexpectedly. These checks catch
that kind of misconfiguration fast.

Run the full suite separately:
    python -m pytest tests/ -v
"""

import sys
import numpy as np


def run_startup_checks():
    """
    Runs all checks. Prints a summary. Raises SystemExit if anything fails
    so the sim doesn't start in a broken state.
    """
    checks = [
        _check_brain_forward,
        _check_sensor_length,
        _check_agent_energy_decay,
        _check_spike_cooldown,
        _check_food_energy_cap,
        _check_fitness_range,
        _check_conservation_10_frames,
        _check_config_consistency,
    ]

    passed = 0
    failed = []

    print("[startup checks] Running pre-launch sanity checks...")
    for check in checks:
        try:
            check()
            passed += 1
        except Exception as e:
            failed.append((check.__name__, str(e)))

    if failed:
        print(f"[startup checks] {passed}/{len(checks)} passed — FAILURES:")
        for name, msg in failed:
            print(f"  FAIL  {name}: {msg}")
        print("[startup checks] Fix the above before running the sim.")
        print("[startup checks] Full test suite: python -m pytest tests/ -v")
        sys.exit(1)
    else:
        print(f"[startup checks] All {passed}/{len(checks)} checks passed ✓")


# ── Individual checks ─────────────────────────────────────────────────────────

def _check_brain_forward():
    """Brain produces correct output shape and bounded values."""
    from simulation.brain import Brain
    from config import BRAIN_INPUTS, BRAIN_OUTPUTS
    np.random.seed(0)
    b   = Brain()
    out = b.forward(np.zeros(BRAIN_INPUTS, dtype=np.float32))
    assert out.shape == (BRAIN_OUTPUTS,), f"Expected shape ({BRAIN_OUTPUTS},), got {out.shape}"
    assert np.all(np.abs(out) <= 1.0), f"Output out of [-1,1]: {out}"


def _check_sensor_length():
    """Sensor vector has exactly 20 channels."""
    from simulation.sensors import compute_sensors
    from simulation.food import Food
    np.random.seed(0)
    from simulation.agent import Agent
    ag   = Agent(x=400, y=300)
    food = [Food(x=500, y=300)]
    s    = compute_sensors(ag, food, [ag])
    assert len(s) == 20, f"Expected 20 sensors, got {len(s)}"


def _check_agent_energy_decay():
    """Agent energy decreases each frame and death triggers correctly."""
    import config
    np.random.seed(0)
    from simulation.agent import Agent
    ag = Agent(x=400, y=300)
    ag.energy = config.AGENT_ENERGY_DECAY * 0.5  # less than one frame of decay
    ag.update(np.zeros(20, dtype=np.float32))
    assert not ag.alive, "Agent with sub-decay energy should die after one update"


def _check_spike_cooldown():
    """Spike only damages agent once per cooldown window."""
    from simulation.spikes import Spike
    from config import SPIKE_ENERGY_DAMAGE, AGENT_MAX_ENERGY
    np.random.seed(0)
    from simulation.agent import Agent
    ag    = Agent(x=400, y=300)
    ag.energy = AGENT_MAX_ENERGY
    spike = Spike(x=400, y=300)
    hit1  = spike.try_hit(ag)
    hit2  = spike.try_hit(ag)
    assert hit1,  "First overlap should hit"
    assert not hit2, "Immediate second hit should be blocked"
    assert abs(ag.energy - (AGENT_MAX_ENERGY - SPIKE_ENERGY_DAMAGE)) < 1e-5


def _check_food_energy_cap():
    """Eating food does not exceed AGENT_MAX_ENERGY."""
    from simulation.food import Food
    from config import AGENT_MAX_ENERGY
    np.random.seed(0)
    from simulation.agent import Agent
    ag       = Agent(x=400, y=300)
    ag.energy = AGENT_MAX_ENERGY
    food     = Food(x=400, y=300)
    ag.eat(food)
    assert ag.energy <= AGENT_MAX_ENERGY, (
        f"Energy {ag.energy} exceeded AGENT_MAX_ENERGY {AGENT_MAX_ENERGY} after eating"
    )


def _check_fitness_range():
    """Fitness is always in [0, 1] for extreme agent states."""
    from config import AGENT_MAX_AGE
    np.random.seed(0)
    from simulation.agent import Agent
    for age, food, dist in [
        (0, 0, 0),
        (AGENT_MAX_AGE, 0, 0),
        (0, 100, 0),
        (0, 0, 99999),
        (AGENT_MAX_AGE, 100, 99999),
    ]:
        ag = Agent(x=400, y=300)
        ag.age, ag.food_eaten, ag.distance_traveled = age, food, dist
        f = ag.fitness()
        assert 0.0 <= f <= 1.0, f"Fitness {f} out of [0,1] for age={age}, food={food}"


def _check_conservation_10_frames():
    """alive + deaths == total_born after 10 frames."""
    from simulation.world import World
    np.random.seed(0)
    world = make_minimal_world()

    death_count = 0
    for _ in range(10):
        prev = {a.id for a in world.agents}
        world.update()
        curr = {a.id for a in world.agents}
        death_count += len(prev - curr)

    alive   = len(world.agents)
    born    = world.evo_mgr.total_born
    recorded = world.evo_mgr.total_deaths

    assert alive + death_count == born, (
        f"Conservation broken: alive({alive}) + dead({death_count}) != born({born})"
    )
    assert recorded == death_count, (
        f"Recorded deaths ({recorded}) != independently counted ({death_count})"
    )


def _check_config_consistency():
    """Config values are self-consistent and within sensible ranges."""
    import config
    assert config.BRAIN_INPUTS == 20, (
        f"BRAIN_INPUTS={config.BRAIN_INPUTS} but sensors.py produces 20 channels. "
        "Update BRAIN_INPUTS if you added/removed sensors."
    )
    assert config.BRAIN_OUTPUTS == 2, (
        f"BRAIN_OUTPUTS must be 2 (turn, speed), got {config.BRAIN_OUTPUTS}"
    )
    assert 0.0 < config.MUTATION_RATE <= 1.0, (
        f"MUTATION_RATE={config.MUTATION_RATE} should be in (0, 1]"
    )
    assert config.POPULATION_MIN >= 2, (
        f"POPULATION_MIN={config.POPULATION_MIN} must be >= 2"
    )
    assert config.AGENT_ENERGY_DECAY > 0, (
        f"AGENT_ENERGY_DECAY={config.AGENT_ENERGY_DECAY} must be > 0"
    )
    total_fitness_weight = (config.FITNESS_LIFESPAN_W + config.FITNESS_FOOD_W
                            + config.FITNESS_DISTANCE_W)
    assert abs(total_fitness_weight - 1.0) < 0.01, (
        f"Fitness weights sum to {total_fitness_weight:.3f}, expected ~1.0"
    )
    assert isinstance(config.BRAIN_HIDDEN_LAYERS, list), (
        f"BRAIN_HIDDEN_LAYERS must be a list, got {type(config.BRAIN_HIDDEN_LAYERS)}"
    )


# ── Helper ─────────────────────────────────────────────────────────────────────

def make_minimal_world():
    """World with logging disabled for speed."""
    import config
    orig_log = config.LOG_INTERVAL
    orig_chk = config.CHECKPOINT_INTERVAL
    config.LOG_INTERVAL        = 0
    config.CHECKPOINT_INTERVAL = 0
    from simulation.world import World
    w = World(logger=None)
    config.LOG_INTERVAL        = orig_log
    config.CHECKPOINT_INTERVAL = orig_chk
    return w


if __name__ == "__main__":
    run_startup_checks()
