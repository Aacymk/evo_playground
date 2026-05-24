"""
tests/test_lifecycle.py

Lifecycle tests — verify the exact mechanics of birth, death, and state
transitions for individual agents. These are the simplest tests but catch
the most bugs in evolutionary systems.

1. Spawning N agents creates exactly N agents
2. Killing K agents records exactly K deaths and leaves N-K alive
3. Energy depletion kills agent and records death
4. Age limit kills agent and records death
5. An agent dead in gen N that gets record_death called in gen N+1 is still
   logged (the dangerous state transition)
6. Spike hit drains energy correctly with cooldown
7. Eating food increases energy correctly and caps at AGENT_MAX_ENERGY
8. Dead agents are removed from world.agents exactly one frame after dying
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.conftest import seed, make_agent, make_agents, make_world, null_sensors
import numpy as np


def test_spawn_count():
    """world.agents has exactly AGENT_INITIAL_COUNT agents after init."""
    from config import AGENT_INITIAL_COUNT
    seed(0)
    world = make_world()
    assert len(world.agents) == AGENT_INITIAL_COUNT, (
        f"Expected {AGENT_INITIAL_COUNT} agents, got {len(world.agents)}"
    )
    assert world.evo_mgr.total_born == AGENT_INITIAL_COUNT


def test_manual_kill_exact_count():
    """
    Spawn 10 agents. Kill 3 manually. Verify exactly 3 deaths recorded,
    exactly 7 alive, no duplicates.
    """
    seed(0)
    from simulation.world import World
    from simulation.agent import Agent

    # Build a minimal world with exactly 10 agents
    world = make_world()
    world.agents = make_agents(10, generation=1)
    world.evo_mgr.total_born = 10
    world.evo_mgr.total_deaths = 0

    targets = world.agents[:3]
    for ag in targets:
        ag.alive = False

    # Simulate the death-handling step
    for ag in world.agents:
        if not ag.alive:
            world.evo_mgr.record_death(ag)
    world.agents = [a for a in world.agents if a.alive]

    assert len(world.agents) == 7, f"Expected 7 alive, got {len(world.agents)}"
    assert world.evo_mgr.total_deaths == 3, (
        f"Expected 3 deaths, got {world.evo_mgr.total_deaths}"
    )

    alive_ids = [a.id for a in world.agents]
    assert len(alive_ids) == len(set(alive_ids)), "Duplicate IDs in alive list"

    dead_ids = {a.id for a in targets}
    assert not dead_ids.intersection(set(alive_ids)), (
        "Killed agents still appear in alive list"
    )


def test_energy_depletion_kills_agent():
    """Agent with energy=0 should have alive=False after update."""
    seed(0)
    agent = make_agent(energy=0.001)  # just above zero
    # One update should drain it below zero
    agent.update(null_sensors())
    assert not agent.alive, "Agent with near-zero energy should die after update"


def test_energy_never_kills_before_zero():
    """Agent with full energy should not die on first update."""
    seed(0)
    from config import AGENT_MAX_ENERGY
    agent = make_agent(energy=AGENT_MAX_ENERGY)
    agent.update(null_sensors())
    assert agent.alive, "Agent with full energy should not die on first update"


def test_age_limit_kills_agent():
    """Agent at AGENT_MAX_AGE - 1 should die on next update."""
    from config import AGENT_MAX_AGE, AGENT_MAX_ENERGY
    seed(0)
    agent = make_agent(energy=AGENT_MAX_ENERGY, age=AGENT_MAX_AGE - 1)
    agent.update(null_sensors())
    assert not agent.alive, (
        f"Agent at age {AGENT_MAX_AGE - 1} should die after one more update"
    )


def test_eating_food_increases_energy():
    """Agent energy increases by FOOD_ENERGY when eating, capped at AGENT_MAX_ENERGY."""
    from config import FOOD_ENERGY, AGENT_MAX_ENERGY
    from simulation.food import Food
    seed(0)

    agent = make_agent(energy=50.0)
    food  = Food(x=400, y=300)

    agent.eat(food)

    assert food.alive is False, "Food should be marked dead after eating"
    assert agent.food_eaten == 1
    expected = min(AGENT_MAX_ENERGY, 50.0 + FOOD_ENERGY)
    assert abs(agent.energy - expected) < 1e-6, (
        f"Expected energy {expected}, got {agent.energy}"
    )


def test_eating_food_caps_at_max_energy():
    """Eating food when near max energy should not exceed AGENT_MAX_ENERGY."""
    from config import FOOD_ENERGY, AGENT_MAX_ENERGY
    from simulation.food import Food
    seed(0)

    agent = make_agent(energy=AGENT_MAX_ENERGY - 1.0)
    food  = Food(x=400, y=300)
    agent.eat(food)

    assert agent.energy <= AGENT_MAX_ENERGY, (
        f"Energy {agent.energy} exceeded AGENT_MAX_ENERGY {AGENT_MAX_ENERGY}"
    )
    assert agent.energy == AGENT_MAX_ENERGY


def test_dead_agents_removed_next_frame():
    """
    An agent with energy <= 0 should be absent from world.agents
    after the very next world.update() call.
    """
    seed(0)
    world = make_world()
    target = world.agents[0]
    target.energy = 0.0  # will die on next update

    world.update()

    ids = [a.id for a in world.agents]
    assert target.id not in ids, (
        f"Agent {target.id} with energy=0 should have been removed after update"
    )


def test_death_recorded_across_generation_boundary():
    """
    State transition test: if an agent born in generation N survives into the
    frame where generation is incremented to N+1, and then dies, it should
    still be correctly recorded in the dead pool (not silently dropped).

    This is the exact bug pattern described in the test spec.
    """
    seed(0)
    world = make_world()

    # Find the current generation
    gen_before = world.evo_mgr.generation

    # Run until at least one generation increment occurs
    for _ in range(5000):
        world.update()
        if world.evo_mgr.generation > gen_before:
            break

    gen_after = world.evo_mgr.generation
    assert gen_after > gen_before, "No generation increment happened during run"

    # Now take an agent born in the NEW generation and kill it
    new_gen_agents = [a for a in world.agents if a.generation == gen_after]
    if not new_gen_agents:
        # Spawn one explicitly if none exist yet
        from simulation.agent import Agent
        ag = Agent(x=400, y=300, generation=gen_after)
        world.agents.append(ag)
        new_gen_agents = [ag]

    deaths_before = world.evo_mgr.total_deaths
    victim = new_gen_agents[0]
    victim.energy = 0.0  # will die next update

    world.update()

    assert world.evo_mgr.total_deaths == deaths_before + 1 or \
           world.evo_mgr.total_deaths > deaths_before, (
        "Death of cross-generation agent was not recorded"
    )


def test_spike_cooldown_prevents_rapid_damage():
    """
    A spike should only damage an agent once per SPIKE_HIT_COOLDOWN frames,
    not every frame while overlapping.
    """
    from config import SPIKE_ENERGY_DAMAGE, SPIKE_HIT_COOLDOWN, AGENT_MAX_ENERGY
    from simulation.spikes import Spike
    seed(0)

    agent = make_agent(x=400, y=300, energy=AGENT_MAX_ENERGY)
    spike = Spike(x=400, y=300)  # place directly on top of agent

    # First hit should apply
    hit1 = spike.try_hit(agent)
    assert hit1, "First overlap should register a hit"
    energy_after_hit1 = agent.energy
    assert abs(energy_after_hit1 - (AGENT_MAX_ENERGY - SPIKE_ENERGY_DAMAGE)) < 1e-6

    # Immediate second hit should be blocked by cooldown
    hit2 = spike.try_hit(agent)
    assert not hit2, "Second immediate hit should be blocked by cooldown"
    assert agent.energy == energy_after_hit1, "Energy should not change during cooldown"

    # After cooldown expires, should hit again
    for _ in range(SPIKE_HIT_COOLDOWN + 1):
        spike.tick_cooldowns()

    hit3 = spike.try_hit(agent)
    assert hit3, "Hit after cooldown should register"
    assert agent.energy < energy_after_hit1, "Energy should decrease after cooldown expires"


def test_negative_energy_clamped_by_spike():
    """
    Spike damage should not push energy below 0 (it should clamp, handled in
    spikes.py). Agent dying from spike should have energy at or below 0, not
    some large negative number that would indicate unclamped subtraction.
    """
    from config import SPIKE_ENERGY_DAMAGE
    from simulation.spikes import Spike
    seed(0)

    agent = make_agent(energy=1.0)  # less than SPIKE_ENERGY_DAMAGE
    spike = Spike(x=400, y=300)
    spike.try_hit(agent)

    assert agent.energy >= 0.0, (
        f"Energy should be clamped to >= 0 after spike hit, got {agent.energy}"
    )


if __name__ == "__main__":
    print("Running lifecycle tests...")
    test_spawn_count()
    test_manual_kill_exact_count()
    test_energy_depletion_kills_agent()
    test_energy_never_kills_before_zero()
    test_age_limit_kills_agent()
    test_eating_food_increases_energy()
    test_eating_food_caps_at_max_energy()
    test_dead_agents_removed_next_frame()
    test_death_recorded_across_generation_boundary()
    test_spike_cooldown_prevents_rapid_damage()
    test_negative_energy_clamped_by_spike()
    print("All lifecycle tests passed.")
