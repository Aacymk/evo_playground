"""
tests/test_conservation.py

Conservation and cohort accounting tests.
Uses fast_config so agents die quickly and multiple generations occur in ~500 frames.

1. alive + dead == total_born at all times
2. Death count and total_born are monotonically non-decreasing
3. No duplicate agent IDs in the alive list
4. No alive agent has alive=False stuck in world.agents
5. Cohort invariant: born[gen] == dead[gen] + alive[gen] for completed generations
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.conftest import seed, make_world, fast_config, restore_config


def test_alive_plus_dead_equals_total_born():
    """alive + independently counted deaths == total_born at end of run."""
    orig = fast_config()
    seed(42)
    world = make_world()

    independent_deaths = 0
    for _ in range(600):
        prev_ids = {a.id for a in world.agents}
        world.update()
        curr_ids = {a.id for a in world.agents}
        independent_deaths += len(prev_ids - curr_ids)

    alive       = len(world.agents)
    total_born  = world.evo_mgr.total_born
    rec_deaths  = world.evo_mgr.total_deaths

    assert alive + independent_deaths == total_born, (
        f"alive({alive}) + dead({independent_deaths}) != total_born({total_born})"
    )
    assert rec_deaths == independent_deaths, (
        f"evo_mgr.total_deaths({rec_deaths}) != counted deaths({independent_deaths})"
    )
    restore_config(orig)
    print(f"  total_born={total_born}, alive={alive}, dead={independent_deaths} ✓")


def test_no_duplicate_alive_ids():
    """No agent ID appears twice in world.agents at any frame."""
    orig = fast_config()
    seed(42)
    world = make_world()

    for frame in range(500):
        ids = [a.id for a in world.agents]
        assert len(ids) == len(set(ids)), (
            f"Frame {frame}: duplicate IDs: {[i for i in ids if ids.count(i) > 1]}"
        )
        world.update()
    restore_config(orig)


def test_no_dead_agent_stuck_in_alive_list():
    """No agent with alive=False should remain in world.agents after update."""
    orig = fast_config()
    seed(42)
    world = make_world()

    for frame in range(500):
        world.update()
        for ag in world.agents:
            assert ag.alive, (
                f"Frame {frame}: agent {ag.id} has alive=False but is in world.agents"
            )
    restore_config(orig)


def test_death_count_monotonic():
    """total_deaths must never decrease."""
    orig = fast_config()
    seed(42)
    world = make_world()
    prev = 0
    for _ in range(500):
        world.update()
        assert world.evo_mgr.total_deaths >= prev, (
            f"total_deaths went {prev} → {world.evo_mgr.total_deaths}"
        )
        prev = world.evo_mgr.total_deaths
    restore_config(orig)


def test_total_born_monotonic():
    """total_born must never decrease."""
    orig = fast_config()
    seed(42)
    world = make_world()
    prev = 0
    for _ in range(500):
        world.update()
        assert world.evo_mgr.total_born >= prev, (
            f"total_born went {prev} → {world.evo_mgr.total_born}"
        )
        prev = world.evo_mgr.total_born
    restore_config(orig)


def test_cohort_accounting():
    """
    For every completed generation:
    agents_born_in_gen == agents_dead_from_gen + agents_still_alive_from_gen
    """
    orig = fast_config()
    seed(42)
    world = make_world()

    births_by_gen = {}
    deaths_by_gen = {}

    for ag in world.agents:
        births_by_gen[ag.generation] = births_by_gen.get(ag.generation, 0) + 1

    for _ in range(800):
        prev = {a.id: a for a in world.agents}
        world.update()
        curr_ids = {a.id for a in world.agents}

        # New births
        for a in world.agents:
            if a.id not in prev:
                births_by_gen[a.generation] = births_by_gen.get(a.generation, 0) + 1

        # New deaths
        for aid in (set(prev) - curr_ids):
            g = prev[aid].generation
            deaths_by_gen[g] = deaths_by_gen.get(g, 0) + 1

    current_gen = world.evo_mgr.generation
    alive_by_gen = {}
    for ag in world.agents:
        alive_by_gen[ag.generation] = alive_by_gen.get(ag.generation, 0) + 1

    checked = 0
    for gen, born in births_by_gen.items():
        if gen >= current_gen:
            continue
        dead  = deaths_by_gen.get(gen, 0)
        alive = alive_by_gen.get(gen, 0)
        assert born == dead + alive, (
            f"Cohort gen {gen}: born={born}, dead={dead}, alive={alive} — mismatch"
        )
        checked += 1

    assert checked > 0, "No completed generations to verify — increase frame count"
    restore_config(orig)
    print(f"  Cohort accounting verified across {checked} completed generations ✓")


if __name__ == "__main__":
    print("Running conservation tests...")
    test_alive_plus_dead_equals_total_born()
    test_no_duplicate_alive_ids()
    test_no_dead_agent_stuck_in_alive_list()
    test_death_count_monotonic()
    test_total_born_monotonic()
    test_cohort_accounting()
    print("All conservation tests passed.")
