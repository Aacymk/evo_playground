"""
tests/test_sensors_world.py

Tests for sensor correctness and world boundary behavior.

Sensor tests:
  1. Food directly ahead in cone → food_dist > 0, food_angle ≈ 0
  2. Food behind agent → food_dist == 0 (not visible)
  3. Food outside vision radius → food_dist == 0
  4. Agent at left wall → wall_west ≈ 1, wall_east ≈ 0
  5. Agent at center → pos_x ≈ 0.5, pos_y ≈ 0.5
  6. Sensor vector is always length 20
  7. All sensor values in expected ranges

World boundary tests:
  8. Agents cannot leave the world bounds (wall bounce)
  9. Agents spawned by safe_spawn are not inside spikes
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.conftest import seed, make_agent
import numpy as np


def test_sensor_vector_length():
    """compute_sensors always returns a 20-element vector."""
    from simulation.sensors import compute_sensors
    from simulation.food import Food
    seed(0)
    agent = make_agent(x=400, y=300)
    food  = [Food(x=500, y=300)]
    s = compute_sensors(agent, food, [agent])
    assert len(s) == 20, f"Expected 20 sensors, got {len(s)}"
    assert s.dtype == np.float32


def test_food_directly_ahead_detected():
    """Food placed directly ahead in cone should produce food_dist > 0."""
    from simulation.sensors import compute_sensors
    from simulation.food import Food
    from config import VISION_RADIUS
    seed(0)

    agent = make_agent(x=400, y=300)
    agent.angle = 0.0   # facing right

    # Place food directly to the right, within vision radius
    food = [Food(x=400 + VISION_RADIUS // 2, y=300)]
    s = compute_sensors(agent, food, [agent])

    assert s[0] > 0.0, f"food_dist should be > 0 with food ahead, got {s[0]}"
    assert abs(s[1]) < 0.1, f"food_angle should be near 0 when food is straight ahead, got {s[1]}"


def test_food_behind_agent_not_detected():
    """Food placed directly behind agent should not be detected (outside FOV)."""
    from simulation.sensors import compute_sensors
    from simulation.food import Food
    from config import VISION_RADIUS
    seed(0)

    agent = make_agent(x=400, y=300)
    agent.angle = 0.0   # facing right

    # Food directly to the LEFT (behind the agent)
    food = [Food(x=400 - VISION_RADIUS // 2, y=300)]
    s = compute_sensors(agent, food, [agent])

    assert s[0] == 0.0, (
        f"food_dist should be 0 for food behind agent, got {s[0]}"
    )


def test_food_outside_radius_not_detected():
    """Food beyond VISION_RADIUS should not be detected."""
    from simulation.sensors import compute_sensors
    from simulation.food import Food
    from config import VISION_RADIUS
    seed(0)

    agent = make_agent(x=400, y=300)
    agent.angle = 0.0

    food = [Food(x=400 + VISION_RADIUS + 50, y=300)]  # just outside radius
    s = compute_sensors(agent, food, [agent])

    assert s[0] == 0.0, (
        f"food_dist should be 0 beyond VISION_RADIUS, got {s[0]}"
    )


def test_wall_sensor_at_left_wall():
    """Agent at left wall: wall_west should be high, wall_east should be low."""
    from simulation.sensors import compute_sensors
    from config import AGENT_RADIUS
    seed(0)

    # Agent exactly at the wall boundary (where bounce clamps it)
    agent = make_agent(x=float(AGENT_RADIUS), y=300)
    s = compute_sensors(agent, [], [agent])

    assert s[11] > 0.8, f"wall_west (s[11]) should be near 1 at left wall, got {s[11]}"
    assert s[10] < 0.3, f"wall_east (s[10]) should be near 0 at left wall, got {s[10]}"
    assert s[14] == 1.0, f"touch_wall (s[14]) should be 1 at wall, got {s[14]}"


def test_position_sensors_at_center():
    """Agent at arena center: pos_x and pos_y should both be ≈ 0.5."""
    from simulation.sensors import compute_sensors
    from config import WORLD_WIDTH, WORLD_HEIGHT
    seed(0)

    agent = make_agent(x=WORLD_WIDTH / 2, y=WORLD_HEIGHT / 2)
    s = compute_sensors(agent, [], [agent])

    assert abs(s[12] - 0.5) < 0.01, f"pos_x should be 0.5 at center, got {s[12]}"
    assert abs(s[13] - 0.5) < 0.01, f"pos_y should be 0.5 at center, got {s[13]}"


def test_all_sensor_values_in_expected_ranges():
    """
    Run 500 frames and verify all sensor channels stay within expected ranges.
    Channel 19 (noise) is [-0.2, 0.2]; most others [0, 1]; angles [-1, 1].
    """
    from simulation.world import World
    seed(0)
    world = World()

    for frame in range(500):
        world.update()
        for agent in world.agents:
            s = agent.last_sensors
            assert len(s) == 20, f"Sensor length changed to {len(s)}"

            # Binary channels
            for i in [14, 15, 16]:
                assert s[i] in (0.0, 1.0), f"Frame {frame} s[{i}]={s[i]} not binary"

            # [0,1] bounded channels
            for i in [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 17, 18]:
                assert 0.0 <= s[i] <= 1.0, (
                    f"Frame {frame} agent {agent.id} s[{i}]={s[i]} out of [0,1]"
                )

            # Angle channels [-1, 1]
            for i in [1, 4]:
                assert -1.0 <= s[i] <= 1.0, (
                    f"Frame {frame} agent {agent.id} s[{i}]={s[i]} out of [-1,1]"
                )

            # Position channels [0, 1]
            assert 0.0 <= s[12] <= 1.0
            assert 0.0 <= s[13] <= 1.0

            # Noise channel [-0.2, 0.2]
            assert -0.21 <= s[19] <= 0.21, (
                f"Frame {frame} noise s[19]={s[19]} out of expected range"
            )


def test_agents_stay_within_bounds():
    """Agents should never leave the world bounds after a wall bounce."""
    from config import WORLD_WIDTH, WORLD_HEIGHT, AGENT_RADIUS
    seed(0)

    from simulation.world import World
    world = World()

    for frame in range(1000):
        world.update()
        for agent in world.agents:
            assert agent.x >= 0, f"Frame {frame}: agent x={agent.x} < 0"
            assert agent.x <= WORLD_WIDTH, f"Frame {frame}: agent x={agent.x} > {WORLD_WIDTH}"
            assert agent.y >= 0, f"Frame {frame}: agent y={agent.y} < 0"
            assert agent.y <= WORLD_HEIGHT, f"Frame {frame}: agent y={agent.y} > {WORLD_HEIGHT}"


def test_safe_spawn_avoids_spikes():
    """Agents spawned via _safe_spawn should not overlap with spike positions."""
    from simulation.agent import _safe_spawn
    from simulation.spikes import SpikeManager
    seed(0)

    sm = SpikeManager()
    forbidden = sm.positions()
    min_clear = 20

    for _ in range(200):
        x, y = _safe_spawn(margin=20, forbidden_positions=forbidden, min_clear=min_clear)
        for fx, fy in forbidden:
            dist = np.hypot(x - fx, y - fy)
            assert dist >= min_clear - 1, (  # -1 for float tolerance
                f"Spawned at ({x:.1f},{y:.1f}), spike at ({fx:.1f},{fy:.1f}), "
                f"dist={dist:.2f} < min_clear={min_clear}"
            )


if __name__ == "__main__":
    print("Running sensor/world tests...")
    test_sensor_vector_length()
    test_food_directly_ahead_detected()
    test_food_behind_agent_not_detected()
    test_food_outside_radius_not_detected()
    test_wall_sensor_at_left_wall()
    test_position_sensors_at_center()
    test_all_sensor_values_in_expected_ranges()
    test_agents_stay_within_bounds()
    test_safe_spawn_avoids_spikes()
    print("All sensor/world tests passed.")
