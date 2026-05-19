"""
Sensor system for agents.

Sensor vector layout (9 values):
  [0]  food_distance   (normalized 0-1, 1 = nothing nearby)
  [1]  food_angle      (normalized -1 to 1, relative to heading)
  [2]  wall_distance   (normalized 0-1)
  [3]  touch_wall      (0 or 1)
  [4]  touch_agent     (0 or 1)
  [5]  touch_food      (0 or 1)
  [6]  sound_level     (0-1, nearby agent movement noise)
  [7]  energy_level    (0-1)
  [8]  noise           (random, adds stochasticity)
"""

import numpy as np
from config import (VISION_RADIUS, VISION_FOV, SOUND_RADIUS,
                    WALL_SENSOR_RADIUS, AGENT_RADIUS, FOOD_RADIUS,
                    AGENT_MAX_ENERGY, WORLD_WIDTH, WORLD_HEIGHT)
from utils.math_utils import distance, angle_diff, normalize_angle


def compute_sensors(agent, food_list, all_agents):
    """
    Returns a numpy array of sensor values for the given agent.
    """
    sensors = np.zeros(9, dtype=np.float32)

    ax, ay = agent.x, agent.y
    heading = agent.angle

    # ── 1. Food vision ──────────────────────────────────────────────────────
    best_dist = VISION_RADIUS
    best_angle = 0.0
    found_food = False

    for food in food_list:
        if not food.alive:
            continue
        dx = food.x - ax
        dy = food.y - ay
        dist = np.hypot(dx, dy)
        if dist > VISION_RADIUS:
            continue
        # Check if within FOV cone
        ang_to_food = np.arctan2(dy, dx)
        rel_ang = normalize_angle(ang_to_food - heading)
        if abs(rel_ang) <= VISION_FOV / 2:
            if dist < best_dist:
                best_dist = dist
                best_angle = rel_ang
                found_food = True

    if found_food:
        sensors[0] = 1.0 - (best_dist / VISION_RADIUS)   # close = 1
        sensors[1] = best_angle / (VISION_FOV / 2)        # normalized angle
    else:
        sensors[0] = 0.0
        sensors[1] = 0.0

    # ── 2. Wall distance ────────────────────────────────────────────────────
    dist_left  = ax
    dist_right = WORLD_WIDTH - ax
    dist_top   = ay
    dist_bot   = WORLD_HEIGHT - ay
    min_wall   = min(dist_left, dist_right, dist_top, dist_bot)
    sensors[2] = max(0.0, 1.0 - min_wall / WALL_SENSOR_RADIUS)

    # ── 3. Touch sensors ────────────────────────────────────────────────────
    touch_wall  = ax <= AGENT_RADIUS or ax >= WORLD_WIDTH - AGENT_RADIUS or \
                  ay <= AGENT_RADIUS or ay >= WORLD_HEIGHT - AGENT_RADIUS
    sensors[3] = 1.0 if touch_wall else 0.0

    touch_agent = False
    for other in all_agents:
        if other is agent:
            continue
        if distance(ax, ay, other.x, other.y) < AGENT_RADIUS * 2.2:
            touch_agent = True
            break
    sensors[4] = 1.0 if touch_agent else 0.0

    touch_food = False
    for food in food_list:
        if not food.alive:
            continue
        if distance(ax, ay, food.x, food.y) < AGENT_RADIUS + FOOD_RADIUS:
            touch_food = True
            break
    sensors[5] = 1.0 if touch_food else 0.0

    # ── 4. Sound sensor ─────────────────────────────────────────────────────
    total_sound = 0.0
    for other in all_agents:
        if other is agent:
            continue
        d = distance(ax, ay, other.x, other.y)
        if d < SOUND_RADIUS and d > 0:
            # Sound intensity falls off with distance; proportional to speed
            intensity = (other.speed / 3.5) * (1.0 - d / SOUND_RADIUS)
            total_sound += intensity
    sensors[6] = min(1.0, total_sound)

    # ── 5. Internal state ───────────────────────────────────────────────────
    sensors[7] = agent.energy / AGENT_MAX_ENERGY
    sensors[8] = np.random.uniform(-0.2, 0.2)  # small noise input

    return sensors
