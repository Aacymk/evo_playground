"""
Sensor system — 20 input channels.

Index  Name               Range       Description
─────────────────────────────────────────────────────────────────────────────
  0    food_dist          0–1         Nearest food in FOV; 0=none/far, 1=touching
  1    food_angle         -1–1        Relative angle to that food (neg=left, pos=right)
  2    food_count_fov     0–1         # food objects visible in FOV, normalised to cap of 10
  3    spike_dist         0–1         Nearest spike in FOV; 0=none/far, 1=touching
  4    spike_angle        -1–1        Relative angle to nearest spike in FOV
  5    spike_count_fov    0–1         # spikes visible in FOV, normalised to cap of 5
  6    agent_count_fov    0–1         # other agents visible in FOV, normalised to cap of 10
  7    wall_dist_nearest  0–1         Nearest wall proximity (any direction)
  8    wall_dist_north    0–1         Distance to top border, normalised
  9    wall_dist_south    0–1         Distance to bottom border, normalised
 10    wall_dist_east     0–1         Distance to right border, normalised
 11    wall_dist_west     0–1         Distance to left border, normalised
 12    pos_x              0–1         Normalised X position (0=left, 1=right)
 13    pos_y              0–1         Normalised Y position (0=top, 1=bottom)
 14    touch_wall         0 or 1      Binary: currently at wall boundary
 15    touch_agent        0 or 1      Binary: body overlapping another agent
 16    touch_food         0 or 1      Binary: body overlapping food
 17    sound              0–1         Nearby agent movement noise
 18    energy_lvl         0–1         Current energy / max energy
 19    noise              +-0.2       Random jitter each frame
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
from config import (VISION_RADIUS, VISION_FOV, SOUND_RADIUS,
                    WALL_SENSOR_RADIUS, AGENT_RADIUS, FOOD_RADIUS,
                    AGENT_MAX_ENERGY, WORLD_WIDTH, WORLD_HEIGHT,
                    SPIKE_RADIUS)
from utils.math_utils import distance, normalize_angle

# Normalisation caps for count sensors
_FOOD_COUNT_CAP  = 10
_SPIKE_COUNT_CAP = 5
_AGENT_COUNT_CAP = 10


def compute_sensors(agent, food_list, all_agents, spike_list=None):
    """
    Returns a numpy float32 array of length 20.
    spike_list: list of Spike objects (or None / empty for backward-compat).
    """
    if spike_list is None:
        spike_list = []

    s = np.zeros(20, dtype=np.float32)
    ax, ay = agent.x, agent.y
    heading = agent.angle
    half_fov = VISION_FOV / 2

    def in_cone(tx, ty):
        """Returns (visible, dist, rel_angle)."""
        dx, dy = tx - ax, ty - ay
        dist = np.hypot(dx, dy)
        if dist > VISION_RADIUS:
            return False, dist, 0.0
        rel_ang = normalize_angle(np.arctan2(dy, dx) - heading)
        return abs(rel_ang) <= half_fov, dist, rel_ang

    # ── [0,1] Nearest food in FOV + [2] food count ─────────────────────────
    best_food_dist = VISION_RADIUS
    best_food_ang  = 0.0
    found_food     = False
    food_count_fov = 0

    for food in food_list:
        if not food.alive:
            continue
        visible, dist, rel_ang = in_cone(food.x, food.y)
        if visible:
            food_count_fov += 1
            if dist < best_food_dist:
                best_food_dist = dist
                best_food_ang  = rel_ang
                found_food     = True

    if found_food:
        s[0] = 1.0 - (best_food_dist / VISION_RADIUS)
        s[1] = best_food_ang / half_fov
    s[2] = min(1.0, food_count_fov / _FOOD_COUNT_CAP)

    # ── [3,4] Nearest spike in FOV + [5] spike count ───────────────────────
    best_spike_dist = VISION_RADIUS
    best_spike_ang  = 0.0
    found_spike     = False
    spike_count_fov = 0

    for spike in spike_list:
        visible, dist, rel_ang = in_cone(spike.x, spike.y)
        if visible:
            spike_count_fov += 1
            if dist < best_spike_dist:
                best_spike_dist = dist
                best_spike_ang  = rel_ang
                found_spike     = True

    if found_spike:
        s[3] = 1.0 - (best_spike_dist / VISION_RADIUS)
        s[4] = best_spike_ang / half_fov
    s[5] = min(1.0, spike_count_fov / _SPIKE_COUNT_CAP)

    # ── [6] Agent count in FOV ──────────────────────────────────────────────
    agent_count_fov = 0
    for other in all_agents:
        if other is agent:
            continue
        visible, _, _ = in_cone(other.x, other.y)
        if visible:
            agent_count_fov += 1
    s[6] = min(1.0, agent_count_fov / _AGENT_COUNT_CAP)

    # ── [7] Nearest wall (any direction) ────────────────────────────────────
    d_north = ay
    d_south = WORLD_HEIGHT - ay
    d_east  = WORLD_WIDTH - ax
    d_west  = ax
    min_wall = min(d_north, d_south, d_east, d_west)
    s[7] = max(0.0, 1.0 - min_wall / WALL_SENSOR_RADIUS)

    # ── [8-11] Cardinal wall distances (normalised by half-dimension) ────────
    s[8]  = 1.0 - min(1.0, d_north / (WORLD_HEIGHT / 2))
    s[9]  = 1.0 - min(1.0, d_south / (WORLD_HEIGHT / 2))
    s[10] = 1.0 - min(1.0, d_east  / (WORLD_WIDTH  / 2))
    s[11] = 1.0 - min(1.0, d_west  / (WORLD_WIDTH  / 2))

    # ── [12,13] World position ──────────────────────────────────────────────
    s[12] = ax / WORLD_WIDTH
    s[13] = ay / WORLD_HEIGHT

    # ── [14,15,16] Touch sensors ────────────────────────────────────────────
    touch_wall = (ax <= AGENT_RADIUS or ax >= WORLD_WIDTH  - AGENT_RADIUS or
                  ay <= AGENT_RADIUS or ay >= WORLD_HEIGHT - AGENT_RADIUS)
    s[14] = 1.0 if touch_wall else 0.0

    touch_agent = any(
        other is not agent and distance(ax, ay, other.x, other.y) < AGENT_RADIUS * 2.2
        for other in all_agents
    )
    s[15] = 1.0 if touch_agent else 0.0

    touch_food = any(
        food.alive and distance(ax, ay, food.x, food.y) < AGENT_RADIUS + FOOD_RADIUS
        for food in food_list
    )
    s[16] = 1.0 if touch_food else 0.0

    # ── [17] Sound ──────────────────────────────────────────────────────────
    total_sound = 0.0
    for other in all_agents:
        if other is agent:
            continue
        d = distance(ax, ay, other.x, other.y)
        if 0 < d < SOUND_RADIUS:
            total_sound += (other.speed / 3.5) * (1.0 - d / SOUND_RADIUS)
    s[17] = min(1.0, total_sound)

    # ── [18] Energy level ───────────────────────────────────────────────────
    s[18] = agent.energy / AGENT_MAX_ENERGY

    # ── [19] Random noise ───────────────────────────────────────────────────
    s[19] = np.random.uniform(-0.2, 0.2)

    return s
