import numpy as np
import pygame
from config import (AGENT_RADIUS, AGENT_INITIAL_ENERGY, AGENT_MAX_ENERGY,
                    AGENT_ENERGY_DECAY, AGENT_SPEED_DECAY_COST, AGENT_MAX_AGE,
                    AGENT_MOVE_SPEED_MAX, AGENT_TURN_SPEED_MAX, FOOD_ENERGY,
                    WORLD_WIDTH, WORLD_HEIGHT, ENERGY_BAR_WIDTH, ENERGY_BAR_HEIGHT,
                    ENERGY_BAR_OFFSET, SELECTED_COLOR, FOOD_RADIUS)
from simulation.brain import Brain
from utils.math_utils import clamp, random_direction


# Color palette for agents - distinguishable across generations
PALETTE = [
    (220, 100, 100), (100, 180, 220), (220, 200, 80),
    (180, 100, 220), (100, 220, 160), (220, 140, 60),
    (160, 220, 220), (220, 160, 200), (140, 200, 100),
]


_agent_id_counter = 0


def _next_id():
    global _agent_id_counter
    _agent_id_counter += 1
    return _agent_id_counter


def _safe_spawn(margin, forbidden_positions=None, min_clear=20, max_attempts=50):
    """
    Pick a random (x, y) that isn't within min_clear pixels of any forbidden
    position (spike locations). Falls back to pure random after max_attempts.
    """
    for _ in range(max_attempts):
        x = np.random.uniform(margin, WORLD_WIDTH - margin)
        y = np.random.uniform(margin, WORLD_HEIGHT - margin)
        if not forbidden_positions:
            return x, y
        too_close = any(
            np.hypot(x - fx, y - fy) < min_clear
            for fx, fy in forbidden_positions
        )
        if not too_close:
            return x, y
    # Fallback — just return the last attempt, better than an infinite loop
    return x, y


class Agent:
    def __init__(self, x=None, y=None, brain=None, color=None, generation=0,
                 forbidden_positions=None):
        self.id = _next_id()
        self.generation = generation

        margin = AGENT_RADIUS + 20
        if x is not None and y is not None:
            self.x, self.y = x, y
        else:
            self.x, self.y = _safe_spawn(margin, forbidden_positions)
        self.angle = random_direction()
        self.speed = 0.0

        self.energy = AGENT_INITIAL_ENERGY
        self.age = 0
        self.alive = True

        self.brain = brain if brain is not None else Brain()
        self.color = color if color is not None else PALETTE[self.id % len(PALETTE)]

        # Stats for fitness calculation
        self.food_eaten = 0
        self.distance_traveled = 0.0
        self.spike_hits = 0

        # Last sensor + output (for UI inspection)
        self.last_sensors = np.zeros(20, dtype=np.float32)
        self.last_outputs = np.zeros(2, dtype=np.float32)

        # Action history for logging (sampled every N frames to keep memory bounded)
        self.action_history: list = []
        self._action_sample_interval = 10  # record every 10 frames

    # ── Update ───────────────────────────────────────────────────────────────

    def update(self, sensors):
        """Apply brain outputs to movement, drain energy, age."""
        self.last_sensors = sensors
        outputs = self.brain.forward(sensors)
        self.last_outputs = outputs

        # outputs[0]: turn   in [-1, 1]
        # outputs[1]: speed  in [-1, 1] (we map to [0, max])
        turn = float(outputs[0]) * AGENT_TURN_SPEED_MAX
        raw_speed = float(outputs[1])
        self.speed = clamp((raw_speed + 1.0) / 2.0 * AGENT_MOVE_SPEED_MAX, 0.0, AGENT_MOVE_SPEED_MAX)

        # Sample action history periodically (every N frames)
        if self.age % self._action_sample_interval == 0:
            self.action_history.append([float(outputs[0]), float(outputs[1])])

        self.angle += turn
        dx = np.cos(self.angle) * self.speed
        dy = np.sin(self.angle) * self.speed

        new_x = self.x + dx
        new_y = self.y + dy

        # Wall bounce
        if new_x < AGENT_RADIUS:
            new_x = AGENT_RADIUS
            self.angle = np.pi - self.angle
        elif new_x > WORLD_WIDTH - AGENT_RADIUS:
            new_x = WORLD_WIDTH - AGENT_RADIUS
            self.angle = np.pi - self.angle
        if new_y < AGENT_RADIUS:
            new_y = AGENT_RADIUS
            self.angle = -self.angle
        elif new_y > WORLD_HEIGHT - AGENT_RADIUS:
            new_y = WORLD_HEIGHT - AGENT_RADIUS
            self.angle = -self.angle

        dist = np.hypot(new_x - self.x, new_y - self.y)
        self.distance_traveled += dist
        self.x = new_x
        self.y = new_y

        # Energy decay (moving costs a tiny bit more)
        decay = AGENT_ENERGY_DECAY + self.speed * AGENT_SPEED_DECAY_COST
        self.energy -= decay
        self.age += 1

        if self.energy <= 0 or self.age >= AGENT_MAX_AGE:
            self.alive = False

    def eat(self, food):
        """Called when agent overlaps with a food item."""
        food.alive = False
        self.energy = min(AGENT_MAX_ENERGY, self.energy + FOOD_ENERGY)
        self.food_eaten += 1

    def fitness(self):
        from config import FITNESS_LIFESPAN_W, FITNESS_FOOD_W, FITNESS_DISTANCE_W
        lifespan_score = self.age / AGENT_MAX_AGE
        food_score = min(1.0, self.food_eaten / 20.0)
        dist_score = min(1.0, self.distance_traveled / 5000.0)
        return (FITNESS_LIFESPAN_W * lifespan_score +
                FITNESS_FOOD_W * food_score +
                FITNESS_DISTANCE_W * dist_score)

    # ── Drawing ──────────────────────────────────────────────────────────────

    def draw(self, surface, selected=False):
        ix, iy = int(self.x), int(self.y)

        # Glow ring for selected
        if selected:
            pygame.draw.circle(surface, SELECTED_COLOR, (ix, iy), AGENT_RADIUS + 5, 2)

        # Body
        pygame.draw.circle(surface, self.color, (ix, iy), AGENT_RADIUS)

        # Direction indicator
        tip_x = ix + int(np.cos(self.angle) * (AGENT_RADIUS + 4))
        tip_y = iy + int(np.sin(self.angle) * (AGENT_RADIUS + 4))
        pygame.draw.line(surface, (255, 255, 255), (ix, iy), (tip_x, tip_y), 2)

        # Energy bar
        self._draw_energy_bar(surface, ix, iy)

    def _draw_energy_bar(self, surface, ix, iy):
        ratio = self.energy / AGENT_MAX_ENERGY
        bar_y = iy - ENERGY_BAR_OFFSET
        bar_x = ix - ENERGY_BAR_WIDTH // 2

        # Background
        pygame.draw.rect(surface, (60, 60, 60),
                         (bar_x, bar_y, ENERGY_BAR_WIDTH, ENERGY_BAR_HEIGHT))
        # Fill
        fill_w = int(ENERGY_BAR_WIDTH * ratio)
        if fill_w > 0:
            r = int(255 * (1 - ratio))
            g = int(200 * ratio)
            pygame.draw.rect(surface, (r, g, 40),
                             (bar_x, bar_y, fill_w, ENERGY_BAR_HEIGHT))

    def draw_vision_cone(self, surface):
        """Draw the FOV cone for this agent (call only for selected agent)."""
        from config import VISION_RADIUS, VISION_FOV, VISION_CONE_ALPHA
        import math

        num_segments = 30
        half_fov = VISION_FOV / 2
        points = [(int(self.x), int(self.y))]

        for i in range(num_segments + 1):
            a = self.angle - half_fov + (VISION_FOV * i / num_segments)
            px = self.x + math.cos(a) * VISION_RADIUS
            py = self.y + math.sin(a) * VISION_RADIUS
            points.append((int(px), int(py)))

        cone_surf = pygame.Surface((WORLD_WIDTH, WORLD_HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(cone_surf, (100, 200, 255, VISION_CONE_ALPHA), points)
        surface.blit(cone_surf, (0, 0))

        # Outline
        pygame.draw.polygon(surface, (80, 160, 220), points, 1)
