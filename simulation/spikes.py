import numpy as np
import pygame
from config import (SPIKE_RADIUS, SPIKE_COLOR, SPIKE_COUNT,
                    SPIKE_ENERGY_DAMAGE, SPIKE_HIT_COOLDOWN,
                    WORLD_WIDTH, WORLD_HEIGHT, AGENT_RADIUS)


class Spike:
    """
    A stationary hazard. Damages any agent that overlaps with it.
    A per-agent cooldown prevents rapid repeated damage from a single spike.
    """

    def __init__(self, x=None, y=None):
        margin = SPIKE_RADIUS + 20
        self.x = x if x is not None else np.random.uniform(margin, WORLD_WIDTH - margin)
        self.y = y if y is not None else np.random.uniform(margin, WORLD_HEIGHT - margin)
        self.radius = SPIKE_RADIUS
        # Maps agent.id -> frames_until_can_hit_again
        self._cooldowns: dict[int, int] = {}

    def tick_cooldowns(self):
        """Decrement all active cooldowns; remove expired ones."""
        expired = [aid for aid, t in self._cooldowns.items() if t <= 1]
        for aid in expired:
            del self._cooldowns[aid]
        for aid in self._cooldowns:
            self._cooldowns[aid] -= 1

    def try_hit(self, agent) -> bool:
        """
        Returns True and applies damage if agent is touching this spike
        and the cooldown has expired. Otherwise returns False.
        """
        dist = np.hypot(agent.x - self.x, agent.y - self.y)
        if dist >= AGENT_RADIUS + self.radius:
            return False
        if agent.id in self._cooldowns:
            return False  # still in cooldown
        # Hit!
        agent.energy = max(0.0, agent.energy - SPIKE_ENERGY_DAMAGE)
        self._cooldowns[agent.id] = SPIKE_HIT_COOLDOWN
        return True

    def draw(self, surface):
        cx, cy = int(self.x), int(self.y)
        r = self.radius

        # Dark fill
        pygame.draw.circle(surface, (80, 20, 20), (cx, cy), r)

        # Spike points radiating outward
        num_spikes = 8
        for i in range(num_spikes):
            angle = (2 * np.pi * i / num_spikes)
            # Inner point (on body edge)
            ix = cx + int(np.cos(angle) * r)
            iy = cy + int(np.sin(angle) * r)
            # Outer tip
            tip_len = r + 4
            ox = cx + int(np.cos(angle) * tip_len)
            oy = cy + int(np.sin(angle) * tip_len)
            pygame.draw.line(surface, SPIKE_COLOR, (ix, iy), (ox, oy), 2)

        # Center circle
        pygame.draw.circle(surface, SPIKE_COLOR, (cx, cy), max(2, r - 1), 1)


class SpikeManager:
    def __init__(self):
        self.items = [Spike() for _ in range(SPIKE_COUNT)]

    def update(self, agents):
        """Tick cooldowns and check all agent–spike collisions."""
        for spike in self.items:
            spike.tick_cooldowns()
            for agent in agents:
                if agent.alive:
                    spike.try_hit(agent)

    def draw(self, surface):
        for spike in self.items:
            spike.draw(surface)

    def count(self):
        return len(self.items)
