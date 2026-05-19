import numpy as np
from simulation.agent import Agent
from simulation.food import FoodManager
from simulation.evolution import EvolutionManager
from simulation.sensors import compute_sensors
from config import (AGENT_INITIAL_COUNT, WORLD_WIDTH, WORLD_HEIGHT,
                    AGENT_RADIUS, FOOD_RADIUS)


class World:
    def __init__(self):
        self.food_mgr = FoodManager()
        self.evo_mgr = EvolutionManager()
        self.agents = [Agent() for _ in range(AGENT_INITIAL_COUNT)]
        self.evo_mgr.total_born = AGENT_INITIAL_COUNT
        self.frame = 0

    # ── Main update ──────────────────────────────────────────────────────────

    def update(self):
        self.frame += 1
        self.food_mgr.update()

        food_list = self.food_mgr.items

        # Sense → Think → Act
        for agent in self.agents:
            sensors = compute_sensors(agent, food_list, self.agents)
            agent.update(sensors)

        # Food consumption
        for agent in self.agents:
            if not agent.alive:
                continue
            for food in food_list:
                if not food.alive:
                    continue
                dist = np.hypot(agent.x - food.x, agent.y - food.y)
                if dist < AGENT_RADIUS + FOOD_RADIUS:
                    agent.eat(food)

        # Handle deaths
        for agent in self.agents:
            if not agent.alive:
                self.evo_mgr.record_death(agent)

        self.agents = [a for a in self.agents if a.alive]

        # Repopulate
        new_agents = self.evo_mgr.maybe_repopulate(self.agents)
        self.agents.extend(new_agents)

    # ── Accessors ────────────────────────────────────────────────────────────

    def agent_at(self, mx, my):
        """Return the agent closest to (mx, my) within click radius, or None."""
        for agent in self.agents:
            if np.hypot(agent.x - mx, agent.y - my) < AGENT_RADIUS + 8:
                return agent
        return None

    @property
    def generation(self):
        return self.evo_mgr.generation

    @property
    def population(self):
        return len(self.agents)

    @property
    def food_count(self):
        return self.food_mgr.count()
