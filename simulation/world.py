import numpy as np
from simulation.agent import Agent
from simulation.food import FoodManager
from simulation.spikes import SpikeManager
from simulation.evolution import EvolutionManager
from simulation.sensors import compute_sensors
from config import (AGENT_INITIAL_COUNT, WORLD_WIDTH, WORLD_HEIGHT,
                    AGENT_RADIUS, FOOD_RADIUS, LOG_INTERVAL, CHECKPOINT_INTERVAL)


class World:
    def __init__(self, logger=None):
        self.food_mgr  = FoodManager()
        self.spike_mgr = SpikeManager()
        self.evo_mgr   = EvolutionManager()
        self.logger    = logger
        self.agents    = [Agent() for _ in range(AGENT_INITIAL_COUNT)]
        self.evo_mgr.total_born = AGENT_INITIAL_COUNT
        self.frame = 0

        # Track generation at last checkpoint so we fire once per gen interval
        self._last_checkpoint_gen = -1

    # ── Main update ──────────────────────────────────────────────────────────

    def update(self):
        self.frame += 1
        self.food_mgr.update()

        food_list  = self.food_mgr.items
        spike_list = self.spike_mgr.items

        # Sense → Think → Act
        for agent in self.agents:
            sensors = compute_sensors(agent, food_list, self.agents, spike_list)
            agent.update(sensors)

        # Food consumption
        for agent in self.agents:
            if not agent.alive:
                continue
            for food in food_list:
                if not food.alive:
                    continue
                if np.hypot(agent.x - food.x, agent.y - food.y) < AGENT_RADIUS + FOOD_RADIUS:
                    agent.eat(food)

        # Spike damage
        for spike in spike_list:
            spike.tick_cooldowns()
            for agent in self.agents:
                if agent.alive:
                    hit = spike.try_hit(agent)
                    if hit:
                        agent.spike_hits += 1

        # Handle deaths
        for agent in self.agents:
            if not agent.alive:
                self.evo_mgr.record_death(agent)

        self.agents = [a for a in self.agents if a.alive]

        # Repopulate
        new_agents = self.evo_mgr.maybe_repopulate(
            self.agents, spike_mgr=self.spike_mgr
        )
        self.agents.extend(new_agents)

        # ── Frame-interval logging ────────────────────────────────────────────
        if self.logger and LOG_INTERVAL > 0 and self.frame % LOG_INTERVAL == 0:
            self.logger.log_frame(self)
            self.evo_mgr.reset_gen_counters()

        # ── Generation-interval checkpoint ────────────────────────────────────
        if (self.logger and CHECKPOINT_INTERVAL > 0
                and self.evo_mgr.generation != self._last_checkpoint_gen
                and self.evo_mgr.generation % CHECKPOINT_INTERVAL == 0):
            self._last_checkpoint_gen = self.evo_mgr.generation
            self.logger.save_checkpoint(self.evo_mgr.generation, self.frame,
                                        self.evo_mgr.dead_pool)

    # ── Accessors ────────────────────────────────────────────────────────────

    def agent_at(self, mx, my):
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

    @property
    def spike_count(self):
        return self.spike_mgr.count()
