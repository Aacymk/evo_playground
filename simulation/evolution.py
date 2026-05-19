import numpy as np
from config import (POPULATION_MIN, POPULATION_MAX, ELITE_FRACTION,
                    WORLD_WIDTH, WORLD_HEIGHT)
from simulation.agent import Agent


class EvolutionManager:
    """
    Tracks fitness of dead agents and spawns new ones when population is low.
    Simple elitist selection + mutation — no crossover.
    """

    def __init__(self):
        self.generation = 1
        self.dead_pool = []     # (fitness, brain, color, generation) of past agents
        self.total_born = 0

    def record_death(self, agent):
        self.dead_pool.append({
            'fitness': agent.fitness(),
            'brain': agent.brain.clone(),
            'color': agent.color,
            'generation': agent.generation,
        })
        # Keep pool bounded
        if len(self.dead_pool) > 200:
            self.dead_pool.sort(key=lambda d: d['fitness'])
            self.dead_pool = self.dead_pool[50:]  # prune weak

    def maybe_repopulate(self, agents):
        """
        If alive population is below POPULATION_MIN, spawn children from
        the fittest dead agents.
        """
        alive_count = len(agents)
        if alive_count >= POPULATION_MIN:
            return []

        spawn_count = min(POPULATION_MIN - alive_count + 5,
                          POPULATION_MAX - alive_count)
        if spawn_count <= 0:
            return []

        if len(self.dead_pool) == 0:
            # No history yet — spawn random agents
            new_agents = [Agent(generation=self.generation) for _ in range(spawn_count)]
            self.generation += 1
            self.total_born += len(new_agents)
            return new_agents

        # Select top performers
        sorted_pool = sorted(self.dead_pool, key=lambda d: d['fitness'], reverse=True)
        elite_count = max(1, int(len(sorted_pool) * ELITE_FRACTION))
        parents = sorted_pool[:elite_count]

        new_agents = []
        self.generation += 1

        for _ in range(spawn_count):
            parent = parents[np.random.randint(len(parents))]
            child_brain = parent['brain'].mutate()
            # Slightly shift color toward parent's color
            child_color = _mutate_color(parent['color'])
            child = Agent(brain=child_brain, color=child_color,
                          generation=self.generation)
            new_agents.append(child)

        self.total_born += len(new_agents)
        return new_agents


def _mutate_color(color):
    """Small random shift in RGB, clamped."""
    r, g, b = color
    delta = np.random.randint(-20, 21, 3)
    r = int(np.clip(r + delta[0], 40, 255))
    g = int(np.clip(g + delta[1], 40, 255))
    b = int(np.clip(b + delta[2], 40, 255))
    return (r, g, b)
