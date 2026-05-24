import numpy as np
from config import POPULATION_MIN, SPAWN_BATCH_MAX, ELITE_FRACTION
from simulation.agent import Agent


class EvolutionManager:
    """
    Tracks fitness of dead agents and spawns new ones when population is low.
    Simple elitist selection + mutation — no crossover.

    Logging is handled externally by GenerationLogger via World.update(),
    not here — so this class stays focused on selection logic only.
    """

    def __init__(self):
        self.generation  = 1
        self.dead_pool   = []   # rolling pruned pool used for parent selection
        self.total_born  = 0
        self.total_deaths = 0

        # Cumulative counters reset each generation for delta tracking
        self.gen_births  = 0
        self.gen_deaths  = 0
        self.gen_food_consumed   = 0
        self.gen_spike_collisions = 0

        # Track unique parents used in last spawn wave
        self.last_unique_parents = 0

    def record_death(self, agent):
        entry = {
            'fitness':           agent.fitness(),
            'brain':             agent.brain.clone(),
            'color':             agent.color,
            'generation':        agent.generation,
            'age':               agent.age,
            'food_eaten':        agent.food_eaten,
            'distance_traveled': agent.distance_traveled,
            'spike_hits':        agent.spike_hits,
            'last_outputs':      agent.last_outputs.tolist(),
            'action_history':    list(agent.action_history),
        }
        self.dead_pool.append(entry)
        if len(self.dead_pool) > 200:
            self.dead_pool.sort(key=lambda d: d['fitness'])
            self.dead_pool = self.dead_pool[50:]

        self.total_deaths += 1
        self.gen_deaths   += 1
        self.gen_food_consumed    += agent.food_eaten
        self.gen_spike_collisions += agent.spike_hits

    def maybe_repopulate(self, agents, spike_mgr=None):
        """
        If alive population is below POPULATION_MIN, spawn a new wave of children.
        Reshuffles spikes for the new generation.
        Returns list of new Agent objects (may be empty).
        """
        if len(agents) >= POPULATION_MIN:
            return []

        # Spawn exactly enough to bring alive count up to POPULATION_MIN + SPAWN_BATCH_MAX.
        # No overshoot: the target ceiling is the only hard limit.
        # This also prevents multi-frame re-firing from accumulating extra batches —
        # once the first wave lands, alive >= POPULATION_MIN and the guard above exits.
        target      = POPULATION_MIN + SPAWN_BATCH_MAX
        spawn_count = min(target - len(agents), SPAWN_BATCH_MAX)
        if spawn_count <= 0:
            return []

        # Reshuffle spikes so layout varies each generation
        if spike_mgr is not None:
            spike_mgr.reshuffle()
        forbidden = spike_mgr.positions() if spike_mgr is not None else []

        # Increment generation FIRST so children are tagged to the new generation
        self.generation += 1

        if len(self.dead_pool) == 0:
            new_agents = [
                Agent(generation=self.generation, forbidden_positions=forbidden)
                for _ in range(spawn_count)
            ]
        else:
            sorted_pool = sorted(self.dead_pool, key=lambda d: d['fitness'], reverse=True)
            elite_count = max(1, int(len(sorted_pool) * ELITE_FRACTION))
            parents     = sorted_pool[:elite_count]

            parent_ids_used = set()
            new_agents = []
            for _ in range(spawn_count):
                idx = np.random.randint(len(parents))
                parent_ids_used.add(idx)
                parent      = parents[idx]
                child_brain = parent['brain'].mutate()
                child_color = _mutate_color(parent['color'])
                new_agents.append(Agent(
                    brain=child_brain, color=child_color,
                    generation=self.generation,
                    forbidden_positions=forbidden,
                ))
            self.last_unique_parents = len(parent_ids_used)

        self.total_born  += len(new_agents)
        self.gen_births  += len(new_agents)
        return new_agents

    def reset_gen_counters(self):
        """Call after logging a generation row to get per-interval deltas."""
        self.gen_births           = 0
        self.gen_deaths           = 0
        self.gen_food_consumed    = 0
        self.gen_spike_collisions = 0


def _mutate_color(color):
    r, g, b = color
    delta = np.random.randint(-20, 21, 3)
    return (
        int(np.clip(r + delta[0], 40, 255)),
        int(np.clip(g + delta[1], 40, 255)),
        int(np.clip(b + delta[2], 40, 255)),
    )
