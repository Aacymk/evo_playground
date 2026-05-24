"""
evolog/generation_logger.py

Writes one CSV row every LOG_INTERVAL frames (a live snapshot of the world),
and saves full population weight checkpoints every CHECKPOINT_INTERVAL generations.

Output structure:
    runs/<run_id>/
    ├── generations.csv       — one row per LOG_INTERVAL frames
    ├── checkpoints.csv       — summary row per checkpoint
    ├── best_agents/
    │   ├── frame_000500.npz  — best alive agent's weights at that frame
    │   └── frame_000500.json — metadata sidecar
    └── checkpoints/
        ├── chk_000050.npz    — full population weights at gen 50
        └── ...
"""

import csv
import json
import os
from datetime import datetime

import numpy as np
from config import MUTATION_RATE, MUTATION_STRENGTH


CSV_COLUMNS = [
    # ── Identity ──────────────────────────────────────────────────────────────
    "generation",
    "frame",
    # ── Population counts ─────────────────────────────────────────────────────
    "alive_population",
    "total_agents_ever_born",
    "total_agents_dead",
    # ── Age of living agents ──────────────────────────────────────────────────
    "current_oldest_alive_age",
    "average_alive_age",
    "median_alive_age",
    # ── Fitness of living agents ──────────────────────────────────────────────
    "average_alive_fitness",
    "best_alive_fitness",
    "fitness_std",
    # ── Fitness of mature agents only (age >= FITNESS_MIN_AGE) ───────────────
    "mature_agent_count",
    "mature_average_fitness",
    "fitness_10th_percentile",
    "fitness_50th_percentile",
    "fitness_75th_percentile",
    "fitness_90th_percentile",
    # ── Behavioral metrics ────────────────────────────────────────────────────
    "behavioral_diversity_score",
    "action_entropy",
    "mean_turn_output",
    "std_turn_output",
    "mean_speed_output",
    "std_speed_output",
    # ── Behavioral archetypes (fraction of population) ────────────────────────
    "pct_stationary",
    "pct_straight_runner",
    "pct_erratic",
    "pct_left_turner",
    "pct_right_turner",
    "pct_balanced_explorer",
    # ── Generation-interval deltas ────────────────────────────────────────────
    "births",
    "deaths",
    "unique_parent_count",
    # ── Genome metrics ────────────────────────────────────────────────────────
    "genome_diversity_score",
    "mean_weight_magnitude",
    "weight_std",
    "mean_pairwise_weight_distance",
    # ── World-event counters (since last log row) ─────────────────────────────
    "food_consumed",
    "spike_collisions",
]

CHECKPOINT_CSV_COLUMNS = [
    "generation", "frame",
    "best_fitness", "average_fitness",
    "oldest_agent_age", "genome_diversity_score",
    "weights_file",
]


class GenerationLogger:
    def __init__(self, run_id: str = None, runs_dir: str = "runs"):
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id

        self.run_dir        = os.path.join(runs_dir, run_id)
        self.best_agent_dir = os.path.join(self.run_dir, "best_agents")
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.best_agent_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.csv_path        = os.path.join(self.run_dir, "generations.csv")
        self.checkpoint_path = os.path.join(self.run_dir, "checkpoints.csv")
        self._init_csv(self.csv_path, CSV_COLUMNS)
        self._init_csv(self.checkpoint_path, CHECKPOINT_CSV_COLUMNS)

        print(f"[Logger] Run directory: {self.run_dir}")

    # ── Public API ────────────────────────────────────────────────────────────

    def log_frame(self, world):
        """
        Called every LOG_INTERVAL frames from world.update().
        Captures a live snapshot of all currently alive agents.
        """
        agents   = world.agents
        evo      = world.evo_mgr

        if not agents:
            return

        row = self._compute_row(world.generation, world.frame, agents, evo)
        self._append_csv(self.csv_path, CSV_COLUMNS, row)

        # Save best alive agent's weights
        best = max(agents, key=lambda a: a.fitness())
        self._save_best_agent(world.frame, best, row)

    def save_checkpoint(self, generation: int, frame: int, dead_pool: list,
                        alive_count: int = 0):
        """
        Called every CHECKPOINT_INTERVAL generations from world.update().
        Saves elite dead_pool weights as the gene pool for replay/resume.
        alive_population is intentionally NOT saved — it is always deterministic
        from config (POPULATION_MIN + SPAWN_BATCH_MAX) and saving it would be
        misleading. Replay uses config to determine spawn size.
        """
        if not dead_pool:
            return

        tag      = f"chk_{generation:06d}"
        npz_path = os.path.join(self.checkpoint_dir, f"{tag}.npz")

        weight_data = {}
        for i, d in enumerate(dead_pool):
            Ws, bs = d['brain'].get_weights()
            for j, (W, b) in enumerate(zip(Ws, bs)):
                weight_data[f"agent{i}_W{j}"] = W
                weight_data[f"agent{i}_b{j}"] = b
            weight_data[f"agent{i}_num_layers"]  = np.array([len(Ws)])
            weight_data[f"agent{i}_layer_sizes"] = np.array(d['brain'].layer_sizes)
            weight_data[f"agent{i}_color"]       = np.array(d['color'])
            weight_data[f"agent{i}_fitness"]     = np.array([d['fitness']])

        # Save pool size so we know how many weight sets are in the file,
        # but this is the dead pool count, NOT alive_population.
        weight_data['pool_size']   = np.array([len(dead_pool)])
        weight_data['generation']  = np.array([generation])
        weight_data['frame']       = np.array([frame])
        np.savez(npz_path, **weight_data)

        fitnesses = [d['fitness'] for d in dead_pool]
        chk_row = {
            "generation":           generation,
            "frame":                frame,
            "best_fitness":         round(max(fitnesses), 4),
            "average_fitness":      round(float(np.mean(fitnesses)), 4),
            "oldest_agent_age":     max(d.get('age', 0) for d in dead_pool),
            "genome_diversity_score": round(_genome_diversity(dead_pool), 4),
            "weights_file":         f"checkpoints/{tag}.npz",
        }
        self._append_csv(self.checkpoint_path, CHECKPOINT_CSV_COLUMNS, chk_row)

        print(f"[Logger] Checkpoint: gen {generation} → {tag}.npz  "
              f"(pool={len(dead_pool)}, best_fit={chk_row['best_fitness']})")

    # ── Row computation ───────────────────────────────────────────────────────

    def _compute_row(self, generation, frame, agents, evo):
        # ── Age ───────────────────────────────────────────────────────────────
        ages = [a.age for a in agents]

        # ── Fitness of LIVING agents ──────────────────────────────────────────
        # Full population — used for average and best (always complete picture)
        fits     = [a.fitness() for a in agents]
        fits_arr = np.array(fits)

        # Mature agents only — used for percentiles so freshly-spawned agents
        # (whose fitness is near-zero by construction) don't drag every
        # percentile to zero after a repopulation wave.
        from config import FITNESS_MIN_AGE
        mature   = [a for a in agents if a.age >= FITNESS_MIN_AGE]
        if mature:
            mature_fits = np.array([a.fitness() for a in mature])
            p10  = round(float(np.percentile(mature_fits, 10)),  4)
            p50  = round(float(np.percentile(mature_fits, 50)),  4)
            p75  = round(float(np.percentile(mature_fits, 75)),  4)
            p90  = round(float(np.percentile(mature_fits, 90)),  4)
            mavg = round(float(np.mean(mature_fits)), 4)
        else:
            # No mature agents in this snapshot (e.g. very early in a run)
            p10 = p50 = p75 = p90 = mavg = None

        # ── Action outputs ────────────────────────────────────────────────────
        # Use mean action history vector per agent (sampled during their life)
        turn_means, speed_means = [], []
        for a in agents:
            if a.action_history:
                arr = np.array(a.action_history)
                turn_means.append(float(np.mean(arr[:, 0])))
                speed_means.append(float(np.mean(arr[:, 1])))
            else:
                turn_means.append(float(a.last_outputs[0]))
                speed_means.append(float(a.last_outputs[1]))

        turn_arr  = np.array(turn_means)
        speed_arr = np.array(speed_means)

        # ── Archetypes ────────────────────────────────────────────────────────
        archetype_counts = {
            "stationary": 0, "straight_runner": 0, "erratic": 0,
            "left_turner": 0, "right_turner": 0, "balanced_explorer": 0,
        }
        for t, s in zip(turn_means, speed_means):
            archetype_counts[_classify(t, s, np.std(turn_means))] += 1
        n = len(agents)
        pct = {k: round(v / n, 4) for k, v in archetype_counts.items()}

        # ── Genome metrics ────────────────────────────────────────────────────
        genome_div   = _genome_diversity_from_agents(agents)
        wt_mag, wt_std, mean_pw_dist = _weight_stats_from_agents(agents)

        # ── Behavioral diversity ──────────────────────────────────────────────
        beh_div = _behavioral_diversity(turn_means, speed_means)
        entropy = _action_entropy(turn_means, speed_means)

        return {
            "generation":                generation,
            "frame":                     frame,
            "alive_population":          n,
            "total_agents_ever_born":    evo.total_born,
            "total_agents_dead":         evo.total_deaths,
            "current_oldest_alive_age":  int(max(ages)),
            "average_alive_age":         round(float(np.mean(ages)), 1),
            "median_alive_age":          round(float(np.median(ages)), 1),
            "average_alive_fitness":     round(float(np.mean(fits_arr)), 4),
            "best_alive_fitness":        round(float(np.max(fits_arr)), 4),
            "fitness_std":               round(float(np.std(fits_arr)), 4),
            "mature_agent_count":        len(mature),
            "mature_average_fitness":    mavg,
            "fitness_10th_percentile":   p10,
            "fitness_50th_percentile":   p50,
            "fitness_75th_percentile":   p75,
            "fitness_90th_percentile":   p90,
            "behavioral_diversity_score":round(beh_div, 4),
            "action_entropy":            round(entropy, 4),
            "mean_turn_output":          round(float(np.mean(turn_arr)), 4),
            "std_turn_output":           round(float(np.std(turn_arr)), 4),
            "mean_speed_output":         round(float(np.mean(speed_arr)), 4),
            "std_speed_output":          round(float(np.std(speed_arr)), 4),
            "pct_stationary":            pct["stationary"],
            "pct_straight_runner":       pct["straight_runner"],
            "pct_erratic":               pct["erratic"],
            "pct_left_turner":           pct["left_turner"],
            "pct_right_turner":          pct["right_turner"],
            "pct_balanced_explorer":     pct["balanced_explorer"],
            "births":                    evo.gen_births,
            "deaths":                    evo.gen_deaths,
            "unique_parent_count":       evo.last_unique_parents,
            "genome_diversity_score":    round(genome_div, 4),
            "mean_weight_magnitude":     round(wt_mag, 4),
            "weight_std":                round(wt_std, 4),
            "mean_pairwise_weight_distance": round(mean_pw_dist, 4),
            "food_consumed":             evo.gen_food_consumed,
            "spike_collisions":          evo.gen_spike_collisions,
        }

    # ── CSV helpers ───────────────────────────────────────────────────────────

    def _init_csv(self, path, columns):
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=columns).writeheader()

    def _append_csv(self, path, columns, row):
        with open(path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=columns).writerow(row)

    # ── Best agent snapshot ───────────────────────────────────────────────────

    def _save_best_agent(self, frame: int, agent, stats_row: dict):
        tag       = f"frame_{frame:08d}"
        npz_path  = os.path.join(self.best_agent_dir, f"{tag}.npz")
        json_path = os.path.join(self.best_agent_dir, f"{tag}.json")

        Ws, bs = agent.brain.get_weights()
        save_dict = {f'W{i}': W for i, W in enumerate(Ws)}
        save_dict.update({f'b{i}': b for i, b in enumerate(bs)})
        save_dict['layer_sizes'] = np.array(agent.brain.layer_sizes)
        np.savez(npz_path, **save_dict)

        action_summary = _summarize_actions(agent.action_history)
        meta = {
            "frame":             frame,
            "fitness":           round(agent.fitness(), 4),
            "age":               agent.age,
            "food_eaten":        agent.food_eaten,
            "spike_hits":        agent.spike_hits,
            "distance_traveled": round(agent.distance_traveled, 1),
            "color":             list(agent.color),
            "action_summary":    action_summary,
            "snapshot_stats":    {k: stats_row[k] for k in [
                "generation", "alive_population", "average_alive_fitness",
                "best_alive_fitness", "behavioral_diversity_score",
            ]},
        }
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)


# ── Metric helpers ────────────────────────────────────────────────────────────

def _classify(mean_turn, mean_speed, pop_turn_std):
    """Assign a behavioral archetype label to one agent."""
    if mean_speed < -0.3:
        return "stationary"
    if pop_turn_std < 0.15 and abs(mean_turn) < 0.1:
        return "straight_runner"
    if pop_turn_std > 0.6:
        return "erratic"
    if mean_turn < -0.2:
        return "left_turner"
    if mean_turn > 0.2:
        return "right_turner"
    return "balanced_explorer"


def _behavioral_diversity(turn_means, speed_means):
    """Mean pairwise L2 distance between agents' average action vectors, norm to [0,1]."""
    if len(turn_means) < 2:
        return 0.0
    vecs = np.column_stack([turn_means, speed_means])
    n = len(vecs)
    idx = np.random.choice(n, size=min(n, 50), replace=False)
    s = vecs[idx]
    dists = [np.linalg.norm(s[i] - s[j])
             for i in range(len(s)) for j in range(i + 1, len(s))]
    return float(np.mean(dists) / 2.83) if dists else 0.0


def _action_entropy(turn_means, speed_means):
    """
    Shannon entropy of discretized (turn, speed) pairs.
    Higher = more spread of behaviors across the population.
    """
    if len(turn_means) < 2:
        return 0.0
    # Discretize into a 5×5 grid
    t_bins = np.digitize(turn_means,  np.linspace(-1, 1, 6))
    s_bins = np.digitize(speed_means, np.linspace(-1, 1, 6))
    keys   = list(zip(t_bins, s_bins))
    counts = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    total = len(keys)
    probs = np.array([v / total for v in counts.values()])
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def _flat_weights_from_agent(agent):
    Ws, bs = agent.brain.get_weights()
    parts  = [W.ravel() for W in Ws] + [b.ravel() for b in bs]
    return np.concatenate(parts)


def _flat_weights_from_dict(d):
    Ws, bs = d['brain'].get_weights()
    parts  = [W.ravel() for W in Ws] + [b.ravel() for b in bs]
    return np.concatenate(parts)


def _genome_diversity_from_agents(agents):
    """Mean pairwise L2 weight distance, normalized by expected random distance."""
    if len(agents) < 2:
        return 0.0
    vecs = [_flat_weights_from_agent(a) for a in agents]
    return _mean_pairwise_dist_normalized(vecs)


def _genome_diversity(pool):
    """Same but from dead_pool dicts."""
    if len(pool) < 2:
        return 0.0
    vecs = [_flat_weights_from_dict(d) for d in pool]
    return _mean_pairwise_dist_normalized(vecs)


def _mean_pairwise_dist_normalized(vecs):
    n   = len(vecs)
    idx = np.random.choice(n, size=min(n, 30), replace=False)
    s   = [vecs[i] for i in idx]
    dists = [np.linalg.norm(s[i] - s[j])
             for i in range(len(s)) for j in range(i + 1, len(s))]
    if not dists:
        return 0.0
    expected = float(np.sqrt(len(vecs[0]) * 2 / 3))
    return float(np.clip(np.mean(dists) / expected, 0, 2))


def _weight_stats_from_agents(agents):
    """Returns (mean_abs, std, mean_pairwise_dist) for the live population."""
    if not agents:
        return 0.0, 0.0, 0.0
    vecs   = [_flat_weights_from_agent(a) for a in agents]
    all_w  = np.concatenate(vecs)
    mag    = float(np.mean(np.abs(all_w)))
    std    = float(np.std(all_w))

    n   = len(vecs)
    idx = np.random.choice(n, size=min(n, 30), replace=False)
    s   = [vecs[i] for i in idx]
    dists = [float(np.linalg.norm(s[i] - s[j]))
             for i in range(len(s)) for j in range(i + 1, len(s))]
    mpd = float(np.mean(dists)) if dists else 0.0
    return mag, std, mpd


def _summarize_actions(action_history):
    if not action_history:
        return {"note": "no history"}
    arr   = np.array(action_history)
    turns = arr[:, 0]
    spds  = arr[:, 1]
    mt, st = float(np.mean(turns)), float(np.std(turns))
    ms, ss = float(np.mean(spds)),  float(np.std(spds))

    if ms < -0.3:          archetype = "mostly_stationary"
    elif st < 0.15 and abs(mt) < 0.1: archetype = "straight_runner"
    elif st > 0.6:         archetype = "erratic_wanderer"
    elif mt > 0.2:         archetype = "consistent_right_turner"
    elif mt < -0.2:        archetype = "consistent_left_turner"
    else:                  archetype = "balanced_explorer"

    return {
        "samples":    len(action_history),
        "mean_turn":  round(mt, 4), "std_turn":  round(st, 4),
        "mean_speed": round(ms, 4), "std_speed": round(ss, 4),
        "archetype":  archetype,
    }
