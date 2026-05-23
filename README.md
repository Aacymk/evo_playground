# EVO-SIM — 2D Evolutionary Emergence Sandbox

A minimal artificial life simulation. Agents with small neural network brains are born, eat food, avoid hazards, and die. The ones that survive longest and eat the most food produce the next generation — with slightly mutated weights. Over thousands of generations, useful movement behaviors emerge from nothing.

---

## How It Works

### The World

The simulation runs in a 1200×800 arena. Three things exist in it:

- **Food** — small green dots. Randomly placed. They respawn periodically. Eating one restores energy.
- **Spikes** — red hazard balls fixed in place. Contact with a spike immediately drains a chunk of energy. Spike positions are randomized at the start of each new generation so agents cannot memorize a fixed layout.
- **Agents** — the evolving organisms.

### Agents

Each agent is a circle with a direction indicator and an energy bar. Every agent independently tracks:

- **Position and heading** — where it is and which way it's facing
- **Energy** — drains at a flat rate each frame, plus a small additional cost proportional to movement speed. Eating food restores energy. Reaching zero energy kills the agent.
- **Age** — frame counter. Agents die of old age at `AGENT_MAX_AGE` frames even if they still have energy.
- **Statistics** — food eaten, distance traveled, spike hits — used for fitness scoring and logging.

### Sensors

Every frame, each agent reads 20 sensor channels that describe what it can perceive:

| Group | Channels | What it detects |
|---|---|---|
| Food vision | 0–2 | Distance and angle to nearest food in FOV cone; count of food visible |
| Spike vision | 3–5 | Distance and angle to nearest spike in FOV cone; count of spikes visible |
| Agent density | 6 | Number of other agents visible in FOV cone |
| Wall proximity | 7–11 | Nearest wall distance; individual N/S/E/W border distances |
| World position | 12–13 | Normalized X and Y coordinates in the arena |
| Touch | 14–16 | Binary contact with wall, another agent, food |
| Sound | 17 | Movement noise from nearby agents |
| Internal state | 18–19 | Current energy level; random noise |

All values are normalized to roughly [-1, 1] or [0, 1] before being fed to the brain.

The **vision cone** only detects objects within `VISION_RADIUS` pixels AND within the `VISION_FOV` angle centered on the agent's heading (120° by default). Objects behind the agent are invisible.

### The Brain

Each agent has a small feedforward neural network built with NumPy:

```
20 sensor inputs
      ↓  tanh
  hidden layer(s)       ← configurable via BRAIN_HIDDEN_LAYERS
      ↓  tanh
  2 outputs: [turn, speed]
```

- **Turn output** in [-1, 1]: multiplied by `AGENT_TURN_SPEED_MAX` to get radians/frame. Negative = left, positive = right.
- **Speed output** in [-1, 1]: remapped to [0, `AGENT_MOVE_SPEED_MAX`]. Agents always move forward.

The network weights **are the genome**. There is no learning during an agent's lifetime — weights are fixed at birth and only change through mutation between generations.

### Evolution

Evolution is **demand-driven**, not time-boxed. Each frame, if the alive population drops below `POPULATION_MIN`, a new wave of agents is spawned immediately. This means generation length is an emergent property of how long agents survive, not a fixed clock.

**Fitness** is computed when an agent dies:
```
fitness = 0.4 × (age / max_age)
        + 0.5 × min(1, food_eaten / 20)
        + 0.1 × min(1, distance_traveled / 5000)
```

Spike hits penalize fitness indirectly — they drain energy and shorten lifespan, which reduces the age and food components.

**Parent selection** draws from a rolling pool of the most recent 200 dead agents. The top `ELITE_FRACTION` (30% by default) are eligible as parents. Each child picks one parent at random from that elite set — there is no crossover, one parent per child.

**Mutation** applies independently to every weight:
```
if random() < MUTATION_RATE:
    weight += gaussian(0, MUTATION_STRENGTH)
    weight = clamp(weight, -3, 3)
```

Agents also inherit a slightly color-drifted version of their parent's color, so loosely related lineages share similar hues over time.

### Logging

The simulation writes a CSV row every `LOG_INTERVAL` frames — a live snapshot of all currently alive agents. This captures real population state rather than aggregating over a generation window. It also saves weight checkpoints and best-agent snapshots at configurable intervals.

---

## Installation

```bash
pip install pygame numpy
```

---

## Running

### Watch it live
```bash
python main.py
```
Opens a 1200×800 window. Click any agent to inspect its sensors and brain outputs in the right panel. Press **ESC** to quit.

### Headless fast training
```bash
python fast_train.py                          # runs until Ctrl+C
python fast_train.py --generations 10000      # stop at a generation target
python fast_train.py --generations 500 --run-id my_experiment
python fast_train.py --print-every 25         # progress line every 25 gens
```
No window, no FPS cap. Roughly 2× faster than visual mode. All output files are written identically.

### Replay a checkpoint
```bash
python replay.py --list runs/20240101_120000              # see what's available
python replay.py --run  runs/20240101_120000              # load latest checkpoint
python replay.py --checkpoint runs/.../chk_000500.npz    # load a specific one
python replay.py --checkpoint chk_000500.npz --freeze    # no mutation, pure playback
python replay.py --checkpoint chk_000500.npz --mutation-rate 0.05
```

---

## Output Files

```
runs/
└── 20240101_120000/
    ├── generations.csv       — one row every LOG_INTERVAL frames
    ├── checkpoints.csv       — summary row per checkpoint
    ├── best_agents/
    │   ├── frame_00000500.npz
    │   ├── frame_00000500.json
    │   └── ...
    └── checkpoints/
        ├── chk_000050.npz
        └── ...
```

### generations.csv — all columns

Each row is a **live snapshot** of the world at that frame. All agent statistics describe the agents currently alive at that moment.

| Column | Description |
|---|---|
| `generation` | Current generation number |
| `frame` | World frame when this row was written |
| `alive_population` | Number of agents alive right now |
| `total_agents_ever_born` | Cumulative births since run started |
| `total_agents_dead` | Cumulative deaths since run started |
| `current_oldest_alive_age` | Age in frames of the longest-surviving living agent |
| `average_alive_age` | Mean age of all living agents |
| `median_alive_age` | Median age of all living agents |
| `average_alive_fitness` | Mean fitness of living agents (computed at snapshot time) |
| `best_alive_fitness` | Highest fitness among living agents |
| `fitness_10th_percentile` | 10th percentile fitness |
| `fitness_50th_percentile` | Median fitness |
| `fitness_75th_percentile` | 75th percentile fitness |
| `fitness_90th_percentile` | 90th percentile fitness |
| `fitness_std` | Standard deviation of fitness |
| `behavioral_diversity_score` | 0–1. Mean pairwise distance between agents' average action vectors. Higher = more varied movement strategies |
| `action_entropy` | Shannon entropy of discretized (turn, speed) pairs across the population. Higher = strategies are more spread out |
| `mean_turn_output` | Mean of each agent's average turn output |
| `std_turn_output` | Std deviation of mean turn outputs across agents |
| `mean_speed_output` | Mean of each agent's average speed output |
| `std_speed_output` | Std deviation of mean speed outputs across agents |
| `pct_stationary` | Fraction of agents classified as mostly stationary |
| `pct_straight_runner` | Fraction that move mostly straight |
| `pct_erratic` | Fraction with high turn variance |
| `pct_left_turner` | Fraction that consistently turn left |
| `pct_right_turner` | Fraction that consistently turn right |
| `pct_balanced_explorer` | Fraction that don't fit a strong pattern |
| `births` | Agents born since the last log row |
| `deaths` | Agents that died since the last log row |
| `unique_parent_count` | Number of distinct parents used in the last spawn wave |
| `genome_diversity_score` | Mean pairwise L2 weight distance across living agents, normalized by expected random distance. 0 = clones, ~1 = as diverse as random init |
| `mean_weight_magnitude` | Mean absolute value of all weights in living agents |
| `weight_std` | Std deviation of all weights in living agents |
| `mean_pairwise_weight_distance` | Raw (unnormalized) mean L2 distance between genomes |
| `food_consumed` | Food eaten since the last log row |
| `spike_collisions` | Spike hits taken since the last log row |

---

## Configuration

Everything is in `config.py`.

### World
| Parameter | Default | Effect |
|---|---|---|
| `WORLD_WIDTH` | 1200 | Arena width in pixels |
| `WORLD_HEIGHT` | 800 | Arena height in pixels |
| `FPS_TARGET` | 60 | Target frame rate for `main.py` |

### Food
| Parameter | Default | Effect |
|---|---|---|
| `FOOD_COUNT_MAX` | 80 | Max food particles at once |
| `FOOD_ENERGY` | 40.0 | Energy gained per food eaten |
| `FOOD_RESPAWN_INTERVAL` | 30 | Frames between respawn attempts |
| `FOOD_RESPAWN_BATCH` | 3 | Particles added per respawn event |

### Spikes
| Parameter | Default | Effect |
|---|---|---|
| `SPIKE_COUNT` | 15 | Number of spikes. Repositioned each generation |
| `SPIKE_RADIUS` | 5 | Collision and visual radius |
| `SPIKE_ENERGY_DAMAGE` | 25.0 | Energy drained on contact |
| `SPIKE_HIT_COOLDOWN` | 30 | Frames before the same spike can hit the same agent again |

### Agents
| Parameter | Default | Effect |
|---|---|---|
| `AGENT_INITIAL_COUNT` | 30 | Starting population |
| `AGENT_INITIAL_ENERGY` | 80.0 | Energy at spawn |
| `AGENT_MAX_ENERGY` | 150.0 | Energy cap |
| `AGENT_ENERGY_DECAY` | 0.05 | Flat energy drain per frame |
| `AGENT_SPEED_DECAY_COST` | 0.01 | Additional drain per unit of speed. Total decay = `AGENT_ENERGY_DECAY + speed × AGENT_SPEED_DECAY_COST` |
| `AGENT_MAX_AGE` | 3000 | Frames before natural death |
| `AGENT_MOVE_SPEED_MAX` | 3.5 | Max pixels per frame |
| `AGENT_TURN_SPEED_MAX` | 0.12 | Max radians per frame |

### Sensors
| Parameter | Default | Effect |
|---|---|---|
| `VISION_RADIUS` | 150 | How far ahead agents can see |
| `VISION_FOV` | 2.094 (120°) | Cone width in radians |
| `SOUND_RADIUS` | 100 | Range for hearing nearby movement |
| `WALL_SENSOR_RADIUS` | 80 | Distance at which wall sensors start firing |

### Brain
| Parameter | Default | Notes |
|---|---|---|
| `BRAIN_INPUTS` | 20 | Must match sensor count. Don't change unless adding/removing sensors |
| `BRAIN_HIDDEN_LAYERS` | `[8]` | See below |
| `BRAIN_OUTPUTS` | 2 | Always 2: turn and speed |

```python
BRAIN_HIDDEN_LAYERS = []           # no hidden layers — direct input→output
BRAIN_HIDDEN_LAYERS = [8]          # one hidden layer, 8 neurons (default)
BRAIN_HIDDEN_LAYERS = [16, 8]      # two hidden layers
BRAIN_HIDDEN_LAYERS = [32, 16, 8]  # three hidden layers
```

Weight matrices resize automatically. No other changes needed.

### Evolution
| Parameter | Default | Effect |
|---|---|---|
| `POPULATION_MIN` | 15 | Alive count that triggers a new spawn wave |
| `SPAWN_BATCH_MAX` | 60 | Max agents per spawn wave. Does not cap live population — agents only arrive in waves, not mid-generation |
| `MUTATION_RATE` | 0.15 | Per-weight chance of mutation |
| `MUTATION_STRENGTH` | 0.3 | Std deviation of Gaussian noise on mutated weights |
| `ELITE_FRACTION` | 0.3 | Top fraction of dead pool eligible as parents |
| `FITNESS_LIFESPAN_W` | 0.4 | Fitness weight: lifespan |
| `FITNESS_FOOD_W` | 0.5 | Fitness weight: food eaten |
| `FITNESS_DISTANCE_W` | 0.1 | Fitness weight: distance traveled |

### Logging
| Parameter | Default | Effect |
|---|---|---|
| `LOG_INTERVAL` | 500 | Write one CSV row every N frames. `0` = disabled |
| `CHECKPOINT_INTERVAL` | 50 | Save full population weights every N generations. `0` = disabled |

---

## Suggested Experiments

**Dangerous world**
```python
SPIKE_COUNT = 35
SPIKE_ENERGY_DAMAGE = 30.0
FOOD_COUNT_MAX = 30
```

**Reward longevity**
```python
FITNESS_LIFESPAN_W = 0.7
FITNESS_FOOD_W = 0.2
FITNESS_DISTANCE_W = 0.1
```

**Faster, noisier evolution**
```python
MUTATION_STRENGTH = 0.6
AGENT_MAX_AGE = 1000
```

**Narrow vision**
```python
VISION_FOV = 0.698   # ~40 degrees
```

**Deeper network**
```python
BRAIN_HIDDEN_LAYERS = [32, 16, 8]
```

**Continue a good run with less mutation**
```bash
python replay.py --run runs/my_experiment --mutation-rate 0.05
```

**Watch trained behavior with no further evolution**
```bash
python replay.py --run runs/my_experiment --freeze
```

---

## Project Structure

```
evo_sim/
├── main.py               — Visual simulation entry point
├── fast_train.py         — Headless training entry point
├── replay.py             — Load and continue from a checkpoint
├── config.py             — All parameters
│
├── simulation/
│   ├── world.py          — Orchestrates each frame; drives logging
│   ├── agent.py          — Agent state, movement, energy, drawing
│   ├── brain.py          — NumPy neural network (variable hidden layers)
│   ├── sensors.py        — 20-channel sensor system
│   ├── food.py           — Food spawning and respawning
│   ├── spikes.py         — Spike hazards, collision, cooldown
│   └── evolution.py      — Fitness, selection, mutation, repopulation
│
├── visualization/
│   ├── renderer.py       — World drawing
│   └── ui.py             — Right-side inspection panel
│
├── evolog/
│   └── generation_logger.py  — Frame-interval CSV logging, checkpoints
│
└── utils/
    └── math_utils.py     — angle_diff, clamp, normalize_angle, etc.
```
