# EVO-SIM — 2D Evolutionary Emergence Sandbox

A minimal artificial life simulation where simple agents survive, eat food, avoid hazards, and evolve movement behaviors through mutation over generations. No reinforcement learning, no NEAT, no ML frameworks — just NumPy math, Pygame visuals, and natural selection doing its thing.

---

## Quick Start

```bash
pip install pygame numpy
python main.py
```

Click any agent to inspect its live sensor readings, brain outputs, and stats. Press **ESC** or close the window to quit.

---

## What You're Looking At

The world is a 1200×800 dark arena with three kinds of objects:

**Food** — small green dots scattered randomly. They respawn periodically. Eating one restores energy.

**Spikes** — red spiky hazards fixed in place throughout the world. Any agent that runs into one loses a chunk of energy instantly. They don't move and don't disappear. Agents that learn to avoid them survive longer.

**Agents** — colored circles with a direction arrow and an energy bar above them. Every agent runs its own neural network every frame, deciding how fast to move and which way to turn based on what its 20 sensors are detecting. They bounce off walls. They eat food by overlapping with it. They die when energy hits zero or they get too old. When the population drops low enough, the best performers from history are used to spawn the next wave — with their weights slightly mutated.

Over generations, agents that wander randomly and blunder into spikes die quickly. Agents whose weights happen to steer toward food and away from spikes survive longer and produce more descendants. Useful behaviors accumulate through this selection pressure.

---

## Project Structure

```
evo_sim/
├── main.py                   — Main loop, event handling, clock
├── config.py                 — Every tunable parameter in one file
│
├── simulation/
│   ├── world.py              — Orchestrates all updates each frame
│   ├── agent.py              — Agent state, movement, drawing
│   ├── brain.py              — NumPy neural network
│   ├── sensors.py            — All 20 sensor channels
│   ├── food.py               — Food particles and respawning
│   ├── spikes.py             — Spike hazards and collision/damage logic
│   └── evolution.py          — Fitness tracking, selection, spawning
│
├── visualization/
│   ├── renderer.py           — World drawing, background grid
│   └── ui.py                 — Right-side inspection panel
│
└── utils/
    └── math_utils.py         — angle_diff, clamp, heading_vector, etc.
```

The simulation logic lives entirely in `simulation/`. The visualization layer in `visualization/` only reads state — it never writes to it. This makes it straightforward to add a headless mode or swap the renderer later.

---

## The Simulation Loop

Each frame runs in this order (`world.py`):

1. **Food update** — remove eaten food, respawn a small batch if below max
2. **Sense** — for each agent, compute all 20 sensor values from current world state
3. **Think** — feed sensors into the agent's neural network, get turn + speed outputs
4. **Move** — apply outputs to position and angle, bounce off walls
5. **Eat** — check overlaps between agents and food, transfer energy
6. **Spike damage** — check overlaps between agents and spikes, drain energy (with per-agent cooldown)
7. **Decay** — drain a small amount of energy each frame; moving faster costs slightly more
8. **Death** — mark agents as dead if energy ≤ 0 or age ≥ max_age
9. **Record** — log dead agents' fitness into the evolution pool
10. **Repopulate** — if alive count dropped below `POPULATION_MIN`, spawn children from top performers
11. **Render** — draw everything, then draw the UI panel on top

---

## Agents

Each agent tracks:

| Property | Description |
|---|---|
| `x, y` | Position in world space |
| `angle` | Current heading in radians |
| `speed` | Current movement speed (set each frame by brain output) |
| `energy` | Starts at 80, capped at 150, drained every frame |
| `age` | Frame counter; natural death at 3000 frames by default |
| `brain` | Neural network whose weights are the genome |
| `color` | RGB color inherited (with small random drift) from parent |
| `food_eaten` | Running count; used in fitness scoring |
| `distance_traveled` | Accumulated movement; used in fitness scoring |
| `spike_hits` | Running count of spike collisions; displayed in UI |

Agents lose a flat `AGENT_ENERGY_DECAY` energy per frame, plus a small additional cost proportional to speed. Standing still is the cheapest strategy, but agents that don't move can't reach food — selection pushes toward efficient movement rather than either extreme.

Wall collisions cause a bounce (the perpendicular velocity component is negated). No energy penalty for hitting a wall.

---

## Spike Hazards

File: `simulation/spikes.py`

Spikes are stationary objects placed randomly at world creation. They never move or respawn. Any agent whose body overlaps a spike loses `SPIKE_ENERGY_DAMAGE` (default 25) energy immediately.

To prevent a single collision from draining an agent in rapid successive hits, each spike maintains a per-agent cooldown dictionary. Once an agent is hit, that spike cannot damage it again for `SPIKE_HIT_COOLDOWN` frames (default 30). The cooldown is per-spike-per-agent, so running into a different spike is still a fresh hit.

Spikes are drawn below food and agents in the render order so they read as embedded terrain rather than foreground objects.

---

## Sensor System

Every frame, `sensors.py` builds a 20-element NumPy float32 array for each agent. All values are normalized before being passed to the brain.

| Index | Name | Range | Description |
|---|---|---|---|
| 0 | `food_dist` | 0–1 | Proximity of nearest food in FOV. 0 = nothing seen, 1 = touching |
| 1 | `food_angle` | -1–1 | Relative angle to that food, normalized by half-FOV. Negative = left, positive = right |
| 2 | `food_count_fov` | 0–1 | Number of food items visible in cone, normalized to a cap of 10 |
| 3 | `spike_dist` | 0–1 | Proximity of nearest spike in FOV. 0 = nothing seen, 1 = touching |
| 4 | `spike_angle` | -1–1 | Relative angle to that spike, same normalization as food_angle |
| 5 | `spike_count_fov` | 0–1 | Number of spikes visible in cone, normalized to a cap of 5 |
| 6 | `agent_count_fov` | 0–1 | Number of other agents visible in cone, normalized to a cap of 10 |
| 7 | `wall_nearest` | 0–1 | Proximity to the closest wall in any direction. 0 = far, 1 = touching |
| 8 | `wall_north` | 0–1 | Distance to the top border, normalized by half world height |
| 9 | `wall_south` | 0–1 | Distance to the bottom border, normalized by half world height |
| 10 | `wall_east` | 0–1 | Distance to the right border, normalized by half world width |
| 11 | `wall_west` | 0–1 | Distance to the left border, normalized by half world width |
| 12 | `pos_x` | 0–1 | Normalized X position in the world. 0 = left edge, 1 = right edge |
| 13 | `pos_y` | 0–1 | Normalized Y position in the world. 0 = top edge, 1 = bottom edge |
| 14 | `touch_wall` | 0 or 1 | Binary: agent is currently at a wall boundary |
| 15 | `touch_agent` | 0 or 1 | Binary: another agent is within ~2 body radii |
| 16 | `touch_food` | 0 or 1 | Binary: body is overlapping a food particle right now |
| 17 | `sound` | 0–1 | Sum of movement noise from nearby agents within `SOUND_RADIUS`. Faster neighbors = louder |
| 18 | `energy_lvl` | 0–1 | Current energy divided by max energy. Agents "know" how hungry they are |
| 19 | `noise` | ±0.2 | Pure random jitter added every frame. Prevents fully deterministic behavior |

**Vision cone:** Sensors 0–6 all use the same cone check — a target is only detected if it is within `VISION_RADIUS` pixels AND within the `VISION_FOV` half-angle (120° by default) centered on the agent's current heading. Objects behind the agent are invisible.

**Directional wall distances (8–11):** Unlike the nearest-wall sensor (7) which only reports the single closest wall, these four channels give independent readings for each cardinal direction. This lets the brain distinguish "close to the top edge" from "close to the bottom edge" and develop direction-specific avoidance behaviors.

**World position (12–13):** Raw normalized X/Y coordinates. Allows behaviors that depend on location in the arena, such as preferring the center or learning that food clusters near certain areas.

**Sound (17):** Each agent emits noise proportional to its movement speed. Nearby agents accumulate this into a single intensity reading. An agent surrounded by fast-moving neighbors gets a high reading; a lone agent in an empty area gets near zero.

---

## The Neural Network (Brain)

File: `simulation/brain.py`

A two-layer feedforward network built with NumPy matrices. No PyTorch, no TensorFlow, no autograd.

```
Architecture:

  20 inputs
      ↓  (W1: 8×20, b1: 8)
  8 hidden neurons   ← tanh activation
      ↓  (W2: 2×8, b2: 2)
  2 outputs          ← tanh activation
```

**Activation function: `tanh` on both layers.**

`tanh` outputs values in (-1, 1), which maps cleanly to signed quantities like turn direction and speed. It's smooth and zero-centered, which also means it would behave well if you ever want to add a learning rule on top.

The forward pass:

```python
h   = tanh(W1 @ sensors + b1)   # hidden layer
out = tanh(W2 @ h + b2)         # output layer
```

**Output mapping:**

| Output | Raw range | Interpretation |
|---|---|---|
| `out[0]` (turn) | -1 to 1 | Multiplied by `AGENT_TURN_SPEED_MAX` (0.12 rad/frame). Negative = turn left, positive = turn right |
| `out[1]` (speed) | -1 to 1 | Remapped to (0, `AGENT_MOVE_SPEED_MAX`) via `(out + 1) / 2 * max_speed`. Always moves forward |

There is **no backpropagation**. Weights are never updated during an agent's lifetime. The only way the network ever changes is through mutation at reproduction.

Total weight count: `(20×8 + 8) + (8×2 + 2) = 186` parameters per agent.

---

## Evolution & Reproduction

File: `simulation/evolution.py`

### When reproduction happens

Reproduction is **not generational** — it's continuous and demand-driven. The population is checked every frame. If living agents drop below `POPULATION_MIN` (default 15), a batch of children is spawned immediately to bring the count back up, slightly overshooting (`POPULATION_MIN - alive + 5` children, capped at `POPULATION_MAX`).

### Fitness scoring

When an agent dies, its fitness is computed from three components:

```
fitness = 0.4 × (age / max_age)
        + 0.5 × min(1.0, food_eaten / 20)
        + 0.1 × min(1.0, distance_traveled / 5000)
```

Food eaten is weighted most heavily (0.5) because it most directly represents successful foraging. Lifespan (0.4) rewards any survival strategy, including efficient spike avoidance. Distance (0.1) gives a small nudge toward exploration. All three weights are tunable in `config.py`.

Note that `spike_hits` does not appear directly in the fitness formula — it penalizes fitness indirectly because getting hit drains energy and shortens lifespan, which reduces both the age and food components.

### Parent selection

Dead agents accumulate in a `dead_pool` (capped at 200 entries; the weakest are pruned when it overflows). When spawning, the pool is sorted by fitness and the top `ELITE_FRACTION` (default 30%) form the eligible parent set. Each child picks one parent at random from that elite set.

There is **no crossover**. Each child has exactly one parent. The child's genome is the parent's weights plus mutation.

### Mutation

Each weight is independently mutated:

```python
for each weight w:
    if random() < MUTATION_RATE:           # 15% chance per weight
        w += gaussian(0, MUTATION_STRENGTH)  # std dev = 0.3
        w = clamp(w, -3, 3)                # prevent runaway weights
```

This applies to all four parameter arrays: `W1`, `b1`, `W2`, `b2`. Biases mutate by the same rule.

The generation counter increments each time a new wave of children is spawned. Agents also inherit a slightly color-drifted version of their parent's RGB color (±20 per channel), so loosely related lineages tend to share similar hues over time.

---

## All Tunable Parameters

Everything lives in `config.py`. No other file needs to be touched for parameter changes.

### World
| Parameter | Default | Effect |
|---|---|---|
| `WORLD_WIDTH` | 1200 | Arena width in pixels |
| `WORLD_HEIGHT` | 800 | Arena height in pixels |
| `FPS_TARGET` | 60 | Target frame rate |

### Food
| Parameter | Default | Effect |
|---|---|---|
| `FOOD_COUNT_MAX` | 80 | Max food in the world at once. Lower = more scarcity pressure |
| `FOOD_ENERGY` | 40.0 | Energy gained per food eaten |
| `FOOD_RESPAWN_INTERVAL` | 30 | Frames between respawn attempts |
| `FOOD_RESPAWN_BATCH` | 3 | Food particles added per respawn event |

### Spikes
| Parameter | Default | Effect |
|---|---|---|
| `SPIKE_COUNT` | 15 | Number of spike hazards placed at startup |
| `SPIKE_RADIUS` | 5 | Collision and visual radius (same as food by default) |
| `SPIKE_ENERGY_DAMAGE` | 25.0 | Energy drained per hit |
| `SPIKE_HIT_COOLDOWN` | 30 | Frames before the same spike can damage the same agent again |

### Agents
| Parameter | Default | Effect |
|---|---|---|
| `AGENT_INITIAL_COUNT` | 30 | Starting population |
| `AGENT_INITIAL_ENERGY` | 80.0 | Energy each agent starts with |
| `AGENT_MAX_ENERGY` | 150.0 | Energy cap |
| `AGENT_ENERGY_DECAY` | 0.05 | Flat energy drain per frame |
| `AGENT_MAX_AGE` | 3000 | Max frames before natural death |
| `AGENT_MOVE_SPEED_MAX` | 3.5 | Pixels per frame at full output |
| `AGENT_TURN_SPEED_MAX` | 0.12 | Radians per frame at full turn output |

### Sensors
| Parameter | Default | Effect |
|---|---|---|
| `VISION_RADIUS` | 150 | How far ahead agents can see into the cone |
| `VISION_FOV` | 2.094 (120°) | Width of the vision cone in radians. Narrower = harder to spot things |
| `SOUND_RADIUS` | 100 | Range at which agents can hear each other move |
| `WALL_SENSOR_RADIUS` | 80 | Distance at which the nearest-wall sensor starts firing |

### Brain
| Parameter | Default | Effect |
|---|---|---|
| `BRAIN_INPUTS` | 20 | Size of the sensor vector — update if you add/remove sensors |
| `BRAIN_HIDDEN` | 8 | Hidden layer neuron count. More = more expressive behavior possible |
| `BRAIN_OUTPUTS` | 2 | Always 2: turn and speed |

### Evolution
| Parameter | Default | Effect |
|---|---|---|
| `POPULATION_MIN` | 15 | Population floor that triggers reproduction |
| `POPULATION_MAX` | 60 | Hard cap on population |
| `MUTATION_RATE` | 0.15 | Per-weight probability of mutation (0–1) |
| `MUTATION_STRENGTH` | 0.3 | Std deviation of Gaussian noise applied to mutated weights |
| `ELITE_FRACTION` | 0.3 | Fraction of dead pool eligible as parents |
| `FITNESS_LIFESPAN_W` | 0.4 | Fitness weight for lifespan |
| `FITNESS_FOOD_W` | 0.5 | Fitness weight for food eaten |
| `FITNESS_DISTANCE_W` | 0.1 | Fitness weight for distance traveled |

---

## Suggested Experiments

**Dangerous world — lots of spikes, scarce food**
```python
SPIKE_COUNT = 35
SPIKE_ENERGY_DAMAGE = 30.0
FOOD_COUNT_MAX = 30
```

**Make spikes hurt less but linger longer (punishment is slower)**
```python
SPIKE_ENERGY_DAMAGE = 10.0
SPIKE_HIT_COOLDOWN = 5
```

**Reward spike avoidance explicitly by making lifespan matter more**
```python
FITNESS_LIFESPAN_W = 0.7
FITNESS_FOOD_W = 0.2
FITNESS_DISTANCE_W = 0.1
```

**Fast evolution with high mutation**
```python
MUTATION_STRENGTH = 0.6
AGENT_MAX_AGE = 1000
FOOD_COUNT_MAX = 40
```

**Narrow vision — precise steering required**
```python
VISION_FOV = 0.698   # ~40 degrees
VISION_RADIUS = 100
```

**Bigger brains**
```python
BRAIN_HIDDEN = 16
```
No other changes needed — weight matrices resize automatically.

---

## What to Add Next

The architecture is intentionally minimal so extensions slot in cleanly:

- **Predators** — agents with a separate genome that gain fitness by catching prey. Add a predator class in `simulation/`, a "predator nearby" sensor channel, and a second fitness function.
- **Pheromones** — a 2D float grid in `world.py` that agents write to when eating or moving. Add a sensor reading the local gradient so agents can follow or avoid chemical trails.
- **Memory / recurrence** — pass the previous hidden layer activations back as extra inputs (Elman network). Add a `hidden_state` field to `Agent` and thread it through `brain.forward()`.
- **Crossover** — in `evolution.py`, select two parents and blend their weight arrays (e.g. uniform crossover) before mutating.
- **Speciation** — cluster agents by weight-space similarity before selection so diverse lineages can't be outcompeted by a single dominant strategy.
- **Communication** — add a broadcast output neuron whose value nearby agents can read as an extra sensor input, enabling rudimentary signaling.
- **Moving spikes** — give spikes a slow drift velocity in `spikes.py` to create a more dynamic hazard landscape.
- **Energy-based reproduction** — let high-energy agents spawn children directly by spending energy, removing the need for a centralized dead pool entirely.

---

## Known Limitations

- Sensor computation is O(agents × food + agents × spikes) each frame. With very large populations or spike counts this can slow down. A spatial grid (divide the world into cells, only check nearby cells) would fix this.
- There is no crossover, so genetic diversity relies entirely on mutation. Early generations can be slow to find useful behaviors if all initial random weights are equally bad.
- The dead pool prunes the weakest entries when it exceeds 200, but does not track genetic diversity — a single highly-fit lineage can crowd out unique strategies that might be better in the long run.