# EVO-SIM — 2D Evolutionary Emergence Sandbox

A minimal artificial life simulation where simple agents survive, eat food, and evolve movement behaviors through mutation over generations. No reinforcement learning, no NEAT, no ML frameworks — just NumPy math, Pygame visuals, and natural selection doing its thing.

---

## Quick Start

```bash
pip install pygame numpy
python main.py
```

Click any agent to inspect its live sensor readings, brain outputs, and stats. Press **ESC** or close the window to quit.

---

## What You're Looking At

The world is a 1200×800 dark arena populated by two things:

**Food** — small green dots scattered randomly across the world. They respawn periodically. Eating food restores energy.

**Agents** — colored circles with a small direction arrow and an energy bar above them. Every agent is running its own neural network every frame, deciding how fast to move and which way to turn based on what its sensors are picking up. They bounce off walls. They eat food by overlapping with it. They die when their energy hits zero or they get too old. When enough agents die, the best performers from the historical record are used to spawn the next wave — with their weights slightly mutated.

Over many generations, agents that wander aimlessly die quickly. Agents whose random initial weights happen to steer toward food live longer and eat more, so their mutated descendants populate the next wave. Useful behaviors accumulate gradually through this selection pressure.

---

## Project Structure

```
evo_sim/
├── main.py                   — Main loop, event handling, clock
├── config.py                 — Every tunable parameter in one file
│
├── simulation/
│   ├── world.py              — Orchestrates updates each frame
│   ├── agent.py              — Agent state, movement, drawing
│   ├── brain.py              — NumPy neural network
│   ├── sensors.py            — All 9 sensor channels
│   ├── food.py               — Food particles and respawning
│   └── evolution.py          — Fitness tracking, selection, spawning
│
├── visualization/
│   ├── renderer.py           — World drawing, background grid
│   └── ui.py                 — Right-side inspection panel
│
└── utils/
    └── math_utils.py         — angle_diff, clamp, heading_vector, etc.
```

The simulation logic lives entirely in `simulation/`. The visualization layer in `visualization/` only reads state — it never writes to it. This separation makes it easy to add a headless mode or swap out the renderer later.

---

## The Simulation Loop

Each frame runs in this order (see `world.py`):

1. **Food update** — remove eaten food, respawn a small batch if below max
2. **Sense** — for each agent, compute all 9 sensor values from the current world state
3. **Think** — feed sensors into the agent's neural network, get turn + speed outputs
4. **Move** — apply outputs to position and angle, bounce off walls
5. **Eat** — check overlaps between agents and food, transfer energy
6. **Decay** — drain energy each frame (moving faster costs slightly more)
7. **Death** — mark agents as dead if energy ≤ 0 or age ≥ max_age
8. **Record** — log dead agents' fitness scores into the evolution pool
9. **Repopulate** — if alive count dropped below `POPULATION_MIN`, spawn children from top performers
10. **Render** — draw everything, then draw the UI panel on top

---

## Agents

Each agent has:

| Property | Description |
|---|---|
| `x, y` | Position in world space |
| `angle` | Current heading in radians |
| `speed` | Current movement speed (set each frame by brain) |
| `energy` | Starts at 80, capped at 150, drained each frame |
| `age` | Frame counter; death at 3000 frames by default |
| `brain` | Neural network whose weights are the genome |
| `color` | RGB color inherited (with drift) from parent |
| `food_eaten` | Running count, used in fitness |
| `distance_traveled` | Accumulated distance, used in fitness |

Agents lose a flat `AGENT_ENERGY_DECAY` energy per frame, plus a small additional cost proportional to their speed. Standing still is the cheapest survival strategy, but agents that don't move can't find food, so selection pushes toward efficient foraging rather than either extreme.

Wall collisions cause a reflection: the relevant velocity component is negated. No energy penalty is applied for hitting a wall.

---

## Sensor System

Every frame, `sensors.py` builds a 9-element NumPy array for each agent. All values are normalized before being passed to the brain.

| Index | Name | Range | Description |
|---|---|---|---|
| 0 | `food_dist` | 0 – 1 | How close the nearest visible food is. 0 = nothing seen, 1 = touching |
| 1 | `food_angle` | -1 – 1 | Relative angle to that food, normalized by half the FOV. Negative = left, positive = right |
| 2 | `wall_dist` | 0 – 1 | Proximity to the nearest wall. 0 = far away, 1 = right against it |
| 3 | `touch_wall` | 0 or 1 | Binary: agent is currently at the wall boundary |
| 4 | `touch_agent` | 0 or 1 | Binary: another agent is within ~2 agent radii |
| 5 | `touch_food` | 0 or 1 | Binary: overlapping with a food particle right now |
| 6 | `sound` | 0 – 1 | Sum of movement noise from nearby agents within `SOUND_RADIUS`. Faster neighbors = louder |
| 7 | `energy_lvl` | 0 – 1 | Current energy divided by max energy. Agents "know" how hungry they are |
| 8 | `noise` | -0.2 – 0.2 | Pure random jitter added every frame. Prevents perfectly deterministic behavior and helps exploration |

**Food vision** uses a cone check: the sensor only fires if food is within `VISION_RADIUS` pixels AND within the `VISION_FOV` angle (120° by default) centered on the agent's heading. Food behind the agent is invisible.

**Sound** is proportional to neighbor speed and falls off linearly with distance. An agent sprinting nearby is louder than one drifting slowly.

---

## The Neural Network (Brain)

File: `simulation/brain.py`

The network is a simple two-layer feedforward network built entirely with NumPy matrices. There is no PyTorch, no TensorFlow, no autograd.

```
Architecture:

  9 inputs
     ↓  (W1: 8×9, b1: 8)
  8 hidden neurons   ← tanh activation
     ↓  (W2: 2×8, b2: 2)
  2 outputs          ← tanh activation
```

**Activation function: `tanh` on both layers.**

`tanh` was chosen because it outputs values in the range (-1, 1), which maps cleanly to signed quantities like "turn left vs right" and "slow vs fast." It's also smooth and zero-centered, which gives better gradient properties if you ever want to add learning later.

The forward pass is:

```python
h   = tanh(W1 @ sensors + b1)   # hidden layer
out = tanh(W2 @ h + b2)         # output layer
```

**Output mapping:**

| Output | Raw range | Interpretation |
|---|---|---|
| `out[0]` (turn) | -1 to 1 | Multiplied by `AGENT_TURN_SPEED_MAX` (0.12 rad/frame). Negative = left, positive = right |
| `out[1]` (speed) | -1 to 1 | Remapped to (0, `AGENT_MOVE_SPEED_MAX`) via `(out + 1) / 2 * max_speed`. Always forward |

There is **no backpropagation**. Weights are never updated during an agent's lifetime. The only way the network ever changes is through mutation at reproduction time.

Total weight count: `(9×8 + 8) + (8×2 + 2) = 90` parameters per agent.

---

## Evolution & Reproduction

File: `simulation/evolution.py`

### When reproduction happens

Reproduction is **not generational** — it's continuous and demand-driven. The population is checked every frame. If the number of living agents drops below `POPULATION_MIN` (default: 15), a batch of children is spawned immediately to bring the count back up, overshooting slightly (spawns `POPULATION_MIN - alive + 5` children, capped at `POPULATION_MAX`).

### Fitness scoring

When an agent dies, its fitness is computed from three components:

```
fitness = 0.4 × (age / max_age)
        + 0.5 × min(1.0, food_eaten / 20)
        + 0.1 × min(1.0, distance_traveled / 5000)
```

Food eaten is weighted most heavily (0.5) because it most directly represents successful foraging. Lifespan (0.4) rewards surviving without food too. Distance (0.1) gives a small nudge toward exploration. The weights are tunable in `config.py`.

### Parent selection

All dead agents accumulate in a `dead_pool` (capped at 200 entries; the weakest are pruned when it overflows). When spawning, the pool is sorted by fitness and the top `ELITE_FRACTION` (default 30%) become the eligible parent set. Each child picks one parent at random from that elite set.

There is **no crossover**. Each child has exactly one parent. The child's genome is the parent's weights plus mutation.

### Mutation

Each weight in the genome is independently mutated:

```python
for each weight w:
    if random() < MUTATION_RATE:      # 15% chance per weight
        w += gaussian(0, MUTATION_STRENGTH)   # std dev = 0.3
        w = clamp(w, -3, 3)           # prevent runaway weights
```

This applies to all four parameter arrays: `W1`, `b1`, `W2`, `b2`. Biases mutate by the same rule as weights.

The generation counter increments every time a new wave of children is spawned. Agents inherit a slightly color-drifted version of their parent's color (±20 per RGB channel), so loosely related lineages tend to share similar hues over time.

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
| `FOOD_ENERGY` | 40.0 | Energy gained per food eaten. Higher = eating matters more |
| `FOOD_RESPAWN_INTERVAL` | 30 | Frames between respawn attempts. Higher = food is scarcer |
| `FOOD_RESPAWN_BATCH` | 3 | Food particles added per respawn event |

### Agents
| Parameter | Default | Effect |
|---|---|---|
| `AGENT_INITIAL_COUNT` | 30 | Starting population |
| `AGENT_INITIAL_ENERGY` | 80.0 | Energy each agent starts with |
| `AGENT_MAX_ENERGY` | 150.0 | Energy cap (overeating has no further benefit) |
| `AGENT_ENERGY_DECAY` | 0.05 | Flat energy drain per frame. Higher = agents die faster |
| `AGENT_MAX_AGE` | 3000 | Max frames before natural death |
| `AGENT_MOVE_SPEED_MAX` | 3.5 | Pixels per frame at full output |
| `AGENT_TURN_SPEED_MAX` | 0.12 | Radians per frame at full turn output |

### Sensors
| Parameter | Default | Effect |
|---|---|---|
| `VISION_RADIUS` | 150 | How far ahead agents can see food |
| `VISION_FOV` | 2.094 (120°) | Width of the vision cone in radians. Narrower = harder to spot food |
| `SOUND_RADIUS` | 100 | Range at which agents can hear each other move |
| `WALL_SENSOR_RADIUS` | 80 | Distance at which wall sensor starts firing |

### Brain
| Parameter | Default | Effect |
|---|---|---|
| `BRAIN_INPUTS` | 9 | Size of the sensor vector — change if you add/remove sensors |
| `BRAIN_HIDDEN` | 8 | Hidden layer neuron count. More = more expressive, slower |
| `BRAIN_OUTPUTS` | 2 | Always 2: turn and speed |

### Evolution
| Parameter | Default | Effect |
|---|---|---|
| `POPULATION_MIN` | 15 | Population floor that triggers reproduction |
| `POPULATION_MAX` | 60 | Hard cap on population |
| `MUTATION_RATE` | 0.15 | Per-weight probability of mutation (0–1) |
| `MUTATION_STRENGTH` | 0.3 | Std deviation of Gaussian noise applied to mutated weights |
| `ELITE_FRACTION` | 0.3 | Fraction of dead pool used as parents |
| `FITNESS_LIFESPAN_W` | 0.4 | Fitness weight for lifespan |
| `FITNESS_FOOD_W` | 0.5 | Fitness weight for food eaten |
| `FITNESS_DISTANCE_W` | 0.1 | Fitness weight for distance traveled |

---

## Suggested Experiments

**Make evolution faster and wilder**
```python
MUTATION_STRENGTH = 0.6
AGENT_MAX_AGE = 1000
FOOD_COUNT_MAX = 40
```

**Make food extremely scarce and see if foraging improves faster**
```python
FOOD_COUNT_MAX = 20
FOOD_RESPAWN_BATCH = 1
FOOD_RESPAWN_INTERVAL = 90
```

**Narrow vision to require more precise steering**
```python
VISION_FOV = 0.698   # ~40 degrees
VISION_RADIUS = 100
```

**Reward exploration, not eating**
```python
FITNESS_LIFESPAN_W = 0.2
FITNESS_FOOD_W = 0.1
FITNESS_DISTANCE_W = 0.7
```

**Bigger brains**
```python
BRAIN_HIDDEN = 16
```
Requires no other changes — the weight matrices resize automatically.

---

## What to Add Next

The architecture is intentionally minimal so extensions slot in cleanly:

- **Predators** — agents with a different genome that gain fitness by catching prey. Add a predator class in `simulation/`, a new sensor channel for "predator nearby," and a second fitness function.
- **Pheromones** — a 2D grid in `world.py` that agents write to when eating or moving, and a new sensor reading the local gradient.
- **Memory / recurrence** — pass the previous hidden layer state back in as extra inputs (making it an Elman network). Add a `hidden_state` field to `Agent` and pass it through `brain.forward()`.
- **Crossover** — in `evolution.py`, select two parents and mix their weight arrays (e.g., uniform crossover) before mutating.
- **Speciation** — cluster agents by weight similarity before selection so diverse lineages don't all collapse to one strategy.
- **Communication** — add a "broadcast" output neuron whose value nearby agents can read as an extra sensor input.
- **Energy-based reproduction** — instead of centralized repopulation, let high-energy agents spawn a child directly by spending energy, removing the need for a dead pool entirely.

---

## Known Limitations

- Sensor computation is O(agents × food) each frame. With large populations and food counts this can slow down. A spatial grid (divide the world into cells, only check nearby cells) would fix this if needed.
- There is no crossover, so genetic diversity relies entirely on mutation. Early generations can be slow to converge if initial random weights are uniformly bad.
- The dead pool prunes the weakest entries when it exceeds 200, but it does not currently track diversity — highly fit but genetically similar agents can crowd out unique strategies.