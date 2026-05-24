# ====================
# WORLD
# ====================
WORLD_WIDTH = 1200
WORLD_HEIGHT = 800
BG_COLOR = (10, 10, 15)
FPS_TARGET = 60

# ====================
# FOOD
# ====================
FOOD_COUNT_MAX = 80          # Max food particles at once
FOOD_RADIUS = 5
FOOD_COLOR = (50, 220, 80)
FOOD_ENERGY = 40.0           # Energy gained when eaten
FOOD_RESPAWN_INTERVAL = 30   # Frames between respawn attempts
FOOD_RESPAWN_BATCH = 3       # How many to respawn at once

# ====================
# AGENTS
# ====================
AGENT_RADIUS = 8
AGENT_INITIAL_COUNT = 30
AGENT_INITIAL_ENERGY = 80.0
AGENT_MAX_ENERGY = 150.0
AGENT_ENERGY_DECAY = 0.05    # Energy lost per frame
AGENT_MAX_AGE = 3000         # Frames before natural death
AGENT_MOVE_SPEED_MAX = 3.5
AGENT_TURN_SPEED_MAX = 0.12  # Radians per frame
AGENT_SPEED_DECAY_COST = 0.01  # Extra energy drain per unit of speed

# ====================
# SENSORS
# ====================
VISION_RADIUS = 150
VISION_FOV = 2.094           # 120 degrees in radians
SOUND_RADIUS = 100
WALL_SENSOR_RADIUS = 80      # Max distance for wall sensor

# ====================
# SPIKES
# ====================
SPIKE_COUNT = 15             # Number of spike balls in the world
SPIKE_RADIUS = 5             # Visual/collision radius (same as food by default)
SPIKE_COLOR = (220, 60, 60)  # Red-ish
SPIKE_ENERGY_DAMAGE = 25.0   # Energy drained on contact per hit
SPIKE_HIT_COOLDOWN = 30      # Frames before the same spike can damage the same agent again

# ====================
# BRAIN
# ====================
BRAIN_INPUTS = 20            # Must match number of sensor channels in sensors.py
BRAIN_HIDDEN_LAYERS = [8]    # List of hidden layer sizes.
                             #   []         = no hidden layers (direct input→output)
                             #   [8]        = one hidden layer, 8 neurons (default)
                             #   [16, 8]    = two hidden layers
                             #   [32, 16, 8] = three hidden layers
BRAIN_OUTPUTS = 2            # Always 2: turn and speed

# ====================
# EVOLUTION
# ====================
POPULATION_MIN = 15          # Trigger reproduction below this
SPAWN_BATCH_MAX = 60         # Max agents spawned in a single repopulation wave.
                             # Does NOT cap the live population — agents only spawn
                             # in waves, so live count is bounded by how many
                             # survive, not by this value.
MUTATION_RATE = 0.15         # Probability of mutating each weight
MUTATION_STRENGTH = 0.3      # Std dev of Gaussian noise
ELITE_FRACTION = 0.3         # Top fraction used as parents
FITNESS_LIFESPAN_W = 0.4
FITNESS_FOOD_W = 0.5
FITNESS_DISTANCE_W = 0.1

# ====================
# TESTING
# ====================
RUN_TESTS_ON_STARTUP = False  # Set True to run the fast sanity checks before
                               # the sim window opens. Catches regressions without
                               # needing to run pytest manually.
                               # Full suite: python -m pytest tests/ -v
HEADLESS_UPDATES_PER_TICK = 10  # Simulation steps per frame in fast-train mode
                                 # (irrelevant when running fully headless)

# ====================
# LOGGING
# ====================
LOG_INTERVAL = 500           # Write one CSV row every N frames (0 = disabled)
CHECKPOINT_INTERVAL = 50     # Save full population weights every N generations (0 = disabled)
FITNESS_MIN_AGE = 200        # Minimum agent age (frames) to be included in fitness
                             # percentile calculations. Agents younger than this are
                             # excluded because their fitness is near-zero by definition
                             # (haven't had time to eat or travel). Does not affect
                             # average_alive_fitness or best_alive_fitness, which always
                             # include the full population.
ENERGY_BAR_WIDTH = 20
ENERGY_BAR_HEIGHT = 4
ENERGY_BAR_OFFSET = 12       # Pixels above agent center
UI_PANEL_WIDTH = 260
UI_FONT_SIZE = 14
VISION_CONE_ALPHA = 30       # Transparency of vision cone overlay
SELECTED_COLOR = (255, 255, 100)
