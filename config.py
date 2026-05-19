# ====================
# WORLD
# ====================
WORLD_WIDTH = 1250
WORLD_HEIGHT = 700
BG_COLOR = (10, 10, 15)
FPS_TARGET = 60

# ====================
# FOOD
# ====================
FOOD_COUNT_MAX = 80          # Max food particles at once
FOOD_RADIUS = 4
FOOD_COLOR = (50, 220, 80)
FOOD_ENERGY = 40.0           # Energy gained when eaten
FOOD_RESPAWN_INTERVAL = 30   # Frames between respawn attempts
FOOD_RESPAWN_BATCH = 3       # How many to respawn at once

# ====================
# AGENTS
# ====================
AGENT_RADIUS = 4
AGENT_INITIAL_COUNT = 55
AGENT_INITIAL_ENERGY = 80.0
AGENT_MAX_ENERGY = 150.0
AGENT_ENERGY_DECAY = 0.05    # Energy lost per frame
AGENT_MAX_AGE = 3000         # Frames before natural death
AGENT_MOVE_SPEED_MAX = 2.0
AGENT_TURN_SPEED_MAX = 0.12  # Radians per frame
AGENT_SPEED_DECAY_COST = 0.00

# ====================
# SENSORS
# ====================
VISION_RADIUS = 120
VISION_FOV = 2.094           # 120 degrees in radians
SOUND_RADIUS = 100
WALL_SENSOR_RADIUS = 80      # Max distance for wall sensor

# ====================
# SPIKES
# ====================
SPIKE_COUNT = 12             # Number of spike balls in the world
SPIKE_RADIUS = 4             # Visual/collision radius (same as food by default)
SPIKE_COLOR = (220, 60, 60)  # Red-ish
SPIKE_ENERGY_DAMAGE = 35.0   # Energy drained on contact per hit
SPIKE_HIT_COOLDOWN = 30      # Frames before the same spike can damage the same agent again

# ====================
# BRAIN
# ====================
BRAIN_INPUTS = 20            # See sensors.py for full breakdown
BRAIN_HIDDEN = 10
BRAIN_OUTPUTS = 2            # [turn, speed]

# ====================
# EVOLUTION
# ====================
POPULATION_MIN = 24          # Trigger reproduction below this
POPULATION_MAX = 80
MUTATION_RATE = 0.05         # Probability of mutating each weight
MUTATION_STRENGTH = 0.2      # Std dev of Gaussian noise
ELITE_FRACTION = 0.3         # Top fraction used as parents
FITNESS_LIFESPAN_W = 0.4
FITNESS_FOOD_W = 0.5
FITNESS_DISTANCE_W = 0.1

# ====================
# VISUALIZATION
# ====================
ENERGY_BAR_WIDTH = 20
ENERGY_BAR_HEIGHT = 4
ENERGY_BAR_OFFSET = 12       # Pixels above agent center
UI_PANEL_WIDTH = 260
UI_FONT_SIZE = 14
VISION_CONE_ALPHA = 30       # Transparency of vision cone overlay
SELECTED_COLOR = (255, 255, 100)
