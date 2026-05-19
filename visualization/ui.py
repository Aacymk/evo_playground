import pygame
from config import UI_PANEL_WIDTH, UI_FONT_SIZE, WORLD_WIDTH, WORLD_HEIGHT


# Must match sensors.py index order exactly
SENSOR_LABELS = [
    "food_dist",       # 0
    "food_angle",      # 1
    "food_count_fov",  # 2
    "spike_dist",      # 3
    "spike_angle",     # 4
    "spike_count_fov", # 5
    "agent_count_fov", # 6
    "wall_nearest",    # 7
    "wall_north",      # 8
    "wall_south",      # 9
    "wall_east",       # 10
    "wall_west",       # 11
    "pos_x",           # 12
    "pos_y",           # 13
    "touch_wall",      # 14
    "touch_agent",     # 15
    "touch_food",      # 16
    "sound",           # 17
    "energy_lvl",      # 18
    "noise",           # 19
]
OUTPUT_LABELS = ["turn", "speed"]

# Color hints per sensor group
_SENSOR_COLORS = {
    range(0, 3):   (100, 220, 100),   # food — green
    range(3, 6):   (220, 80,  80),    # spikes — red
    range(6, 7):   (120, 180, 255),   # agent density — blue
    range(7, 14):  (180, 160, 220),   # spatial — purple
    range(14, 17): (220, 200, 100),   # touch — yellow
    range(17, 18): (100, 200, 220),   # sound — cyan
    range(18, 20): (180, 180, 180),   # internal — grey
}


def _sensor_color(idx):
    for r, col in _SENSOR_COLORS.items():
        if idx in r:
            return col
    return (200, 200, 200)


class UIPanel:
    def __init__(self):
        pygame.font.init()
        self.font       = pygame.font.SysFont("monospace", UI_FONT_SIZE)
        self.small_font = pygame.font.SysFont("monospace", 11)
        self.panel_rect = pygame.Rect(
            WORLD_WIDTH - UI_PANEL_WIDTH, 0, UI_PANEL_WIDTH, WORLD_HEIGHT
        )

    def draw(self, surface, world, selected_agent, fps):
        panel_surf = pygame.Surface((UI_PANEL_WIDTH, WORLD_HEIGHT), pygame.SRCALPHA)
        panel_surf.fill((15, 15, 25, 210))
        surface.blit(panel_surf, (WORLD_WIDTH - UI_PANEL_WIDTH, 0))

        x = WORLD_WIDTH - UI_PANEL_WIDTH + 10
        y = 10

        def text(msg, color=(220, 220, 220), font=None):
            nonlocal y
            f = font or self.font
            surf = f.render(msg, True, color)
            surface.blit(surf, (x, y))
            y += f.get_height() + 2

        def divider():
            nonlocal y
            pygame.draw.line(surface, (60, 60, 80),
                             (x - 5, y), (x + UI_PANEL_WIDTH - 15, y), 1)
            y += 5

        # ── Header ───────────────────────────────────────────────────────────
        text("[ EVO-SIM ]", (100, 220, 180))
        divider()
        text(f"FPS:        {fps:.0f}", (180, 180, 180))
        text(f"Frame:      {world.frame}")
        text(f"Generation: {world.generation}", (200, 200, 100))
        text(f"Population: {world.population}", (100, 200, 255))
        text(f"Food:       {world.food_count}", (80, 220, 80))
        text(f"Spikes:     {world.spike_count}", (220, 80, 80))
        divider()

        if selected_agent is None:
            text("Click agent to inspect", (120, 120, 140))
            return

        ag = selected_agent
        # ── Agent overview ───────────────────────────────────────────────────
        text(f"Agent #{ag.id}", tuple(ag.color))
        text(f"Gen {ag.generation}")
        text(f"Energy:     {ag.energy:.1f}", _energy_color(ag.energy))
        text(f"Age:        {ag.age} / {_cfg('AGENT_MAX_AGE')}")
        text(f"Food eaten: {ag.food_eaten}", (100, 220, 100))
        text(f"Spike hits: {ag.spike_hits}", (220, 100, 100))
        text(f"Speed:      {ag.speed:.2f}")
        text(f"Fitness:    {ag.fitness():.3f}", (200, 200, 80))
        divider()

        # ── Sensors ──────────────────────────────────────────────────────────
        text("SENSORS", (160, 160, 255))
        for i, label in enumerate(SENSOR_LABELS):
            val = float(ag.last_sensors[i]) if i < len(ag.last_sensors) else 0.0
            bar = _bar(val)
            col = _sensor_color(i)
            text(f" {label:<15} {val:+.2f} {bar}", col, self.small_font)
        divider()

        # ── Outputs ──────────────────────────────────────────────────────────
        text("BRAIN OUT", (255, 180, 100))
        for i, label in enumerate(OUTPUT_LABELS):
            val = float(ag.last_outputs[i]) if i < len(ag.last_outputs) else 0.0
            bar = _bar((val + 1) / 2)
            text(f"  {label:<8} {val:+.3f} {bar}", (220, 200, 140), self.small_font)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _bar(value, width=7):
    filled = int(max(0.0, min(1.0, value)) * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _energy_color(energy):
    import config
    ratio = max(0.0, min(1.0, energy / config.AGENT_MAX_ENERGY))
    r = int(255 * (1 - ratio))
    g = int(200 * ratio)
    return (r, g, 40)


def _val_color(val):
    if val > 0.6:
        return (100, 255, 120)
    if val > 0.2:
        return (220, 220, 100)
    return (180, 180, 180)


def _cfg(key):
    import config
    return getattr(config, key, "?")
