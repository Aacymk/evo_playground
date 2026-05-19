import pygame
from config import UI_PANEL_WIDTH, UI_FONT_SIZE, WORLD_WIDTH, WORLD_HEIGHT


SENSOR_LABELS = [
    "food_dist", "food_angle", "wall_dist",
    "touch_wall", "touch_agent", "touch_food",
    "sound", "energy_lvl", "noise"
]
OUTPUT_LABELS = ["turn", "speed"]


class UIPanel:
    def __init__(self):
        pygame.font.init()
        self.font = pygame.font.SysFont("monospace", UI_FONT_SIZE)
        self.small_font = pygame.font.SysFont("monospace", 12)
        self.panel_rect = pygame.Rect(
            WORLD_WIDTH - UI_PANEL_WIDTH, 0, UI_PANEL_WIDTH, WORLD_HEIGHT
        )

    def draw(self, surface, world, selected_agent, fps):
        # Semi-transparent panel background
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
            y += f.get_height() + 3

        def divider():
            nonlocal y
            pygame.draw.line(surface, (60, 60, 80),
                             (x - 5, y), (x + UI_PANEL_WIDTH - 15, y), 1)
            y += 6

        # ── Header ───────────────────────────────────────────────────────────
        text("[ EVO-SIM ]", (100, 220, 180))
        divider()
        text(f"FPS:        {fps:.0f}", (180, 180, 180))
        text(f"Frame:      {world.frame}")
        text(f"Generation: {world.generation}", (200, 200, 100))
        text(f"Population: {world.population}", (100, 200, 255))
        text(f"Food:       {world.food_count}", (80, 220, 80))
        divider()

        if selected_agent is None:
            text("Click agent to inspect", (120, 120, 140))
            return

        ag = selected_agent
        # ── Agent overview ───────────────────────────────────────────────────
        text(f"Agent #{ag.id}", tuple(ag.color))
        text(f"Gen {ag.generation}")
        text(f"Energy: {ag.energy:.1f}", _energy_color(ag.energy))
        text(f"Age:    {ag.age} / {from_config('AGENT_MAX_AGE')}")
        text(f"Food:   {ag.food_eaten}")
        text(f"Speed:  {ag.speed:.2f}")
        text(f"Fitness:{ag.fitness():.3f}", (200, 200, 80))
        divider()

        # ── Sensors ──────────────────────────────────────────────────────────
        text("SENSORS", (160, 160, 255))
        for i, label in enumerate(SENSOR_LABELS):
            val = ag.last_sensors[i]
            bar = _bar(val)
            col = _val_color(val)
            text(f"  {label:<12} {val:+.2f} {bar}", col, self.small_font)
        divider()

        # ── Outputs ──────────────────────────────────────────────────────────
        text("BRAIN OUT", (255, 180, 100))
        for i, label in enumerate(OUTPUT_LABELS):
            val = ag.last_outputs[i] if i < len(ag.last_outputs) else 0.0
            bar = _bar((val + 1) / 2)
            text(f"  {label:<8} {val:+.3f} {bar}", (220, 200, 140), self.small_font)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _bar(value, width=8):
    """ASCII progress bar for a 0-1 value."""
    filled = int(max(0.0, min(1.0, value)) * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _energy_color(energy):
    from config import AGENT_MAX_ENERGY
    r = int(255 * (1 - energy / AGENT_MAX_ENERGY))
    g = int(200 * energy / AGENT_MAX_ENERGY)
    return (r, g, 40)


def _val_color(val):
    if val > 0.6:
        return (100, 255, 120)
    if val > 0.2:
        return (220, 220, 100)
    return (180, 180, 180)


def from_config(key):
    import config
    return getattr(config, key, "?")
