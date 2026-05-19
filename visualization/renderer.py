import pygame
from config import WORLD_WIDTH, WORLD_HEIGHT, BG_COLOR, UI_PANEL_WIDTH


class Renderer:
    def __init__(self, surface):
        self.surface = surface
        # World area (excluding UI panel on the right)
        self.world_rect = pygame.Rect(0, 0, WORLD_WIDTH - UI_PANEL_WIDTH, WORLD_HEIGHT)

    def begin_frame(self):
        self.surface.fill(BG_COLOR)
        # Subtle grid
        self._draw_grid()

    def _draw_grid(self):
        grid_color = (18, 22, 28)
        step = 80
        for x in range(0, WORLD_WIDTH - UI_PANEL_WIDTH, step):
            pygame.draw.line(self.surface, grid_color, (x, 0), (x, WORLD_HEIGHT))
        for y in range(0, WORLD_HEIGHT, step):
            pygame.draw.line(self.surface, grid_color, (0, y), (WORLD_WIDTH - UI_PANEL_WIDTH, y))

    def draw_world(self, world, selected_agent):
        # Vision cone first (drawn under agents)
        if selected_agent is not None and selected_agent.alive:
            selected_agent.draw_vision_cone(self.surface)

        # Spikes (drawn under food and agents so they feel embedded in world)
        world.spike_mgr.draw(self.surface)

        # Food
        world.food_mgr.draw(self.surface)

        # Agents
        for agent in world.agents:
            is_selected = (selected_agent is not None and agent.id == selected_agent.id)
            agent.draw(self.surface, selected=is_selected)

    def end_frame(self):
        pygame.display.flip()
