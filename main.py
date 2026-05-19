"""
EVO-SIM — 2D Evolutionary Emergence Sandbox
============================================
Run:   python main.py
Click an agent to inspect its sensors, brain, and stats.
Close the window or press ESC to quit.
"""

import sys
import pygame
from config import WORLD_WIDTH, WORLD_HEIGHT, FPS_TARGET
from simulation.world import World
from visualization.renderer import Renderer
from visualization.ui import UIPanel


def main():
    pygame.init()
    screen = pygame.display.set_mode((WORLD_WIDTH, WORLD_HEIGHT))
    pygame.display.set_caption("EVO-SIM — Evolutionary Emergence Sandbox")
    clock = pygame.time.Clock()

    world = World()
    renderer = Renderer(screen)
    ui = UIPanel()

    selected_agent = None
    fps = 0.0
    running = True

    print("EVO-SIM started. Click an agent to inspect it. ESC to quit.")

    while running:
        # ── Events ───────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                clicked = world.agent_at(mx, my)
                if clicked is not None:
                    selected_agent = clicked
                else:
                    selected_agent = None  # click empty space deselects

        # Keep selected agent reference valid
        if selected_agent is not None and not selected_agent.alive:
            selected_agent = None

        # ── Simulation step ──────────────────────────────────────────────────
        world.update()

        # ── Render ───────────────────────────────────────────────────────────
        renderer.begin_frame()
        renderer.draw_world(world, selected_agent)
        ui.draw(screen, world, selected_agent, fps)
        renderer.end_frame()

        dt = clock.tick(FPS_TARGET)
        fps = clock.get_fps()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
