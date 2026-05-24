"""
EVO-SIM — 2D Evolutionary Emergence Sandbox
============================================
Run:   python main.py
Click an agent to inspect its sensors, brain, and stats.
Close the window or press ESC to quit.

Results are saved automatically to runs/<timestamp>/
Set RUN_TESTS_ON_STARTUP = True in config.py to run sanity checks first.
"""

import sys
import pygame
from config import WORLD_WIDTH, WORLD_HEIGHT, FPS_TARGET, RUN_TESTS_ON_STARTUP
from simulation.world import World
from visualization.renderer import Renderer
from visualization.ui import UIPanel
from evolog.generation_logger import GenerationLogger


def _run_startup_tests():
    """
    Run the fast test modules before opening the window.
    Skips the slow logging/determinism tests so startup stays under ~60s.
    """
    import subprocess
    fast_modules = [
        "tests/test_brain_evolution.py",
        "tests/test_lifecycle.py",
        "tests/test_sensors_world.py",
        "tests/test_conservation.py",
    ]
    print("=" * 55)
    print("RUN_TESTS_ON_STARTUP = True — running sanity checks...")
    print("=" * 55)
    result = subprocess.run(
        [sys.executable, "-m", "pytest"] + fast_modules + ["-v", "--tb=short"],
        capture_output=False,
    )
    if result.returncode != 0:
        print("\n[STARTUP] Tests FAILED — fix issues before running the sim.")
        print("[STARTUP] Set RUN_TESTS_ON_STARTUP = False in config.py to skip.")
        sys.exit(1)
    print("=" * 55)
    print("[STARTUP] All checks passed. Starting simulation...")
    print("=" * 55)


def main():
    if RUN_TESTS_ON_STARTUP:
        _run_startup_tests()

    pygame.init()
    screen = pygame.display.set_mode((WORLD_WIDTH, WORLD_HEIGHT))
    pygame.display.set_caption("EVO-SIM — Evolutionary Emergence Sandbox")
    clock = pygame.time.Clock()

    logger = GenerationLogger()
    world = World(logger=logger)
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
