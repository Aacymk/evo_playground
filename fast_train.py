"""
fast_train.py — Headless evolutionary training, no Pygame rendering.
=====================================================================
Runs the simulation as fast as the CPU allows (no FPS cap, no drawing).
Progress is printed to the terminal every PRINT_INTERVAL generations.
Results are saved to runs/<run_id>/ exactly as in main.py.

Usage:
    python fast_train.py                          # run until Ctrl+C
    python fast_train.py --generations 10000      # stop at generation 10000
    python fast_train.py --generations 500 --run-id my_experiment

Arguments:
    --generations N     Stop after N generations  (default: run forever)
    --run-id NAME       Custom run folder name     (default: timestamp)
    --print-every N     Print progress every N generations (default: 10)
"""

import argparse
import os
import sys
import time

# Suppress the pygame "Hello from the pygame community" banner
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
# Tell SDL to use a dummy video driver so no window opens
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import pygame
pygame.init()
# Create a minimal 1x1 surface so pygame doesn't complain
pygame.display.set_mode((1, 1))

from simulation.world import World
from evolog.generation_logger import GenerationLogger


def parse_args():
    p = argparse.ArgumentParser(description="EVO-SIM headless fast trainer")
    p.add_argument("--generations", type=int, default=0,
                   help="Stop after this many generations (0 = run until Ctrl+C)")
    p.add_argument("--run-id", type=str, default=None,
                   help="Custom name for the run folder under runs/")
    p.add_argument("--print-every", type=int, default=10,
                   help="Print a progress line every N generations")
    return p.parse_args()


def main():
    args = parse_args()
    max_gen     = args.generations
    print_every = args.print_every

    logger = GenerationLogger(run_id=args.run_id)
    world  = World(logger=logger)

    print(f"[fast_train] Starting headless run → {logger.run_dir}")
    if max_gen > 0:
        print(f"[fast_train] Target: {max_gen} generations")
    else:
        print("[fast_train] Running until Ctrl+C")
    print()

    last_gen    = world.generation
    last_time   = time.perf_counter()
    frame_count = 0
    start_time  = last_time

    try:
        while True:
            world.update()
            frame_count += 1

            current_gen = world.generation

            # ── Progress print ────────────────────────────────────────────────
            if current_gen != last_gen and (current_gen - 1) % print_every == 0:
                now        = time.perf_counter()
                elapsed    = now - last_time
                total      = now - start_time
                fps        = frame_count / total if total > 0 else 0
                last_time  = now

                # Pull best fitness from most recent dead pool entry if available
                pool = world.evo_mgr.dead_pool
                best_fit = max((d['fitness'] for d in pool), default=0.0)
                avg_fit  = (sum(d['fitness'] for d in pool) / len(pool)
                            if pool else 0.0)

                print(
                    f"  gen {current_gen - 1:>6} | "
                    f"frame {world.frame:>8} | "
                    f"pop {world.population:>3} | "
                    f"best_fit {best_fit:.4f} | "
                    f"avg_fit {avg_fit:.4f} | "
                    f"{fps:>7.0f} sim-fps"
                )
                last_gen = current_gen

            # ── Stop condition ────────────────────────────────────────────────
            if max_gen > 0 and world.generation > max_gen:
                break

    except KeyboardInterrupt:
        print("\n[fast_train] Interrupted by user.")

    total_time = time.perf_counter() - start_time
    print(f"\n[fast_train] Done.")
    print(f"  Generations : {world.generation - 1}")
    print(f"  Total frames: {world.frame}")
    print(f"  Wall time   : {total_time:.1f}s")
    print(f"  Avg sim-fps : {world.frame / total_time:.0f}")
    print(f"  Results in  : {logger.run_dir}")

    pygame.quit()


if __name__ == "__main__":
    main()
