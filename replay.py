"""
replay.py — Load a checkpoint and watch it run visually.
=========================================================
Spawns a population seeded from saved weights, then runs the full
Pygame visualization exactly like main.py.

Usage:
    python replay.py --checkpoint runs/20240101_120000/checkpoints/chk_000500.npz
    python replay.py --run runs/20240101_120000            # loads latest checkpoint
    python replay.py --list runs/20240101_120000           # list available checkpoints

Arguments:
    --checkpoint PATH   Direct path to a .npz checkpoint file
    --run PATH          Path to a run folder; loads the latest checkpoint in it
    --list PATH         Print all checkpoints in a run folder and exit
    --mutation-rate F   Override mutation rate for this replay (default: from config)
    --freeze            Disable mutation entirely — pure exploitation, no evolution
"""

import argparse
import os
import sys
import glob

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="EVO-SIM checkpoint replayer")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--checkpoint", type=str,
                       help="Path to a specific .npz checkpoint file")
    group.add_argument("--run", type=str,
                       help="Path to a run folder; uses the latest checkpoint")
    group.add_argument("--list", type=str, metavar="RUN_DIR",
                       help="List available checkpoints in a run folder and exit")
    p.add_argument("--mutation-rate", type=float, default=None,
                   help="Override MUTATION_RATE for this replay session")
    p.add_argument("--freeze", action="store_true",
                   help="Disable all mutation (pure exploitation)")
    return p.parse_args()


def find_latest_checkpoint(run_dir: str) -> str:
    pattern = os.path.join(run_dir, "checkpoints", "chk_*.npz")
    files   = sorted(glob.glob(pattern))
    if not files:
        print(f"[replay] No checkpoints found in {run_dir}/checkpoints/")
        sys.exit(1)
    return files[-1]


def list_checkpoints(run_dir: str):
    pattern = os.path.join(run_dir, "checkpoints", "chk_*.npz")
    files   = sorted(glob.glob(pattern))
    if not files:
        print(f"No checkpoints found in {run_dir}/checkpoints/")
        return
    print(f"Checkpoints in {run_dir}:")
    for f in files:
        data = np.load(f)
        gen       = int(data['generation'][0])
        frame     = int(data['frame'][0])
        pool_size = int(data['pool_size'][0])
        fitnesses = [float(data[f'agent{i}_fitness'][0]) for i in range(pool_size)]
        best_fit  = max(fitnesses)
        print(f"  {os.path.basename(f)}  gen={gen:>6}  frame={frame:>8}  "
              f"pool={pool_size}  best_fit={best_fit:.4f}")


def load_checkpoint(npz_path: str):
    """
    Returns (pool_agents, generation, frame).
    pool_agents is the saved dead pool — a gene pool for seeding replay.
    Spawn count is determined by config, not by how many are in this file.
    """
    data      = np.load(npz_path)
    pool_size = int(data['pool_size'][0])
    agents    = []
    for i in range(pool_size):
        num_layers = int(data[f'agent{i}_num_layers'][0])
        Ws = [data[f'agent{i}_W{j}'] for j in range(num_layers)]
        bs = [data[f'agent{i}_b{j}'] for j in range(num_layers)]
        agents.append({
            'Ws':          Ws,
            'bs':          bs,
            'layer_sizes': data[f'agent{i}_layer_sizes'].tolist(),
            'color':       tuple(int(c) for c in data[f'agent{i}_color']),
            'fitness':     float(data[f'agent{i}_fitness'][0]),
        })
    gen   = int(data['generation'][0])
    frame = int(data['frame'][0])
    return agents, gen, frame


def build_seeded_world(checkpoint_agents, start_gen, logger,
                       mutation_rate_override=None, freeze=False):
    """
    Create a World seeded from checkpoint weights.

    The checkpoint contains a dead pool (gene pool), not a live population.
    We load that pool into evo_mgr.dead_pool, clear the random initial agents,
    then let maybe_repopulate() spawn the correct number of agents from config
    (POPULATION_MIN + SPAWN_BATCH_MAX) — exactly as it would mid-run.
    """
    import config
    from simulation.world import World
    from simulation.brain import Brain
    from simulation.agent import Agent

    if freeze:
        config.MUTATION_RATE = 0.0
        print("[replay] Mutation disabled — pure exploitation mode")
    elif mutation_rate_override is not None:
        config.MUTATION_RATE = mutation_rate_override
        print(f"[replay] Mutation rate overridden to {mutation_rate_override}")

    world = World(logger=logger)

    # Clear the randomly-initialised population
    world.agents.clear()

    # Load checkpoint entries into the dead pool so repopulation has parents
    world.evo_mgr.dead_pool = []
    for ag_data in checkpoint_agents:
        brain = Brain(([W.copy() for W in ag_data['Ws']],
                       [b.copy() for b in ag_data['bs']]))
        world.evo_mgr.dead_pool.append({
            'fitness':           ag_data['fitness'],
            'brain':             brain,
            'color':             ag_data['color'],
            'generation':        start_gen,
            'age':               0,
            'food_eaten':        0,
            'distance_traveled': 0.0,
            'spike_hits':        0,
            'last_outputs':      [0.0, 0.0],
            'action_history':    [],
        })

    world.evo_mgr.generation  = start_gen
    world.evo_mgr.total_born  = 0
    world.evo_mgr.total_deaths = 0

    # Spawn directly to POPULATION_MIN + SPAWN_BATCH_MAX — the maximum that can
    # ever be alive simultaneously — using the loaded gene pool as parents.
    # We don't call maybe_repopulate() because that caps at SPAWN_BATCH_MAX per
    # call; we want the full starting population in one shot.
    from config import POPULATION_MIN, SPAWN_BATCH_MAX, ELITE_FRACTION
    from simulation.brain import Brain as _Brain
    import numpy as _np

    target      = POPULATION_MIN + SPAWN_BATCH_MAX
    sorted_pool = sorted(world.evo_mgr.dead_pool,
                         key=lambda d: d['fitness'], reverse=True)
    elite_count = max(1, int(len(sorted_pool) * ELITE_FRACTION))
    parents     = sorted_pool[:elite_count]
    forbidden   = world.spike_mgr.positions()

    new_agents = []
    for _ in range(target):
        parent      = parents[_np.random.randint(len(parents))]
        child_brain = parent['brain'].mutate()
        child_color = parent['color']
        new_agents.append(Agent(brain=child_brain, color=child_color,
                                generation=start_gen + 1,
                                forbidden_positions=forbidden))

    world.agents.extend(new_agents)
    world.evo_mgr.generation  = start_gen + 1
    world.evo_mgr.total_born  = len(new_agents)

    print(f"[replay] Seeded from gen {start_gen} pool ({len(checkpoint_agents)} genomes) "
          f"→ spawned {len(new_agents)} agents (config: "
          f"POPULATION_MIN={config.POPULATION_MIN} + SPAWN_BATCH_MAX={config.SPAWN_BATCH_MAX})")
    return world


def main():
    args = parse_args()

    # ── List mode ────────────────────────────────────────────────────────────
    if args.list:
        list_checkpoints(args.list)
        return

    # ── Resolve checkpoint path ───────────────────────────────────────────────
    if args.checkpoint:
        npz_path = args.checkpoint
    elif args.run:
        npz_path = find_latest_checkpoint(args.run)
    else:
        print("[replay] Provide --checkpoint, --run, or --list. Use --help for details.")
        sys.exit(1)

    if not os.path.exists(npz_path):
        print(f"[replay] File not found: {npz_path}")
        sys.exit(1)

    print(f"[replay] Loading checkpoint: {npz_path}")
    checkpoint_agents, start_gen, start_frame = load_checkpoint(npz_path)
    print(f"[replay] Generation {start_gen}, frame {start_frame}, "
          f"{len(checkpoint_agents)} agents")

    # ── Build world ───────────────────────────────────────────────────────────
    from evolog.generation_logger import GenerationLogger
    logger = GenerationLogger()   # new run folder for this replay session
    world  = build_seeded_world(
        checkpoint_agents, start_gen, logger,
        mutation_rate_override=args.mutation_rate,
        freeze=args.freeze,
    )

    # ── Run visual loop (same as main.py) ─────────────────────────────────────
    import pygame
    from config import WORLD_WIDTH, WORLD_HEIGHT, FPS_TARGET
    from visualization.renderer import Renderer
    from visualization.ui import UIPanel

    pygame.init()
    screen = pygame.display.set_mode((WORLD_WIDTH, WORLD_HEIGHT))
    caption = f"EVO-SIM REPLAY — from gen {start_gen} | {os.path.basename(npz_path)}"
    pygame.display.set_caption(caption)
    clock = pygame.time.Clock()

    renderer = Renderer(screen)
    ui       = UIPanel()

    selected_agent = None
    fps     = 0.0
    running = True

    print(f"[replay] Visualization started. ESC to quit.")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                clicked = world.agent_at(mx, my)
                selected_agent = clicked if clicked else None

        if selected_agent is not None and not selected_agent.alive:
            selected_agent = None

        world.update()

        renderer.begin_frame()
        renderer.draw_world(world, selected_agent)
        ui.draw(screen, world, selected_agent, fps)
        renderer.end_frame()

        clock.tick(FPS_TARGET)
        fps = clock.get_fps()

    pygame.quit()


if __name__ == "__main__":
    main()
