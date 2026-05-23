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
        gen   = int(data['generation'][0])
        frame = int(data['frame'][0])
        count = int(data['agent_count'][0])
        fitnesses = [float(data[f'agent{i}_fitness'][0]) for i in range(count)]
        best_fit  = max(fitnesses)
        print(f"  {os.path.basename(f)}  gen={gen:>6}  frame={frame:>8}  "
              f"agents={count}  best_fit={best_fit:.4f}")


def load_checkpoint(npz_path: str):
    """
    Returns a list of dicts: [{Ws, bs, layer_sizes, color, fitness}, ...]
    """
    data   = np.load(npz_path)
    count  = int(data['agent_count'][0])
    agents = []
    for i in range(count):
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
    Create a World whose initial population is seeded from checkpoint weights.
    """
    import config
    from simulation.world import World
    from simulation.brain import Brain
    from simulation.agent import Agent

    # Optionally patch mutation rate for this session
    if freeze:
        config.MUTATION_RATE = 0.0
        print("[replay] Mutation disabled — pure exploitation mode")
    elif mutation_rate_override is not None:
        config.MUTATION_RATE = mutation_rate_override
        print(f"[replay] Mutation rate overridden to {mutation_rate_override}")

    world = World(logger=logger)

    # Replace the random initial population with checkpoint agents
    world.agents.clear()
    forbidden = world.spike_mgr.positions()

    for ag_data in checkpoint_agents:
        brain = Brain((ag_data['Ws'], ag_data['bs']))
        agent = Agent(brain=brain, color=ag_data['color'],
                      generation=start_gen, forbidden_positions=forbidden)
        world.agents.append(agent)

    # Seed the evo manager's dead pool with the checkpoint fitness scores
    # so early repopulation has something to select from
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

    world.evo_mgr.generation = start_gen + 1
    world.evo_mgr.total_born = len(checkpoint_agents)
    print(f"[replay] Seeded {len(checkpoint_agents)} agents from gen {start_gen}")
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
