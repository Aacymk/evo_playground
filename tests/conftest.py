"""
tests/conftest.py

Shared setup used by all test modules:
  - Suppress pygame display before any import
  - Provide factory helpers for building isolated agents, worlds, etc.
  - Provide a seeded RNG fixture so deterministic tests are easy
"""

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import pygame
pygame.init()
pygame.display.set_mode((1, 1))

import numpy as np
import sys

# Make sure the project root is on the path when running tests directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Seed helper ───────────────────────────────────────────────────────────────

def seed(n=0):
    """Set numpy random seed for reproducible tests."""
    np.random.seed(n)


# ── Agent factory ─────────────────────────────────────────────────────────────

def make_agent(x=400, y=300, generation=1, energy=None, age=None):
    """Create a minimal agent at a fixed position with deterministic brain."""
    from simulation.agent import Agent
    from simulation.brain import Brain
    seed()
    a = Agent(x=x, y=y, generation=generation)
    if energy is not None:
        a.energy = energy
    if age is not None:
        a.age = age
    return a


def make_agents(n, generation=1):
    """Create n agents spread across the arena."""
    from simulation.agent import Agent
    agents = []
    for i in range(n):
        x = 100 + (i % 10) * 100
        y = 100 + (i // 10) * 100
        agents.append(Agent(x=x, y=y, generation=generation))
    return agents


# ── World factory ─────────────────────────────────────────────────────────────

def make_world(logger=None):
    """Create a World with optional logger."""
    from simulation.world import World
    return World(logger=logger)


# ── Null sensors ─────────────────────────────────────────────────────────────

def fast_config():
    """
    Patch config to make agents die quickly so tests reach multiple generations
    in few frames. Call at the top of any test that needs generational turnover.
    Returns the original values so you can restore with restore_config().
    """
    import config
    orig = {
        'AGENT_MAX_AGE':        config.AGENT_MAX_AGE,
        'AGENT_ENERGY_DECAY':   config.AGENT_ENERGY_DECAY,
        'AGENT_INITIAL_ENERGY': config.AGENT_INITIAL_ENERGY,
        'FOOD_ENERGY':          config.FOOD_ENERGY,
        'LOG_INTERVAL':         config.LOG_INTERVAL,
        'CHECKPOINT_INTERVAL':  config.CHECKPOINT_INTERVAL,
    }
    config.AGENT_MAX_AGE        = 300
    config.AGENT_ENERGY_DECAY   = 0.3
    config.AGENT_INITIAL_ENERGY = 30.0
    config.FOOD_ENERGY          = 15.0
    return orig


def restore_config(orig):
    import config
    for k, v in orig.items():
        setattr(config, k, v)


def null_sensors():
    """20-element zero sensor vector."""
    return np.zeros(20, dtype=np.float32)
