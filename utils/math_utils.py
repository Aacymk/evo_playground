import numpy as np


def angle_diff(a, b):
    """Smallest signed difference between two angles (radians)."""
    diff = (b - a + np.pi) % (2 * np.pi) - np.pi
    return diff


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def distance(ax, ay, bx, by):
    return np.hypot(bx - ax, by - ay)


def normalize_angle(a):
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def random_direction():
    return np.random.uniform(-np.pi, np.pi)


def heading_vector(angle):
    """Unit vector from angle."""
    return np.array([np.cos(angle), np.sin(angle)])
