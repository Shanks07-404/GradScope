"""
GradScope Toy Datasets
Synthetic 2D toy dataset generators implemented in pure Python (zero external dependencies).
Generates Spirals, Moons, and XOR datasets.
"""

from __future__ import annotations
import math
import random
from typing import List, Tuple


def make_spirals(
    n_points: int = 200,
    noise: float = 0.1,
    seed: int | None = 42
) -> Tuple[List[List[float]], List[int]]:
    """
    Generates two intertwined Archimedean spirals.
    Returns:
        points: list of [x, y] coordinates in ~[-1.5, 1.5]
        labels: list of class labels (0 or 1)
    """
    if seed is not None:
        random.seed(seed)

    points: List[List[float]] = []
    labels: List[int] = []
    points_per_class = n_points // 2

    for class_idx in (0, 1):
        for i in range(points_per_class):
            r = (i / points_per_class) * 1.2 + 0.1  # radius
            t = 1.75 * math.pi * (i / points_per_class) * 2.0 + (class_idx * math.pi)  # angle
            nx = random.gauss(0, noise) if noise > 0 else 0.0
            ny = random.gauss(0, noise) if noise > 0 else 0.0
            x = r * math.sin(t) + nx
            y = r * math.cos(t) + ny
            points.append([round(x, 4), round(y, 4)])
            labels.append(class_idx)

    return points, labels


def make_moons(
    n_points: int = 200,
    noise: float = 0.1,
    seed: int | None = 42
) -> Tuple[List[List[float]], List[int]]:
    """
    Generates two interleaving half-circle moons.
    Returns:
        points: list of [x, y] coordinates in ~[-1.5, 1.5]
        labels: list of class labels (0 or 1)
    """
    if seed is not None:
        random.seed(seed)

    points: List[List[float]] = []
    labels: List[int] = []
    n_pts_out = n_points // 2
    n_pts_in = n_points - n_pts_out

    # Upper moon (class 0)
    for i in range(n_pts_out):
        theta = math.pi * (i / max(n_pts_out - 1, 1))
        nx = random.gauss(0, noise) if noise > 0 else 0.0
        ny = random.gauss(0, noise) if noise > 0 else 0.0
        x = math.cos(theta) * 0.9 - 0.4 + nx
        y = math.sin(theta) * 0.9 - 0.2 + ny
        points.append([round(x, 4), round(y, 4)])
        labels.append(0)

    # Lower moon (class 1)
    for i in range(n_pts_in):
        theta = math.pi * (i / max(n_pts_in - 1, 1))
        nx = random.gauss(0, noise) if noise > 0 else 0.0
        ny = random.gauss(0, noise) if noise > 0 else 0.0
        x = (1.0 - math.cos(theta)) * 0.9 - 0.4 + nx
        y = (1.0 - math.sin(theta)) * 0.9 - 0.7 + ny
        points.append([round(x, 4), round(y, 4)])
        labels.append(1)

    return points, labels


def make_xor(
    n_points: int = 200,
    noise: float = 0.1,
    seed: int | None = 42
) -> Tuple[List[List[float]], List[int]]:
    """
    Generates classic 4-quadrant XOR dataset.
    Class 0: Quadrants 1 & 3 (+,+) and (-,-)
    Class 1: Quadrants 2 & 4 (+,-) and (-,+)
    """
    if seed is not None:
        random.seed(seed)

    points: List[List[float]] = []
    labels: List[int] = []
    pts_per_cluster = n_points // 4

    clusters = [
        (0.7, 0.7, 0),    # Top-right (+, +) -> class 0
        (-0.7, -0.7, 0),  # Bottom-left (-, -) -> class 0
        (-0.7, 0.7, 1),   # Top-left (-, +) -> class 1
        (0.7, -0.7, 1),   # Bottom-right (+, -) -> class 1
    ]

    for cx, cy, label in clusters:
        for _ in range(pts_per_cluster):
            nx = random.gauss(0, noise + 0.05) if noise >= 0 else 0.0
            ny = random.gauss(0, noise + 0.05) if noise >= 0 else 0.0
            points.append([round(cx + nx, 4), round(cy + ny, 4)])
            labels.append(label)

    return points, labels
