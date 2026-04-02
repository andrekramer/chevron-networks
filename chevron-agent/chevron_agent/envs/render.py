from __future__ import annotations

from typing import Iterable

import numpy as np


COLORS: dict[int, tuple[float, float, float]] = {
    0: (0.0, 0.0, 0.0),   # empty
    1: (1.0, 1.0, 1.0),   # agent
    2: (0.1, 0.3, 1.0),   # cue blue
    3: (1.0, 0.2, 0.2),   # cue red
    4: (0.2, 0.5, 1.0),   # masked blue cue
    5: (1.0, 0.5, 0.5),   # masked red cue
    6: (0.1, 0.3, 1.0),   # target a
    7: (1.0, 0.2, 0.2),   # target b
    8: (1.0, 0.9, 0.0),   # trap
    9: (0.0, 0.9, 0.2),   # lure
}


def render_layers(size: int, layers: Iterable[tuple[int, tuple[int, int]]]) -> np.ndarray:
    image = np.zeros((3, size, size), dtype=np.float32)
    for token, pos in layers:
        color = COLORS[token]
        y, x = pos
        image[:, y, x] = color
    return image


def render_symbolic_layers(size: int, layers: Iterable[tuple[int, tuple[int, int]]]) -> np.ndarray:
    channels = max(COLORS)  # skip empty channel 0
    image = np.zeros((channels, size, size), dtype=np.float32)
    for token, pos in layers:
        if token == 0:
            continue
        y, x = pos
        image[token - 1, y, x] = 1.0
    return image
