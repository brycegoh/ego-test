"""Composite the robot render over the original egocentric frame.

Alpha-blends the rendered robot RGBA snapshot (from stages/render.py) onto the original
RGB frame, producing the final "robot positioned like the hand" image.

Contract:
    input  -> original RGB frame (H, W, 3), robot RGBA snapshot (H, W, 4)
    output -> composited RGB image (H, W, 3) saved under outputs/

SCAFFOLD: signatures + contracts only.
"""

from __future__ import annotations

import numpy as np


def overlay(frame_rgb: np.ndarray, robot_rgba: np.ndarray) -> np.ndarray:
    """Alpha-composite `robot_rgba` over `frame_rgb` and return the result (H, W, 3)."""
    raise NotImplementedError
