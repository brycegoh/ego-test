"""Composite the robot render over the original egocentric frame.

Alpha-blends the rendered robot RGBA snapshot (from stages/render.py) onto the original
RGB frame, producing the final "robot positioned like the hand" image.

Contract:
    input  -> original RGB frame (H, W, 3), robot RGBA snapshot (H, W, 4)
    output -> composited RGB image (H, W, 3) saved under outputs/
"""

from __future__ import annotations

import numpy as np


def overlay(frame_rgb: np.ndarray, robot_rgba: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    """Alpha-composite ``robot_rgba`` over ``frame_rgb`` and return the result (H, W, 3).

    ``opacity`` scales the robot's alpha (1.0 = fully opaque where rendered). Both inputs
    must share (H, W). Output dtype matches ``frame_rgb`` (typically uint8).
    """
    frame_rgb = np.asarray(frame_rgb)
    robot_rgba = np.asarray(robot_rgba)
    if frame_rgb.shape[:2] != robot_rgba.shape[:2]:
        raise ValueError(
            f"Size mismatch: frame {frame_rgb.shape[:2]} vs robot {robot_rgba.shape[:2]}."
        )
    if robot_rgba.shape[2] != 4:
        raise ValueError("robot_rgba must have 4 channels (RGBA).")

    alpha = (robot_rgba[..., 3:4].astype(np.float64) / 255.0) * float(opacity)
    blended = frame_rgb[..., :3].astype(np.float64) * (1.0 - alpha) + robot_rgba[..., :3].astype(
        np.float64
    ) * alpha
    return np.clip(blended, 0, 255).astype(frame_rgb.dtype)
