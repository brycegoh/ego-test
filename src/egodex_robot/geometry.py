"""Camera + SE(3) geometry helpers shared across stages.

EgoDex provides a 3x3 intrinsics matrix and per-frame 4x4 camera extrinsics in the ARKit
origin frame. These helpers convert between frames and project 3D points to pixels so the
robot render lines up with the original egocentric image.

SCAFFOLD: signatures + contracts only.
"""

from __future__ import annotations

import numpy as np


def project(points_cam: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Project (N, 3) camera-frame points to (N, 2) pixel coordinates."""
    raise NotImplementedError


def transform_points(points: np.ndarray, se3: np.ndarray) -> np.ndarray:
    """Apply a 4x4 SE(3) transform to (N, 3) points."""
    raise NotImplementedError


def invert_se3(se3: np.ndarray) -> np.ndarray:
    """Return the inverse of a 4x4 SE(3) transform."""
    raise NotImplementedError
