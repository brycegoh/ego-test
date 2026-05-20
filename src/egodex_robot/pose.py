"""Hand-pose decoding from EgoDex/LeRobot annotations.

Primary source of truth: the dataset's ARKit 3D hand tracking, packed into the 48-dim
`observation.state`. This module decodes that vector into structured per-hand poses and
keypoints, expressed in the camera/ARKit frame, for use by viz, grasp, and render.

(HAMER is an optional RGB-only alternative for the same outputs; see stages/hamer.py.)

SCAFFOLD: signatures + contracts only. The exact index layout of the 48-dim state must be
confirmed against the dataset's printed schema before implementing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HandPose:
    """One hand in the camera/ARKit frame."""

    wrist_position: np.ndarray  # (3,) xyz, meters
    wrist_rotation: np.ndarray  # (3, 3) rotation matrix
    keypoints: np.ndarray       # (J, 3) finger-joint positions
    confidence: np.ndarray | None = None  # (J,) optional per-joint confidence


def decode_state(state: np.ndarray) -> dict[str, HandPose]:
    """Decode a 48-dim `observation.state` into {"left": HandPose, "right": HandPose}."""
    raise NotImplementedError


def contact_points(hand: HandPose) -> np.ndarray:
    """Return candidate fingertip/contact points (N, 3) used to seed grasp clustering."""
    raise NotImplementedError
