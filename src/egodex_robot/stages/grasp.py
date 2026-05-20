"""Parallel-gripper grasp generation.

Two paths:
- Preferred (GPU): GraspGen (https://github.com/NVlabs/GraspGen) generates 6-DOF
  parallel-gripper grasps from the reconstructed object mesh (+ hand-pose context).
- Fallback (CPU): KMeans(n_clusters=2) over the hand's fingertip/contact points -> two
  jaw positions -> a gripper pose (midpoint + approach/closing axes). Lets the render and
  overlay stages run without any ML.

Output contract (both paths):
    Grasp { position (3,), rotation (3, 3), width (float) } in the camera/ARKit frame.

SCAFFOLD: signatures + contracts only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Grasp:
    position: np.ndarray  # (3,) gripper center
    rotation: np.ndarray  # (3, 3) gripper orientation
    width: float          # jaw opening, meters


def grasp_from_contacts(contact_points: np.ndarray) -> Grasp:
    """CPU fallback: cluster (N, 3) contact points into 2 jaws -> a parallel-gripper Grasp."""
    raise NotImplementedError


def grasp_from_graspgen(object_mesh: dict[str, Any], hand_pose: Any | None = None) -> Grasp:
    """GPU path: run GraspGen on the object mesh. Requires the GraspGen environment."""
    raise NotImplementedError(
        "GraspGen is not wired up. Set up its GPU environment (see AGENTS.md), "
        "or use grasp_from_contacts() for a CPU fallback."
    )
