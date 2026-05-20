"""Render the Mobile ALOHA robot posed to the grasp(s), through the egocentric camera.

Loads the Mobile ALOHA URDF (from `assets/urdf/`, fetched by `scripts/fetch_urdf.sh`) into
rerun using rerun's built-in URDF loader (rerun-sdk >= 0.29).
Mobile ALOHA is bimanual (two ViperX 300 6-DOF arms with parallel-jaw grippers on a mobile
base), so left/right grasps map to the two arms. Each arm's gripper is placed at the
target wrist/grasp pose (direct end-effector placement first; inverse kinematics for joint
angles is a follow-up TODO). A pinhole camera matching the dataset intrinsics/extrinsics is
set up so a snapshot aligns with the original frame.

URDF source: agilexrobotics/mobile_aloha_sim (ready-made flat URDF, fetched by
scripts/fetch_urdf.sh). Meshes are referenced as `package://aloha/...`; if rerun cannot
resolve `package://`, rewrite those refs to relative paths (see fetch_urdf.sh note).

Contract:
    input  -> {"left": Grasp, "right": Grasp} (+ optional object mesh),
              camera intrinsics/extrinsics
    output -> rendered robot RGBA snapshot (H, W, 4) saved under outputs/

SCAFFOLD: signatures + contracts only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_URDF = Path("assets/urdf/aloha/urdf/aloha.urdf")


def render_robot(
    grasps: dict[str, Any],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    urdf_path: Path = DEFAULT_URDF,
) -> np.ndarray:
    """Pose the SO-101 URDF to `grasps` and return an RGBA snapshot from the camera."""
    raise NotImplementedError
