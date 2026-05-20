"""Render the SO-101 robot posed to the grasp, through the egocentric camera.

Loads the SO-100/SO-101 URDF (from `assets/urdf/`, fetched by `scripts/fetch_urdf.sh`)
into rerun using rerun's built-in URDF loader (rerun-sdk >= 0.29). Each arm's gripper is
placed at the target wrist/grasp pose (direct end-effector placement first; inverse
kinematics for joint angles is a follow-up TODO). A pinhole camera matching the dataset
intrinsics/extrinsics is set up so a snapshot aligns with the original frame.

URDF source: TheRobotStudio/SO-ARM100, Simulation/SO101/so101_new_calib.urdf

Contract:
    input  -> Grasp(s) (+ optional object mesh), camera intrinsics/extrinsics
    output -> rendered robot RGBA snapshot (H, W, 4) saved under outputs/

SCAFFOLD: signatures + contracts only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_URDF = Path("assets/urdf/so101_new_calib.urdf")


def render_robot(
    grasps: dict[str, Any],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    urdf_path: Path = DEFAULT_URDF,
) -> np.ndarray:
    """Pose the SO-101 URDF to `grasps` and return an RGBA snapshot from the camera."""
    raise NotImplementedError
