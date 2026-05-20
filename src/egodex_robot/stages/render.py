"""Render the Mobile ALOHA robot posed to the grasp(s), through the egocentric camera.

Loads the Mobile ALOHA URDF (from `assets/urdf/`, fetched by `scripts/fetch_urdf.sh`) into
rerun using rerun's built-in URDF loader (rerun-sdk >= 0.29). Each arm's gripper is placed
at the target wrist/grasp pose (direct end-effector placement first; inverse kinematics for
joint angles is a follow-up TODO). A pinhole camera matching the dataset
intrinsics/extrinsics is set up so a snapshot aligns with the original frame.

URDF source: agilexrobotics/mobile_aloha_sim, `aloha_tracer2_dabai_dark.urdf` (a flat URDF:
two 6-DOF parallel-gripper arms on an AgileX Tracer base). The model has four arm chains:
`fl_`/`fr_` (front = followers, the manipulators we retarget) and `bl_`/`br_` (back =
leaders). Map the left/right hand grasps onto the `fl_`/`fr_` follower arms.

Meshes are referenced as `package://<pkg>/...`. rerun's URDF loader resolves these via
`ROS_PACKAGE_PATH` (NOT relative paths) -- it must include PACKAGE_ROOT (the dir holding
the `aloha_new_description/` and `tracer2_description/` package folders). Validated: loading
with `ROS_PACKAGE_PATH=<PACKAGE_ROOT>` embeds all 99 meshes.

Contract:
    input  -> {"left": Grasp, "right": Grasp} (+ optional object mesh),
              camera intrinsics/extrinsics
    output -> rendered robot RGBA snapshot (H, W, 4) saved under outputs/

SCAFFOLD: signatures + contracts only.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

PACKAGE_ROOT = Path("assets/urdf")  # set ROS_PACKAGE_PATH here so package:// meshes resolve
DEFAULT_URDF = PACKAGE_ROOT / "aloha_new_description/urdf/aloha_tracer2_dabai_dark.urdf"
FOLLOWER_ARM_PREFIXES = {"left": "fl_", "right": "fr_"}


def ensure_package_path(root: Path = PACKAGE_ROOT) -> None:
    """Prepend `root` to ROS_PACKAGE_PATH so rerun resolves `package://` mesh refs."""
    abs_root = str(root.resolve())
    existing = os.environ.get("ROS_PACKAGE_PATH", "")
    if abs_root not in existing.split(os.pathsep):
        os.environ["ROS_PACKAGE_PATH"] = (
            f"{abs_root}{os.pathsep}{existing}" if existing else abs_root
        )


def render_robot(
    grasps: dict[str, Any],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    urdf_path: Path = DEFAULT_URDF,
) -> np.ndarray:
    """Pose the Mobile ALOHA URDF to `grasps` and return an RGBA snapshot from the camera."""
    raise NotImplementedError
