"""Rerun visualization of the dataset.

Logs the egocentric RGB stream, the camera pinhole/frustum (from intrinsics + extrinsics),
and the 3D hand keypoints/skeleton per frame. Follows lerobot's own rerun-based dataset
visualization approach.

SCAFFOLD: signatures + contracts only.
"""

from __future__ import annotations

from typing import Any


def log_episode(dataset: Any, episode: int = 0, recording_name: str = "egodex") -> None:
    """Stream one episode (RGB + camera + 3D hand keypoints) to a rerun recording."""
    raise NotImplementedError
