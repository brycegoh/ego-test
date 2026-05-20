"""Rerun visualization: a three-panel side-by-side comparison.

The viz lays out, left-to-right, three synchronized panels via a rerun blueprint:

    ┌──────────────┬──────────────┬──────────────┐
    │   3D scene   │ edited video │ orig. video  │
    │ (Spatial3D)  │ (Spatial2D)  │ (Spatial2D)  │
    └──────────────┴──────────────┴──────────────┘

- 3D scene: the reconstructed scene — camera frustum, 3D hand keypoints/skeleton,
  object mesh (SAM-3D), and the posed SO-101 URDF (rerun's built-in URDF loader).
- edited video: the per-frame overlay output (robot composited over the frame).
- original video: the raw egocentric RGB stream.

All three share the dataset timeline so scrubbing stays in sync. Follows lerobot's
rerun-based dataset visualization approach.

SCAFFOLD: signatures + contracts only.
"""

from __future__ import annotations

from typing import Any

# Entity path roots for the three panels (kept stable so the blueprint can target them).
SCENE_3D = "scene"        # camera, hand keypoints, object mesh, SO-101 URDF
EDITED_2D = "edited"      # robot-overlaid frame
ORIGINAL_2D = "original"  # raw egocentric frame


def build_blueprint() -> Any:
    """Return a rerun blueprint: 3D scene | edited video | original video, side by side."""
    raise NotImplementedError


def log_episode(dataset: Any, episode: int = 0, recording_name: str = "egodex") -> None:
    """Stream one episode into the three-panel layout.

    Per frame, logs: original RGB (ORIGINAL_2D), the overlay render (EDITED_2D), and the
    3D scene contents (SCENE_3D), all on the shared dataset timeline.
    """
    raise NotImplementedError
