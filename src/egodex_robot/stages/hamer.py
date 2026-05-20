"""Optional hand-mesh reconstruction from RGB via HAMER.

https://github.com/geopavlakos/hamer

This is an OPTIONAL, RGB-only alternative to the dataset's ARKit hand annotations
(see ../pose.py), useful when ground-truth pose is unavailable or for comparison. HAMER
needs a GPU and its own environment + weights; it is import-gated so that importing this
module never breaks the CPU-only stages.

Contract:
    input  -> RGB frame (H, W, 3)
    output -> MANO hand mesh (vertices, faces) + 3D keypoints in the camera frame

SCAFFOLD: not implemented. Run HAMER in a dedicated environment; see AGENTS.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def reconstruct_hands(rgb: np.ndarray) -> dict[str, Any]:
    """Return per-hand {vertices, faces, keypoints}. Requires the HAMER environment."""
    raise NotImplementedError(
        "HAMER is not wired up. Set up the HAMER GPU environment (see AGENTS.md), "
        "or use the dataset's ARKit hand annotations via pose.decode_state()."
    )
