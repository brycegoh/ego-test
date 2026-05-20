"""Object 3D reconstruction from RGB via SAM-3D-Objects.

https://github.com/facebookresearch/sam-3d-objects

Reconstructs the manipulated object's 3D mesh from the egocentric frame so it can be
placed in the rerun scene and used as grasp context. Needs a GPU + its own environment +
weights; import-gated.

Contract:
    input  -> RGB frame (H, W, 3) [+ optional 2D prompt/mask for the target object]
    output -> object mesh (vertices, faces) + 6-DOF pose in the camera frame

SCAFFOLD: not implemented. Run SAM-3D-Objects in a dedicated environment; see AGENTS.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def reconstruct_object(rgb: np.ndarray, prompt: Any | None = None) -> dict[str, Any]:
    """Return {vertices, faces, pose}. Requires the SAM-3D-Objects environment."""
    raise NotImplementedError(
        "SAM-3D-Objects is not wired up. Set up its GPU environment (see AGENTS.md)."
    )
