"""Object 3D reconstruction from RGB via SAM-3D-Objects.

https://github.com/facebookresearch/sam-3d-objects

Reconstructs the manipulated object from the egocentric frame so it can be placed in the
rerun scene and used as grasp context for GraspGen. Needs a GPU + its own environment +
weights; everything is import-gated.

Contract:
    input  -> RGB frame (H, W, 3) uint8 + a binary object mask (H, W) (e.g. from SAM)
    output -> {"vertices": (N, 3), "faces": (F, 3) | None, "pose": (4, 4)} in the camera
              frame. SAM 3D Objects emits a Gaussian splat plus a local->camera layout
              (rotation quaternion + translation + scale); we expose the splat points as the
              object geometry and pack the layout into ``pose``.

Environment (do NOT add to the core env; see AGENTS.md):
    git clone https://github.com/facebookresearch/sam-3d-objects && cd sam-3d-objects
    pip install -e . && download the checkpoints into checkpoints/hf/
    (see the repo's notebook/ for mask prompting; masks come from SAM/SAM2.)

This wrapper mirrors sam-3d-objects/demo.py (`Inference(config).__call__(image, mask)`). It
is GPU-only and cannot be exercised in the CPU sandbox where it was written. Note: SAM 3D
outputs a splat, not a watertight mesh -- proper meshing uses the repo's mesh-postprocess /
notebooks; here ``faces`` is a convex hull over the splat points as a usable stand-in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_SETUP_HINT = (
    "SAM-3D-Objects is not available. Set up its GPU environment (clone "
    "facebookresearch/sam-3d-objects, pip install -e ., fetch checkpoints/hf/). See AGENTS.md."
)

DEFAULT_CONFIG = "checkpoints/hf/pipeline.yaml"


def _quat_trans_to_se3(quat_wxyz: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Build a 4x4 from a (w, x, y, z) quaternion + translation."""
    w, x, y, z = quat_wxyz
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = np.asarray(trans, dtype=float).reshape(3)
    return out


def reconstruct_object(
    rgb: np.ndarray,
    mask: np.ndarray | None = None,
    config_path: str | Path = DEFAULT_CONFIG,
    seed: int = 42,
) -> dict[str, Any]:
    """Return {vertices, faces, pose} for the masked object in the camera frame. Requires GPU."""
    try:
        import sys

        # The repo ships its inference helpers under notebook/; make them importable.
        repo_notebook = Path(config_path).resolve().parents[2] / "notebook"
        if repo_notebook.is_dir() and str(repo_notebook) not in sys.path:
            sys.path.append(str(repo_notebook))
        from inference import Inference  # noqa: E402  (from sam-3d-objects/notebook)
    except ImportError as exc:  # pragma: no cover - GPU env only
        raise NotImplementedError(_SETUP_HINT) from exc

    if mask is None:
        raise ValueError(
            "SAM-3D-Objects needs a binary object mask (H, W). Prompt SAM/SAM2 to segment "
            "the manipulated object, then pass its mask here."
        )

    inference = Inference(str(config_path), compile=False)
    output = inference(np.asarray(rgb).astype(np.uint8), np.asarray(mask) > 0, seed=seed)

    gaussian = output["gaussian"][0]
    vertices = gaussian.get_xyz.detach().cpu().numpy()

    faces = None
    try:  # convex hull is a usable stand-in for the splat's surface
        import trimesh

        faces = trimesh.convex.convex_hull(vertices).faces
    except Exception:  # pragma: no cover
        faces = None

    quat = output["rotation"][0].detach().cpu().numpy().reshape(-1)  # (w, x, y, z), local->cam
    trans = output["translation"][0].detach().cpu().numpy().reshape(3)
    pose = _quat_trans_to_se3(quat, trans)

    return {"vertices": np.asarray(vertices, dtype=float), "faces": faces, "pose": pose}
