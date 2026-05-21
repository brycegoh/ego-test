"""Object segmentation via SAM 2 (GPU).

https://github.com/facebookresearch/sam2

Produces the binary object mask that SAM-3D-Objects (stages/sam3d.py) needs, so the
object -> grasp chain can run automatically. First-class GPU stage: ``torch`` / ``sam2`` are
imported at module top (the ``sam2`` extra installs them); imported only when the object
backend is enabled, so the CPU stages stay torch-free.

Contract:
    input  -> RGB frame (H, W, 3) uint8, optional (u, v) pixel point prompt
    output -> binary mask (H, W) bool for the manipulated object

Prompt strategy: when a point is given (the pipeline passes the hand-contact centroid
projected to pixels) SAM 2 is prompted with that positive point so it grabs the object the
hand is holding. Without a prompt it falls back to the automatic mask generator and keeps the
largest mask nearest the image center. Both are heuristics that want validation on real
frames.

Environment (its own venv; see AGENTS.md):
    git clone https://github.com/facebookresearch/sam2 && cd sam2 && pip install -e .
    # download a checkpoint, e.g. checkpoints/sam2.1_hiera_large.pt
"""

from __future__ import annotations

import numpy as np
import torch
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

DEFAULT_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def segment_object(
    rgb: np.ndarray,
    point_prompt: tuple[float, float] | None = None,
    checkpoint: str = DEFAULT_CHECKPOINT,
    config: str = DEFAULT_CONFIG,
    device: str | None = None,
) -> np.ndarray:
    """Return a binary (H, W) object mask for the egocentric frame. Requires GPU."""
    rgb = np.asarray(rgb).astype(np.uint8)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sam2 = build_sam2(config, checkpoint, device=device)

    if point_prompt is not None:
        predictor = SAM2ImagePredictor(sam2)
        predictor.set_image(rgb)
        coords = np.array([[float(point_prompt[0]), float(point_prompt[1])]])
        labels = np.array([1])  # positive point
        masks, scores, _ = predictor.predict(
            point_coords=coords, point_labels=labels, multimask_output=True
        )
        return masks[int(np.argmax(scores))].astype(bool)

    # No prompt: keep the largest automatic mask nearest the image center.
    generator = SAM2AutomaticMaskGenerator(sam2)
    anns = generator.generate(rgb)
    if not anns:
        raise RuntimeError("SAM 2 produced no masks for this frame.")
    h, w = rgb.shape[:2]
    center = np.array([w / 2.0, h / 2.0])

    def score(ann: dict) -> float:
        x, y, bw, bh = ann["bbox"]
        centroid = np.array([x + bw / 2.0, y + bh / 2.0])
        dist = np.linalg.norm(centroid - center) / np.linalg.norm(center)
        return ann["area"] * (1.0 - 0.5 * dist)

    return max(anns, key=score)["segmentation"].astype(bool)
