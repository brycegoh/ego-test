"""Optional hand-mesh reconstruction from RGB via HAMER.

https://github.com/geopavlakos/hamer

OPTIONAL, RGB-only alternative to the dataset's ARKit hand annotations (see ../pose.py),
useful when ground-truth pose is unavailable or for comparison. HAMER needs a GPU and its
own environment + weights (plus a detectron2 ViTDet detector and ViTPose); everything is
import-gated so importing this module never breaks the CPU-only stages.

Contract:
    input  -> RGB frame (H, W, 3), uint8
    output -> {"left"/"right": {vertices (778,3), faces (F,3), keypoints (21,3)}} in the
              HAMER camera frame (metric, +x right / +y down / +z forward). Lift to the
              world/ARKit frame with the dataset extrinsics if you need to mix with the
              ARKit pose path.

Environment (do NOT add to the core env; see AGENTS.md):
    git clone https://github.com/geopavlakos/hamer && cd hamer
    pip install -e .[all] && pip install -v -e third-party/ViTPose
    bash fetch_demo_data.sh        # downloads checkpoints into _DATA/

This wrapper mirrors hamer/demo.py: detector -> ViTPose hand boxes -> ViTDetDataset ->
HAMER model. It is GPU-only and cannot be exercised in the CPU sandbox where it was written.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_SETUP_HINT = (
    "HAMER is not available. Set up its GPU environment (clone geopavlakos/hamer, "
    "pip install -e .[all], install third-party/ViTPose, run fetch_demo_data.sh), "
    "or use the dataset's ARKit hand annotations via pose.decode_state(). See AGENTS.md."
)


def _load_models(device: str):
    """Load the HAMER model + ViTDet detector + ViTPose, following hamer/demo.py."""
    from pathlib import Path

    import torch
    from hamer.configs import CACHE_DIR_HAMER
    from hamer.models import DEFAULT_CHECKPOINT, download_models, load_hamer
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    from vitpose_model import ViTPoseModel

    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)
    model = model.to(device).eval()

    import hamer
    from detectron2.config import LazyConfig

    cfg_path = Path(hamer.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = (
        "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/"
        "f328730692/model_final_f05665.pkl"
    )
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)
    cpm = ViTPoseModel(device)
    return model, model_cfg, detector, cpm


def _detect_hand_boxes(rgb: np.ndarray, detector: Any, cpm: Any) -> tuple[np.ndarray, np.ndarray]:
    """Run the body detector + ViTPose to get hand boxes and a left/right flag."""
    # detectron2 expects BGR
    det_out = detector(rgb[:, :, ::-1])
    instances = det_out["instances"]
    keep = (instances.pred_classes == 0) & (instances.scores > 0.5)
    boxes = instances.pred_boxes.tensor[keep].cpu().numpy()
    scores = instances.scores[keep].cpu().numpy()

    vitposes_out = cpm.predict_pose(
        rgb, [np.concatenate([boxes, scores[:, None]], axis=1)]
    )

    hand_boxes, is_right = [], []
    for vitposes in vitposes_out:
        for side, slc in (("left", slice(-42, -21)), ("right", slice(-21, None))):
            keyp = vitposes["keypoints"][slc]
            valid = keyp[:, 2] > 0.5
            if valid.sum() > 3:
                hand_boxes.append(
                    [keyp[valid, 0].min(), keyp[valid, 1].min(),
                     keyp[valid, 0].max(), keyp[valid, 1].max()]
                )
                is_right.append(1 if side == "right" else 0)
    if not hand_boxes:
        return np.empty((0, 4)), np.empty((0,), dtype=int)
    return np.stack(hand_boxes), np.stack(is_right)


def reconstruct_hands(rgb: np.ndarray, device: str | None = None) -> dict[str, Any]:
    """Return per-hand {vertices, faces, keypoints} in the HAMER camera frame. Requires GPU."""
    try:
        import torch
        from hamer.datasets.vitdet_dataset import ViTDetDataset
        from hamer.utils import recursive_to
    except ImportError as exc:  # pragma: no cover - GPU env only
        raise NotImplementedError(_SETUP_HINT) from exc

    rgb = np.asarray(rgb)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg, detector, cpm = _load_models(device)

    boxes, right = _detect_hand_boxes(rgb, detector, cpm)
    if len(boxes) == 0:
        return {}

    dataset = ViTDetDataset(model_cfg, rgb[:, :, ::-1], boxes, right)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    faces = np.asarray(model.mano.faces)

    hands: dict[str, Any] = {}
    for batch in loader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)
        verts = out["pred_vertices"].cpu().numpy()
        keyp = out["pred_keypoints_3d"].cpu().numpy()
        cam_t = out["pred_cam_t"].cpu().numpy()
        is_right = batch["right"].cpu().numpy()
        for n in range(verts.shape[0]):
            side = "right" if is_right[n] > 0.5 else "left"
            # MANO is built right-handed; mirror x for the left hand (as hamer/demo.py does).
            sign = 1.0 if side == "right" else -1.0
            v = verts[n].copy(); v[:, 0] *= sign
            k = keyp[n].copy(); k[:, 0] *= sign
            hands[side] = {
                "vertices": v + cam_t[n],
                "faces": faces,
                "keypoints": k + cam_t[n],
            }
    return hands
