"""Shared per-frame retargeting pipeline used by both the CLI and the rerun viz.

One frame in, one posed-and-rendered robot out. ``Backends`` selects which stage
implementations to use; the **default is the full GPU chain** (HAMER hands -> SAM 2 mask ->
SAM-3D object -> GraspGen grasp), with a CPU fallback (``Backends.cpu()``: ARKit pose +
KMeans grasp) that runs without a GPU.

Only the selected backend's stage module is imported, and lazily -- the three GPU stacks
(HAMER / SAM-3D / GraspGen) have conflicting CUDA pins and cannot coexist in one process, so
you install and run one at a time. Importing this module never pulls in torch.

Frame convention: the world frame is the ARKit origin frame. ARKit hand poses are already in
world; HAMER/SAM-3D outputs are in a CV camera frame and are lifted to world with the
dataset extrinsics (``geometry.cv_camera_*_to_world``). All grasp candidates and IK targets
are therefore compared and solved in one consistent world frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import geometry, pose
from .ik import EE_FRAMES, RobotIK
from .stages import grasp as grasp_stage
from .stages import render as render_stage


@dataclass
class Backends:
    """Which stage implementations to run. Defaults to the full GPU chain."""

    hand: str = "hamer"          # "hamer" (GPU) | "arkit" (CPU)
    use_object: bool = True      # SAM 2 mask -> SAM-3D object reconstruction (GPU)
    grasp: str = "graspgen"      # "graspgen" (GPU, needs use_object) | "kmeans" (CPU)
    gripper_config: str = ""     # GraspGen gripper YAML (required for grasp="graspgen")
    sam2_checkpoint: str = "checkpoints/sam2.1_hiera_large.pt"
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam3d_config: str = "checkpoints/hf/pipeline.yaml"

    @classmethod
    def cpu(cls) -> "Backends":
        """The no-GPU fallback: ARKit hand pose + KMeans grasp, no object reconstruction."""
        return cls(hand="arkit", use_object=False, grasp="kmeans")


@dataclass
class RetargetResult:
    hands: dict[str, pose.HandPose]
    grasps: dict[str, grasp_stage.Grasp]
    ik: Any                       # ik.IKResult
    rgba: np.ndarray              # (H, W, 4) robot render
    object_mesh: dict[str, Any] | None = None


def _lift_handpose(hand: pose.HandPose, extrinsics: np.ndarray) -> pose.HandPose:
    """Lift a HandPose from the (HAMER) CV camera frame into the world frame."""
    rot_world = extrinsics[:3, :3] @ geometry.GL_CV_FLIP @ hand.wrist_rotation
    return pose.HandPose(
        wrist_position=geometry.cv_camera_points_to_world(hand.wrist_position, extrinsics)[0],
        wrist_rotation=rot_world,
        keypoints=geometry.cv_camera_points_to_world(hand.keypoints, extrinsics),
        confidence=hand.confidence,
    )


def _decode_hands(frame: dict, backends: Backends) -> dict[str, pose.HandPose]:
    if backends.hand == "arkit":
        return pose.decode_state(frame["observation.state"])
    if backends.hand == "hamer":
        from .stages import hamer  # GPU; imported only when selected

        recon = hamer.reconstruct_hands(frame["rgb"])
        return {
            side: _lift_handpose(hamer.hamer_to_handpose(data), frame["extrinsics"])
            for side, data in recon.items()
        }
    raise ValueError(f"Unknown hand backend {backends.hand!r} (expected 'hamer' or 'arkit').")


def _reconstruct_object(frame: dict, hands: dict, backends: Backends) -> dict[str, Any]:
    """SAM 2 mask -> SAM-3D mesh, lifted to the world frame."""
    from .stages import sam3d, segment  # GPU; imported only when selected

    # Prompt SAM 2 at the hand-contact centroid projected to pixels (the held object).
    point_prompt = None
    if hands:
        contacts = np.vstack([pose.contact_points(h) for h in hands.values()])
        centroid_world = contacts.mean(axis=0, keepdims=True)
        uv = geometry.pixels_from_world(centroid_world, frame["intrinsics"], frame["extrinsics"])
        point_prompt = (float(uv[0, 0]), float(uv[0, 1]))

    mask = segment.segment_object(
        frame["rgb"], point_prompt, backends.sam2_checkpoint, backends.sam2_config
    )
    obj_cam = sam3d.reconstruct_object(frame["rgb"], mask=mask, config_path=backends.sam3d_config)
    return {
        "vertices": geometry.cv_camera_points_to_world(obj_cam["vertices"], frame["extrinsics"]),
        "faces": obj_cam["faces"],
        "pose": geometry.cv_camera_pose_to_world(obj_cam["pose"], frame["extrinsics"]),
    }


def _grasps(
    hands: dict, object_mesh: dict | None, backends: Backends
) -> dict[str, grasp_stage.Grasp]:
    if backends.grasp == "kmeans":
        return {
            side: grasp_stage.grasp_from_contacts(pose.contact_points(h))
            for side, h in hands.items()
        }
    if backends.grasp == "graspgen":
        from .stages import graspgen  # GPU; imported only when selected

        if object_mesh is None:
            raise ValueError("grasp='graspgen' requires use_object=True (an object mesh).")
        candidates = graspgen.graspgen_candidates(object_mesh, backends.gripper_config)
        return {side: grasp_stage.select_grasp(candidates, h) for side, h in hands.items()}
    raise ValueError(f"Unknown grasp backend {backends.grasp!r} (expected 'graspgen' or 'kmeans').")


def retarget_frame(
    frame: dict,
    backends: Backends | None = None,
    ik: RobotIK | None = None,
) -> RetargetResult:
    """Run one frame end-to-end: hands -> (object) -> grasp -> IK -> render.

    ``frame`` is a dataset.get_frame dict (rgb, observation.state, intrinsics, extrinsics).
    Pass a reused ``RobotIK`` when looping over an episode. Returns the posed result plus the
    rendered robot RGBA.
    """
    backends = backends or Backends()
    ik = ik or RobotIK()

    hands = _decode_hands(frame, backends)
    object_mesh = _reconstruct_object(frame, hands, backends) if backends.use_object else None
    grasps = _grasps(hands, object_mesh, backends)

    targets = {side: grasps[side].as_se3() for side in EE_FRAMES if side in grasps}
    base = geometry.robot_base_from_camera(frame["extrinsics"])
    result = ik.solve(targets, base_transform=base)

    height, width = frame["rgb"].shape[:2]
    rgba = render_stage.render_robot(
        result.link_transforms, frame["intrinsics"], frame["extrinsics"], (width, height), ik.urdf_path
    )
    return RetargetResult(hands=hands, grasps=grasps, ik=result, rgba=rgba, object_mesh=object_mesh)
