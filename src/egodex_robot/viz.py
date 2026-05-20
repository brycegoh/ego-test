"""Rerun visualization: a three-panel side-by-side comparison.

The viz lays out, left-to-right, three synchronized panels via a rerun blueprint:

    ┌──────────────┬──────────────┬──────────────┐
    │   3D scene   │ edited video │ orig. video  │
    │ (Spatial3D)  │ (Spatial2D)  │ (Spatial2D)  │
    └──────────────┴──────────────┴──────────────┘

- 3D scene: camera frustum (Pinhole + extrinsics), 3D hand keypoints, and the posed Mobile
  ALOHA robot. The robot is drawn by placing each visual mesh at its **placo FK world
  transform** -- the exact same transforms that drive the pyrender overlay -- so the 3D
  panel and the edited panel can never disagree (the plan's "single kinematics source of
  truth"). rerun is a viewer here, not a kinematics engine.
- edited video: the per-frame overlay output (robot composited over the frame).
- original video: the raw egocentric RGB stream.

All three share the dataset timeline so scrubbing stays in sync.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import geometry, pose
from .ik import EE_FRAMES, RobotIK
from .stages import grasp as grasp_stage
from .stages import overlay as overlay_stage
from .stages import render as render_stage

# Entity path roots for the three panels (kept stable so the blueprint can target them).
SCENE_3D = "scene"        # camera, hand keypoints, robot meshes
EDITED_2D = "edited"      # robot-overlaid frame
ORIGINAL_2D = "original"  # raw egocentric frame

_ROBOT_ROOT = f"{SCENE_3D}/robot"
_TIMELINE = "frame"


def build_blueprint() -> Any:
    """Return a rerun blueprint: 3D scene | edited video | original video, side by side."""
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin=SCENE_3D, name="3D scene"),
            rrb.Spatial2DView(origin=EDITED_2D, name="edited"),
            rrb.Spatial2DView(origin=ORIGINAL_2D, name="original"),
        ),
        collapse_panels=True,
    )


def log_robot_meshes_static(visuals: list[render_stage.Visual]) -> None:
    """Log each visual mesh once (static); per-frame transforms reposition them."""
    import rerun as rr
    import trimesh  # noqa: F401  (load happens in render_stage helper)

    cache = render_stage._load_mesh_cache(visuals)
    for i, visual in enumerate(visuals):
        mesh = cache[visual.mesh_path]
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        if not np.allclose(visual.scale, 1.0):
            verts = verts * visual.scale.astype(np.float32)
        rr.log(
            f"{_ROBOT_ROOT}/{visual.link}/visual_{i}",
            rr.Mesh3D(
                vertex_positions=verts,
                triangle_indices=np.asarray(mesh.faces, dtype=np.uint32),
            ),
            static=True,
        )


def log_robot_pose(visuals: list[render_stage.Visual], link_transforms: dict[str, np.ndarray]) -> None:
    """Per-frame: place each visual at ``link_world @ link->visual origin`` (placo FK)."""
    import rerun as rr

    for i, visual in enumerate(visuals):
        link_world = link_transforms.get(visual.link)
        if link_world is None:
            continue
        world = np.asarray(link_world, dtype=float) @ visual.origin
        rr.log(
            f"{_ROBOT_ROOT}/{visual.link}/visual_{i}",
            rr.Transform3D(translation=world[:3, 3], mat3x3=world[:3, :3]),
        )


def log_camera(intrinsics: np.ndarray, extrinsics: np.ndarray, width: int, height: int) -> None:
    """Log the egocentric camera frustum (pinhole + world pose) into the 3D panel."""
    import rerun as rr

    extrinsics = np.asarray(extrinsics, dtype=float)
    rr.log(
        f"{SCENE_3D}/camera",
        rr.Transform3D(translation=extrinsics[:3, 3], mat3x3=extrinsics[:3, :3]),
    )
    rr.log(
        f"{SCENE_3D}/camera/image",
        rr.Pinhole(
            image_from_camera=np.asarray(intrinsics, dtype=float),
            width=width,
            height=height,
            camera_xyz=rr.ViewCoordinates.RUB,  # ARKit/OpenGL: x right, y up, z back
        ),
    )


def log_hands(hands: dict[str, pose.HandPose]) -> None:
    """Log per-hand wrist + fingertip keypoints into the 3D panel."""
    import rerun as rr

    colors = {"left": [80, 160, 255], "right": [255, 160, 80]}
    for side, hand in hands.items():
        pts = np.vstack([hand.wrist_position.reshape(1, 3), hand.keypoints.reshape(-1, 3)])
        rr.log(f"{SCENE_3D}/hand/{side}", rr.Points3D(pts, colors=colors.get(side), radii=0.008))


def log_episode(dataset: Any, episode: int = 0, recording_name: str = "egodex") -> None:
    """Stream one episode into the three-panel layout.

    Per frame: decode the ARKit hand pose, place the robot base relative to the camera,
    solve IK for both follower arms, render the robot through the camera, overlay it on the
    RGB, and log original RGB (ORIGINAL_2D), the overlay (EDITED_2D) and the 3D scene
    (camera + hands + posed robot) on the shared dataset timeline.

    Requires a loaded ``LeRobotDataset`` (so HF + lerobot must be available).
    """
    import rerun as rr

    from .dataset import iter_episode_frames

    rr.init(recording_name, spawn=True)
    rr.send_blueprint(build_blueprint())

    ik = RobotIK()
    visuals = render_stage.parse_visuals(ik.urdf_path)
    log_robot_meshes_static(visuals)

    for frame_index, frame in iter_episode_frames(dataset, episode):
        rr.set_time(_TIMELINE, sequence=frame_index)
        rgb = frame["rgb"]
        height, width = rgb.shape[:2]
        intrinsics = frame["intrinsics"]
        extrinsics = frame["extrinsics"]

        rr.log(ORIGINAL_2D, rr.Image(rgb))
        log_camera(intrinsics, extrinsics, width, height)

        hands = pose.decode_state(frame["observation.state"])
        log_hands(hands)

        base = geometry.robot_base_from_camera(extrinsics)
        targets = {
            side: grasp_stage.grasp_from_contacts(pose.contact_points(hands[side])).as_se3()
            for side in EE_FRAMES
            if side in hands
        }
        result = ik.solve(targets, base_transform=base)
        log_robot_pose(visuals, result.link_transforms)

        robot_rgba = render_stage.render_robot(
            result.link_transforms, intrinsics, extrinsics, (width, height), ik.urdf_path
        )
        rr.log(EDITED_2D, rr.Image(overlay_stage.overlay(rgb, robot_rgba)))
