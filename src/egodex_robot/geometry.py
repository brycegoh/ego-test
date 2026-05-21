"""Camera + SE(3) geometry helpers shared across stages.

EgoDex provides a 3x3 intrinsics matrix and per-frame 4x4 camera extrinsics in the ARKit
origin frame. These helpers convert between frames and project 3D points to pixels so the
robot render lines up with the original egocentric image.

Frame conventions (the alignment crux -- everything downstream depends on these):

- **World frame** = the ARKit origin frame. Hand poses, camera extrinsics, the robot base
  and all posed robot links are expressed here, so the 3D panel and the overlay can never
  drift apart.
- **Camera extrinsics** are ``T_world_cam`` (camera-to-world), in the ARKit/OpenGL camera
  convention: +x right, +y up, **-z forward** (the camera looks down its own -z). This is
  the convention pyrender's camera node also uses, so the extrinsics feed pyrender directly.
- **Pinhole projection** uses the computer-vision convention: +x right, +y down, **+z
  forward** (points in front of the camera have positive depth). ``project`` therefore
  expects points already in this CV camera frame; ``pixels_from_world`` does the
  world -> ARKit-cam -> CV-cam -> pixel chain for you.

The flip between the two camera conventions is a 180 deg rotation about x, ``diag(1,-1,-1)``.
"""

from __future__ import annotations

import numpy as np

# OpenGL/ARKit camera (x right, y up, z back) <-> CV camera (x right, y down, z forward).
# Negating y and z is its own inverse, so one constant serves both directions.
GL_CV_FLIP = np.diag([1.0, -1.0, -1.0])


def project(points_cam: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Project (N, 3) CV-camera-frame points (+z forward) to (N, 2) pixel coordinates."""
    pts = np.asarray(points_cam, dtype=float).reshape(-1, 3)
    intrinsics = np.asarray(intrinsics, dtype=float)
    uvw = pts @ intrinsics.T  # (N, 3) = (K @ p)^T
    z = uvw[:, 2:3]
    z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    return uvw[:, :2] / z


def transform_points(points: np.ndarray, se3: np.ndarray) -> np.ndarray:
    """Apply a 4x4 SE(3) transform to (N, 3) points, returning (N, 3)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    se3 = np.asarray(se3, dtype=float)
    return pts @ se3[:3, :3].T + se3[:3, 3]


def invert_se3(se3: np.ndarray) -> np.ndarray:
    """Return the inverse of a 4x4 SE(3) transform (using R^T, -R^T t)."""
    se3 = np.asarray(se3, dtype=float)
    rot = se3[:3, :3]
    trans = se3[:3, 3]
    out = np.eye(4)
    out[:3, :3] = rot.T
    out[:3, 3] = -rot.T @ trans
    return out


def pixels_from_world(
    points_world: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray
) -> np.ndarray:
    """Project (N, 3) world points to pixels given K and ``T_world_cam`` (ARKit/GL camera).

    Chains world -> ARKit camera -> CV camera -> pixels, so callers can hand in world-frame
    hand keypoints / link origins and get image coordinates that line up with the render.
    """
    t_cam_world = invert_se3(extrinsics)
    pts_cam_gl = transform_points(points_world, t_cam_world)
    pts_cam_cv = pts_cam_gl @ GL_CV_FLIP.T
    return project(pts_cam_cv, intrinsics)


def make_se3(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Pack a (3, 3) rotation and (3,) translation into a 4x4 SE(3) matrix."""
    out = np.eye(4)
    out[:3, :3] = np.asarray(rotation, dtype=float)
    out[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return out


# Orientation of the robot base expressed in the camera frame, so the follower arms face
# the scene: the base's forward/up axis (+z, where fl_/fr_link6 reach) maps to camera
# forward (-z), and the base's left-right arm-separation axis (+y) maps to camera right
# (+x) -- so the two grippers appear side-by-side, not stacked. Columns are the images of
# the base x/y/z axes in the camera frame.
BASE_ORIENTATION_IN_CAMERA = np.array(
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
)


def robot_base_from_camera(
    extrinsics: np.ndarray, offset_xyz: tuple[float, float, float] = (0.0, -0.395, 0.538)
) -> np.ndarray:
    """Place the robot base at a fixed pose relative to the camera (the v1 base placement).

    The base is oriented (``BASE_ORIENTATION_IN_CAMERA``) so the follower arms face the same
    way the camera looks, and translated by ``offset_xyz`` in the ARKit camera frame (x
    right, y up, z back). The default drops the torso ~0.40 m **down** and ~0.54 m **back**,
    which puts the two neutral grippers ~0.45 m in front of the camera, centered and a touch
    below the optical axis -- so the arms reach forward into the egocentric frame toward the
    hands. Deterministic and tunable; the exact offset wants validation on real EgoDex
    frames. Returns ``T_world_base``.
    """
    offset = make_se3(BASE_ORIENTATION_IN_CAMERA, np.asarray(offset_xyz, dtype=float))
    return np.asarray(extrinsics, dtype=float) @ offset


def cv_camera_points_to_world(points_cv: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
    """Lift (N, 3) CV-camera-frame points (+z forward) into the world frame.

    HAMER and SAM-3D emit geometry in a CV camera frame; this flips to the ARKit/GL camera
    convention and applies ``T_world_cam`` (``extrinsics``).
    """
    gl = np.asarray(points_cv, dtype=float).reshape(-1, 3) @ GL_CV_FLIP.T
    return transform_points(gl, extrinsics)


def cv_camera_pose_to_world(pose_cv: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
    """Lift a 4x4 pose expressed in the CV camera frame into the world frame."""
    flip = np.eye(4)
    flip[:3, :3] = GL_CV_FLIP
    return np.asarray(extrinsics, dtype=float) @ flip @ np.asarray(pose_cv, dtype=float)


def rotation_geodesic(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    """Geodesic angle (radians) between two rotation matrices."""
    rel = np.asarray(rot_a, dtype=float).T @ np.asarray(rot_b, dtype=float)
    cos = (np.trace(rel) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))
