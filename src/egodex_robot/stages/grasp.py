"""Parallel-gripper grasp: shared types + the CPU path.

This module holds the frame-agnostic grasp pieces used by every backend:
- ``Grasp`` dataclass + ``_orthonormal_frame`` (the gripper frame builder),
- ``grasp_from_contacts`` -- the CPU fallback: KMeans(2) over the hand's fingertip/contact
  points -> two jaw positions -> a gripper pose (midpoint + closing/approach axes),
- ``select_grasp`` -- generate-then-select: pick the candidate closest to the human grasp.

The GPU path (GraspGen) lives in ``stages/graspgen.py`` so importing this module never pulls
in ``grasp_gen``/``torch`` (the CPU stages depend on it).

Output contract (every backend):
    Grasp { position (3,), rotation (3, 3), width (float) } in the world (ARKit) frame.

Gripper frame convention (columns of ``rotation``):
    x = approach (toward the object), y = closing (jaw-to-jaw / width), z = x cross y.
This matches the end-effector target consumed by ik.RobotIK.solve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import geometry


@dataclass
class Grasp:
    position: np.ndarray  # (3,) gripper center
    rotation: np.ndarray  # (3, 3) gripper orientation
    width: float          # jaw opening, meters

    def as_se3(self) -> np.ndarray:
        """Return the 4x4 world transform for this grasp (the IK end-effector target)."""
        return geometry.make_se3(self.rotation, self.position)


def _orthonormal_frame(approach: np.ndarray, closing: np.ndarray) -> np.ndarray:
    """Build a right-handed rotation with x~approach, y~closing (Gram-Schmidt)."""
    x = approach / (np.linalg.norm(approach) + 1e-12)
    y = closing - np.dot(closing, x) * x
    ny = np.linalg.norm(y)
    if ny < 1e-9:  # closing parallel to approach; pick any perpendicular
        helper = np.array([1.0, 0.0, 0.0]) if abs(x[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        y = helper - np.dot(helper, x) * x
        ny = np.linalg.norm(y)
    y = y / ny
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


def grasp_from_contacts(contact_points: np.ndarray) -> Grasp:
    """CPU fallback: cluster (N, 3) contact points into 2 jaws -> a parallel-gripper Grasp."""
    from sklearn.cluster import KMeans

    pts = np.asarray(contact_points, dtype=float).reshape(-1, 3)
    if pts.shape[0] < 2:
        raise ValueError("Need at least 2 contact points to form a parallel grasp.")

    labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(pts)
    jaw_a = pts[labels == 0].mean(axis=0)
    jaw_b = pts[labels == 1].mean(axis=0)

    center = (jaw_a + jaw_b) / 2.0
    closing = jaw_b - jaw_a
    width = float(np.linalg.norm(closing))

    # Approach = the contact cloud's main spread perpendicular to the closing axis.
    centered = pts - center
    closing_unit = closing / (width + 1e-12)
    perp = centered - np.outer(centered @ closing_unit, closing_unit)
    if np.linalg.norm(perp) > 1e-9:
        _, _, vh = np.linalg.svd(perp, full_matrices=False)
        approach = vh[0]
    else:
        approach = np.array([0.0, 0.0, 1.0])

    rotation = _orthonormal_frame(approach, closing)
    return Grasp(position=center, rotation=rotation, width=width)


def select_grasp(
    candidates: list[Grasp],
    hand_pose: Any,
    contacts: np.ndarray | None = None,
    position_weight: float = 1.0,
    orientation_weight: float = 0.3,
) -> Grasp:
    """Generate-then-select: pick the candidate closest to the human grasp in SE(3).

    The human grasp reference is the hand's wrist pose (``hand_pose.wrist_position`` /
    ``.wrist_rotation``). Distance = ``position_weight * ||dp|| + orientation_weight *
    geodesic(dR)``. Candidates and the hand pose must already be in the **same frame**
    (the world/ARKit frame); GraspGen output expressed in object/camera frame must be
    transformed into it before calling this.
    """
    if not candidates:
        raise ValueError("No grasp candidates to select from.")

    ref_pos = np.asarray(hand_pose.wrist_position, dtype=float)
    ref_rot = np.asarray(hand_pose.wrist_rotation, dtype=float)

    def cost(g: Grasp) -> float:
        dp = float(np.linalg.norm(np.asarray(g.position, float) - ref_pos))
        dr = geometry.rotation_geodesic(np.asarray(g.rotation, float), ref_rot)
        return position_weight * dp + orientation_weight * dr

    return min(candidates, key=cost)
