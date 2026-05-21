"""Parallel-gripper grasp generation.

Two paths:
- Preferred (GPU): GraspGen (https://github.com/NVlabs/GraspGen) generates 6-DOF
  parallel-gripper grasp candidates from the reconstructed object mesh; ``select_grasp``
  then picks the candidate closest (in SE(3)) to the human grasp.
- Fallback (CPU): KMeans(n_clusters=2) over the hand's fingertip/contact points -> two
  jaw positions -> a gripper pose (midpoint + closing/approach axes). Lets the IK, render,
  and overlay stages run without any ML.

Output contract (both paths):
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


# GraspGen reports grasp poses in a base-link frame where +z is the approach (into the
# object) and the jaws open along +x. Our Grasp uses x=approach, y=closing -- so we read
# GraspGen's z column as approach and its x column as the closing axis. This axis mapping is
# the same retarget the GraspGen README describes for a new gripper and should be validated
# against the fl_/fr_link6 frame on real grasps.
_GRASPGEN_SETUP_HINT = (
    "GraspGen is not available. Set up its GPU environment (clone NVlabs/GraspGen, "
    "`uv pip install -e .`, download checkpoints + a gripper config), or use "
    "grasp_from_contacts() for a CPU fallback. See AGENTS.md."
)


def _graspgen_pose_to_grasp(pose: np.ndarray, width: float) -> Grasp:
    """Convert a GraspGen 4x4 grasp pose into our Grasp (x=approach, y=closing)."""
    pose = np.asarray(pose, dtype=float)
    approach = pose[:3, 2]   # GraspGen +z
    closing = pose[:3, 0]    # GraspGen +x (jaw-opening axis)
    return Grasp(position=pose[:3, 3].copy(), rotation=_orthonormal_frame(approach, closing), width=width)


def graspgen_candidates(
    object_mesh: dict[str, Any],
    gripper_config: str,
    num_grasps: int = 200,
    topk: int = 100,
    num_sample_points: int = 2000,
) -> list[Grasp]:
    """GPU: run GraspGen on the object mesh, returning ranked Grasp candidates.

    ``object_mesh`` is the ``{"vertices", "faces", ...}`` dict from stages/sam3d.py.
    ``gripper_config`` is a GraspGen gripper YAML (e.g. ``graspgen_robotiq_2f_140.yml``).
    Candidates are returned in the **same frame as ``object_mesh["vertices"]``** (GraspGen's
    mean-centering is undone here), so the caller must express the hand pose in that same
    frame before calling ``select_grasp``.
    """
    try:
        import torch  # noqa: F401
        import trimesh
        import trimesh.transformations as tra
        from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
    except ImportError as exc:  # pragma: no cover - GPU env only
        raise NotImplementedError(_GRASPGEN_SETUP_HINT) from exc

    verts = np.asarray(object_mesh["vertices"], dtype=float)
    faces = object_mesh.get("faces")
    if faces is not None:
        mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces), process=False)
        xyz, _ = trimesh.sample.sample_surface(mesh, num_sample_points)
        xyz = np.asarray(xyz)
    else:
        xyz = verts

    t_center = tra.translation_matrix(-xyz.mean(axis=0))
    xyz_centered = tra.transform_points(xyz, t_center)

    grasp_cfg = load_grasp_cfg(gripper_config)
    sampler = GraspGenSampler(grasp_cfg)
    grasps, conf = GraspGenSampler.run_inference(
        xyz_centered, sampler, grasp_threshold=-1.0, num_grasps=num_grasps,
        topk_num_grasps=topk, remove_outliers=False,
    )
    if len(grasps) == 0:
        return []

    grasps = grasps.cpu().numpy()
    order = np.argsort(-conf.cpu().numpy())  # best confidence first
    width = float(getattr(getattr(grasp_cfg, "data", object()), "gripper_width", 0.08))
    t_uncenter = tra.inverse_matrix(t_center)
    return [_graspgen_pose_to_grasp(t_uncenter @ grasps[i], width) for i in order]


def grasp_from_graspgen(
    object_mesh: dict[str, Any],
    hand_pose: Any | None = None,
    gripper_config: str = "",
) -> Grasp:
    """GPU path: generate GraspGen candidates and select the one nearest the human grasp.

    With ``hand_pose`` given, returns the candidate closest to the wrist pose (the plan's
    generate-then-select); otherwise returns GraspGen's top-confidence candidate. Requires
    the GraspGen environment and a ``gripper_config`` YAML.
    """
    if not gripper_config:
        raise ValueError("grasp_from_graspgen needs a GraspGen gripper_config YAML path.")
    candidates = graspgen_candidates(object_mesh, gripper_config)
    if not candidates:
        raise RuntimeError("GraspGen returned no grasps for this object.")
    if hand_pose is None:
        return candidates[0]
    return select_grasp(candidates, hand_pose)
