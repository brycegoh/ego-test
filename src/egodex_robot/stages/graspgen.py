"""GraspGen 6-DOF parallel-gripper grasp generation (GPU).

https://github.com/NVlabs/GraspGen

Generates ranked grasp candidates from a reconstructed object mesh (from stages/sam3d.py);
``grasp.select_grasp`` then picks the candidate closest to the human grasp.

First-class GPU stage: ``torch`` / ``trimesh`` / ``grasp_gen`` are imported at module top, so
importing this module requires the ``graspgen`` extra (``pip install '.[graspgen]'``) plus
GraspGen's checkpoints and a gripper config YAML. It is only imported when the GraspGen
backend is selected, so the CPU stages stay torch-free.

Environment (its own venv -- conflicting CUDA pins with HAMER/SAM-3D; see AGENTS.md):
    git clone https://github.com/NVlabs/GraspGen && cd GraspGen && uv pip install -e .
    # download checkpoints + a gripper config (e.g. graspgen_robotiq_2f_140.yml)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import trimesh
import trimesh.transformations as tra
from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg

from .grasp import Grasp, _orthonormal_frame, select_grasp


def _graspgen_pose_to_grasp(pose: np.ndarray, width: float) -> Grasp:
    """Convert a GraspGen 4x4 grasp pose into our Grasp (x=approach, y=closing).

    GraspGen reports poses in a base-link frame where +z is the approach (into the object)
    and the jaws open along +x; our Grasp uses x=approach, y=closing. This axis mapping is
    the retarget the GraspGen README describes for a new gripper and should be validated
    against the fl_/fr_link6 frame on real grasps.
    """
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
    """Run GraspGen on the object mesh, returning ranked Grasp candidates.

    ``object_mesh`` is the ``{"vertices", "faces", ...}`` dict from stages/sam3d.py.
    ``gripper_config`` is a GraspGen gripper YAML (e.g. ``graspgen_robotiq_2f_140.yml``).
    Candidates are returned in the **same frame as ``object_mesh["vertices"]``** (GraspGen's
    mean-centering is undone here), so the caller must express the hand pose in that same
    frame before calling ``grasp.select_grasp``.
    """
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
    """Generate GraspGen candidates and select the one nearest the human grasp.

    With ``hand_pose`` given, returns the candidate closest to the wrist pose (the plan's
    generate-then-select); otherwise returns GraspGen's top-confidence candidate.
    """
    if not gripper_config:
        raise ValueError("grasp_from_graspgen needs a GraspGen gripper_config YAML path.")
    candidates = graspgen_candidates(object_mesh, gripper_config)
    if not candidates:
        raise RuntimeError("GraspGen returned no grasps for this object.")
    if hand_pose is None:
        return candidates[0]
    return select_grasp(candidates, hand_pose)
