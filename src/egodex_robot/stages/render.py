"""Render the posed Mobile ALOHA robot through the egocentric camera, with pyrender.

rerun cannot rasterise a posed robot to an image (it is a viewer/logger, not an offscreen
renderer), so the robot pixels are produced here with **pyrender** and only *displayed* in
rerun. The geometry is driven entirely by placo's FK: ``ik.RobotIK.solve`` returns a world
transform for every link, and this stage places each link's visual mesh at that transform.

Pipeline:
    placo FK link transforms (world)         from ik.py
        + per-link visual meshes (URDF)      parsed here, loaded with trimesh
        + camera intrinsics K + extrinsics   from the dataset (geometry conventions)
        -> offscreen pyrender                 -> RGBA (H, W, 4) uint8

Alpha comes from the depth buffer (foreground = rendered geometry), because pyrender's
RGBA flag is not honoured under the OSMesa backend used for headless rendering.

Headless GL: pyrender needs an offscreen GL context. Set ``PYOPENGL_PLATFORM=egl`` (if an
EGL library is present) or ``PYOPENGL_PLATFORM=osmesa`` (CPU, needs ``libosmesa6``). This
module defaults the variable to ``osmesa`` if it is unset.

Meshes are ``package://<pkg>/...`` and resolve under the package root (``assets/urdf``).
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import geometry

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

PACKAGE_ROOT = Path("assets/urdf")
DEFAULT_URDF = PACKAGE_ROOT / "aloha_new_description/urdf/aloha_tracer2_dabai_dark.urdf"


def ensure_package_path(root: Path = PACKAGE_ROOT) -> None:
    """Prepend ``root`` to ``ROS_PACKAGE_PATH`` (kept for callers that resolve meshes)."""
    abs_root = str(Path(root).resolve())
    existing = os.environ.get("ROS_PACKAGE_PATH", "")
    if abs_root not in existing.split(os.pathsep):
        os.environ["ROS_PACKAGE_PATH"] = (
            f"{abs_root}{os.pathsep}{existing}" if existing else abs_root
        )


def _rpy_to_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _resolve_mesh(filename: str, package_root: Path) -> Path:
    if filename.startswith("package://"):
        return Path(package_root).resolve() / filename[len("package://"):]
    return Path(filename)


@dataclass
class Visual:
    """One visual mesh attached to a link, with its constant link->visual offset."""

    link: str
    mesh_path: Path
    scale: np.ndarray            # (3,)
    origin: np.ndarray           # (4, 4) link -> visual transform


def parse_visuals(urdf_path: Path = DEFAULT_URDF, package_root: Path = PACKAGE_ROOT) -> list[Visual]:
    """Parse every ``<visual><geometry><mesh>`` in the URDF into ``Visual`` records."""
    root = ET.parse(urdf_path).getroot()
    visuals: list[Visual] = []
    for link in root.findall("link"):
        link_name = link.get("name")
        for visual in link.findall("visual"):
            geom = visual.find("geometry")
            mesh = geom.find("mesh") if geom is not None else None
            if mesh is None:
                continue
            scale_attr = mesh.get("scale", "1 1 1")
            scale = np.array([float(s) for s in scale_attr.split()], dtype=float)
            origin_el = visual.find("origin")
            xyz = (0.0, 0.0, 0.0)
            rpy = (0.0, 0.0, 0.0)
            if origin_el is not None:
                xyz = tuple(float(v) for v in origin_el.get("xyz", "0 0 0").split())
                rpy = tuple(float(v) for v in origin_el.get("rpy", "0 0 0").split())
            origin = geometry.make_se3(_rpy_to_matrix(rpy), np.array(xyz))
            visuals.append(
                Visual(
                    link=link_name,
                    mesh_path=_resolve_mesh(mesh.get("filename"), package_root),
                    scale=scale,
                    origin=origin,
                )
            )
    return visuals


def _load_mesh_cache(visuals: list[Visual]) -> dict[Path, object]:
    import trimesh

    cache: dict[Path, object] = {}
    for visual in visuals:
        if visual.mesh_path not in cache:
            cache[visual.mesh_path] = trimesh.load(str(visual.mesh_path), force="mesh")
    return cache


def render_robot(
    link_transforms: dict[str, np.ndarray],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    image_size: tuple[int, int],
    urdf_path: Path = DEFAULT_URDF,
    package_root: Path = PACKAGE_ROOT,
    bg_alpha_from_depth: bool = True,
) -> np.ndarray:
    """Rasterise the posed robot and return an RGBA snapshot ``(H, W, 4)`` uint8.

    ``link_transforms`` maps link name -> (4, 4) world transform (from ``ik.RobotIK.solve``).
    ``intrinsics`` is the 3x3 K; ``extrinsics`` is ``T_world_cam`` (ARKit/GL camera, which is
    also pyrender's camera convention, so it is used directly as the camera node pose).
    ``image_size`` is ``(width, height)`` in pixels.
    """
    import pyrender
    import trimesh

    width, height = image_size
    intrinsics = np.asarray(intrinsics, dtype=float)
    extrinsics = np.asarray(extrinsics, dtype=float)

    visuals = parse_visuals(urdf_path, package_root)
    mesh_cache = _load_mesh_cache(visuals)

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.6, 0.6, 0.6])

    for visual in visuals:
        link_world = link_transforms.get(visual.link)
        if link_world is None:
            continue
        base_mesh = mesh_cache[visual.mesh_path]
        mesh = base_mesh.copy()
        if not np.allclose(visual.scale, 1.0):
            mesh.apply_scale(visual.scale)
        node_pose = np.asarray(link_world, dtype=float) @ visual.origin
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False), pose=node_pose)

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=100.0)
    scene.add(camera, pose=extrinsics)
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0), pose=extrinsics)

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    try:
        color, depth = renderer.render(scene)
    finally:
        renderer.delete()

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = color[..., :3]
    if bg_alpha_from_depth:
        rgba[..., 3] = np.where(depth > 0, 255, 0).astype(np.uint8)
    else:
        rgba[..., 3] = 255
    return rgba
