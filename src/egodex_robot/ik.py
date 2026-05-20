"""placo inverse-kinematics wrapper for the Mobile ALOHA follower arms.

placo is the single kinematics source of truth: it parses the URDF once and provides both
IK (joint angles that put a gripper at a target pose) and FK (world transforms for every
link). Those same link transforms drive *both* display backends -- pyrender (the rasterised
overlay) and rerun (the 3D panel) -- so the two views can never disagree.

We retarget the left/right human hands onto the front *follower* arms (``fl_``/``fr_``),
each a 6-DOF revolute chain ending at ``fl_link6`` / ``fr_link6`` (the wrist/end-effector),
with prismatic ``joint7``/``joint8`` jaws. The floating base is fixed (masked) at a caller-
supplied ``T_world_base``; placo reports a per-arm position/orientation residual so callers
can flag targets that the chain cannot reach within its joint limits.

Meshes are referenced as ``package://<pkg>/...``; placo (via pinocchio) resolves these
through ``ROS_PACKAGE_PATH``, which must contain the package root (``assets/urdf``).
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import geometry

PACKAGE_ROOT = Path("assets/urdf")
DEFAULT_URDF = PACKAGE_ROOT / "aloha_new_description/urdf/aloha_tracer2_dabai_dark.urdf"

# Human hand -> follower-arm end-effector frames.
EE_FRAMES: dict[str, str] = {"left": "fl_link6", "right": "fr_link6"}
ARM_PREFIXES: dict[str, str] = {"left": "fl_", "right": "fr_"}

# Residual above which a solve is flagged as not reaching the target (clamped to nearest).
REACH_POS_TOL = 0.02  # meters
REACH_ROT_TOL = np.deg2rad(10.0)


def ensure_package_path(root: Path = PACKAGE_ROOT) -> None:
    """Prepend ``root`` to ``ROS_PACKAGE_PATH`` so ``package://`` mesh refs resolve."""
    abs_root = str(Path(root).resolve())
    existing = os.environ.get("ROS_PACKAGE_PATH", "")
    if abs_root not in existing.split(os.pathsep):
        os.environ["ROS_PACKAGE_PATH"] = (
            f"{abs_root}{os.pathsep}{existing}" if existing else abs_root
        )


def _link_names(urdf_path: Path) -> list[str]:
    root = ET.parse(urdf_path).getroot()
    return [link.get("name") for link in root.findall("link")]


@dataclass
class ArmSolution:
    """One arm's IK result."""

    joints: dict[str, float]          # actuated joint name -> angle (rad) / position (m)
    ee_transform: np.ndarray          # (4, 4) achieved end-effector pose in world
    target: np.ndarray                # (4, 4) requested target pose in world
    position_residual: float          # meters
    orientation_residual: float       # radians
    reachable: bool


@dataclass
class IKResult:
    """Full solve: per-arm solutions plus FK world transforms for every link."""

    arms: dict[str, ArmSolution]
    link_transforms: dict[str, np.ndarray]  # link name -> (4, 4) world transform
    base_transform: np.ndarray               # (4, 4) world transform of the fixed base
    joints: dict[str, float] = field(default_factory=dict)  # all actuated joints


class RobotIK:
    """Loads the Mobile ALOHA URDF in placo and solves the follower-arm IK."""

    def __init__(self, urdf_path: Path = DEFAULT_URDF):
        import placo  # imported here so the package imports without placo installed

        self.urdf_path = Path(urdf_path)
        if not self.urdf_path.exists():
            raise FileNotFoundError(
                f"URDF not found at {self.urdf_path}. Run scripts/fetch_urdf.sh first."
            )
        ensure_package_path()
        self._placo = placo
        self.robot = placo.RobotWrapper(str(self.urdf_path), placo.Flags.ignore_collisions)
        self.link_names = _link_names(self.urdf_path)
        self._actuated = list(self.robot.joint_names())

    def _arm_joint_names(self, side: str) -> list[str]:
        prefix = ARM_PREFIXES[side]
        return [j for j in self._actuated if j.startswith(prefix)]

    def solve(
        self,
        targets: dict[str, np.ndarray],
        base_transform: np.ndarray | None = None,
        max_iters: int = 600,
        dt: float = 0.1,
        position_weight: float = 1.0,
        orientation_weight: float = 0.001,
    ) -> IKResult:
        """Solve IK so each arm's end-effector reaches its target pose in ``targets``.

        ``targets`` maps ``"left"``/``"right"`` to a (4, 4) world-frame target pose. The
        floating base is fixed at ``base_transform`` (identity if omitted). Velocity IK runs
        for ``max_iters`` steps; ``dt`` around 0.1 keeps the QP stable (larger steps
        oscillate).

        Position is weighted ~1000x above orientation. A genuinely reachable pose then
        converges to zero on *both*; for a pose whose exact orientation the 6-DOF chain
        cannot hold at that position, the gripper still lands where the hand is (position
        tight) and orientation clamps to the nearest feasible value -- exactly what the
        overlay needs, since correctness is "gripper where the hand is". The achieved-vs-
        target residuals are reported per arm so unreachable targets can be flagged.
        """
        robot = self.robot
        robot.reset()
        base = np.eye(4) if base_transform is None else np.asarray(base_transform, float)
        robot.set_T_world_fbase(base)
        robot.update_kinematics()

        solver = robot.make_solver()
        solver.mask_fbase(True)
        solver.enable_joint_limits(True)
        solver.dt = dt

        tasks: dict[str, np.ndarray] = {}
        for side, target in targets.items():
            if side not in EE_FRAMES:
                raise ValueError(f"Unknown arm side {side!r}; expected one of {list(EE_FRAMES)}")
            frame = EE_FRAMES[side]
            target = np.asarray(target, dtype=float)
            task = solver.add_frame_task(frame, target)
            task.configure(frame, "soft", position_weight, orientation_weight)
            tasks[side] = target

        for _ in range(max_iters):
            robot.update_kinematics()
            solver.solve(True)
        robot.update_kinematics()

        arms: dict[str, ArmSolution] = {}
        for side, target in tasks.items():
            frame = EE_FRAMES[side]
            achieved = np.asarray(robot.get_T_world_frame(frame), dtype=float).copy()
            pos_res = float(np.linalg.norm(achieved[:3, 3] - target[:3, 3]))
            rot_res = geometry.rotation_geodesic(achieved[:3, :3], target[:3, :3])
            arms[side] = ArmSolution(
                joints={j: float(robot.get_joint(j)) for j in self._arm_joint_names(side)},
                ee_transform=achieved,
                target=target,
                position_residual=pos_res,
                orientation_residual=rot_res,
                reachable=(pos_res <= REACH_POS_TOL and rot_res <= REACH_ROT_TOL),
            )

        link_transforms = {
            name: np.asarray(robot.get_T_world_frame(name), dtype=float).copy()
            for name in self.link_names
        }
        all_joints = {j: float(robot.get_joint(j)) for j in self._actuated}
        return IKResult(
            arms=arms,
            link_transforms=link_transforms,
            base_transform=base,
            joints=all_joints,
        )
