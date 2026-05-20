"""EgoDex -> SO-101 robot retargeting.

Turn the EgoDex egocentric human-hand dataset (`pepijn223/egodex-test`, LeRobot format)
into a robot-centric view: reconstruct the scene in 3D, place an SO-100/SO-101
parallel-gripper robot where the hands are, choose a grasp, render it through the
original camera, and overlay the robot on the egocentric frame.

This package is currently a SCAFFOLD: module/CLI structure and input/output contracts
are defined, but the stage logic is not implemented yet (see AGENTS.md).
"""

__version__ = "0.1.0"
