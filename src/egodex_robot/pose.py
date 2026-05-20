"""Hand-pose decoding from EgoDex/LeRobot annotations.

Primary source of truth: the dataset's ARKit 3D hand tracking, packed into the 48-dim
``observation.state``. This module decodes that vector into structured per-hand poses and
keypoints, expressed in the world (ARKit) frame, for use by viz, grasp, and IK.

(HAMER is an optional RGB-only alternative for the same outputs; see stages/hamer.py.)

The 48-dim layout below is **unconfirmed** -- Hugging Face is blocked in the sandbox where
this was built, so the indices could not be checked against the real ``meta/info.json``. It
follows the README description ("per-hand wrist position + rotation + finger joints") with
the cleanest split that sums to 48:

    48 = 2 hands x 24, and 24 = wrist_pos(3) + wrist_rot_6d(6) + 5 fingertips x 3 (15).

Left hand is ``state[0:24]``, right hand is ``state[24:48]``. The 6-D rotation is the
first two columns of the rotation matrix (Zhou et al. 2019), recovered with Gram-Schmidt.
Call ``assert_state_layout(info)`` once where HF is reachable to gate these indices against
the real schema before trusting them; ``LAYOUT`` is the single place to fix them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Single source of truth for the (unconfirmed) 48-dim layout. Edit here once confirmed.
STATE_DIM = 48
PER_HAND = 24
HANDS = ("left", "right")
LAYOUT = {
    "wrist_pos": slice(0, 3),       # xyz, meters, world frame
    "wrist_rot6d": slice(3, 9),     # 6-D continuous rotation (two matrix columns)
    "fingertips": slice(9, 24),     # 5 fingertips x 3 (thumb, index, middle, ring, little)
}
NUM_FINGERTIPS = 5


@dataclass
class HandPose:
    """One hand in the world (ARKit) frame."""

    wrist_position: np.ndarray  # (3,) xyz, meters
    wrist_rotation: np.ndarray  # (3, 3) rotation matrix
    keypoints: np.ndarray       # (J, 3) finger-joint positions
    confidence: np.ndarray | None = None  # (J,) optional per-joint confidence


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Recover a (3, 3) rotation matrix from a 6-D rotation (two columns + Gram-Schmidt)."""
    rot6d = np.asarray(rot6d, dtype=float).reshape(6)
    a1, a2 = rot6d[:3], rot6d[3:]
    b1 = a1 / (np.linalg.norm(a1) + 1e-12)
    a2_proj = a2 - np.dot(b1, a2) * b1
    b2 = a2_proj / (np.linalg.norm(a2_proj) + 1e-12)
    b3 = np.cross(b1, b2)
    return np.column_stack([b1, b2, b3])


def _decode_hand(chunk: np.ndarray) -> HandPose:
    pos = np.asarray(chunk[LAYOUT["wrist_pos"]], dtype=float)
    rot = rot6d_to_matrix(chunk[LAYOUT["wrist_rot6d"]])
    fingertips = np.asarray(chunk[LAYOUT["fingertips"]], dtype=float).reshape(NUM_FINGERTIPS, 3)
    return HandPose(wrist_position=pos, wrist_rotation=rot, keypoints=fingertips)


def decode_state(state: np.ndarray) -> dict[str, HandPose]:
    """Decode a 48-dim ``observation.state`` into ``{"left": HandPose, "right": HandPose}``."""
    state = np.asarray(state, dtype=float).reshape(-1)
    if state.shape[0] != STATE_DIM:
        raise ValueError(
            f"Expected a {STATE_DIM}-dim observation.state, got {state.shape[0]}. "
            "The layout is unconfirmed; run assert_state_layout(info) against meta/info.json."
        )
    return {
        side: _decode_hand(state[i * PER_HAND : (i + 1) * PER_HAND])
        for i, side in enumerate(HANDS)
    }


def contact_points(hand: HandPose) -> np.ndarray:
    """Return candidate fingertip/contact points (N, 3) used to seed grasp clustering."""
    return np.asarray(hand.keypoints, dtype=float).reshape(-1, 3)


def assert_state_layout(info: dict[str, Any]) -> None:
    """Gate the (unconfirmed) layout against ``meta/info.json`` features.

    Run this once in an environment where Hugging Face is reachable. It checks that
    ``observation.state`` exists and is ``STATE_DIM``-wide; if the real width differs,
    update ``LAYOUT``/``PER_HAND`` here rather than guessing downstream.
    """
    features = info.get("features", info)
    state = features.get("observation.state")
    if state is None:
        raise AssertionError("meta/info.json has no 'observation.state' feature.")
    shape = state.get("shape", state.get("shapes"))
    dim = int(np.prod(shape)) if shape is not None else None
    if dim != STATE_DIM:
        raise AssertionError(
            f"observation.state width is {dim}, but pose.LAYOUT assumes {STATE_DIM}. "
            "Update pose.LAYOUT / PER_HAND to match the real schema before decoding."
        )
