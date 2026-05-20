"""CPU-verifiable tests for pose decoding, KMeans grasp, grasp selection, and overlay."""

import numpy as np
import pytest

from egodex_robot import geometry, pose
from egodex_robot.stages import grasp as grasp_stage
from egodex_robot.stages import overlay as overlay_stage


# ----------------------------------------------------------------------------- pose
def test_rot6d_to_matrix_is_orthonormal():
    rng = np.random.default_rng(0)
    rot6d = rng.normal(size=6)
    R = pose.rot6d_to_matrix(rot6d)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
    assert abs(np.linalg.det(R) - 1.0) < 1e-9


def test_rot6d_recovers_known_rotation():
    rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])  # 90 deg about z
    rot6d = np.concatenate([rz[:, 0], rz[:, 1]])
    assert np.allclose(pose.rot6d_to_matrix(rot6d), rz, atol=1e-9)


def test_decode_state_splits_two_hands():
    rng = np.random.default_rng(1)
    state = rng.normal(size=pose.STATE_DIM)
    hands = pose.decode_state(state)
    assert set(hands) == {"left", "right"}
    assert np.allclose(hands["left"].wrist_position, state[0:3])
    assert np.allclose(hands["right"].wrist_position, state[24:27])
    assert hands["left"].keypoints.shape == (pose.NUM_FINGERTIPS, 3)


def test_decode_state_rejects_wrong_width():
    with pytest.raises(ValueError):
        pose.decode_state(np.zeros(40))


# ----------------------------------------------------------------------------- grasp
def test_grasp_from_contacts_two_clusters():
    # two tight clusters 8 cm apart along x -> jaws at +/-0.04, width ~0.08
    rng = np.random.default_rng(2)
    left = rng.normal(scale=0.002, size=(10, 3)) + np.array([-0.04, 0, 0])
    right = rng.normal(scale=0.002, size=(10, 3)) + np.array([0.04, 0, 0])
    g = grasp_stage.grasp_from_contacts(np.vstack([left, right]))
    assert np.allclose(g.position, [0, 0, 0], atol=0.01)
    assert abs(g.width - 0.08) < 0.01
    # rotation is a proper orthonormal frame
    assert np.allclose(g.rotation.T @ g.rotation, np.eye(3), atol=1e-6)
    assert abs(np.linalg.det(g.rotation) - 1.0) < 1e-6


def test_grasp_as_se3_round_trips():
    g = grasp_stage.grasp_from_contacts(np.array([[-0.05, 0, 0], [0.05, 0, 0]]))
    T = g.as_se3()
    assert np.allclose(T[:3, 3], g.position)
    assert np.allclose(T[:3, :3], g.rotation)


def test_grasp_from_contacts_needs_two_points():
    with pytest.raises(ValueError):
        grasp_stage.grasp_from_contacts(np.array([[0.0, 0.0, 0.0]]))


def test_select_grasp_picks_closest_to_hand():
    hand = pose.HandPose(
        wrist_position=np.array([0.0, 0.0, 0.0]),
        wrist_rotation=np.eye(3),
        keypoints=np.zeros((5, 3)),
    )
    near = grasp_stage.Grasp(position=np.array([0.01, 0, 0]), rotation=np.eye(3), width=0.05)
    far = grasp_stage.Grasp(position=np.array([1.0, 0, 0]), rotation=np.eye(3), width=0.05)
    chosen = grasp_stage.select_grasp([far, near], hand)
    assert chosen is near


# --------------------------------------------------------------------------- overlay
def test_overlay_full_alpha_replaces_pixels():
    frame = np.full((4, 4, 3), 100, np.uint8)
    robot = np.zeros((4, 4, 4), np.uint8)
    robot[..., :3] = 200
    robot[..., 3] = 255  # fully opaque everywhere
    out = overlay_stage.overlay(frame, robot)
    assert np.all(out == 200)


def test_overlay_zero_alpha_keeps_frame():
    frame = np.full((4, 4, 3), 100, np.uint8)
    robot = np.full((4, 4, 4), 200, np.uint8)
    robot[..., 3] = 0
    out = overlay_stage.overlay(frame, robot)
    assert np.all(out == 100)


def test_overlay_half_alpha_blends():
    frame = np.zeros((2, 2, 3), np.uint8)
    robot = np.zeros((2, 2, 4), np.uint8)
    robot[..., :3] = 200
    robot[..., 3] = 128
    out = overlay_stage.overlay(frame, robot)
    assert np.all(np.abs(out.astype(int) - int(200 * 128 / 255)) <= 1)


def test_overlay_rejects_size_mismatch():
    with pytest.raises(ValueError):
        overlay_stage.overlay(np.zeros((4, 4, 3), np.uint8), np.zeros((5, 5, 4), np.uint8))
