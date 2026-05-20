"""Synthetic SE(3) / projection tests for geometry.py (fully verifiable on CPU)."""

import numpy as np

from egodex_robot import geometry


def _random_se3(rng):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(-np.pi, np.pi)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rot = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)
    t = rng.uniform(-2, 2, size=3)
    return geometry.make_se3(rot, t)


def test_invert_se3_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        T = _random_se3(rng)
        assert np.allclose(T @ geometry.invert_se3(T), np.eye(4), atol=1e-10)
        assert np.allclose(geometry.invert_se3(geometry.invert_se3(T)), T, atol=1e-10)


def test_invert_se3_is_orthonormal_inverse():
    rng = np.random.default_rng(1)
    T = _random_se3(rng)
    Tinv = geometry.invert_se3(T)
    # inverse rotation is the transpose
    assert np.allclose(Tinv[:3, :3], T[:3, :3].T, atol=1e-12)


def test_transform_points_matches_homogeneous():
    rng = np.random.default_rng(2)
    T = _random_se3(rng)
    pts = rng.normal(size=(50, 3))
    got = geometry.transform_points(pts, T)
    hom = np.hstack([pts, np.ones((50, 1))]) @ T.T
    assert np.allclose(got, hom[:, :3], atol=1e-10)


def test_transform_then_inverse_recovers_points():
    rng = np.random.default_rng(3)
    T = _random_se3(rng)
    pts = rng.normal(size=(30, 3))
    back = geometry.transform_points(geometry.transform_points(pts, T), geometry.invert_se3(T))
    assert np.allclose(back, pts, atol=1e-9)


def test_project_pinhole():
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    # a point 2 m in front, offset right+down by known amounts
    pts = np.array([[0.0, 0.0, 2.0], [0.4, 0.2, 2.0]])
    uv = geometry.project(pts, K)
    assert np.allclose(uv[0], [320, 240])               # on the principal axis -> center
    assert np.allclose(uv[1], [320 + 500 * 0.2, 240 + 500 * 0.1])


def test_pixels_from_world_with_identity_extrinsics():
    # T_world_cam = identity means world == ARKit camera frame (z back).
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    # point in front of an ARKit camera is at -z; the GL->CV flip makes z positive.
    world_pt = np.array([[0.1, 0.05, -2.0]])
    uv = geometry.pixels_from_world(world_pt, K, np.eye(4))
    # CV point = (0.1, -0.05, 2.0): u = 320 + 500*0.05, v = 240 - 500*0.025
    assert np.allclose(uv[0], [320 + 500 * 0.05, 240 - 500 * 0.025])


def test_robot_base_from_camera_offset_in_camera_frame():
    rng = np.random.default_rng(5)
    T_world_cam = _random_se3(rng)
    offset = (0.0, -0.35, 0.30)
    base = geometry.robot_base_from_camera(T_world_cam, offset)
    expected = T_world_cam @ geometry.make_se3(geometry.BASE_ORIENTATION_IN_CAMERA, np.array(offset))
    assert np.allclose(base, expected)


def test_base_orientation_is_a_proper_rotation():
    R = geometry.BASE_ORIENTATION_IN_CAMERA
    assert np.allclose(R.T @ R, np.eye(3))
    assert abs(np.linalg.det(R) - 1.0) < 1e-12
    # base forward/up axis (+z) maps to camera forward (-z)
    assert np.allclose(R @ np.array([0, 0, 1.0]), [0, 0, -1.0])


def test_rotation_geodesic():
    rng = np.random.default_rng(6)
    T = _random_se3(rng)
    R = T[:3, :3]
    assert geometry.rotation_geodesic(R, R) < 1e-9
    # 90 deg about z
    rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    assert abs(geometry.rotation_geodesic(np.eye(3), rz) - np.pi / 2) < 1e-9
