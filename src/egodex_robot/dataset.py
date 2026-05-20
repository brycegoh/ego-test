"""Dataset loading + inspection for `pepijn223/egodex-test` (LeRobot v3 format).

Reuses `lerobot.datasets.lerobot_dataset.LeRobotDataset` rather than re-implementing
parquet/video parsing. ``lerobot`` is imported lazily inside the functions so that importing
this module (and the rest of the package) never requires lerobot or network access.

Known shape (to confirm at runtime against `meta/info.json`):
- single egocentric RGB video (Apple Vision Pro), 30 fps;
- `observation.state`: 48-dim float32 (wrist xyz + rotation + finger joints);
- ~3 episodes / ~632 frames / 3 tasks in this test slice.

Camera intrinsics/extrinsics: EgoDex's source HDF5 carries a 3x3 K and per-frame 4x4
extrinsics; whether the LeRobot slice re-exports them is **unconfirmed** (HF is blocked in
the sandbox where this was built). ``get_frame`` returns them when present and otherwise
falls back to a centered pinhole derived from the image size, flagged via ``intrinsics_src``.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

DATASET_REPO_ID = "pepijn223/egodex-test"

# Keys the EgoDex->LeRobot export *may* expose for camera geometry (checked at runtime).
_INTRINSICS_KEYS = ("observation.camera.intrinsics", "camera_intrinsics", "intrinsics")
_EXTRINSICS_KEYS = ("observation.camera.extrinsics", "camera_extrinsics", "extrinsics")
_STATE_KEY = "observation.state"


def load_dataset(repo_id: str = DATASET_REPO_ID) -> Any:
    """Return a `LeRobotDataset` for `repo_id` (downloads from the Hub on first use)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(repo_id)


def print_schema(dataset: Any) -> None:
    """Print `meta/info.json`: features (keys/dtypes/shapes), fps, robot_type, counts."""
    meta = dataset.meta
    info = getattr(meta, "info", {})
    print(f"repo_id      : {getattr(dataset, 'repo_id', '?')}")
    print(f"robot_type   : {info.get('robot_type')}")
    print(f"fps          : {info.get('fps')}")
    print(f"episodes     : {info.get('total_episodes')}")
    print(f"frames       : {info.get('total_frames')}")
    print(f"tasks        : {info.get('total_tasks')}")
    print("features:")
    for key, feat in info.get("features", {}).items():
        print(f"  {key:32s} dtype={feat.get('dtype'):10s} shape={feat.get('shape')}")

    # Gate the (unconfirmed) 48-dim observation.state layout against the real schema.
    from . import pose

    try:
        pose.assert_state_layout(info)
        print(f"\nobservation.state layout check: OK ({pose.STATE_DIM}-dim).")
    except AssertionError as err:
        print(f"\nWARNING: observation.state layout mismatch -> {err}")


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _rgb_from_row(row: dict[str, Any]) -> np.ndarray:
    """Pull the single egocentric RGB frame, normalised to (H, W, 3) uint8."""
    img = None
    for key, value in row.items():
        if "image" in key or "rgb" in key or key.startswith("observation.images"):
            img = value
            break
    if img is None:
        raise KeyError("No image/rgb feature found in the frame.")
    arr = _to_numpy(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
    return arr


def _default_intrinsics(width: int, height: int) -> np.ndarray:
    """Centered pinhole with a ~60 deg horizontal FOV, as a fallback only."""
    focal = 0.5 * width / np.tan(np.deg2rad(60.0) / 2.0)
    return np.array([[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1.0]])


def get_frame(dataset: Any, index: int) -> dict[str, Any]:
    """Return a single frame dict: rgb, observation.state, intrinsics, extrinsics, task, ..."""
    row = dataset[index]
    rgb = _rgb_from_row(row)
    height, width = rgb.shape[:2]

    state = _first_present(row, (_STATE_KEY,))
    intrinsics = _first_present(row, _INTRINSICS_KEYS)
    extrinsics = _first_present(row, _EXTRINSICS_KEYS)

    intrinsics_src = "dataset"
    if intrinsics is None:
        intrinsics = _default_intrinsics(width, height)
        intrinsics_src = "fallback-centered-pinhole"
    else:
        intrinsics = _to_numpy(intrinsics).reshape(3, 3)

    if extrinsics is None:
        extrinsics = np.eye(4)
    else:
        extrinsics = _to_numpy(extrinsics).reshape(4, 4)

    return {
        "rgb": rgb,
        "observation.state": _to_numpy(state) if state is not None else None,
        "intrinsics": np.asarray(intrinsics, dtype=float),
        "extrinsics": np.asarray(extrinsics, dtype=float),
        "intrinsics_src": intrinsics_src,
        "task": row.get("task"),
        "timestamp": _to_numpy(row["timestamp"]).item() if "timestamp" in row else None,
        "episode_index": int(_to_numpy(row["episode_index"]).item()) if "episode_index" in row else None,
    }


def iter_episode_frames(dataset: Any, episode: int = 0) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(global_frame_index, frame_dict)`` for every frame in ``episode`` in order."""
    from_idx = int(dataset.episode_data_index["from"][episode].item())
    to_idx = int(dataset.episode_data_index["to"][episode].item())
    for index in range(from_idx, to_idx):
        yield index, get_frame(dataset, index)
