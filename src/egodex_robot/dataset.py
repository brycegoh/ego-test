"""Dataset loading + inspection for `pepijn223/egodex-test` (LeRobot v3 format).

Reuses `lerobot.datasets.lerobot_dataset.LeRobotDataset` rather than re-implementing
parquet/video parsing.

Known shape (to confirm at runtime against `meta/info.json`):
- single egocentric RGB video (Apple Vision Pro), 30 fps;
- `observation.state`: 48-dim float32 (wrist xyz + rotation + finger joints);
- ~3 episodes / ~632 frames / 3 tasks in this test slice.

SCAFFOLD: signatures + contracts only.
"""

from __future__ import annotations

from typing import Any

DATASET_REPO_ID = "pepijn223/egodex-test"


def load_dataset(repo_id: str = DATASET_REPO_ID) -> Any:
    """Return a `LeRobotDataset` for `repo_id` (downloads from the Hub on first use)."""
    raise NotImplementedError


def print_schema(dataset: Any) -> None:
    """Print `meta/info.json`: features (keys/dtypes/shapes), fps, robot_type, counts."""
    raise NotImplementedError


def get_frame(dataset: Any, index: int) -> dict[str, Any]:
    """Return a single frame: {"rgb", "observation.state", "task", "timestamp", ...}."""
    raise NotImplementedError
