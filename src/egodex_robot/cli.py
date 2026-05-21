"""`egodex` command-line interface.

Each subcommand maps to one pipeline stage. Stages run independently and read/write small
artifacts under `outputs/`, so they can be invoked one at a time or chained with
`egodex run`.

    egodex load                 # download + inspect pepijn223/egodex-test, dump a frame
    egodex viz                  # rerun: 3D scene | edited video | original video
    egodex hand --frame N       # hand pose (HAMER by default; --arkit for annotations)
    egodex object --frame N     # reconstruct the manipulated object (SAM 2 + SAM-3D) [GPU]
    egodex grasp --frame N      # parallel-gripper grasp (GraspGen by default; --kmeans CPU)
    egodex render --frame N     # IK the Mobile ALOHA arms + render through the camera
    egodex overlay --frame N    # composite the robot render over the original frame
    egodex run --frame N        # chain the stages end-to-end

The pipeline runs the **full GPU chain by default** (HAMER hands -> SAM 2 mask -> SAM-3D
object -> GraspGen grasp). Pass `--cpu` (or `--arkit` / `--kmeans`) to use the no-GPU
fallback (ARKit pose + KMeans grasp). GPU stages need their per-stage extras + weights;
each lives in its own env (conflicting CUDA pins) -- see AGENTS.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer

app = typer.Typer(
    add_completion=False,
    help="EgoDex -> Mobile ALOHA robot retargeting pipeline.",
)

DATASET_REPO_ID = "pepijn223/egodex-test"
OUTPUTS = Path("outputs")


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _save_rgba(path: Path, rgba: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))


def _backends(cpu: bool, gripper_config: str = ""):
    """Build a pipeline.Backends from CLI flags (GPU default unless --cpu)."""
    from .pipeline import Backends

    if cpu:
        return Backends.cpu()
    backends = Backends()
    if gripper_config:
        backends.gripper_config = gripper_config
    if not backends.gripper_config:
        raise typer.BadParameter(
            "The default GPU grasp backend (GraspGen) needs --gripper-config <yaml>. "
            "Pass --cpu to use the KMeans fallback instead."
        )
    return backends


@app.command()
def load(
    repo_id: str = typer.Option(DATASET_REPO_ID, help="LeRobot dataset repo id."),
    frame: int = typer.Option(0, help="Frame index to dump a sample for."),
) -> None:
    """Download + inspect the dataset and dump a sample frame to `outputs/`."""
    from . import dataset as ds

    dataset = ds.load_dataset(repo_id)
    ds.print_schema(dataset)
    sample = ds.get_frame(dataset, frame)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    _save_rgb(OUTPUTS / f"frame_{frame:05d}.png", sample["rgb"])
    np.savez(
        OUTPUTS / f"frame_{frame:05d}.npz",
        state=sample["observation.state"],
        intrinsics=sample["intrinsics"],
        extrinsics=sample["extrinsics"],
    )
    print(f"\nintrinsics source: {sample['intrinsics_src']}")
    print(f"wrote {OUTPUTS / f'frame_{frame:05d}.png'} and .npz")


@app.command()
def viz(
    repo_id: str = typer.Option(DATASET_REPO_ID, help="LeRobot dataset repo id."),
    episode: int = typer.Option(0, help="Episode index to visualize."),
    cpu: bool = typer.Option(False, help="Use the CPU fallback (ARKit pose + KMeans grasp)."),
    gripper_config: str = typer.Option("", help="GraspGen gripper YAML (GPU default)."),
) -> None:
    """Open a 3-panel rerun view: 3D scene | edited (overlaid) video | original video."""
    from . import dataset as ds
    from . import viz as viz_mod

    dataset = ds.load_dataset(repo_id)
    viz_mod.log_episode(dataset, episode=episode, backends=_backends(cpu, gripper_config))


@app.command()
def hand(
    frame: int = typer.Option(..., help="Frame index."),
    repo_id: str = typer.Option(DATASET_REPO_ID, help="LeRobot dataset repo id."),
    arkit: bool = typer.Option(False, help="Use the dataset's ARKit annotations instead of HAMER."),
) -> None:
    """Extract the hand pose for a frame (HAMER [GPU] by default; --arkit for annotations)."""
    from . import dataset as ds
    from . import pose

    dataset = ds.load_dataset(repo_id)
    sample = ds.get_frame(dataset, frame)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS / f"hand_{frame:05d}.npz"

    if not arkit:
        from .stages.hamer import reconstruct_hands  # GPU + weights; see AGENTS.md

        recon = reconstruct_hands(sample["rgb"])
        np.savez(
            out,
            **{f"{side}_{field}": data[field] for side, data in recon.items() for field in data},
        )
        for side, data in recon.items():
            print(f"{side}: {len(data['vertices'])} verts, {len(data['keypoints'])} keypoints (HAMER)")
        print(f"wrote {out}")
        return

    hands = pose.decode_state(sample["observation.state"])
    np.savez(
        out,
        **{
            f"{side}_{field}": getattr(h, field)
            for side, h in hands.items()
            for field in ("wrist_position", "wrist_rotation", "keypoints")
        },
    )
    for side, h in hands.items():
        print(f"{side}: wrist {np.round(h.wrist_position, 3)}, {len(h.keypoints)} fingertips (ARKit)")
    print(f"wrote {out}")


@app.command()
def object(
    frame: int = typer.Option(..., help="Frame index."),
    repo_id: str = typer.Option(DATASET_REPO_ID, help="LeRobot dataset repo id."),
    mask: str = typer.Option("", help="Object mask PNG; if omitted, SAM 2 segments automatically."),
    config: str = typer.Option("checkpoints/hf/pipeline.yaml", help="SAM-3D pipeline config."),
) -> None:
    """Reconstruct the manipulated object's 3D mesh (SAM 2 mask -> SAM-3D-Objects). [GPU]"""
    from . import dataset as ds
    from .stages.sam3d import reconstruct_object

    dataset = ds.load_dataset(repo_id)
    sample = ds.get_frame(dataset, frame)
    if mask:
        import cv2

        mask_arr = cv2.imread(mask, cv2.IMREAD_GRAYSCALE) > 0
    else:
        from .stages.segment import segment_object  # GPU; SAM 2

        mask_arr = segment_object(sample["rgb"])
    recon = reconstruct_object(sample["rgb"], mask=mask_arr, config_path=config)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS / f"object_{frame:05d}.npz"
    np.savez(out, vertices=recon["vertices"], faces=recon["faces"], pose=recon["pose"])
    print(f"object: {len(recon['vertices'])} verts; wrote {out}")


@app.command()
def grasp(
    frame: int = typer.Option(..., help="Frame index."),
    repo_id: str = typer.Option(DATASET_REPO_ID, help="LeRobot dataset repo id."),
    kmeans: bool = typer.Option(False, help="Use the CPU KMeans fallback instead of GraspGen."),
    gripper_config: str = typer.Option("", help="GraspGen gripper YAML (required unless --kmeans)."),
    object_npz: str = typer.Option("", help="Object artifact from `egodex object` (else reconstructed)."),
) -> None:
    """Generate a parallel-gripper grasp (GraspGen [GPU] by default; --kmeans CPU fallback)."""
    from . import dataset as ds
    from . import pose
    from .stages import grasp as grasp_stage

    dataset = ds.load_dataset(repo_id)
    sample = ds.get_frame(dataset, frame)
    hands = pose.decode_state(sample["observation.state"])
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS / f"grasp_{frame:05d}.npz"
    payload = {}

    if not kmeans:
        from .stages import graspgen

        if object_npz:
            loaded = np.load(object_npz, allow_pickle=True)
            object_mesh = {"vertices": loaded["vertices"], "faces": loaded["faces"], "pose": loaded["pose"]}
        else:
            from . import geometry
            from .stages.sam3d import reconstruct_object
            from .stages.segment import segment_object

            mask_arr = segment_object(sample["rgb"])
            obj_cam = reconstruct_object(sample["rgb"], mask=mask_arr)
            object_mesh = {
                "vertices": geometry.cv_camera_points_to_world(obj_cam["vertices"], sample["extrinsics"]),
                "faces": obj_cam["faces"],
                "pose": geometry.cv_camera_pose_to_world(obj_cam["pose"], sample["extrinsics"]),
            }
        candidates = graspgen.graspgen_candidates(object_mesh, gripper_config)
        for side, h in hands.items():
            g = grasp_stage.select_grasp(candidates, h)
            payload[f"{side}_position"] = g.position
            payload[f"{side}_rotation"] = g.rotation
            payload[f"{side}_width"] = np.array([g.width])
            print(f"{side}: GraspGen center {np.round(g.position, 3)}, width {g.width:.3f} m")
    else:
        for side, h in hands.items():
            g = grasp_stage.grasp_from_contacts(pose.contact_points(h))
            payload[f"{side}_position"] = g.position
            payload[f"{side}_rotation"] = g.rotation
            payload[f"{side}_width"] = np.array([g.width])
            print(f"{side}: center {np.round(g.position, 3)}, width {g.width:.3f} m")
    np.savez(out, **payload)
    print(f"wrote {out}")


@app.command()
def render(
    frame: int = typer.Option(..., help="Frame index."),
    repo_id: str = typer.Option(DATASET_REPO_ID, help="LeRobot dataset repo id."),
    cpu: bool = typer.Option(False, help="Use the CPU fallback (ARKit pose + KMeans grasp)."),
    gripper_config: str = typer.Option("", help="GraspGen gripper YAML (GPU default)."),
) -> None:
    """Solve IK for the follower arms and render the robot through the camera intrinsics."""
    from . import dataset as ds
    from . import pipeline

    dataset = ds.load_dataset(repo_id)
    sample = ds.get_frame(dataset, frame)
    result = pipeline.retarget_frame(sample, backends=_backends(cpu, gripper_config))
    for side, arm in result.ik.arms.items():
        flag = "" if arm.reachable else "  [clamped: target out of reach]"
        print(
            f"{side}: pos_res={arm.position_residual:.4f} m, "
            f"rot_res={np.degrees(arm.orientation_residual):.1f} deg{flag}"
        )
    out = OUTPUTS / f"render_{frame:05d}.png"
    _save_rgba(out, result.rgba)
    print(f"wrote {out}")


@app.command()
def overlay(
    frame: int = typer.Option(..., help="Frame index."),
    repo_id: str = typer.Option(DATASET_REPO_ID, help="LeRobot dataset repo id."),
    cpu: bool = typer.Option(False, help="Use the CPU fallback (ARKit pose + KMeans grasp)."),
    gripper_config: str = typer.Option("", help="GraspGen gripper YAML (GPU default)."),
) -> None:
    """Composite the robot render over the original egocentric frame."""
    from . import dataset as ds
    from . import pipeline
    from .stages import overlay as overlay_stage

    dataset = ds.load_dataset(repo_id)
    sample = ds.get_frame(dataset, frame)
    result = pipeline.retarget_frame(sample, backends=_backends(cpu, gripper_config))
    composited = overlay_stage.overlay(sample["rgb"], result.rgba)
    out = OUTPUTS / f"overlay_{frame:05d}.png"
    _save_rgb(out, composited)
    print(f"wrote {out}")


@app.command()
def run(
    frame: int = typer.Option(0, help="Frame index."),
    repo_id: str = typer.Option(DATASET_REPO_ID, help="LeRobot dataset repo id."),
    cpu: bool = typer.Option(False, help="Use the CPU fallback (ARKit pose + KMeans grasp)."),
    gripper_config: str = typer.Option("", help="GraspGen gripper YAML (GPU default)."),
) -> None:
    """Chain the stages end-to-end (GPU by default; --cpu for the ARKit + KMeans fallback)."""
    from . import dataset as ds
    from . import pipeline
    from .stages import overlay as overlay_stage

    backends = _backends(cpu, gripper_config)
    dataset = ds.load_dataset(repo_id)
    sample = ds.get_frame(dataset, frame)
    mode = "CPU (ARKit + KMeans)" if cpu else "GPU (HAMER + SAM2 + SAM-3D + GraspGen)"
    print(f"loaded frame {frame} (intrinsics: {sample['intrinsics_src']}); pipeline: {mode}")

    result = pipeline.retarget_frame(sample, backends=backends)
    for side, arm in result.ik.arms.items():
        flag = "" if arm.reachable else "  [clamped]"
        print(f"IK {side}: pos_res={arm.position_residual:.4f} m{flag}")

    composited = overlay_stage.overlay(sample["rgb"], result.rgba)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    _save_rgba(OUTPUTS / f"render_{frame:05d}.png", result.rgba)
    out = OUTPUTS / f"overlay_{frame:05d}.png"
    _save_rgb(out, composited)
    print(f"wrote {out}")


if __name__ == "__main__":
    app()
