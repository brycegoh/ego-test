"""`egodex` command-line interface.

Each subcommand maps to one pipeline stage. Stages run independently and read/write small
artifacts under `outputs/`, so they can be invoked one at a time or chained with
`egodex run`.

    egodex load                 # download + inspect pepijn223/egodex-test, dump a frame
    egodex viz                  # rerun: 3D scene | edited video | original video
    egodex hand --frame N       # decode ARKit hand pose for a frame
    egodex object --frame N     # reconstruct the manipulated object in 3D (SAM-3D)  [GPU]
    egodex grasp --frame N      # parallel-gripper grasp (GraspGen / KMeans fallback)
    egodex render --frame N     # IK the Mobile ALOHA arms + render through the camera
    egodex overlay --frame N    # composite the robot render over the original frame
    egodex run --frame N        # chain the available stages end-to-end

CPU stages (no GPU): load, viz, hand, grasp (KMeans), render, overlay, run.
GPU stages (separate env + weights): object, hand --hamer, grasp --graspgen.
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


def _retarget_frame(frame: dict, urdf_path: Path | None = None):
    """Shared core: ARKit pose -> grasp targets -> IK -> render. Returns (result, rgba, info)."""
    from . import geometry, pose
    from .ik import EE_FRAMES, RobotIK
    from .stages import grasp as grasp_stage
    from .stages import render as render_stage

    hands = pose.decode_state(frame["observation.state"])
    targets = {
        side: grasp_stage.grasp_from_contacts(pose.contact_points(hands[side])).as_se3()
        for side in EE_FRAMES
        if side in hands
    }
    ik = RobotIK() if urdf_path is None else RobotIK(urdf_path)
    base = geometry.robot_base_from_camera(frame["extrinsics"])
    result = ik.solve(targets, base_transform=base)

    height, width = frame["rgb"].shape[:2]
    rgba = render_stage.render_robot(
        result.link_transforms, frame["intrinsics"], frame["extrinsics"], (width, height), ik.urdf_path
    )
    return hands, result, rgba


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
) -> None:
    """Open a 3-panel rerun view: 3D scene | edited (overlaid) video | original video."""
    from . import dataset as ds
    from . import viz as viz_mod

    dataset = ds.load_dataset(repo_id)
    viz_mod.log_episode(dataset, episode=episode)


@app.command()
def hand(
    frame: int = typer.Option(..., help="Frame index."),
    repo_id: str = typer.Option(DATASET_REPO_ID, help="LeRobot dataset repo id."),
    hamer: bool = typer.Option(False, help="Use HAMER (RGB, GPU) instead of ARKit annotations."),
) -> None:
    """Decode the hand pose for a frame (ARKit annotations by default; HAMER with --hamer [GPU])."""
    from . import dataset as ds
    from . import pose

    dataset = ds.load_dataset(repo_id)
    sample = ds.get_frame(dataset, frame)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS / f"hand_{frame:05d}.npz"

    if hamer:
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
        print(f"{side}: wrist {np.round(h.wrist_position, 3)}, {len(h.keypoints)} fingertips")
    print(f"wrote {out}")


@app.command()
def object(
    frame: int = typer.Option(..., help="Frame index."),
    repo_id: str = typer.Option(DATASET_REPO_ID, help="LeRobot dataset repo id."),
    mask: str = typer.Option("", help="Path to a binary object mask PNG (from SAM/SAM2)."),
    config: str = typer.Option("checkpoints/hf/pipeline.yaml", help="SAM-3D pipeline config."),
) -> None:
    """Reconstruct the manipulated object's 3D mesh from RGB (SAM-3D-Objects). [GPU]"""
    from . import dataset as ds
    from .stages.sam3d import reconstruct_object

    dataset = ds.load_dataset(repo_id)
    sample = ds.get_frame(dataset, frame)
    mask_arr = None
    if mask:
        import cv2

        mask_arr = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    recon = reconstruct_object(sample["rgb"], mask=mask_arr, config_path=config)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS / f"object_{frame:05d}.npz"
    np.savez(out, vertices=recon["vertices"], faces=recon["faces"], pose=recon["pose"])
    print(f"object: {len(recon['vertices'])} verts; wrote {out}")


@app.command()
def grasp(
    frame: int = typer.Option(..., help="Frame index."),
    repo_id: str = typer.Option(DATASET_REPO_ID, help="LeRobot dataset repo id."),
    graspgen: bool = typer.Option(False, help="Use GraspGen (GPU) on a reconstructed object mesh."),
    gripper_config: str = typer.Option("", help="GraspGen gripper YAML (required with --graspgen)."),
    object_npz: str = typer.Option("", help="Object mesh artifact from `egodex object` (--graspgen)."),
) -> None:
    """Generate a parallel-gripper grasp (KMeans(2) over ARKit fingertips; GraspGen [GPU])."""
    from . import dataset as ds
    from . import pose
    from .stages import grasp as grasp_stage

    dataset = ds.load_dataset(repo_id)
    sample = ds.get_frame(dataset, frame)
    hands = pose.decode_state(sample["observation.state"])
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS / f"grasp_{frame:05d}.npz"
    payload = {}

    if graspgen:
        loaded = np.load(object_npz, allow_pickle=True)
        object_mesh = {"vertices": loaded["vertices"], "faces": loaded["faces"], "pose": loaded["pose"]}
        # Select per hand against that hand's wrist pose (must share the object's frame).
        for side, h in hands.items():
            g = grasp_stage.grasp_from_graspgen(object_mesh, hand_pose=h, gripper_config=gripper_config)
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
) -> None:
    """Solve IK for the follower arms and render the robot through the camera intrinsics."""
    from . import dataset as ds

    dataset = ds.load_dataset(repo_id)
    sample = ds.get_frame(dataset, frame)
    _, result, rgba = _retarget_frame(sample)
    for side, arm in result.arms.items():
        flag = "" if arm.reachable else "  [clamped: target out of reach]"
        print(
            f"{side}: pos_res={arm.position_residual:.4f} m, "
            f"rot_res={np.degrees(arm.orientation_residual):.1f} deg{flag}"
        )
    out = OUTPUTS / f"render_{frame:05d}.png"
    _save_rgba(out, rgba)
    print(f"wrote {out}")


@app.command()
def overlay(
    frame: int = typer.Option(..., help="Frame index."),
    repo_id: str = typer.Option(DATASET_REPO_ID, help="LeRobot dataset repo id."),
) -> None:
    """Composite the robot render over the original egocentric frame."""
    from . import dataset as ds
    from .stages import overlay as overlay_stage

    dataset = ds.load_dataset(repo_id)
    sample = ds.get_frame(dataset, frame)
    _, _, rgba = _retarget_frame(sample)
    composited = overlay_stage.overlay(sample["rgb"], rgba)
    out = OUTPUTS / f"overlay_{frame:05d}.png"
    _save_rgb(out, composited)
    print(f"wrote {out}")


@app.command()
def run(
    frame: int = typer.Option(0, help="Frame index."),
    repo_id: str = typer.Option(DATASET_REPO_ID, help="LeRobot dataset repo id."),
) -> None:
    """Chain the available CPU stages end-to-end (GPU stages are skipped with a notice)."""
    from . import dataset as ds
    from .stages import overlay as overlay_stage

    print("Skipping GPU stages (object: SAM-3D, hand: HAMER, grasp: GraspGen). See AGENTS.md.")
    dataset = ds.load_dataset(repo_id)
    sample = ds.get_frame(dataset, frame)
    print(f"loaded frame {frame} (intrinsics: {sample['intrinsics_src']})")

    _, result, rgba = _retarget_frame(sample)
    for side, arm in result.arms.items():
        flag = "" if arm.reachable else "  [clamped]"
        print(f"IK {side}: pos_res={arm.position_residual:.4f} m{flag}")

    composited = overlay_stage.overlay(sample["rgb"], rgba)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    _save_rgba(OUTPUTS / f"render_{frame:05d}.png", rgba)
    out = OUTPUTS / f"overlay_{frame:05d}.png"
    _save_rgb(out, composited)
    print(f"wrote {out}")


if __name__ == "__main__":
    app()
