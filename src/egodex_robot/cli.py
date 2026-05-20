"""`egodex` command-line interface.

Each subcommand maps to one pipeline stage. Stages are designed to run independently and
to read/write small artifacts under `outputs/`, so they can be invoked one at a time with
short commands or chained with `egodex run`.

    egodex load                 # download + inspect pepijn223/egodex-test, dump a frame
    egodex viz                  # rerun: 3D scene | edited video | original video
    egodex hand --frame N       # extract hand pose for a frame (ARKit annotations / HAMER)
    egodex object --frame N     # reconstruct the manipulated object in 3D (SAM-3D)  [GPU]
    egodex grasp --frame N      # generate parallel-gripper grasp (GraspGen / KMeans)
    egodex render --frame N     # pose the SO-101 URDF and snapshot via camera intrinsics
    egodex overlay --frame N    # composite the robot render over the original frame
    egodex run --frame N        # chain the available stages end-to-end

SCAFFOLD: command signatures are defined; bodies are not implemented yet.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    add_completion=False,
    help="EgoDex -> SO-101 robot retargeting pipeline (scaffold).",
)

DATASET_REPO_ID = "pepijn223/egodex-test"

_NOT_IMPLEMENTED = "This stage is a scaffold and is not implemented yet. See AGENTS.md."


@app.command()
def load(
    repo_id: str = typer.Option(DATASET_REPO_ID, help="LeRobot dataset repo id."),
    frame: int = typer.Option(0, help="Frame index to dump a sample for."),
) -> None:
    """Download + inspect the dataset and dump a sample frame/pose to `outputs/`."""
    raise NotImplementedError(_NOT_IMPLEMENTED)


@app.command()
def viz(
    repo_id: str = typer.Option(DATASET_REPO_ID, help="LeRobot dataset repo id."),
    episode: int = typer.Option(0, help="Episode index to visualize."),
) -> None:
    """Open a 3-panel rerun view: 3D scene | edited (overlaid) video | original video."""
    raise NotImplementedError(_NOT_IMPLEMENTED)


@app.command()
def hand(frame: int = typer.Option(..., help="Frame index.")) -> None:
    """Extract hand pose for a frame (ARKit annotations by default; HAMER optional)."""
    raise NotImplementedError(_NOT_IMPLEMENTED)


@app.command()
def object(frame: int = typer.Option(..., help="Frame index.")) -> None:
    """Reconstruct the manipulated object's 3D mesh from RGB (SAM-3D-Objects). [GPU]"""
    raise NotImplementedError(_NOT_IMPLEMENTED)


@app.command()
def grasp(frame: int = typer.Option(..., help="Frame index.")) -> None:
    """Generate a parallel-gripper grasp (GraspGen [GPU]; KMeans(2) fallback on CPU)."""
    raise NotImplementedError(_NOT_IMPLEMENTED)


@app.command()
def render(frame: int = typer.Option(..., help="Frame index.")) -> None:
    """Pose the SO-101 URDF to the grasp and snapshot it through the camera intrinsics."""
    raise NotImplementedError(_NOT_IMPLEMENTED)


@app.command()
def overlay(frame: int = typer.Option(..., help="Frame index.")) -> None:
    """Composite the robot render over the original egocentric frame."""
    raise NotImplementedError(_NOT_IMPLEMENTED)


@app.command()
def run(frame: int = typer.Option(0, help="Frame index.")) -> None:
    """Chain the available stages end-to-end (skipping GPU-gated stages with a notice)."""
    raise NotImplementedError(_NOT_IMPLEMENTED)


if __name__ == "__main__":
    app()
