# AGENTS.md

Orientation for agents/developers working in this repo. Read alongside
[README.md](README.md).

## What this is

A `uv` project that retargets the EgoDex egocentric human-hand dataset
(`pepijn223/egodex-test`, LeRobot format) onto an SO-100/SO-101 parallel-gripper robot and
overlays the robot on the original egocentric frame.

**Current state: scaffold.** The package layout, CLI, and per-stage input/output contracts
exist; the stage logic is **not implemented**. Every `.py` under `src/egodex_robot/` is a
placeholder that raises `NotImplementedError`. Implement stages one at a time.

## Layout

```
pyproject.toml            uv project; core CPU deps; defines the `egodex` CLI entry point
src/egodex_robot/
  cli.py                  Typer CLI: load, viz, hand, object, grasp, render, overlay, run
  dataset.py              LeRobotDataset loader + schema inspection + frame extraction
  pose.py                 decode ARKit hand pose from the 48-dim observation.state
  geometry.py             camera intrinsics/extrinsics + SE(3) helpers, projection
  viz.py                  rerun logging (RGB + 3D hand keypoints + camera frustum)
  stages/
    hamer.py              [GPU] optional RGB hand-mesh reconstruction (HAMER)
    sam3d.py              [GPU] object 3D reconstruction (SAM-3D-Objects)
    grasp.py              GraspGen [GPU] + KMeans(2) contact-cluster fallback (CPU)
    render.py             load SO-101 URDF in rerun, pose to grasp, snapshot
    overlay.py            composite robot render over the original frame
scripts/fetch_urdf.sh     download the SO-101 URDF from TheRobotStudio/SO-ARM100
assets/urdf/              URDF + meshes land here (gitignored)
outputs/                  stage artifacts: frames, poses, renders, overlays (gitignored)
```

## Workflow

```bash
uv sync                          # core CPU env (lerobot, rerun-sdk, numpy, opencv, typer, sklearn)
bash scripts/fetch_urdf.sh       # SO-101 URDF -> assets/urdf/
uv run egodex <stage> [--frame N]
```

Each stage reads/writes small artifacts under `outputs/` so stages can be run
independently or chained via `egodex run`.

## Environments: what runs where

**CPU-only (core `uv` env)** — implementable and runnable here:
`load`, `viz`, `grasp` (KMeans fallback), `render`, `overlay`.

**GPU + separate environment + large weights** — do **not** add these to the core env;
they have mutually conflicting torch/CUDA/dependency pins and each expects its own setup:
- **HAMER** — https://github.com/geopavlakos/hamer — hand mesh from RGB.
- **SAM-3D-Objects** — https://github.com/facebookresearch/sam-3d-objects — object mesh.
- **GraspGen** — https://github.com/NVlabs/GraspGen — 6-DOF parallel-gripper grasps.

Recommended pattern: install each in its own venv/conda env, run it as a separate step
that writes a mesh/pose/grasp artifact into `outputs/`, then let the CPU stages
(`render`, `overlay`) consume those artifacts. Keep model weights under `weights/`
(gitignored). The gated modules raise a clear `NotImplementedError` with a pointer here
rather than importing heavy deps at module load.

## Key facts / decisions

- **Hand pose** defaults to the dataset's **ARKit 3D annotations** (`pose.decode_state`),
  not HAMER. HAMER is optional/comparison only.
- **Grasp**: prefer GraspGen; KMeans(2) over fingertip/contact points is the CPU fallback.
- **Robot**: SO-100/SO-101, URDF `Simulation/SO101/so101_new_calib.urdf` from
  `TheRobotStudio/SO-ARM100`.
- **Camera**: single egocentric view (the only view EgoDex provides).
- **Reuse** existing tools — don't re-implement: `LeRobotDataset` (loading), rerun's
  built-in URDF loader (rerun-sdk ≥ 0.29) and lerobot's rerun viz approach (visualization).
- The 48-dim `observation.state` layout must be **confirmed at runtime** against
  `meta/info.json` (via `egodex load`) before relying on specific indices in `pose.py`.

## Conventions

- CLI via Typer; one subcommand per stage; stages take `--frame`/`--episode` and print the
  output paths they wrote.
- Heavy/optional deps are **import-gated** inside the stage that needs them — importing
  `egodex_robot` must never require a GPU or pull torch from HAMER/SAM-3D/GraspGen.
- Write artifacts to `outputs/`; weights to `weights/`; both are gitignored.

## Environment caveats

- This sandbox blocks the Hugging Face host, so `egodex load` (dataset download) and some
  installs may fail here — run them where network access is available.
- GPU stages cannot run in CPU-only sandboxes; validate them on a GPU host.

## Git

- Develop on branch `claude/egodex-dataset-project-iHcFt`.
- Commit with clear messages; push with `git push -u origin claude/egodex-dataset-project-iHcFt`.
- Do not open a pull request unless explicitly asked.
