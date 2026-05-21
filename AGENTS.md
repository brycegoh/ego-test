# AGENTS.md

Orientation for agents/developers working in this repo. Read alongside
[README.md](README.md).

## What this is

A `uv` project that retargets the EgoDex egocentric human-hand dataset
(`pepijn223/egodex-test`, LeRobot format) onto a Mobile ALOHA bimanual parallel-gripper
robot and overlays the robot on the original egocentric frame.

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
  viz.py                  rerun 3-panel view: 3D scene | edited video | original video
  stages/
    hamer.py              [GPU] optional RGB hand-mesh reconstruction (HAMER)
    sam3d.py              [GPU] object 3D reconstruction (SAM-3D-Objects)
    grasp.py              GraspGen [GPU] + KMeans(2) contact-cluster fallback (CPU)
    render.py             load Mobile ALOHA URDF in rerun, pose to grasp, snapshot
    overlay.py            composite robot render over the original frame
scripts/fetch_urdf.sh     fetch Mobile ALOHA flat URDF -> assets/urdf/aloha/urdf/aloha.urdf
assets/urdf/              URDF + meshes land here (gitignored)
outputs/                  stage artifacts: frames, poses, renders, overlays (gitignored)
```

## Workflow

```bash
uv sync                          # core CPU env (lerobot, rerun-sdk, numpy, opencv, typer, sklearn)
bash scripts/fetch_urdf.sh       # Mobile ALOHA flat URDF -> assets/urdf/aloha/urdf/aloha.urdf
uv run egodex <stage> [--frame N]
```

Each stage reads/writes small artifacts under `outputs/` so stages can be run
independently or chained via `egodex run`.

## Environments: what runs where

The pipeline runs the **full GPU chain by default** (HAMER hands → SAM 2 mask → SAM-3D
object → GraspGen grasp). A **CPU fallback** (ARKit pose + KMeans grasp, no object recon)
runs without a GPU via `--cpu` (or `--arkit` / `--kmeans`) and covers `load`, `viz`,
`render`, `overlay`, `run`.

The GPU stages are **first-class code**: their heavy imports live at the top of each stage
module (no `try/except`, no `NotImplementedError`), and they are declared as **per-stage
optional extras** in `pyproject.toml`:
- **HAMER** — https://github.com/geopavlakos/hamer — `pip install '.[hamer]'`
- **SAM 2** — https://github.com/facebookresearch/sam2 — `pip install '.[sam2]'`
- **SAM-3D-Objects** — https://github.com/facebookresearch/sam-3d-objects — `pip install '.[sam3d]'`
- **GraspGen** — https://github.com/NVlabs/GraspGen — `pip install '.[graspgen]'`

Install each into **its own environment** — they have mutually conflicting torch/CUDA pins,
so they cannot coexist in one process (the shared `pipeline.py` imports only the *selected*
backend, lazily). The extras are best-effort: some sub-deps don't express as plain
requirements (HAMER's `third-party/ViTPose` + a detectron2 build; SAM-3D's pytorch3d) and
need the manual steps in each stage module's docstring. Keep model weights under `weights/`
or the repo-specified `checkpoints/` (gitignored). Importing `egodex_robot` and the CPU
stages never pulls in torch — only a selected GPU stage module does.

## Key facts / decisions

- **Hand pose** defaults to **HAMER** (`stages/hamer.py`); the dataset's **ARKit 3D
  annotations** (`pose.decode_state`) are the CPU fallback (`--arkit` / `Backends.cpu()`).
- **Grasp**: defaults to **GraspGen** (`stages/graspgen.py`, needs the SAM 2 → SAM-3D object);
  KMeans(2) over fingertip/contact points is the CPU fallback (`--kmeans`).
- **Object**: **SAM 2** (`stages/segment.py`) masks the held object → **SAM-3D-Objects**
  (`stages/sam3d.py`) reconstructs it. Camera-frame outputs are lifted to the world (ARKit)
  frame in `pipeline.py` so grasps and the hand pose are compared in one frame.
- **Robot**: Mobile ALOHA (bimanual 6-DOF parallel-jaw arms on an AgileX Tracer base). URDF
  from `agilexrobotics/mobile_aloha_sim` (master), file
  `aloha_new_description/urdf/aloha_tracer2_dabai_dark.urdf` — a flat URDF, no xacro/ROS
  needed. `scripts/fetch_urdf.sh` lays it out under `assets/urdf/` (validated: 52 links, 36
  actuated joints, 99 meshes, all resolve). The model has four arm chains: `fl_`/`fr_`
  (front = follower manipulators we retarget) and `bl_`/`br_` (back = leaders).
- **Mesh resolution**: meshes are `package://<pkg>/...`. rerun resolves these via
  `ROS_PACKAGE_PATH`, which must include `assets/urdf/` (the dir holding the package-named
  folders). `render.ensure_package_path()` sets this; validated that rerun then embeds all
  meshes. (No mesh-path rewriting needed.)
- **Camera**: single egocentric view (the only view EgoDex provides).
- **Reuse** existing tools — don't re-implement: `LeRobotDataset` (loading), rerun's
  built-in URDF loader (rerun-sdk ≥ 0.29) and lerobot's rerun viz approach (visualization).
- The 48-dim `observation.state` layout must be **confirmed at runtime** against
  `meta/info.json` (via `egodex load`) before relying on specific indices in `pose.py`.

## Conventions

- CLI via Typer; one subcommand per stage; stages take `--frame`/`--episode` and print the
  output paths they wrote.
- GPU stages import their deps at module top (installed via per-stage extras). The CPU path
  and `import egodex_robot` stay torch-free because `pipeline.py` / the CLI import a GPU
  stage module only when its backend is selected. Don't reintroduce module-level GPU imports
  into the shared modules (`pipeline.py`, `stages/grasp.py`, `stages/render.py`).
- Write artifacts to `outputs/`; weights to `weights/`; both are gitignored.

## Environment caveats

- This sandbox blocks the Hugging Face host, so `egodex load` (dataset download) and some
  installs may fail here — run them where network access is available.
- GPU stages cannot run in CPU-only sandboxes; validate them on a GPU host.

## Git

- Develop on branch `claude/egodex-dataset-project-iHcFt`.
- Commit with clear messages; push with `git push -u origin claude/egodex-dataset-project-iHcFt`.
- Do not open a pull request unless explicitly asked.
