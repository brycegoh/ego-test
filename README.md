# EgoDex → Mobile ALOHA robot retargeting

Turn an **egocentric human-hand** manipulation dataset into something usable for a
**bimanual mobile manipulator with parallel grippers** (Mobile ALOHA): reconstruct each
frame's scene in 3D, place the robot where the hands are, pick a sensible grasp, render it
through the original camera, and overlay the robot on the egocentric frame.

> Status: **scaffold**. The project structure, CLI, and per-stage contracts are in place;
> the stage logic is not implemented yet. See [AGENTS.md](AGENTS.md).

## Dataset

[`pepijn223/egodex-test`](https://huggingface.co/datasets/pepijn223/egodex-test) — a
[LeRobot](https://github.com/huggingface/lerobot)-format slice of Apple's
[EgoDex](https://github.com/apple/ml-egodex).

- **Single egocentric RGB camera** (Apple Vision Pro), 30 fps.
- `observation.state`: **48-dim float32** — wrist position + rotation + finger joints,
  derived from EgoDex's ARKit hand tracking.
- ~3 episodes / ~632 frames / 3 tasks in this test slice.
- EgoDex source annotations (HDF5): a 3×3 camera intrinsics matrix, per-frame 4×4 camera
  extrinsics, 68 joint 4×4 SE(3) transforms in the ARKit origin frame, optional per-joint
  confidence, and natural-language task labels.

> Exact feature keys/shapes are confirmed at runtime from `meta/info.json` (`egodex load`).

## Problem & goal

The data shows *human hands*, not a robot, and the hand is present in the camera frame.
We want, for a given egocentric frame, a recreated image in which a Mobile ALOHA arm is
positioned like the hand and — when grasping — chooses a sensible gripper orientation.
Mobile ALOHA is bimanual (two ViperX 300 6-DOF arms with parallel-jaw grippers on a mobile
base), so the left/right hands map naturally onto its left/right arms.

Challenges:
1. **Hand → gripper pose translation** (human hand vs. 2-finger parallel gripper).
2. **The hand occupies the camera frame** and must be replaced by the robot.

> **Camera note:** EgoDex is a *single egocentric view*. An earlier draft mentioned
> "top / left wrist / right wrist" cameras — those describe a **future target robot**, not
> this dataset. This project produces the single egocentric view with the robot overlaid.

## Pipeline

```
                  egocentric RGB frame  (+ ARKit hand annotations, camera intrinsics)
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                            ▼
  hand pose                                     object 3D
  • ARKit annotations  (default, in dataset)    • SAM-3D-Objects   [GPU]
  • HAMER from RGB     (optional)  [GPU]           reconstruct mesh + pose
        │                                            │
        └──────────────────────┬─────────────────────┘
                               ▼
                          grasp gen
                          • GraspGen 6-DOF grasp        [GPU]
                          • KMeans(2) contact clusters  (CPU fallback)
                               │
                               ▼
                     Mobile ALOHA URDF render in rerun
                     (pose arms to grasp, snapshot via camera intrinsics)
                               │
                               ▼
                     overlay robot render on original frame
```

Design decisions:
- **Hand pose** comes from the dataset's **ARKit 3D annotations** by default (accurate and
  already present). [HAMER](https://github.com/geopavlakos/hamer) is an optional RGB-only
  alternative / comparison.
- **Object** is reconstructed with
  [SAM-3D-Objects](https://github.com/facebookresearch/sam-3d-objects).
- **Grasp** is generated with [GraspGen](https://github.com/NVlabs/GraspGen) (preferred);
  a KMeans(2) clustering of fingertip/contact points into the two gripper jaws is the
  no-GPU fallback.
- **Render** uses [rerun](https://rerun.io)'s built-in URDF loader with the **Mobile
  ALOHA** description from
  [agilexrobotics/mobile_aloha_sim](https://github.com/agilexrobotics/mobile_aloha_sim)
  (a ready-made flat URDF — no xacro needed — fetched by `scripts/fetch_urdf.sh`).

## Visualization

`egodex viz` opens a [rerun](https://rerun.io) window laid out as three synchronized
panels (sharing the dataset timeline so scrubbing stays in sync):

```
┌──────────────┬──────────────────┬──────────────────┐
│   3D scene   │   edited video   │  original video  │
│ hand kpts +  │  robot overlaid  │   raw egocentric │
│ object mesh +│  on the frame    │      RGB         │
│ ALOHA URDF   │                  │                  │
└──────────────┴──────────────────┴──────────────────┘
```

- **3D scene** — camera frustum, 3D hand keypoints/skeleton, reconstructed object mesh,
  and the posed Mobile ALOHA URDF.
- **edited video** — the overlay output (robot composited over the frame).
- **original video** — the raw egocentric RGB stream.

## Commands (intended UX)

```bash
uv sync                          # install the core CPU env
bash scripts/fetch_urdf.sh       # fetch the Mobile ALOHA URDF + meshes into assets/urdf/
export ROS_PACKAGE_PATH="$PWD/assets/urdf"   # so rerun resolves package:// mesh refs

uv run egodex load               # download + inspect the dataset, dump a sample frame
uv run egodex viz                # rerun: 3D scene | edited video | original video
uv run egodex hand   --frame 0   # hand pose (ARKit annotations / HAMER)
uv run egodex object --frame 0   # object 3D reconstruction (SAM-3D)        [GPU]
uv run egodex grasp  --frame 0   # parallel-gripper grasp (GraspGen / KMeans)
uv run egodex render --frame 0   # pose Mobile ALOHA URDF + snapshot
uv run egodex overlay --frame 0  # composite robot over the original frame
uv run egodex run    --frame 0   # chain the available stages
```

CPU-only stages: `load`, `viz`, `grasp` (KMeans fallback), `render`, `overlay`.
GPU + separate environment + weights: `object`, `hand` (HAMER), `grasp` (GraspGen).
See [AGENTS.md](AGENTS.md) for environment setup.

## Existing literature

- **DexUMI** — inpaints the human hand with a robotic hand as a manipulation interface:
  [PDF](https://dex-umi.github.io/static/pdfs/DexUMI:%20Using%20Human%20Hand%20as%20the%20Universal%20Manipulation%20Interface%20for%20Dexterous%20Manipulation.pdf)
