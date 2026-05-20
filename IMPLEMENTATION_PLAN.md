# Implementation Plan: EgoDex → Mobile ALOHA retargeting

Turning the scaffold into a working pipeline that places a Mobile ALOHA arm where a
human hand is in an egocentric frame and overlays it on the original image.

This plan supersedes the "direct end-effector placement" and "rerun snapshot" assumptions
in the original scaffold, which verification showed don't hold (see
[Why the scaffold's contracts changed](#why-the-scaffolds-contracts-changed)).

## Decisions locked in

- **Posing:** full-arm inverse kinematics on the `fl_`/`fr_` follower chains via
  [**placo**](https://github.com/Rhoban/placo).
- **Rendering:** rerun *cannot* rasterize the posed robot to an image, so the robot frames
  are produced **externally with pyrender** and then **fed into rerun for display**.
  pyrender makes pixels; rerun shows them.
- **Single kinematics source of truth:** placo parses the URDF and provides both IK
  (joint angles) and FK (link world transforms). Those same transforms drive *both*
  display backends — pyrender (rasterized overlay) and rerun (3D panel) — so the 3D view
  and the overlay can never drift apart.
- **Grasp:** generate-then-select. Preferred path uses GraspGen candidates selected by
  proximity to the human grasp; KMeans(2) on contact points is the CPU fallback.
- **New dependencies:** `placo`, `pyrender`, `trimesh` (all verified installable on
  PyPI, Python 3.11 compatible).

## Why the scaffold's contracts changed

Verified in this environment (rerun 0.32.2):

1. **rerun's URDF API is forward-kinematics only** — `UrdfJoint.compute_transform(value)`
   plus an optional base via `UrdfTree.from_file_path(..., static_transform_entity_path=)`.
   "Direct end-effector placement" is not meaningful for a kinematic tree; reaching a
   target wrist pose requires IK. → use placo.
2. **rerun has no offscreen render-to-image API** (no screenshot/snapshot/raster symbols in
   the SDK). It is a viewer/logger, not a rasterizer, so it cannot fulfil the
   `render.py` → `(H, W, 4)` RGBA contract that `overlay.py` consumes. → use pyrender for
   the pixels, rerun only for display.

Also verified: the Mobile ALOHA URDF loads through rerun with all meshes resolved; the
`fl_`/`fr_` follower chains exist with `joint1..joint6` (6-DOF arm) + `joint7/joint8`
(gripper). placo, pyrender, trimesh all install cleanly.

## The alignment crux

The overlay only lines up if pyrender's camera uses the **same intrinsics and per-frame
extrinsics** as the EgoDex camera, and every stage works in **one consistent world frame**.
`geometry.py` (frame plumbing) is therefore the foundation everything else builds on and
must be implemented and tested first.

## Open design problems (resolved for v1)

1. **Base placement → fixed offset relative to the camera.** `base = camera_pose ×
   constant_offset` (down and back, approximating a torso), so the arm appears to come
   "from the person" toward the hand. Deterministic and debuggable. placo reports a
   reachability residual per solve; unreachable targets clamp to the nearest reachable pose
   and are flagged. (Solving *for* the base is ambiguous and deferred.)
2. **Gripper target → raw wrist pose first, grasp pose later.** Validate IK → render →
   overlay alignment against the always-present, accurate ARKit wrist pose (with a hand→
   gripper axis remap) before coupling correctness to grasp quality. The runnable CPU path
   never depends on a GPU grasp stage.
3. **48-dim `observation.state` layout — unconfirmed here.** Hugging Face is blocked in this
   sandbox (`403 host_not_allowed`), so `pose.decode_state` indices cannot be verified.
   Implement against the README layout (per-hand wrist xyz + rotation + finger joints) and
   gate it behind a runtime assert against `meta/info.json`, to be run once where HF is
   reachable.

## Grasp: generate-then-select

- **Preferred (GPU):** SAM-3D mesh → GraspGen candidate set → pick the candidate minimizing
  an SE(3) distance to the human grasp (weighted position + orientation, optionally an
  approach-direction term and alignment of the gripper closing axis to the fingertip
  contact-cluster axis). To pin down when building: the **distance metric** and **frame
  alignment** (GraspGen grasps in object/camera frame must be compared in the same frame as
  the hand pose).
- **Fallback (CPU, runnable here):** KMeans(2) over fingertip/contact points → jaw midpoint
  + approach/closing axes + width.
- New `grasp.py` contract: add `select_grasp(candidates, hand_pose, contacts) -> Grasp`
  alongside the existing `grasp_from_graspgen` / `grasp_from_contacts`.

## Stage-by-stage

| File | Plan |
|---|---|
| `pyproject.toml` | add `placo`, `pyrender`, `trimesh`. Document headless GL: set `PYOPENGL_PLATFORM=egl` (or `osmesa`) for pyrender offscreen rendering. |
| `geometry.py` | implement `project` (pinhole `K·[R\|t]`), `transform_points`, `invert_se3`; add ARKit-world ↔ camera ↔ robot-world conversions. Unit-tested on synthetic SE(3). **Fully verifiable here.** |
| `pose.py` | `decode_state` → per-hand `HandPose` (wrist pos/rot + keypoints); `contact_points` → fingertips. Indices gated by a runtime schema check against `meta/info.json`. |
| `grasp.py` | implement `grasp_from_contacts` (KMeans(2)); add `select_grasp` (generate-then-select). KMeans path **verifiable on synthetic points here**; GraspGen stays GPU-gated. |
| `ik.py` *(new)* | placo wrapper: load URDF, set base transform, solve `fl_`/`fr_` 6-DOF for left/right gripper targets → joint angles + FK link world transforms + reachability residual. |
| `render.py` | rewrite: take placo FK link transforms → load each visual mesh (trimesh) → place in a pyrender scene → camera = dataset **intrinsics + extrinsics** → offscreen render → RGBA `(H, W, 4)`. rerun is no longer the renderer. |
| `overlay.py` | `overlay()` alpha-composite robot RGBA over the original RGB. **Verifiable here** on synthetic RGBA. |
| `viz.py` | rerun 3-panel: 3D = `UrdfTree` + per-joint `compute_transform` (FK from IK angles) + hand keypoints + camera frustum; "edited" + "original" = image streams (externally-produced overlay + raw RGB), all on the shared timeline. |
| `cli.py` / `run` | wire stages; `run` chains load → pose → grasp → ik → render → overlay, skipping GPU stages with a notice. |

## What's verifiable where

- **Here (CPU, now):** `geometry` math, KMeans grasp, overlay compositing, URDF load,
  placo IK against the real URDF, pyrender offscreen render of the posed robot (synthetic
  targets). The whole **render path** can be smoke-tested without the dataset.
- **Needs HF access (elsewhere):** dataset download, 48-dim `observation.state` layout
  confirmation, real per-frame intrinsics/extrinsics, end-to-end on actual frames.
- **GPU + separate env (unchanged):** HAMER, SAM-3D-Objects, GraspGen.

## Implementation order

1. `geometry.py` + tests (foundation).
2. `ik.py` (placo) — prove IK solves to the URDF's `fl_`/`fr_` chains.
3. `render.py` (pyrender) — prove a posed robot rasterizes through given intrinsics.
4. `pose.py` + `grasp.py` (CPU logic, schema-gated).
5. `overlay.py` + `viz.py` (display: external frames fed into rerun).
6. `cli` / `run` wiring + dataset stages (validated once HF is reachable).
