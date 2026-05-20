#!/usr/bin/env bash
# Fetch the Mobile ALOHA robot description and produce a flat URDF in assets/urdf/.
#
# Mobile ALOHA is bimanual (two Interbotix ViperX 300 6-DOF arms with parallel-jaw
# grippers) on a mobile base -- a good match for the project goal. The render stage
# (src/egodex_robot/stages/render.py) loads `assets/urdf/aloha.urdf` via rerun's built-in
# URDF loader, which needs a *flat* .urdf (not xacro), so we expand the xacro here.
#
# Requires network access to github.com (may be blocked in sandboxed environments) and the
# `xacro` tool (from ROS 2) on PATH. Exact in-repo paths should be verified on first fetch.
#
# Alternatives if ROS/xacro is unavailable:
#   - `pip install robot_descriptions` and use its ALOHA assets, or
#   - google-deepmind/mujoco_menagerie `aloha` (MJCF; needs URDF conversion for rerun).
#
# Usage: bash scripts/fetch_urdf.sh
set -euo pipefail

DEST="assets/urdf"
REPO="https://github.com/Interbotix/aloha.git"   # maintained Mobile ALOHA / ALOHA 2 stack
mkdir -p "${DEST}"

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

git clone --depth 1 "${REPO}" "${TMP}/aloha"

# Locate the top-level bimanual ALOHA xacro and its meshes (path verified on fetch).
XACRO="$(find "${TMP}/aloha" -name 'aloha*.urdf.xacro' | head -n1)"
if [[ -z "${XACRO}" ]]; then
  echo "Could not find an aloha*.urdf.xacro in ${REPO}." >&2
  echo "Inspect the clone under ${TMP}/aloha and update this script's path." >&2
  exit 1
fi

# Copy meshes referenced by the description, then expand xacro -> flat URDF.
DESC_DIR="$(dirname "$(dirname "${XACRO}")")"   # .../<pkg>/urdf/foo.xacro -> .../<pkg>
cp -r "${DESC_DIR}/." "${DEST}/"
xacro "${XACRO}" > "${DEST}/aloha.urdf"

echo "Mobile ALOHA URDF written to ${DEST}/aloha.urdf"
