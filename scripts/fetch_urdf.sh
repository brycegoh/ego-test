#!/usr/bin/env bash
# Fetch the Mobile ALOHA URDF + meshes from AgileX's simulation repo into assets/urdf/.
#
#   https://github.com/agilexrobotics/mobile_aloha_sim   (branch: master)
#
# Mobile ALOHA is bimanual (two 6-DOF Piper arms with parallel-jaw grippers) on an AgileX
# Tracer mobile base. We use `aloha_tracer2_dabai_dark.urdf`: a *flat* URDF (no xacro /
# `$(find ...)` includes) covering both arms + base. Its meshes live in two ROS packages,
# `aloha_new_description` and `tracer2_description`, referenced as `package://<pkg>/...`.
#
# Layout produced (package root = assets/urdf/, so `package://<pkg>/x` -> assets/urdf/<pkg>/x):
#   assets/urdf/aloha_new_description/urdf/aloha_tracer2_dabai_dark.urdf   <- DEFAULT_URDF
#   assets/urdf/aloha_new_description/meshes/...
#   assets/urdf/tracer2_description/meshes/...
#
# rerun's URDF loader resolves package:// via ROS_PACKAGE_PATH, so the render stage sets
#   ROS_PACKAGE_PATH=<abs path to assets/urdf>
# before logging. Validated: with that set, rerun embeds all 99 meshes.
#
# Requires network access to github.com (may be blocked in sandboxed environments).
# Usage: bash scripts/fetch_urdf.sh
set -euo pipefail

DEST="assets/urdf"
REPO="https://github.com/agilexrobotics/mobile_aloha_sim.git"
mkdir -p "${DEST}"

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

# Sparse-checkout only the two description packages the chosen URDF needs.
git clone --depth 1 --filter=blob:none --sparse "${REPO}" "${TMP}/mobile_aloha_sim"
git -C "${TMP}/mobile_aloha_sim" sparse-checkout set \
  aloha_new_description tracer2_description

cp -r "${TMP}/mobile_aloha_sim/aloha_new_description" "${DEST}/"
cp -r "${TMP}/mobile_aloha_sim/tracer2_description" "${DEST}/"

URDF="${DEST}/aloha_new_description/urdf/aloha_tracer2_dabai_dark.urdf"
if [[ -f "${URDF}" ]]; then
  echo "Mobile ALOHA URDF ready at ${URDF}"
  echo "For rerun: export ROS_PACKAGE_PATH=\"$(cd "${DEST}" && pwd)\"  # resolves package:// meshes"
else
  echo "Expected ${URDF} not found; inspect ${DEST} and update render.DEFAULT_URDF." >&2
  exit 1
fi
