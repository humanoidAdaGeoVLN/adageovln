#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

MASTER_PORT=$((RANDOM % 101 + 20000))
NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi --list-gpus 2>/dev/null | wc -l)}"
if [ "${NPROC_PER_NODE}" -lt 1 ]; then
  NPROC_PER_NODE=1
fi

CHECKPOINT="${CHECKPOINT:-checkpoints/adageovln}"
GEOMETRY_ENCODER_PATH="${GEOMETRY_ENCODER_PATH:-checkpoints/VGGT-1B}"
OUTPUT_PATH="${OUTPUT_PATH:-evaluation/vln}"
SCENE_IDS="${SCENE_IDS:-EU6Fwq7SyZv}"
CONFIG="${CONFIG:-config/vln_r2r.yaml}"
EVAL_SPLIT="${EVAL_SPLIT:-val_unseen}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
# VGGT streaming KV-cache window (frames of geometry history kept during eval).
export VGGT_KV_START="${VGGT_KV_START:-8}"
export VGGT_KV_RECENT="${VGGT_KV_RECENT:-48}"
# Cache frame-strict geometry after geo_ln/MLP/gate/scale so historical frames
# do not repeat the fusion projection. Set to 0 to use legacy raw-feature caching.
export VLN_PROJECTED_GEOMETRY_CACHE="${VLN_PROJECTED_GEOMETRY_CACHE:-1}"
# Oracle-stop diagnostic: auto-STOP within success_distance of the goal (isolates
# navigation quality from the stop policy). Set VLN_ORACLE_STOP=1 to enable.
export VLN_ORACLE_STOP="${VLN_ORACLE_STOP:-0}"
export GEOMETRY_ENCODER_PATH

echo "CHECKPOINT: ${CHECKPOINT}"
echo "GEOMETRY_ENCODER_PATH: ${GEOMETRY_ENCODER_PATH}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
echo "CONFIG: ${CONFIG}"
echo "EVAL_SPLIT: ${EVAL_SPLIT}"
echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "SAVE_VIDEO: ${SAVE_VIDEO}"
echo "VGGT_KV: start=${VGGT_KV_START} recent=${VGGT_KV_RECENT}"
echo "VLN_PROJECTED_GEOMETRY_CACHE: ${VLN_PROJECTED_GEOMETRY_CACHE}"
echo "VLN_ORACLE_STOP: ${VLN_ORACLE_STOP}"

mkdir -p "${OUTPUT_PATH}"

extra_args=()
if [ "${SAVE_VIDEO}" = "1" ]; then
  extra_args+=(--save_video)
fi

torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" src/evaluation.py \
  --model_path "${CHECKPOINT}" \
  --geometry_encoder_path "${GEOMETRY_ENCODER_PATH}" \
  --habitat_config_path "${CONFIG}" \
  --eval_split "${EVAL_SPLIT}" \
  --scene_ids "${SCENE_IDS}" \
  --output_path "${OUTPUT_PATH}" \
  "${extra_args[@]}"
