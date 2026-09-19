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
SCENE_IDS="${SCENE_IDS:-none}"   # comma-separated scene ids to SKIP; "none" = evaluate all
CONFIG="${CONFIG:-config/vln_r2r.yaml}"
EVAL_SPLIT="${EVAL_SPLIT:-val_unseen}"
SAVE_VIDEO="${SAVE_VIDEO:-0}"
VLN_SEGMENT_TRANSITION_WEIGHTS_PATH="${VLN_SEGMENT_TRANSITION_WEIGHTS_PATH:-config/adageovln.json}"
VGGT_TOTAL_BUDGET="${VGGT_TOTAL_BUDGET:-900000}"
MAX_STEPS="${MAX_STEPS:-400}"
OUTPUT_PATH="${OUTPUT_PATH:-evaluation/adageovln}"

# Enable the budgeted AdaGeoKV cache with navigation-aware additive scoring.
export USE_ADAGEO_KV_CACHE="${USE_ADAGEO_KV_CACHE:-1}"
export ADAGEO_KV_SCORE_MODE="${ADAGEO_KV_SCORE_MODE:-vln_segment_transition}"
export VLN_SEGMENT_TRANSITION_WEIGHTS_PATH
export VGGT_TOTAL_BUDGET
export VGGT_KV_START="${VGGT_KV_START:-8}"
export VGGT_KV_RECENT="${VGGT_KV_RECENT:-56}"
VLN_SCORE_PROBE="${VLN_SCORE_PROBE:-0}"
VLN_SCORE_PROBE_DIR="${VLN_SCORE_PROBE_DIR:-${OUTPUT_PATH}/probe}"
VLN_SCORE_PROBE_EVERY="${VLN_SCORE_PROBE_EVERY:-10}"
VLN_SCORE_PROBE_LAYERS="${VLN_SCORE_PROBE_LAYERS:-0,11,23}"
export VLN_SCORE_PROBE
export VLN_SCORE_PROBE_DIR
export VLN_SCORE_PROBE_EVERY
export VLN_SCORE_PROBE_LAYERS
export VLN_ORACLE_STOP="${VLN_ORACLE_STOP:-0}"
export GEOMETRY_ENCODER_PATH

echo "CHECKPOINT: ${CHECKPOINT}"
echo "SCENE_IDS: ${SCENE_IDS}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
echo "CONFIG: ${CONFIG}"
echo "EVAL_SPLIT: ${EVAL_SPLIT}"
echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "USE_ADAGEO_KV_CACHE: ${USE_ADAGEO_KV_CACHE}"
echo "ADAGEO_KV_SCORE_MODE: ${ADAGEO_KV_SCORE_MODE}"
echo "VLN_SEGMENT_TRANSITION_WEIGHTS_PATH: ${VLN_SEGMENT_TRANSITION_WEIGHTS_PATH}"
echo "VGGT_TOTAL_BUDGET: ${VGGT_TOTAL_BUDGET}"
echo "MAX_STEPS: ${MAX_STEPS}"
echo "VGGT_KV: start=${VGGT_KV_START} recent=${VGGT_KV_RECENT}"
echo "VLN_ORACLE_STOP: ${VLN_ORACLE_STOP}"
echo "VLN_SCORE_PROBE: ${VLN_SCORE_PROBE}"
echo "VLN_SCORE_PROBE_DIR: ${VLN_SCORE_PROBE_DIR}"
echo "VLN_SCORE_PROBE_EVERY: ${VLN_SCORE_PROBE_EVERY}"
echo "VLN_SCORE_PROBE_LAYERS: ${VLN_SCORE_PROBE_LAYERS}"

mkdir -p "${OUTPUT_PATH}"

extra_args=()
if [ "${SAVE_VIDEO}" = "1" ]; then
  extra_args+=(--save_video)
fi

torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" src/evaluation_scene.py \
  --model_path "${CHECKPOINT}" \
  --geometry_encoder_path "${GEOMETRY_ENCODER_PATH}" \
  --habitat_config_path "${CONFIG}" \
  --eval_split "${EVAL_SPLIT}" \
  --scene_ids "${SCENE_IDS}" \
  --output_path "${OUTPUT_PATH}" \
  --max_steps "${MAX_STEPS}" \
  "${extra_args[@]}"
