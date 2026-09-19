#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

CHECKPOINT="${CHECKPOINT:-checkpoints/adageovln}"
GEOMETRY_ENCODER_PATH="${GEOMETRY_ENCODER_PATH:-checkpoints/VGGT-1B}"
OUTPUT_PATH="${OUTPUT_PATH:-evaluation/vln_multiprocess}"
CONFIG="${CONFIG:-config/vln_r2r.yaml}"
EVAL_SPLIT="${EVAL_SPLIT:-val_unseen}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"

# Comma-separated physical GPU ids. One model process is launched per shard;
# when CHUNKS exceeds the number of GPU ids, assignment wraps round.
if [ -z "${GPU_IDS:-}" ]; then
  GPU_IDS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, - || true)"
  GPU_IDS="${GPU_IDS:-0}"
fi
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
GPU_COUNT="${#GPU_ARRAY[@]}"
PROCESSES_PER_GPU="${PROCESSES_PER_GPU:-2}"
CHUNKS="${CHUNKS:-$((GPU_COUNT * PROCESSES_PER_GPU))}"

export VGGT_KV_START="${VGGT_KV_START:-8}"
export VGGT_KV_RECENT="${VGGT_KV_RECENT:-48}"
export VLN_PROJECTED_GEOMETRY_CACHE="${VLN_PROJECTED_GEOMETRY_CACHE:-1}"
export VLN_ORACLE_STOP="${VLN_ORACLE_STOP:-0}"
export GEOMETRY_ENCODER_PATH

if [ "${CHUNKS}" -lt 1 ]; then
  echo "CHUNKS must be at least 1" >&2
  exit 2
fi
if [ "${PROCESSES_PER_GPU}" -lt 1 ]; then
  echo "PROCESSES_PER_GPU must be at least 1" >&2
  exit 2
fi
if [ "${#GPU_ARRAY[@]}" -lt 1 ] || [ -z "${GPU_ARRAY[0]}" ]; then
  echo "GPU_IDS must contain at least one GPU id" >&2
  exit 2
fi

echo "CHECKPOINT: ${CHECKPOINT}"
echo "GEOMETRY_ENCODER_PATH: ${GEOMETRY_ENCODER_PATH}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
echo "CONFIG: ${CONFIG}"
echo "EVAL_SPLIT: ${EVAL_SPLIT}"
echo "CHUNKS: ${CHUNKS}"
echo "GPU_IDS: ${GPU_IDS}"
echo "PROCESSES_PER_GPU: ${PROCESSES_PER_GPU}"
echo "SAVE_VIDEO: ${SAVE_VIDEO}"

mkdir -p "${OUTPUT_PATH}/shards"

extra_args=()
if [ "${SAVE_VIDEO}" = "1" ]; then
  extra_args+=(--save_video)
fi

pids=()

cleanup_workers() {
  local pid
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      # Each worker is started with setsid, so the negative id terminates the
      # worker's whole process group, including simulator/video subprocesses.
      kill -TERM -- "-${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
}

handle_interrupt() {
  trap - INT TERM
  echo
  echo "Interrupted. Stopping all evaluation workers..." >&2
  cleanup_workers
  exit 130
}

trap handle_interrupt INT TERM
trap cleanup_workers EXIT

for ((split_id = 0; split_id < CHUNKS; split_id++)); do
  gpu_id="${GPU_ARRAY[$((split_id % GPU_COUNT))]}"
  echo "Launching shard ${split_id}/${CHUNKS} on GPU ${gpu_id}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONUNBUFFERED=1 setsid python src/evaluation_multiprocess.py \
    --split-num "${CHUNKS}" \
    --split-id "${split_id}" \
    --model_path "${CHECKPOINT}" \
    --geometry_encoder_path "${GEOMETRY_ENCODER_PATH}" \
    --habitat_config_path "${CONFIG}" \
    --eval_split "${EVAL_SPLIT}" \
    --output_path "${OUTPUT_PATH}" \
    "${extra_args[@]}" &
  pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "Shard ${index} failed; see the error above." >&2
    failed=1
  fi
done
if [ "${failed}" -ne 0 ]; then
  exit 1
fi

echo "All ${CHUNKS} shards completed. Merging results..."
python src/evaluation_multiprocess.py \
  --split-num "${CHUNKS}" \
  --output_path "${OUTPUT_PATH}" \
  --merge-only
