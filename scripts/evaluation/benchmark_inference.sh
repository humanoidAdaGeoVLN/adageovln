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
CONFIG="${CONFIG:-config/vln_r2r.yaml}"
EVAL_SPLIT="${EVAL_SPLIT:-val_unseen}"
GPU_ID="${GPU_ID:-0}"
BENCHMARK_EPISODES="${BENCHMARK_EPISODES:-10}"
PARALLEL_PROCESSES="${PARALLEL_PROCESSES:-2}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)_$$"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${PROJECT_ROOT}/evaluation_benchmark/${RUN_TAG}}"

export VGGT_KV_START="${VGGT_KV_START:-8}"
export VGGT_KV_RECENT="${VGGT_KV_RECENT:-48}"
export VLN_PROJECTED_GEOMETRY_CACHE="${VLN_PROJECTED_GEOMETRY_CACHE:-1}"
export VLN_ORACLE_STOP="${VLN_ORACLE_STOP:-0}"
export GEOMETRY_ENCODER_PATH

if [ "${BENCHMARK_EPISODES}" -lt 1 ]; then
  echo "BENCHMARK_EPISODES must be at least 1" >&2
  exit 2
fi
if [ "${PARALLEL_PROCESSES}" -lt 2 ]; then
  echo "PARALLEL_PROCESSES must be at least 2" >&2
  exit 2
fi
if [ "${PARALLEL_PROCESSES}" -gt "${BENCHMARK_EPISODES}" ]; then
  echo "PARALLEL_PROCESSES cannot exceed BENCHMARK_EPISODES" >&2
  exit 2
fi

SEQUENTIAL_OUTPUT="${BENCHMARK_ROOT}/sequential"
PARALLEL_OUTPUT="${BENCHMARK_ROOT}/parallel_${PARALLEL_PROCESSES}proc"
mkdir -p "${SEQUENTIAL_OUTPUT}" "${PARALLEL_OUTPUT}/shards"

common_args=(
  --model_path "${CHECKPOINT}"
  --geometry_encoder_path "${GEOMETRY_ENCODER_PATH}"
  --habitat_config_path "${CONFIG}"
  --eval_split "${EVAL_SPLIT}"
  --max-episodes-total "${BENCHMARK_EPISODES}"
)

echo "Benchmarking the same first ${BENCHMARK_EPISODES} episodes on GPU ${GPU_ID}"
echo "Output: ${BENCHMARK_ROOT}"

echo "[1/2] Sequential: 1 process"
sequential_start_ns="$(date +%s%N)"
CUDA_VISIBLE_DEVICES="${GPU_ID}" python src/evaluation_multiprocess.py \
  --split-num 1 \
  --split-id 0 \
  --output_path "${SEQUENTIAL_OUTPUT}" \
  "${common_args[@]}" >"${SEQUENTIAL_OUTPUT}/run.log" 2>&1
python src/evaluation_multiprocess.py \
  --split-num 1 \
  --output_path "${SEQUENTIAL_OUTPUT}" \
  --merge-only >>"${SEQUENTIAL_OUTPUT}/run.log" 2>&1
sequential_end_ns="$(date +%s%N)"

echo "[2/2] Parallel: ${PARALLEL_PROCESSES} processes on GPU ${GPU_ID}"
parallel_start_ns="$(date +%s%N)"
pids=()
for ((split_id = 0; split_id < PARALLEL_PROCESSES; split_id++)); do
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python src/evaluation_multiprocess.py \
    --split-num "${PARALLEL_PROCESSES}" \
    --split-id "${split_id}" \
    --output_path "${PARALLEL_OUTPUT}" \
    "${common_args[@]}" \
    >"${PARALLEL_OUTPUT}/shards/shard_$(printf '%04d' "${split_id}").log" 2>&1 &
  pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "Parallel shard ${index} failed; inspect ${PARALLEL_OUTPUT}/shards" >&2
    failed=1
  fi
done
if [ "${failed}" -ne 0 ]; then
  exit 1
fi

python src/evaluation_multiprocess.py \
  --split-num "${PARALLEL_PROCESSES}" \
  --output_path "${PARALLEL_OUTPUT}" \
  --merge-only >"${PARALLEL_OUTPUT}/merge.log" 2>&1
parallel_end_ns="$(date +%s%N)"

sequential_seconds="$(awk -v start="${sequential_start_ns}" -v end="${sequential_end_ns}" 'BEGIN {printf "%.3f", (end-start)/1000000000}')"
parallel_seconds="$(awk -v start="${parallel_start_ns}" -v end="${parallel_end_ns}" 'BEGIN {printf "%.3f", (end-start)/1000000000}')"
sequential_eps="$(awk -v n="${BENCHMARK_EPISODES}" -v t="${sequential_seconds}" 'BEGIN {printf "%.4f", n/t}')"
parallel_eps="$(awk -v n="${BENCHMARK_EPISODES}" -v t="${parallel_seconds}" 'BEGIN {printf "%.4f", n/t}')"
speedup="$(awk -v sequential="${sequential_seconds}" -v parallel="${parallel_seconds}" 'BEGIN {printf "%.3f", sequential/parallel}')"

echo
echo "================ VLN EVALUATION BENCHMARK ================"
printf "%-24s %12s %15s\n" "Mode" "Time (s)" "Episodes/sec"
printf "%-24s %12s %15s\n" "Sequential (1 proc)" "${sequential_seconds}" "${sequential_eps}"
printf "%-24s %12s %15s\n" "Parallel (${PARALLEL_PROCESSES} proc)" "${parallel_seconds}" "${parallel_eps}"
echo "Speedup: ${speedup}x"
echo "Sequential result: ${SEQUENTIAL_OUTPUT}/result.json"
echo "Parallel result:   ${PARALLEL_OUTPUT}/result.json"
echo "=========================================================="
