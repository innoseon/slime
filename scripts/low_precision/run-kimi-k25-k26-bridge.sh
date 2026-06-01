#!/bin/bash

set -euo pipefail

SLIME_DIR="${SLIME_DIR:-/root/slime}"
KIMI_HF_CKPT="${KIMI_HF_CKPT:-/inspire/hdd/project/ai4education/public/models/Kimi-K2.6}"
SAVE_DIR="${SAVE_DIR:-/inspire/qb-ilm/project/ai4education/p-liuwentao/ly-project/ckpt/kimi-k2.6-bridge-v030}"
PROMPT_DATA="${PROMPT_DATA:-/inspire/hdd/project/ai4education/p-liuwentao/slime/data/dapo_17k/dapo_math_17k.jsonl}"
INPUT_KEY="${INPUT_KEY:-prompt}"
LABEL_KEY="${LABEL_KEY:-label}"

if [[ "${KIMI_HF_CKPT}" == KIMI_HF_CKPT=* ]]; then
    echo "KIMI_HF_CKPT looks double-assigned: ${KIMI_HF_CKPT}" >&2
    echo "Use: export KIMI_HF_CKPT=/path/to/Kimi-K2.6" >&2
    exit 1
fi

for required in config.json tokenizer_config.json model.safetensors.index.json; do
    if [ ! -f "${KIMI_HF_CKPT}/${required}" ]; then
        echo "Missing ${required} in KIMI_HF_CKPT=${KIMI_HF_CKPT}" >&2
        exit 1
    fi
done

if [ ! -f "${PROMPT_DATA}" ]; then
    echo "PROMPT_DATA=${PROMPT_DATA} does not exist" >&2
    exit 1
fi

cd "${SLIME_DIR}"

export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

NNODES="${PET_NNODES:-${NNODES:-1}}"
NODE_RANK="${PET_NODE_RANK:-${NODE_RANK:-0}}"
GPUS_PER_NODE="${PET_NPROC_PER_NODE:-${GPUS_PER_NODE:-8}}"
MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-6379}}"
DASHBOARD_PORT="${DASHBOARD_PORT:-$((MASTER_PORT + 1))}"
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-${NNODES}}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-${GPUS_PER_NODE}}"
USE_EXTERNAL_RAY="${USE_EXTERNAL_RAY:-0}"

export MASTER_ADDR MASTER_PORT
export no_proxy="${no_proxy:-localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [ "${NVLINK_COUNT}" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: ${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"

source "${SLIME_DIR}/scripts/models/kimi-k25-k26.sh"

CKPT_ARGS=(
    --megatron-to-hf-mode bridge
    --hf-checkpoint "${KIMI_HF_CKPT}"
    --load "${LOAD_DIR:-${KIMI_HF_CKPT}}"
    --save "${SAVE_DIR}"
    --save-interval "${SAVE_INTERVAL:-20}"
)

ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_DATA}"
    --input-key "${INPUT_KEY}"
    --label-key "${LABEL_KEY}"
    --apply-chat-template
    --rollout-shuffle
    --rm-type "${RM_TYPE:-math}"
    --num-rollout "${NUM_ROLLOUT:-100}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-8}"
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-16384}"
    --rollout-temperature "${ROLLOUT_TEMPERATURE:-1}"
    --global-batch-size "${GLOBAL_BATCH_SIZE:-32}"
    --balance-data
)

EVAL_ARGS=()
if [ -n "${EVAL_PROMPT_DATA:-}" ]; then
    EVAL_ARGS=(
        --eval-interval "${EVAL_INTERVAL:-100}"
        --skip-eval-before-train
        --eval-prompt-data "${EVAL_NAME:-aime}" "${EVAL_PROMPT_DATA}"
        --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT:-4}"
        --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN:-8192}"
        --eval-top-p "${EVAL_TOP_P:-1}"
    )
fi

PERF_ARGS=(
    --tensor-model-parallel-size "${TP:-8}"
    --sequence-parallel
    --pipeline-model-parallel-size "${PP:-8}"
    --context-parallel-size "${CP:-4}"
    --expert-model-parallel-size "${EP:-32}"
    --expert-tensor-parallel-size "${ETP:-1}"
    --decoder-last-pipeline-num-layers "${DECODER_LAST_PIPELINE_NUM_LAYERS:-5}"
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers "${RECOMPUTE_NUM_LAYERS:-1}"
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-16384}"
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --kl-loss-coef "${KL_LOSS_COEF:-0.00}"
    --kl-loss-type low_var_kl
    --entropy-coef "${ENTROPY_COEF:-0.00}"
    --eps-clip "${EPS_CLIP:-0.2}"
    --eps-clip-high "${EPS_CLIP_HIGH:-0.28}"
    --use-tis
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR:-1e-6}"
    --lr-decay-style constant
    --weight-decay "${WEIGHT_DECAY:-0.1}"
    --adam-beta1 "${ADAM_BETA1:-0.9}"
    --adam-beta2 "${ADAM_BETA2:-0.98}"
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
)

WANDB_ARGS=()
if [ "${USE_TENSORBOARD:-1}" = "1" ]; then
    WANDB_ARGS+=(--use-tensorboard)
    WANDB_ARGS+=(--tb-experiment-name "${TB_EXPERIMENT_NAME:-slime-kimi-k26-v030}")
    WANDB_ARGS+=(--tb-project-name "${TB_PROJECT_NAME:-tensorboard-log}")
fi
if [ "${USE_WANDB:-0}" = "1" ]; then
    WANDB_ARGS+=(--use-wandb)
    WANDB_ARGS+=(--wandb-project "${WANDB_PROJECT:-slime-kimi}")
    WANDB_ARGS+=(--wandb-group "${WANDB_GROUP:-kimi-k26-v030}")
    if [ -n "${WANDB_KEY:-}" ]; then
        WANDB_ARGS+=(--wandb-key "${WANDB_KEY}")
    fi
fi

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE:-16}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.4}"
    --sglang-enable-dp-attention
    --sglang-dp-size "${SGLANG_DP_SIZE:-8}"
    --sglang-moe-dense-tp-size "${SGLANG_MOE_DENSE_TP_SIZE:-1}"
    --sglang-enable-dp-lm-head
    --sglang-ep-size "${SGLANG_EP_SIZE:-16}"
    --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY:-256}"
    --sglang-watchdog-timeout "${SGLANG_WATCHDOG_TIMEOUT:-7200}"
    --sglang-dist-timeout "${SGLANG_DIST_TIMEOUT:-7200}"
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

if [ "${FREEZE_VISION:-1}" = "1" ]; then
    MISC_ARGS+=(--freeze-params-name-list "vision_tower" "mm_projector")
fi

if [ -n "${MULTIMODAL_KEYS:-}" ]; then
    MISC_ARGS+=(--multimodal-keys "${MULTIMODAL_KEYS}")
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"${no_proxy}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"WANDB_MODE\": \"${WANDB_MODE}\",
    \"FLASHINFER_DISABLE_VERSION_CHECK\": \"${FLASHINFER_DISABLE_VERSION_CHECK}\",
    \"NCCL_TIMEOUT_MS\": \"${NCCL_TIMEOUT_MS:-360000000}\",
    \"NVSHMEM_DISABLE_NCCL\": \"${NVSHMEM_DISABLE_NCCL:-1}\",
    \"OPEN_TRAINING_INT4_FAKE_QAT_FLAG\": \"1\",
    \"OPEN_TRAINING_INT4_GROUP_SIZE\": \"32\",
    \"TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC\": \"${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}\",
    \"NCCL_ASYNC_ERROR_HANDLING\": \"1\"
  }
}"

submit_job() {
    ray job submit --address="http://127.0.0.1:${DASHBOARD_PORT}" \
        --runtime-env-json="${RUNTIME_ENV_JSON}" \
        -- python3 train.py \
        --actor-num-nodes "${ACTOR_NUM_NODES}" \
        --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
        --colocate \
        --update-weight-buffer-size "${UPDATE_WEIGHT_BUFFER_SIZE:-2147483648}" \
        "${MODEL_ARGS[@]}" \
        "${CKPT_ARGS[@]}" \
        "${ROLLOUT_ARGS[@]}" \
        "${OPTIMIZER_ARGS[@]}" \
        "${GRPO_ARGS[@]}" \
        "${WANDB_ARGS[@]}" \
        "${PERF_ARGS[@]}" \
        "${EVAL_ARGS[@]}" \
        "${SGLANG_ARGS[@]}" \
        "${MISC_ARGS[@]}"
}

if [ "${USE_EXTERNAL_RAY}" = "1" ]; then
    submit_job
elif [ "${NODE_RANK}" = "0" ]; then
    ray start --head --node-ip-address "${MASTER_ADDR}" --port "${MASTER_PORT}" \
        --num-gpus "${GPUS_PER_NODE}" --disable-usage-stats \
        --dashboard-host=0.0.0.0 --dashboard-port="${DASHBOARD_PORT}"

    python3 - <<PY
import sys
import time

import ray

expected = int("${NNODES}")
deadline = time.time() + int("${RAY_WAIT_TIMEOUT:-600}")
ray.init(address="auto", logging_level="ERROR")
while time.time() < deadline:
    alive = sum(1 for node in ray.nodes() if node.get("Alive"))
    print(f"Ray nodes alive: {alive}/{expected}", flush=True)
    if alive >= expected:
        sys.exit(0)
    time.sleep(5)
print("Ray node wait timed out; submitting anyway.", flush=True)
PY
    submit_job
else
    sleep 5
    ray start --address="${MASTER_ADDR}:${MASTER_PORT}" \
        --num-gpus "${GPUS_PER_NODE}" --disable-usage-stats --block
fi
