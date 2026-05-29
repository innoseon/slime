#!/usr/bin/env bash
set -euo pipefail

# Offline-ready Kimi-K2.5/Kimi-K2.6 bridge-mode RL launch.
# The image must already contain:
#   - patched slime source from this branch
#   - Megatron-Bridge v0.4.0-slime-kimi
#   - the local HF checkpoint and warmed HuggingFace cache
#
# Required at runtime:
#   KIMI_HF_CKPT=/path/to/Kimi-K2.5-or-Kimi-K2.6-HF
#   SAVE_DIR=/path/to/slime-output
#   PROMPT_DATA=/path/to/train.jsonl

: "${KIMI_HF_CKPT:?Set KIMI_HF_CKPT=/path/to/Kimi-K2.5-or-Kimi-K2.6-HF}"
if [[ "${KIMI_HF_CKPT}" == KIMI_HF_CKPT=* ]]; then
    KIMI_HF_CKPT="${KIMI_HF_CKPT#KIMI_HF_CKPT=}"
    export KIMI_HF_CKPT
fi
: "${SAVE_DIR:?Set SAVE_DIR=/path/to/slime-output}"
: "${PROMPT_DATA:?Set PROMPT_DATA=/path/to/train.jsonl}"

set -x
export PYTHONBUFFERED="${PYTHONBUFFERED:-16}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [ "${NVLINK_COUNT}" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: ${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"

eval "$(
python3 - "${KIMI_HF_CKPT}" <<'PY'
import json
import shlex
import sys
from pathlib import Path

ckpt = Path(sys.argv[1])
cfg = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
text = cfg.get("text_config") or cfg
if cfg.get("model_type") != "kimi_k25" or text.get("model_type") != "kimi_k2":
    raise SystemExit(
        "Expected model_type='kimi_k25' and text_config.model_type='kimi_k2', "
        f"got {cfg.get('model_type')!r}/{text.get('model_type')!r}"
    )

num_layers = int(text["num_hidden_layers"])
first_dense = int(text.get("first_k_dense_replace", 0))
moe_layer_freq = "[" + ", ".join("0" if i < first_dense else "1" for i in range(num_layers)) + "]"
rope_scaling = text.get("rope_scaling") or {}
moe_ffn_hidden = int(text["moe_intermediate_size"])
n_shared = int(text.get("n_shared_experts", 0))

values = {
    "KIMI_NUM_LAYERS": num_layers,
    "KIMI_MOE_LAYER_FREQ": moe_layer_freq,
    "KIMI_HIDDEN_SIZE": text["hidden_size"],
    "KIMI_FFN_HIDDEN_SIZE": text["intermediate_size"],
    "KIMI_NUM_ATTENTION_HEADS": text["num_attention_heads"],
    "KIMI_KV_CHANNELS": text.get("kv_channels", text.get("qk_rope_head_dim", 64)),
    "KIMI_NORM_EPSILON": text["rms_norm_eps"],
    "KIMI_VOCAB_SIZE": text["vocab_size"],
    "KIMI_Q_LORA_RANK": text.get("q_lora_rank", 1536),
    "KIMI_KV_LORA_RANK": text["kv_lora_rank"],
    "KIMI_QK_HEAD_DIM": text.get("qk_nope_head_dim", text.get("qk_head_dim", 128)),
    "KIMI_QK_POS_EMB_HEAD_DIM": text.get("qk_rope_head_dim", 64),
    "KIMI_V_HEAD_DIM": text["v_head_dim"],
    "KIMI_ROTARY_SCALING_FACTOR": rope_scaling.get("factor", 1.0),
    "KIMI_ROTARY_BASE": text.get("rope_theta", 50000.0),
    "KIMI_MSCALE": rope_scaling.get("mscale", 1.0),
    "KIMI_MSCALE_ALL_DIM": rope_scaling.get("mscale_all_dim", 1.0),
    "KIMI_NUM_EXPERTS": text["n_routed_experts"],
    "KIMI_MOE_FFN_HIDDEN_SIZE": moe_ffn_hidden,
    "KIMI_MOE_ROUTER_TOPK": text["num_experts_per_tok"],
    "KIMI_MOE_SHARED_EXPERT_INTERMEDIATE_SIZE": moe_ffn_hidden * n_shared,
    "KIMI_MOE_ROUTER_GROUP_TOPK": text.get("topk_group", 1),
    "KIMI_MOE_ROUTER_NUM_GROUPS": text.get("n_group", 1),
    "KIMI_MOE_ROUTER_TOPK_SCALING_FACTOR": text.get("routed_scaling_factor", 2.827),
}

for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

MODEL_ARGS=(
    --disable-bias-linear
    --num-layers "${KIMI_NUM_LAYERS}"
    --hidden-size "${KIMI_HIDDEN_SIZE}"
    --ffn-hidden-size "${KIMI_FFN_HIDDEN_SIZE}"
    --num-attention-heads "${KIMI_NUM_ATTENTION_HEADS}"
    --kv-channels "${KIMI_KV_CHANNELS}"
    --normalization RMSNorm
    --position-embedding-type rope
    --norm-epsilon "${KIMI_NORM_EPSILON}"
    --swiglu
    --untie-embeddings-and-output-weights
    --vocab-size "${KIMI_VOCAB_SIZE}"
    --multi-latent-attention
    --q-lora-rank "${KIMI_Q_LORA_RANK}"
    --kv-lora-rank "${KIMI_KV_LORA_RANK}"
    --qk-head-dim "${KIMI_QK_HEAD_DIM}"
    --qk-pos-emb-head-dim "${KIMI_QK_POS_EMB_HEAD_DIM}"
    --v-head-dim "${KIMI_V_HEAD_DIM}"
    --qk-layernorm
    --rotary-scaling-factor "${KIMI_ROTARY_SCALING_FACTOR}"
    --rotary-base "${KIMI_ROTARY_BASE}"
    --mscale "${KIMI_MSCALE}"
    --mscale-all-dim "${KIMI_MSCALE_ALL_DIM}"
    --attention-softmax-in-fp32
    --no-rope-fusion
    --num-experts "${KIMI_NUM_EXPERTS}"
    --moe-layer-freq "${KIMI_MOE_LAYER_FREQ}"
    --moe-ffn-hidden-size "${KIMI_MOE_FFN_HIDDEN_SIZE}"
    --moe-router-topk "${KIMI_MOE_ROUTER_TOPK}"
    --moe-shared-expert-intermediate-size "${KIMI_MOE_SHARED_EXPERT_INTERMEDIATE_SIZE}"
    --moe-router-pre-softmax
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-load-balancing-type seq_aux_loss
    --moe-token-dispatcher-type alltoall
    --moe-aux-loss-coeff 0
    --moe-router-bias-update-rate 0
    --moe-router-group-topk "${KIMI_MOE_ROUTER_GROUP_TOPK}"
    --moe-router-num-groups "${KIMI_MOE_ROUTER_NUM_GROUPS}"
    --moe-grouped-gemm
    --moe-router-topk-scaling-factor "${KIMI_MOE_ROUTER_TOPK_SCALING_FACTOR}"
    --moe-router-dtype fp32
    --moe-permute-fusion
)

CKPT_ARGS=(
   --megatron-to-hf-mode bridge
   --hf-checkpoint "${KIMI_HF_CKPT}"
   --load "${LOAD_DIR:-${KIMI_HF_CKPT}}"
   --save "${SAVE_DIR}"
   --save-interval "${SAVE_INTERVAL:-20}"
   --no-load-optim
   --no-load-rng
   --finetune
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key "${INPUT_KEY:-prompt}"
   --label-key "${LABEL_KEY:-label}"
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

EVAL_ARGS=(
   --eval-interval "${EVAL_INTERVAL:-100}"
   --skip-eval-before-train
   --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT:-4}"
   --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN:-8192}"
   --eval-top-p "${EVAL_TOP_P:-1}"
)
if [[ -n "${EVAL_PROMPT_DATA:-}" ]]; then
   EVAL_ARGS+=(--eval-prompt-data "${EVAL_NAME:-aime}" "${EVAL_PROMPT_DATA}")
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
if [[ "${USE_KL_LOSS:-0}" == "1" ]]; then
   : "${REF_LOAD:?Set REF_LOAD when USE_KL_LOSS=1}"
   CKPT_ARGS+=(--ref-load "${REF_LOAD}")
   GRPO_ARGS+=(--use-kl-loss)
fi

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

TENSORBOARD_ARGS=()
if [[ "${USE_TENSORBOARD:-1}" == "1" ]]; then
   TENSORBOARD_ARGS+=(--use-tensorboard --tb-experiment-name "${TB_EXPERIMENT_NAME:-kimi-k25-k26-bridge}" --tb-project-name "${TB_PROJECT_NAME:-tensorboard-log}")
fi

WANDB_ARGS=()
if [[ "${USE_WANDB:-0}" == "1" ]]; then
   WANDB_ARGS+=(--use-wandb --wandb-project "${WANDB_PROJECT:-slime-dev}" --wandb-group "${WANDB_GROUP:-kimi-k25-k26-bridge}")
   [[ -n "${WANDB_KEY:-}" ]] && WANDB_ARGS+=(--wandb-key "${WANDB_KEY}")
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

export NNODES="${NNODES:-${PET_NNODES:-1}}"
export NODE_RANK="${NODE_RANK:-${PET_NODE_RANK:-0}}"
export MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-127.0.0.1}}"
export MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-6379}}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-${PET_NPROC_PER_NODE:-8}}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-${NNODES}}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-${GPUS_PER_NODE}}"
export no_proxy="localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR},${no_proxy:-}"

RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-$((MASTER_PORT + 1))}"
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_PATH:-/root/Megatron-LM}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NCCL_TIMEOUT_MS\": \"360000000\",
    \"TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC\": \"7200\",
    \"NCCL_ASYNC_ERROR_HANDLING\": \"1\",
    \"TRANSFORMERS_OFFLINE\": \"${TRANSFORMERS_OFFLINE}\",
    \"HF_HUB_OFFLINE\": \"${HF_HUB_OFFLINE}\",
    \"no_proxy\": \"${no_proxy}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"WANDB_MODE\": \"${WANDB_MODE}\"
  }
}"

if [[ "${NODE_RANK}" == "0" ]]; then
    ray start --head --node-ip-address "${MASTER_ADDR}" --port "${MASTER_PORT}" \
        --num-gpus "${GPUS_PER_NODE}" --disable-usage-stats \
        --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}"

    python3 - <<PY
import sys
import time
import ray
expected = int("${NNODES}")
deadline = time.time() + int("${RAY_WAIT_NODES_TIMEOUT:-600}")
ray.init(address="auto", logging_level="ERROR")
while time.time() < deadline:
    alive = sum(1 for node in ray.nodes() if node.get("Alive"))
    print(f"Ray nodes alive: {alive}/{expected}", flush=True)
    if alive >= expected:
        sys.exit(0)
    time.sleep(5)
print("WARN: proceeding before all Ray nodes joined", flush=True)
PY

    ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
       --runtime-env-json="${RUNTIME_ENV_JSON}" \
       -- python3 "${SLIME_TRAIN_ENTRY:-train.py}" \
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
       "${TENSORBOARD_ARGS[@]}" \
       "${PERF_ARGS[@]}" \
       "${EVAL_ARGS[@]}" \
       "${SGLANG_ARGS[@]}" \
       "${MISC_ARGS[@]}" \
       "$@"
else
    sleep 5
    ray start --address="${MASTER_ADDR}:${MASTER_PORT}" \
        --num-gpus "${GPUS_PER_NODE}" --disable-usage-stats --block
fi
