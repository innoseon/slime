#!/bin/bash

set -euo pipefail

SLIME_DIR="${SLIME_DIR:-/root/slime}"
KIMI_HF_CKPT="${KIMI_HF_CKPT:-/inspire/hdd/project/ai4education/public/models/Kimi-K2.6}"
KIMI_MCORE_CKPT="${KIMI_MCORE_CKPT:-/inspire/qb-ilm/project/ai4education/p-liuwentao/ly-project/ckpt/Kimi-K2.6_torch_dist_bridge}"

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

NNODES="${PET_NNODES:-${NNODES:-1}}"
NODE_RANK="${PET_NODE_RANK:-${NODE_RANK:-0}}"
GPUS_PER_NODE="${PET_NPROC_PER_NODE:-${GPUS_PER_NODE:-8}}"
MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-29500}}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="/root/Megatron-LM/${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}}"

cd "${SLIME_DIR}"

source "${SLIME_DIR}/scripts/models/kimi-k25-k26.sh"

PERF_ARGS=(
    --tensor-model-parallel-size "${TP:-8}"
    --sequence-parallel
    --pipeline-model-parallel-size "${PP:-8}"
    --context-parallel-size "${CP:-2}"
    --expert-model-parallel-size "${EP:-16}"
    --expert-tensor-parallel-size "${ETP:-1}"
    --decoder-last-pipeline-num-layers "${DECODER_LAST_PIPELINE_NUM_LAYERS:-5}"
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

mkdir -p "$(dirname "${KIMI_MCORE_CKPT}")"

python -m torch.distributed.run \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --nproc_per_node "${GPUS_PER_NODE}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    tools/convert_hf_to_torch_dist_bridge.py \
    --megatron-to-hf-mode bridge \
    --hf-checkpoint "${KIMI_HF_CKPT}" \
    --save "${KIMI_MCORE_CKPT}" \
    "${MODEL_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${MISC_ARGS[@]}"
