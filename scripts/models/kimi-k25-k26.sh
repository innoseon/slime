#!/bin/bash

if [ -z "${KIMI_HF_CKPT:-}" ]; then
    echo "KIMI_HF_CKPT must point to the local Kimi-K2.5/K2.6 HuggingFace checkpoint." >&2
    exit 1
fi

if [[ "${KIMI_HF_CKPT}" == KIMI_HF_CKPT=* ]]; then
    echo "KIMI_HF_CKPT looks double-assigned: ${KIMI_HF_CKPT}" >&2
    echo "Use: export KIMI_HF_CKPT=/path/to/Kimi-K2.6" >&2
    exit 1
fi

mapfile -t MODEL_ARGS < <(python3 - "${KIMI_HF_CKPT}" <<'PY'
import json
import os
import sys
from pathlib import Path

ckpt = Path(sys.argv[1])
with (ckpt / "config.json").open() as f:
    cfg = json.load(f)

text = cfg.get("text_config") or cfg


def get(name, default=None):
    return text.get(name, cfg.get(name, default))


def as_int(value, default):
    if value is None:
        value = default
    return int(float(value))


def as_float(value, default):
    if value is None:
        value = default
    return float(value)


num_layers = as_int(get("num_hidden_layers"), 61)
first_dense = as_int(get("first_k_dense_replace"), 1)
moe_layers = max(num_layers - first_dense, 0)
moe_layer_freq = f"[0]*{first_dense}+[1]*{moe_layers}"

rope = get("rope_scaling", {}) or get("rope_parameters", {}) or {}
rope_theta = get("rope_theta", rope.get("rope_theta", 50000))
rope_factor = rope.get("factor", get("rotary_scaling_factor", 64.0))

moe_intermediate = as_int(get("moe_intermediate_size"), 2048)
shared_experts = as_int(get("n_shared_experts"), 1)
seq_length = as_int(os.environ.get("KIMI_SEQ_LENGTH") or get("max_position_embeddings"), 262144)

args = [
    "--disable-bias-linear",
    "--num-layers", str(num_layers),
    "--hidden-size", str(as_int(get("hidden_size"), 7168)),
    "--ffn-hidden-size", str(as_int(get("intermediate_size"), 18432)),
    "--num-attention-heads", str(as_int(get("num_attention_heads"), 64)),
    "--kv-channels", str(as_int(get("kv_channels", get("qk_rope_head_dim")), 64)),
    "--normalization", "RMSNorm",
    "--position-embedding-type", "rope",
    "--norm-epsilon", str(get("rms_norm_eps", "1e-5")),
    "--swiglu",
    "--untie-embeddings-and-output-weights",
    "--vocab-size", str(as_int(get("vocab_size"), 163840)),
    "--seq-length", str(seq_length),
    "--multi-latent-attention",
    "--q-lora-rank", str(as_int(get("q_lora_rank"), 1536)),
    "--kv-lora-rank", str(as_int(get("kv_lora_rank"), 512)),
    "--qk-head-dim", str(as_int(get("qk_nope_head_dim"), 128)),
    "--qk-pos-emb-head-dim", str(as_int(get("qk_rope_head_dim"), 64)),
    "--v-head-dim", str(as_int(get("v_head_dim"), 128)),
    "--qk-layernorm",
    "--rotary-scaling-factor", str(as_float(rope_factor, 64.0)),
    "--rotary-base", str(as_int(rope_theta, 50000)),
    "--mscale", str(as_float(get("mscale"), 1.0)),
    "--mscale-all-dim", str(as_float(get("mscale_all_dim"), 1.0)),
    "--attention-softmax-in-fp32",
    "--no-rope-fusion",
    "--num-experts", str(as_int(get("n_routed_experts", get("num_experts")), 384)),
    "--moe-layer-freq", moe_layer_freq,
    "--moe-ffn-hidden-size", str(moe_intermediate),
    "--moe-router-topk", str(as_int(get("num_experts_per_tok", get("moe_router_topk")), 8)),
    "--moe-shared-expert-intermediate-size", str(moe_intermediate * shared_experts),
    "--moe-router-pre-softmax",
    "--moe-router-score-function", "sigmoid",
    "--moe-router-enable-expert-bias",
    "--moe-router-load-balancing-type", "seq_aux_loss",
    "--moe-token-dispatcher-type", "alltoall",
    "--moe-aux-loss-coeff", str(os.environ.get("KIMI_MOE_AUX_LOSS_COEFF", "0")),
    "--moe-router-bias-update-rate", str(os.environ.get("KIMI_MOE_ROUTER_BIAS_UPDATE_RATE", "0")),
    "--moe-router-group-topk", str(as_int(get("topk_group"), 1)),
    "--moe-router-num-groups", str(as_int(get("n_group"), 1)),
    "--moe-grouped-gemm",
    "--moe-router-topk-scaling-factor", str(as_float(get("routed_scaling_factor"), 2.827)),
    "--moe-router-dtype", "fp32",
    "--moe-permute-fusion",
]

for arg in args:
    print(arg)
PY
)
