#!/bin/bash

set -euo pipefail
set -x

PATCHED_SLIME_SRC="${PATCHED_SLIME_SRC:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
SLIME_DIR="${SLIME_DIR:-/root/slime}"
BRIDGE_DIR="${BRIDGE_DIR:-/root/Megatron-Bridge-radixark}"
MEGATRON_BRIDGE_REPO="${MEGATRON_BRIDGE_REPO:-https://github.com/radixark/Megatron-Bridge.git}"
MEGATRON_BRIDGE_REF="${MEGATRON_BRIDGE_REF:-6fde1c8538ea4ad966c7fba5f759be54f943b598}"

if [ -z "${KIMI_HF_CKPT:-}" ]; then
    echo "KIMI_HF_CKPT must point to the local Kimi-K2.5/K2.6 HuggingFace checkpoint." >&2
    exit 1
fi

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

if [ ! -d "${SLIME_DIR}" ]; then
    echo "SLIME_DIR=${SLIME_DIR} does not exist. Run this inside the slime v0.3.0 Docker image." >&2
    exit 1
fi

install -D "${PATCHED_SLIME_SRC}/slime/utils/megatron_bridge_utils.py" \
    "${SLIME_DIR}/slime/utils/megatron_bridge_utils.py"
install -D "${PATCHED_SLIME_SRC}/slime/backends/megatron_utils/model_provider.py" \
    "${SLIME_DIR}/slime/backends/megatron_utils/model_provider.py"
install -D "${PATCHED_SLIME_SRC}/slime/backends/megatron_utils/actor.py" \
    "${SLIME_DIR}/slime/backends/megatron_utils/actor.py"
install -D "${PATCHED_SLIME_SRC}/slime/backends/megatron_utils/hf_checkpoint_saver.py" \
    "${SLIME_DIR}/slime/backends/megatron_utils/hf_checkpoint_saver.py"
install -D "${PATCHED_SLIME_SRC}/slime/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py" \
    "${SLIME_DIR}/slime/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py"
install -D "${PATCHED_SLIME_SRC}/slime/backends/megatron_utils/megatron_to_hf/processors/quantizer_compressed_tensors.py" \
    "${SLIME_DIR}/slime/backends/megatron_utils/megatron_to_hf/processors/quantizer_compressed_tensors.py"
install -D "${PATCHED_SLIME_SRC}/scripts/models/kimi-k25-k26.sh" \
    "${SLIME_DIR}/scripts/models/kimi-k25-k26.sh"
install -D "${PATCHED_SLIME_SRC}/scripts/low_precision/run-kimi-k25-k26-bridge.sh" \
    "${SLIME_DIR}/scripts/low_precision/run-kimi-k25-k26-bridge.sh"
install -D "${PATCHED_SLIME_SRC}/scripts/diagnostics/ray-network-nccl-test.sh" \
    "${SLIME_DIR}/scripts/diagnostics/ray-network-nccl-test.sh"

if [ ! -d "${BRIDGE_DIR}/.git" ]; then
    rm -rf "${BRIDGE_DIR}"
    git clone "${MEGATRON_BRIDGE_REPO}" "${BRIDGE_DIR}"
fi
git -C "${BRIDGE_DIR}" remote set-url origin "${MEGATRON_BRIDGE_REPO}"
git -C "${BRIDGE_DIR}" fetch origin bridge
git -C "${BRIDGE_DIR}" checkout --detach "${MEGATRON_BRIDGE_REF}"

python -m pip install --no-deps --no-build-isolation -e "${BRIDGE_DIR}"
python -m pip install --no-deps --no-build-isolation -e "${SLIME_DIR}"

python - "${KIMI_HF_CKPT}" <<'PY'
import sys
from pathlib import Path

ckpt = Path(sys.argv[1])

from transformers import AutoConfig, AutoProcessor, AutoTokenizer

config = AutoConfig.from_pretrained(ckpt, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)

print("HF config:", type(config), getattr(config, "model_type", None), getattr(config, "architectures", None))
print("tokenizer:", type(tokenizer), getattr(tokenizer, "vocab_size", None))
print("processor:", type(processor))

from slime.utils import megatron_bridge_utils

quantization_config = megatron_bridge_utils.get_hf_quantization_config(config)
print("quantization_config:", None if quantization_config is None else quantization_config.get("quant_method"))
assert quantization_config is not None, "Kimi compressed-tensors quantization_config was not found"
PY

if [[ "${RUN_BRIDGE_SMOKE:-1}" == "1" ]]; then
    python - "${KIMI_HF_CKPT}" <<'PY'
import sys
from pathlib import Path

ckpt = Path(sys.argv[1])

import megatron.bridge.package_info as pi
from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion.model_bridge import HFWeightTuple

print("bridge version:", pi.__version__)
print("bridge file:", pi.__file__)
print("HFWeightTuple fields:", HFWeightTuple._fields)
assert "megatron_param_name" in HFWeightTuple._fields

import slime_plugins.megatron_bridge  # noqa: F401

print("slime_plugins.megatron_bridge import ok")

bridge = AutoBridge.from_hf_pretrained(ckpt, trust_remote_code=True)
provider = bridge.to_megatron_provider(load_weights=False)
print("provider:", type(provider))
print("provider.seq_length:", getattr(provider, "seq_length", None))
print("provider.freeze_vision_model:", getattr(provider, "freeze_vision_model", None))
PY
fi

python -m compileall -q \
    "${SLIME_DIR}/slime/utils/megatron_bridge_utils.py" \
    "${SLIME_DIR}/slime/backends/megatron_utils/model_provider.py" \
    "${SLIME_DIR}/slime/backends/megatron_utils/actor.py" \
    "${SLIME_DIR}/slime/backends/megatron_utils/hf_checkpoint_saver.py" \
    "${SLIME_DIR}/slime/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py" \
    "${SLIME_DIR}/slime/backends/megatron_utils/megatron_to_hf/processors/quantizer_compressed_tensors.py"

echo "Kimi K2.5/K2.6 v0.3.0 offline image preparation finished."
