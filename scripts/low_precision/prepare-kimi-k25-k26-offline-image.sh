#!/usr/bin/env bash
set -euo pipefail

# Run this once inside the online slime v0.2.4 container, then save the
# resulting container image with your platform UI for the offline training
# machine. This script can live on GPFS; it copies the patched files from the
# checkout containing this script into /root/slime.
#
# Required:
#   KIMI_HF_CKPT=/models/Kimi-K2.6        # final path that will exist offline
#
# Optional download mode:
#   DOWNLOAD_MODEL=1
#   KIMI_MODEL_ID=moonshotai/Kimi-K2.6
#   HF_TOKEN=...

: "${KIMI_HF_CKPT:?Set KIMI_HF_CKPT to the final local HF checkpoint path}"
if [[ "${KIMI_HF_CKPT}" == KIMI_HF_CKPT=* ]]; then
    KIMI_HF_CKPT="${KIMI_HF_CKPT#KIMI_HF_CKPT=}"
    export KIMI_HF_CKPT
fi

SLIME_DIR="${SLIME_DIR:-/root/slime}"
BRIDGE_SPEC="${BRIDGE_SPEC:-git+https://github.com/innoseon/Megatron-Bridge.git@v0.4.0-slime-kimi}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PATCHED_SLIME_SRC="${PATCHED_SLIME_SRC:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

set -x
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${KIMI_HF_CKPT}"

python -m pip install --force-reinstall --no-deps --no-build-isolation "${BRIDGE_SPEC}"

if [[ "${PATCHED_SLIME_SRC}" != "${SLIME_DIR}" ]]; then
    install -D "${PATCHED_SLIME_SRC}/slime/utils/megatron_bridge_utils.py" \
        "${SLIME_DIR}/slime/utils/megatron_bridge_utils.py"
    install -D "${PATCHED_SLIME_SRC}/slime/backends/megatron_utils/actor.py" \
        "${SLIME_DIR}/slime/backends/megatron_utils/actor.py"
    install -D "${PATCHED_SLIME_SRC}/slime/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py" \
        "${SLIME_DIR}/slime/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py"
    install -D "${PATCHED_SLIME_SRC}/slime/backends/megatron_utils/megatron_to_hf/processors/quantizer_compressed_tensors.py" \
        "${SLIME_DIR}/slime/backends/megatron_utils/megatron_to_hf/processors/quantizer_compressed_tensors.py"
    install -D "${PATCHED_SLIME_SRC}/slime_plugins/megatron_bridge/__init__.py" \
        "${SLIME_DIR}/slime_plugins/megatron_bridge/__init__.py"
    install -D "${PATCHED_SLIME_SRC}/slime_plugins/megatron_bridge/kimi_k25_vl.py" \
        "${SLIME_DIR}/slime_plugins/megatron_bridge/kimi_k25_vl.py"
    install -D "${PATCHED_SLIME_SRC}/scripts/low_precision/run-kimi-k25-k26-bridge.sh" \
        "${SLIME_DIR}/scripts/low_precision/run-kimi-k25-k26-bridge.sh"
    install -D "${PATCHED_SLIME_SRC}/scripts/low_precision/prepare-kimi-k25-k26-offline-image.sh" \
        "${SLIME_DIR}/scripts/low_precision/prepare-kimi-k25-k26-offline-image.sh"
    chmod +x "${SLIME_DIR}/scripts/low_precision/run-kimi-k25-k26-bridge.sh"
    chmod +x "${SLIME_DIR}/scripts/low_precision/prepare-kimi-k25-k26-offline-image.sh"
fi

cd "${SLIME_DIR}"
python -m pip install --no-deps --no-build-isolation -e "${SLIME_DIR}"

if [[ "${DOWNLOAD_MODEL:-0}" == "1" ]]; then
    : "${KIMI_MODEL_ID:?Set KIMI_MODEL_ID when DOWNLOAD_MODEL=1}"
    if command -v hf >/dev/null 2>&1; then
        hf download "${KIMI_MODEL_ID}" --local-dir "${KIMI_HF_CKPT}" ${HF_TOKEN:+--token "${HF_TOKEN}"}
    elif command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download "${KIMI_MODEL_ID}" --local-dir "${KIMI_HF_CKPT}" ${HF_TOKEN:+--token "${HF_TOKEN}"}
    else
        python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["KIMI_MODEL_ID"],
    local_dir=os.environ["KIMI_HF_CKPT"],
    token=os.environ.get("HF_TOKEN"),
    local_dir_use_symlinks=False,
)
PY
    fi
fi

python - <<'PY'
import os
from pathlib import Path

ckpt = Path(os.environ["KIMI_HF_CKPT"])
required = ["config.json"]
missing = [name for name in required if not (ckpt / name).exists()]
if missing:
    raise SystemExit(f"Missing files in {ckpt}: {missing}")

has_weights = any(ckpt.glob("*.safetensors")) or any(ckpt.glob("*.bin")) or (ckpt / "model.safetensors.index.json").exists()
if not has_weights:
    raise SystemExit(f"No HF weight files or safetensors index found in {ckpt}")

from transformers import AutoConfig, AutoProcessor, AutoTokenizer

config = AutoConfig.from_pretrained(ckpt, trust_remote_code=True, local_files_only=True)
print("HF config:", type(config), getattr(config, "model_type", None), getattr(config, "architectures", None))

tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True, local_files_only=True)
print("tokenizer:", type(tokenizer), getattr(tokenizer, "vocab_size", None))

try:
    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True, local_files_only=True)
    print("processor:", type(processor))
except Exception as exc:
    print("processor warmup skipped:", repr(exc))

from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion.model_bridge import HFWeightTuple
from megatron.bridge.models.kimi_vl.kimi_k25_vl_bridge import KimiK25VLBridge
from slime.backends.megatron_utils.update_weight.hf_weight_iterator_bridge import HfWeightIteratorBridge
from slime.utils import megatron_bridge_utils
import slime_plugins.megatron_bridge  # noqa: F401

assert "megatron_param_name" in HFWeightTuple._fields, HFWeightTuple._fields
print("HFWeightTuple fields:", HFWeightTuple._fields)
print("Kimi bridge:", KimiK25VLBridge)
print("slime bridge iterator:", HfWeightIteratorBridge)

bridge = megatron_bridge_utils.patch_auto_bridge_hf_config(
    AutoBridge.from_hf_pretrained(ckpt, trust_remote_code=True, local_files_only=True)
)
quantization_config = megatron_bridge_utils.get_hf_quantization_config(config)
annotated = megatron_bridge_utils.annotate_quantization_config_from_bridge(quantization_config, bridge)
print("quant method:", None if not annotated else annotated.get("quant_method"))
print("native packed weights:", 0 if not annotated else len(annotated.get("_slime_quantized_weight_names", [])))

provider = bridge.to_megatron_provider(load_weights=False)
print("provider:", type(provider))
PY

TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python - <<'PY'
import os
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer

ckpt = Path(os.environ["KIMI_HF_CKPT"])
AutoConfig.from_pretrained(ckpt, trust_remote_code=True, local_files_only=True)
AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True, local_files_only=True)
print("offline HF cache smoke ok")
PY

python -m py_compile \
    slime/utils/megatron_bridge_utils.py \
    slime/backends/megatron_utils/actor.py \
    slime/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py \
    slime/backends/megatron_utils/megatron_to_hf/processors/quantizer_compressed_tensors.py \
    slime_plugins/megatron_bridge/kimi_k25_vl.py

echo "Online preparation finished. Save this container image in the platform UI for the offline machine."
