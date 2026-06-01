import copy
from contextlib import contextmanager

try:
    from megatron.core.utils import unwrap_model
except ImportError:
    unwrap_model = None


def _collect_related_hf_configs(hf_config):
    configs = []
    seen_config_ids = set()

    def add_config(config):
        if config is None or id(config) in seen_config_ids:
            return
        seen_config_ids.add(id(config))
        configs.append(config)

    add_config(hf_config)
    add_config(getattr(hf_config, "config", None))

    for config in list(configs):
        add_config(getattr(config, "text_config", None))

    return configs


def patch_hf_config_for_megatron_bridge(hf_config):
    configs = _collect_related_hf_configs(hf_config)

    for config in configs:
        rope_params = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None)
        if isinstance(rope_params, dict) and "rope_theta" in rope_params and not hasattr(config, "rope_theta"):
            config.rope_theta = rope_params["rope_theta"]

    return hf_config


def patch_auto_bridge_hf_config(bridge):
    hf_pretrained = getattr(bridge, "hf_pretrained", None)
    if hf_pretrained is not None:
        patch_hf_config_for_megatron_bridge(hf_pretrained)

    return bridge


def get_hf_quantization_config(hf_config):
    for config in _collect_related_hf_configs(hf_config):
        quantization_config = getattr(config, "quantization_config", None)
        if quantization_config is not None:
            return quantization_config
    return None


def annotate_quantization_config_with_hf_packed_keys(quantization_config, hf_keys):
    if not isinstance(quantization_config, dict):
        return quantization_config
    if quantization_config.get("quant_method") != "compressed-tensors":
        return quantization_config

    quantized_weight_names = sorted(key[: -len("_packed")] for key in hf_keys if key.endswith(".weight_packed"))
    if not quantized_weight_names:
        return quantization_config

    annotated = copy.deepcopy(quantization_config)
    annotated["_slime_quantized_weight_names"] = quantized_weight_names
    return annotated


def annotate_quantization_config_from_bridge(quantization_config, bridge):
    hf_pretrained = getattr(bridge, "hf_pretrained", None)
    hf_state = getattr(hf_pretrained, "state", None)
    hf_source = getattr(hf_state, "source", None)
    if hf_source is None or not hasattr(hf_source, "get_all_keys"):
        return quantization_config

    try:
        hf_keys = hf_source.get_all_keys()
    except Exception:
        return quantization_config

    return annotate_quantization_config_with_hf_packed_keys(quantization_config, hf_keys)


@contextmanager
def patch_megatron_model(model):
    unwrapped_model = unwrap_model(model)[0]
    model_config = unwrapped_model.config
    attribute_was_added = False
    if not hasattr(model_config, "share_embeddings_and_output_weights"):
        model_config.share_embeddings_and_output_weights = unwrapped_model.share_embeddings_and_output_weights
        attribute_was_added = True

    try:
        yield
    finally:
        if attribute_was_added:
            delattr(model_config, "share_embeddings_and_output_weights")
