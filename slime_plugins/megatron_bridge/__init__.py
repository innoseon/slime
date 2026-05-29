import slime_plugins.megatron_bridge.kimi_k25_vl  # noqa: F401  # register Kimi-K2.5/K2.6 bridge

try:
    import slime_plugins.megatron_bridge.glm4v_moe  # noqa: F401  # register GLM-4.6V bridge
except ImportError:
    pass
