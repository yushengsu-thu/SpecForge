"""Draft FFN selection; dense construction keeps the caller's kernel factory."""

from .config import MoEConfig, resolve_moe_config
from .layer import MoELayer
from .runtime import apply_pending_balance_updates, finalize_step_output


def build_ffn(config, *, dense):
    if getattr(config, "moe_preset", None) is None:
        if getattr(config, "n_routed_experts", None) is not None:
            raise ValueError("n_routed_experts requires an explicit moe_preset")
        return dense(config)
    return MoELayer(resolve_moe_config(config), config.hidden_size)


__all__ = [
    "MoEConfig",
    "MoELayer",
    "resolve_moe_config",
    "build_ffn",
    "apply_pending_balance_updates",
    "finalize_step_output",
]
