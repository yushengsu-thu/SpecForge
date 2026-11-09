"""Compute prepared query/key/value tensors without owning learned projections."""

from .eager import eager_attention
from .sdpa import sdpa_attention


def attention_backend(name):
    if name == "eager":
        return eager_attention
    if name == "sdpa":
        return sdpa_attention
    if name == "flex_attention":
        from .flex import flex_attention

        return flex_attention
    raise ValueError(f"unsupported attention backend: {name!r}")


def execute_attention(function, query, key, value, *, attention_mask, **options):
    """Return activations and active query rows, including fully masked sparse rows."""
    from torch import Tensor

    if not isinstance(attention_mask, Tensor):
        return function(
            query,
            key,
            value,
            attention_mask=attention_mask,
            return_valid=True,
            **options,
        )
    output = function(query, key, value, attention_mask=attention_mask, **options)
    return output, attention_mask.any(dim=-1).any(dim=1)


__all__ = ["attention_backend", "eager_attention", "sdpa_attention"]
