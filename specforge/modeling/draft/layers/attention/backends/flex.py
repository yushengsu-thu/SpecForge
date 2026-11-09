"""FlexAttention over explicit visibility, without owning learned projections."""

import torch
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
from torch.nn.attention.flex_attention import flex_attention as torch_flex_attention

from specforge.modeling.draft.layers.attention.backends.policy import (
    flex_attention_backend,
)

_attention = torch.compile(torch_flex_attention, dynamic=True)


def flex_attention(
    query, key, value, *, attention_mask, scale, dropout_p, training, return_valid=False
):
    if training and dropout_p:
        raise ValueError("FlexAttention does not support attention dropout")
    batch, heads, queries, _ = query.shape
    keys = key.shape[-2]
    value_width = value.shape[-1]

    def for_width(width):
        return max(16, 1 << (width - 1).bit_length())

    query = F.pad(query, (0, for_width(query.shape[-1]) - query.shape[-1]))
    key = F.pad(key, (0, for_width(key.shape[-1]) - key.shape[-1]))
    value = F.pad(value, (0, for_width(value_width) - value_width))

    def visible(b, h, q, k):
        return (
            (q < queries)
            & (k < keys)
            & attention_mask[
                b % attention_mask.shape[0],
                h % attention_mask.shape[1],
                q.clamp(max=queries - 1),
                k.clamp(max=keys - 1),
            ]
        )

    backend = flex_attention_backend() or "TRITON"
    block_mask = (
        attention_mask
        if isinstance(attention_mask, BlockMask)
        else create_block_mask(
            visible,
            B=batch,
            H=None if attention_mask.shape[1] == 1 else heads,
            Q_LEN=queries,
            KV_LEN=keys,
            device=query.device,
            BLOCK_SIZE=(256, 128) if backend == "FLASH" else 128,
        )
    )
    options = (
        {"BACKEND": backend}
        if tuple(map(int, torch.__version__.split(".")[:2])) >= (2, 11)
        else {"FORCE_USE_FLEX_ATTENTION": True}
    )
    result = _attention(
        query,
        key,
        value,
        block_mask=block_mask,
        scale=scale,
        enable_gqa=query.shape[1] != key.shape[1],
        kernel_options=options,
        return_lse=return_valid,
    )
    if return_valid:
        output, lse = result
        return output[..., :value_width], torch.isfinite(lse).any(dim=1)
    return result[..., :value_width]
