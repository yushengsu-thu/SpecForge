"""Explicit visibility narrowed by attention-node semantics for each backend."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from specforge.config.draft._validation import boolean, positive_int


@dataclass(frozen=True, eq=False, kw_only=True)
class AttentionLayout:
    """Logical positions [batch, Q/K] and allowed pairs [batch, heads, Q, K].

    Visibility defines padding, block isolation, and context access; it must be
    explicit even for bidirectional attention. Its batch/head axes may broadcast.
    Logical positions determine causality/window distance, independently of RoPE
    coordinates (which may have multiple axes). Do not mutate these tensors while
    their autograd graph is live. Visibility is a boolean tensor or a head-independent
    callable (batch, head, query_index, key_index) -> bool. Flex keeps callable
    visibility sparse; dense backends materialize it on demand.
    """

    query_positions: Tensor
    key_positions: Tensor
    visibility: Tensor | Callable

    def __post_init__(self):
        q, k, allowed = self.query_positions, self.key_positions, self.visibility
        if any(
            p.ndim != 2 or p.dtype not in (torch.int32, torch.int64) for p in (q, k)
        ):
            raise ValueError("logical positions must be rank-two integer tensors")
        if q.device != k.device:
            raise ValueError("query/key positions must share a device")
        if callable(allowed):
            batches = {q.shape[0], k.shape[0]} - {1}
        else:
            if (
                allowed.ndim != 4
                or allowed.dtype != torch.bool
                or allowed.device != q.device
                or allowed.shape[1] < 1
                or allowed.shape[2:] != (q.shape[1], k.shape[1])
            ):
                raise ValueError(
                    "visibility must be boolean [batch, heads, Q, K] on the positions device"
                )
            batches = {q.shape[0], k.shape[0], allowed.shape[0]} - {1}
        if len(batches) > 1:
            raise ValueError("positions and visibility must have broadcastable batches")


def dense_attention_mask(
    layout: AttentionLayout, *, is_causal: bool, sliding_window: int | None
) -> Tensor:
    """Intersect visibility with causal/window constraints; never expand access.

    A window W permits distances strictly smaller than W in either direction.
    Causality additionally removes future keys. Thus W=1 permits equal positions
    only, and still respects the model's original isolation/padding constraints.
    """
    if not isinstance(layout, AttentionLayout):
        raise TypeError("dense attention requires AttentionLayout")
    boolean("is_causal", is_causal)
    if sliding_window is not None:
        positive_int("sliding_window", sliding_window)
    q, k = layout.query_positions, layout.key_positions
    allowed = layout.visibility
    if callable(allowed):
        batch = torch.arange(max(q.shape[0], k.shape[0]), device=q.device)[
            :, None, None, None
        ]
        queries = torch.arange(q.shape[1], device=q.device)[None, None, :, None]
        keys = torch.arange(k.shape[1], device=k.device)[None, None, None, :]
        allowed = allowed(batch, torch.zeros_like(batch), queries, keys)
    q = layout.query_positions.unsqueeze(-1)
    k = layout.key_positions.unsqueeze(-2)
    if is_causal:
        # A boolean comparison avoids an unnecessary int64 Q-by-K allocation.
        allowed = allowed & (q >= k).unsqueeze(1)
    if sliding_window is not None:
        allowed = allowed & ((q - k).abs() < sliding_window).unsqueeze(1)
    return allowed


def prepare_attention_mask(layout, *, is_causal, sliding_window, backend):
    """Keep visibility symbolic for FlexAttention; materialize only for dense kernels."""
    if backend != "flex_attention":
        return dense_attention_mask(
            layout, is_causal=is_causal, sliding_window=sliding_window
        )
    if not isinstance(layout, AttentionLayout):
        raise TypeError("attention requires AttentionLayout")
    boolean("is_causal", is_causal)
    if sliding_window is not None:
        positive_int("sliding_window", sliding_window)
    from torch.nn.attention.flex_attention import create_block_mask

    from specforge.modeling.draft.layers.attention.backends.policy import (
        flex_attention_backend,
    )

    q, k = layout.query_positions, layout.key_positions
    visibility = layout.visibility

    def allowed(b, h, qi, ki):
        within = (qi < q.shape[1]) & (ki < k.shape[1])
        qi, ki = qi.clamp(max=q.shape[1] - 1), ki.clamp(max=k.shape[1] - 1)
        if callable(visibility):
            result = visibility(b, h, qi, ki)
        else:
            result = visibility[
                b % visibility.shape[0], h % visibility.shape[1], qi, ki
            ]
        qp, kp = q[b % q.shape[0], qi], k[b % k.shape[0], ki]
        if is_causal:
            result = result & (qp >= kp)
        if sliding_window is not None:
            result = result & ((qp - kp).abs() < sliding_window)
        return within & result

    return create_block_mask(
        allowed,
        B=max(
            q.shape[0],
            k.shape[0],
            1 if callable(visibility) else visibility.shape[0],
        ),
        H=(
            None
            if callable(visibility) or visibility.shape[1] == 1
            else visibility.shape[1]
        ),
        Q_LEN=q.shape[1],
        KV_LEN=k.shape[1],
        device=q.device,
        BLOCK_SIZE=(256, 128) if flex_attention_backend() == "FLASH" else 128,
    )
