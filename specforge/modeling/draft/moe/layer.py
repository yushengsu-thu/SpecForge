"""Qwen softmax/top-k routing with one sigmoid-gated shared expert."""

import torch
from torch import nn
from torch.nn import functional as F

from .checkpoint import normalize_qwen_expert_keys
from .config import MoEConfig
from .experts import PackedExperts, SharedExpert


class MoELayer(nn.Module):
    def __init__(self, config: MoEConfig, hidden_size: int):
        super().__init__()
        self.moe_config = config
        self.gate = nn.Linear(hidden_size, config.n_routed_experts, bias=False)
        self.experts = PackedExperts(config, hidden_size)
        self.shared_expert = SharedExpert(
            hidden_size, config.shared_expert_intermediate_size, config.swiglu_limit
        )
        self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)
        self._auxiliary_records = []
        self._capture_auxiliary = True
        self.register_load_state_dict_pre_hook(normalize_qwen_expert_keys)
        self.reset_parameters(0.02)

    @torch.no_grad()
    def reset_parameters(self, std: float) -> None:
        for parameter in self.parameters():
            nn.init.normal_(parameter, mean=0.0, std=std)

    def begin_forward(self) -> None:
        self._auxiliary_records = []
        self._capture_auxiliary = True

    def take_auxiliary_records(self):
        records = self._auxiliary_records
        self._auxiliary_records = []
        # Non-reentrant activation checkpoint recomputation occurs after the
        # finalizer. It must not publish a second graph or mutate balance state.
        self._capture_auxiliary = False
        return records

    def _route(self, x):
        """Family routing seam; selection and mixture weights stay distinct."""
        logits = self.gate(x)
        probabilities = F.softmax(logits, dim=-1, dtype=torch.float32)
        weights, indices = probabilities.topk(
            self.moe_config.num_experts_per_tok, dim=-1
        )
        if self.moe_config.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return probabilities, indices, weights.to(logits.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        probabilities, indices, weights = self._route(x)
        routed = self.experts(x, indices, weights)
        shared = self.shared_expert(x) * self.shared_expert_gate(x).sigmoid()
        counts = torch.bincount(
            indices.flatten(), minlength=self.moe_config.n_routed_experts
        ).float()
        token_count = max(x.shape[0], 1)
        fraction = counts / token_count
        # E * sum_e f_e * mean_token p_e, with f_e counting all top-k
        # assignments per token as in Qwen's load_balancing_loss_func.
        # Uniform routing has value top_k.
        # A sum/explicit count keeps the empty-token case finite.
        mean_probability = probabilities.sum(dim=0) / token_count
        auxiliary = (
            self.moe_config.n_routed_experts * (fraction * mean_probability).sum()
        )
        entropy = (
            -(probabilities * probabilities.clamp_min(1e-30).log()).sum() / token_count
        )
        active = (counts > 0).float().mean()
        if self._capture_auxiliary:
            self._auxiliary_records.append(
                (auxiliary, entropy.detach(), active.detach())
            )
        # The auxiliary graph is also reachable through the wrapped output.
        # Its backward must wait for that edge before using expert/router
        # weights, so FSDP's output pre-backward hooks can unshard them first.
        output = routed + shared + auxiliary.to(routed.dtype) * 0
        return output.reshape(shape)
