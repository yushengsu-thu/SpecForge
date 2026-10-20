"""FFN math, auxiliary gradients, and actual trainer normalization."""

import copy
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn
from torch.testing import assert_close
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3Config

from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.dflash2 import DFlash2DraftModel
from specforge.modeling.draft.domino import DominoDraftModel
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.modeling.draft.moe import (
    MoELayer,
    apply_pending_balance_updates,
    build_ffn,
    finalize_step_output,
    resolve_moe_config,
)
from specforge.modeling.draft.moe.runtime import moe_resume_contract
from specforge.runtime.contracts import TrainBatch
from specforge.training.controller import TrainerCore
from specforge.training.strategies.base import StepOutput


def moe_config(**overrides):
    values = dict(
        architectures=["DFlashDraftModel"],
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_hidden_layers=2,
        num_target_layers=4,
        layer_types=["full_attention", "full_attention"],
        block_size=3,
        vocab_size=32,
        moe_preset="qwen3_5_moe",
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        dflash_config={"target_layer_ids": [1], "moe_aux_loss_coeff": 0.03},
    )
    values.update(overrides)
    result = Qwen3Config(**values)
    result._attn_implementation = "eager"
    return result


def test_dense_factory_is_unchanged_and_moe_requires_opt_in():
    config = SimpleNamespace(hidden_size=8)
    dense = mock.Mock(return_value=nn.Identity())
    assert build_ffn(config, dense=dense) is dense.return_value
    dense.assert_called_once_with(config)
    config.n_routed_experts = 4
    with pytest.raises(ValueError, match="explicit moe_preset"):
        build_ffn(config, dense=dense)


@pytest.mark.parametrize(
    "field,value",
    [
        ("n_routed_experts", 0),
        ("num_experts_per_tok", 5),
        ("shared_expert_intermediate_size", 0),
        ("norm_topk_prob", "yes"),
        ("scoring_func", "sqrtsoftplus"),
        ("routed_scaling_factor", 2.5),
        ("swiglu_limit", float("nan")),
        ("moe_preset", "deepseek_v4"),
    ],
)
def test_invalid_or_unsupported_config_fails(field, value):
    with pytest.raises(ValueError):
        resolve_moe_config(moe_config(**{field: value}))


def test_execution_options_are_strict():
    for options in (
        {"moe_router_center": "sample"},
        {"moe_dispatch": "typo"},
        {"moe_aux_loss_coeff": -1},
        {"moe_aux_loss_coeff": float("nan")},
        {"moe_typo": 1},
    ):
        with pytest.raises(ValueError):
            resolve_moe_config(moe_config(dflash_config=options))


def test_resume_contract_persists_math_but_allows_execution_change():
    first = MoELayer(resolve_moe_config(moe_config()), 16)
    grouped_config = moe_config(
        dflash_config={"moe_dispatch": "grouped_mm", "moe_aux_loss_coeff": 0.03}
    )
    grouped = MoELayer(resolve_moe_config(grouped_config), 16)
    assert moe_resume_contract(first) == moe_resume_contract(grouped)
    changed = MoELayer(resolve_moe_config(moe_config(norm_topk_prob=False)), 16)
    assert moe_resume_contract(first) != moe_resume_contract(changed)
    assert moe_resume_contract(nn.Linear(16, 16)) == {}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_matches_official_qwen_forward_and_gradients(dtype):
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeTextConfig,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeSparseMoeBlock,
    )

    torch.manual_seed(17)
    ours = MoELayer(resolve_moe_config(moe_config()), 16).to(dtype)
    official_config = Qwen3_5MoeTextConfig(
        hidden_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
    )
    official = Qwen3_5MoeSparseMoeBlock(official_config).to(dtype)
    official.load_state_dict(ours.state_dict())
    x = torch.randn(2, 5, 16, dtype=dtype, requires_grad=True)
    other_x = x.detach().clone().requires_grad_()
    actual, expected = ours(x), official(other_x)
    tolerance = 1e-6 if dtype == torch.float32 else 2e-4
    assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    actual.float().square().sum().backward()
    expected.float().square().sum().backward()
    assert_close(x.grad, other_x.grad, atol=tolerance, rtol=tolerance)
    for (_, parameter), (_, reference) in zip(
        ours.named_parameters(), official.named_parameters()
    ):
        assert_close(parameter.grad, reference.grad, atol=tolerance, rtol=tolerance)


def test_reference_preserves_input_dtype_under_autocast():
    layer = MoELayer(resolve_moe_config(moe_config()), 16)
    x = torch.randn(2, 3, 16, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = layer(x)
    assert output.dtype == torch.float32
    output.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_auxiliary_matches_qwen_formula_and_reaches_router():
    torch.manual_seed(9)
    layer = MoELayer(resolve_moe_config(moe_config()), 16)
    x = torch.randn(2, 3, 16)
    output = layer(x)
    probabilities = layer.gate(x.flatten(0, 1)).float().softmax(-1)
    selected = probabilities.topk(2, dim=-1).indices
    frequencies = torch.nn.functional.one_hot(selected, 4).float().sum(1).mean(0)
    expected = 4 * (frequencies * probabilities.mean(0)).sum()
    result = finalize_step_output(layer, StepOutput(output.sum() * 0, {}))
    assert_close(result.loss, 0.03 * expected)
    result.loss.backward()
    assert layer.gate.weight.grad.norm() > 0
    assert layer.experts.gate_up_proj.grad.count_nonzero() == 0
    assert layer._auxiliary_records == []


def test_auxiliary_backward_waits_for_wrapped_output_hook():
    layer = MoELayer(resolve_moe_config(moe_config()), 16)
    output = layer(torch.randn(1, 3, 16, requires_grad=True))
    order = []
    output.register_hook(lambda gradient: order.append("output"))
    layer._auxiliary_records[0][0].register_hook(
        lambda gradient: order.append("auxiliary")
    )
    finalized = finalize_step_output(layer, StepOutput(output.square().mean(), {}))
    finalized.loss.backward()
    assert order == ["output", "auxiliary"]


def test_checkpoint_recomputation_preserves_aux_gradient_without_duplicates():
    torch.manual_seed(7)
    normal = MoELayer(resolve_moe_config(moe_config()), 16)
    checked = copy.deepcopy(normal)
    x = torch.randn(2, 5, 16, requires_grad=True)
    other_x = x.detach().clone().requires_grad_()
    for layer, value, use_checkpoint in ((normal, x, False), (checked, other_x, True)):
        apply_pending_balance_updates(layer)
        output = (
            checkpoint(layer, value, use_reentrant=False)
            if use_checkpoint
            else layer(value)
        )
        result = finalize_step_output(layer, StepOutput(output.square().mean(), {}))
        result.loss.backward()
        assert layer._auxiliary_records == []
    assert_close(x.grad, other_x.grad)
    for actual, expected in zip(checked.parameters(), normal.parameters()):
        assert_close(actual.grad, expected.grad)


def test_hf_decoder_checkpointing_and_initialization():
    torch.manual_seed(21)
    normal = DFlashDraftModel(moe_config())
    checked = copy.deepcopy(normal)
    checked.gradient_checkpointing_enable()
    assert all(layer.gradient_checkpointing for layer in checked.layers)
    with pytest.raises(ValueError, match="non-reentrant"):
        checked.gradient_checkpointing_enable({"use_reentrant": True})
    noise = torch.randn(1, 3, 16, requires_grad=True)
    target = torch.randn(1, 4, 16)
    position = torch.arange(7).unsqueeze(0)
    for model in (normal, checked):
        output = model(
            position_ids=position, noise_embedding=noise, target_hidden=target
        )
        finalized = finalize_step_output(model, StepOutput(output.square().mean(), {}))
        finalized.loss.backward()
    for actual, expected in zip(checked.parameters(), normal.parameters()):
        assert_close(actual.grad, expected.grad, atol=2e-6, rtol=2e-5)
    packed = checked.layers[0].mlp.experts.gate_up_proj
    assert torch.isfinite(packed).all() and packed.std() > 0


@pytest.mark.parametrize(
    "model_class,options",
    [
        (DFlashDraftModel, {}),
        (
            DFlash2DraftModel,
            {
                "conv_kernel_size": 2,
                "conv_group_size": 4,
                "selector_rank": 4,
                "selector_top_k": 3,
            },
        ),
        (DominoDraftModel, {"emb_dim": 8, "gru_hidden_dim": 8}),
        (DSparkDraftModel, {"markov_rank": 4}),
    ],
)
def test_dflash_family_backbones_share_the_ffn_and_auxiliary_boundary(
    model_class, options
):
    method = {"target_layer_ids": [1], "moe_aux_loss_coeff": 0.02, **options}
    model = model_class(
        moe_config(architectures=[model_class.__name__], dflash_config=method)
    )
    output = model(
        position_ids=torch.arange(7).unsqueeze(0),
        noise_embedding=torch.randn(1, 3, 16),
        target_hidden=torch.randn(1, 4, 16),
    )
    result = finalize_step_output(model, StepOutput(output.square().mean(), {}))
    result.loss.backward()
    for layer in model.layers:
        assert isinstance(layer.mlp, MoELayer)
        assert torch.isfinite(layer.mlp.gate.weight.grad).all()
        assert layer.mlp.gate.weight.grad.norm() > 0


def test_trainer_accumulation_weights_aux_with_full_objective_denominator():
    torch.manual_seed(30)
    model = MoELayer(resolve_moe_config(moe_config()), 16)
    reference = copy.deepcopy(model)
    inputs = [torch.randn(1, 3, 16), torch.randn(1, 5, 16)]
    denominators = [2.0, 7.0]

    class Strategy:
        def trainable_module(self):
            return model

        def forward_loss(self, batch, ctx):
            apply_pending_balance_updates(model)
            hidden = model(batch.tensors["x"])
            loss = hidden.square().mean()
            denominator = loss.new_tensor(batch.metadata["denominator"])
            return StepOutput(loss, {}, loss_terms=(loss * denominator, denominator))

    class Backend:
        parallel_config = None

        def backward(self, loss, *, is_boundary):
            loss.backward()

        def scale_gradients(self, scale):
            for parameter in model.parameters():
                parameter.grad.mul_(scale)

        def step(self):
            return 0.0

    trainer = TrainerCore(Strategy(), Backend(), accumulation_steps=2)
    expected_numerator = 0
    for value, denominator in zip(inputs, denominators):
        apply_pending_balance_updates(reference)
        output = reference(value)
        expected = finalize_step_output(
            reference, StepOutput(output.square().mean(), {})
        )
        expected_numerator = expected_numerator + expected.loss * denominator
        batch = TrainBatch(
            sample_ids=["sample"],
            strategy="dflash",
            tensors={"x": value},
            metadata={"denominator": denominator},
        )
        result = trainer.train_step(batch)
    (expected_numerator / sum(denominators)).backward()
    assert result.optimizer_stepped
    assert "moe/aux_loss" in result.metrics
    for actual, expected in zip(model.parameters(), reference.parameters()):
        assert_close(actual.grad, expected.grad, atol=1e-7, rtol=1e-5)


def test_empty_tokens_are_finite_and_unused_experts_receive_zero_gradient():
    layer = MoELayer(resolve_moe_config(moe_config()), 16)
    output = layer(torch.empty(1, 0, 16, requires_grad=True))
    finalized = finalize_step_output(layer, StepOutput(output.sum(), {}))
    assert torch.isfinite(finalized.loss)
    finalized.loss.backward()
    assert layer.experts.gate_up_proj.grad.count_nonzero() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA grouped_mm")
def test_grouped_mm_matches_reference_with_empty_and_odd_expert_groups():
    config = moe_config(dflash_config={"moe_dispatch": "grouped_mm"})
    grouped = MoELayer(resolve_moe_config(config), 16).cuda().bfloat16()
    reference = MoELayer(resolve_moe_config(moe_config()), 16).cuda().bfloat16()
    reference.load_state_dict(grouped.state_dict())
    # Five assignments including a zero-sized group, with odd cumulative ends.
    indices = torch.tensor([[0], [1], [1], [3], [3]], device="cuda")
    weights = torch.rand(5, 1, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    other_weights = weights.detach().clone().requires_grad_()
    x = torch.randn(5, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    other_x = x.detach().clone().requires_grad_()
    output = grouped.experts(x, indices, weights)
    expected = reference.experts(other_x, indices, other_weights)
    assert_close(output, expected, atol=0.002, rtol=0.02)
    output.float().square().sum().backward()
    expected.float().square().sum().backward()
    assert_close(x.grad, other_x.grad, atol=0.002, rtol=0.02)
    assert_close(weights.grad, other_weights.grad, atol=0.002, rtol=0.02)
    for actual, expected in zip(
        grouped.experts.parameters(), reference.experts.parameters()
    ):
        assert_close(actual.grad, expected.grad, atol=0.002, rtol=0.02)
