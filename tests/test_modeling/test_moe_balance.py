# coding=utf-8
"""balance.py and its wiring: loss, hooks, objective folds and the step boundary."""

import copy
import math
import unittest
from unittest import mock

import torch
import torch.distributed as dist
from torch import nn
from transformers import Qwen3Config
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    load_balancing_loss_func,
)

from specforge.algorithms.common.dflash_family_model import (
    OnlineDFlashModel,
    OnlineDominoModel,
    OnlineDSparkModel,
)
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.domino import DominoDraftModel
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.modeling.draft.moe.balance import (
    count_tokens_per_expert,
    dispatch_balance,
    load_balance_loss,
    router_metrics,
    set_moe_token_mask,
    step_moe_balancing,
    take_moe_aux_loss,
    take_moe_router_metrics,
)
from specforge.modeling.draft.moe.families.qwen3_5_moe import build_moe
from specforge.modeling.draft.moe.router import TopKRouter
from specforge.optimizer import BF16Optimizer, grad_norm_group
from specforge.runtime.contracts import TrainBatch
from specforge.training.backend import TrainingBackend
from specforge.training.controller import TrainerCore
from specforge.training.strategies.base import DraftTrainStrategy, StepOutput

E, K, B, S, H, V = 8, 2, 2, 6, 16, 32


def _moe(coef=0.5, selection_bias=False, **dflash_config):
    config = Qwen3_5MoeTextConfig(
        num_experts=E,
        num_experts_per_tok=K,
        hidden_size=H,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        dflash_config={
            "moe_router_aux_loss_coef": coef,
            "moe_router_selection_bias": selection_bias,
            **dflash_config,
        },
    )
    return build_moe(config)


def _zero_mean_sign_step(counts, rate=0.001):
    delta = rate * torch.sign(counts.mean() - counts)
    return delta - delta.mean()


def _real_token_mask():
    mask = torch.ones(B, S, dtype=torch.bool)
    mask[1, 4:] = False  # the second sample ends in padding
    return mask


def _loss(scores, indices, mask=None):
    return load_balance_loss(scores, count_tokens_per_expert(indices, mask, E), mask)


def _scores_and_indices(gate, x):
    """The router's scores and picks for ``x`` without touching its training state."""
    scores, indices, _ = gate.eval()(x.reshape(-1, H))
    gate.train()
    return scores, indices


def _tiny_moe_draft(coef=0.5, cls=DFlashDraftModel, **dflash_config):
    """A one-layer DFlash-family draft whose FFN is a 4-expert MoE, on CPU."""
    config = Qwen3Config(
        architectures=["DFlashDraftModel"],
        hidden_size=H,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        num_target_layers=4,
        head_dim=4,
        max_position_embeddings=64,
        vocab_size=V,
        layer_types=["full_attention"],
        block_size=2,
        dflash_config={
            "target_layer_ids": [1],
            "moe_router_aux_loss_coef": coef,
            **dflash_config,
        },
        moe_preset="qwen3_5_moe",
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=0,
        norm_topk_prob=False,
    )
    config._attn_implementation = "eager"
    return cls(config)


class _MaskRecordingDraft(nn.Module):
    """A stand-in draft: one MoE layer over the noise embedding."""

    def __init__(self, moe):
        super().__init__()
        self.moe = moe
        self.sliding_window = None
        self.seen_mask = None

    def forward(self, position_ids, noise_embedding, target_hidden, attention_mask):
        self.seen_mask = self.moe.token_mask
        return self.moe(noise_embedding)


def _online(draft, cls=OnlineDFlashModel, **overrides):
    kwargs = dict(
        draft_model=draft,
        target_lm_head=nn.Linear(H, V, bias=False),
        target_embed_tokens=nn.Embedding(V, H),
        mask_token_id=V - 1,
        block_size=2,
        attention_backend="eager",
        num_anchors=2,
    )
    return cls(**{**kwargs, **overrides})


class LoadBalanceLossTest(unittest.TestCase):
    def test_matches_transformers_load_balancing_loss(self):
        torch.manual_seed(0)
        logits = torch.randn(B * S, E, requires_grad=True)
        scores = logits.softmax(dim=-1)
        indices = scores.topk(K, dim=-1).indices
        for mask in (None, _real_token_mask()):
            with self.subTest(masked=mask is not None):
                # transformers sums the K slots per token: K times our per-token form.
                want = load_balancing_loss_func((logits,), E, K, mask)
                flat = None if mask is None else mask.reshape(-1)
                torch.testing.assert_close(K * _loss(scores, indices, flat), want)
        with self.subTest(masked="every token"):
            loss = _loss(scores, indices, torch.zeros(B * S, dtype=torch.bool))
            self.assertEqual(loss.item(), 0.0)
            loss.backward()
            self.assertTrue(torch.all(logits.grad == 0))


class FP32BuffersTest(unittest.TestCase):
    def test_buffers_keep_fp32_values_when_cast_to_bf16(self):
        router = TopKRouter(
            E, H, 1, norm_topk_prob=False, initializer_range=0.5, selection_bias=True
        )
        bias = torch.linspace(1.001, 1.002, E)  # bf16 would round these to 1.0
        router.selection_bias.copy_(bias)
        router.tokens_per_expert.fill_(1001.0)  # bf16 would round this to 1000
        router = router.bfloat16()
        self.assertEqual(router.selection_bias.dtype, torch.float32)
        self.assertEqual(router.tokens_per_expert.dtype, torch.float32)
        torch.testing.assert_close(router.selection_bias, bias)
        torch.testing.assert_close(router.tokens_per_expert, torch.full((E,), 1001.0))
        self.assertIn("selection_bias", router.state_dict())
        self.assertNotIn("tokens_per_expert", router.state_dict())


class RouterMetricsTest(unittest.TestCase):
    """The router health metrics: how peaked the scores are and how large the
    logits grew, read independently of where the dispatch sent the tokens."""

    BASE = {
        "top1_prob", "prob_balance", "logit_gap", "logit_absmax", "bias_flip_frac",
        "entropy", "logsumexp_sq", "top1_logit",
    }
    NAMES = BASE | {f"mean_prob/e{k}" for k in range(E)}

    def test_metrics_read_uniform_and_saturated_routers(self):
        flat = torch.zeros(6, E)  # every expert equally likely
        scores = flat.softmax(dim=-1)
        indices = scores.topk(K, dim=-1).indices
        m = router_metrics(flat, scores, indices)
        self.assertEqual(set(m), self.NAMES)
        self.assertAlmostEqual(m["top1_prob"].item(), 1 / E, places=6)
        self.assertAlmostEqual(m["prob_balance"].item(), 1.0, places=6)
        self.assertEqual(m["logit_gap"].item(), 0.0)
        self.assertEqual(m["logit_absmax"].item(), 0.0)
        self.assertEqual(m["bias_flip_frac"].item(), 0.0)
        self.assertAlmostEqual(m["entropy"].item(), math.log(E), places=5)
        self.assertAlmostEqual(m["logsumexp_sq"].item(), math.log(E) ** 2, places=5)
        self.assertEqual(m["top1_logit"].item(), 0.0)
        self.assertAlmostEqual(m["mean_prob/e3"].item(), 1 / E, places=6)
        peaked = flat.clone()
        peaked[:, 3] = 20.0  # expert 3 wins every token by 20 nats: saturated
        scores = peaked.softmax(dim=-1)
        indices = scores.topk(K, dim=-1).indices
        m = router_metrics(peaked, scores, indices)
        self.assertAlmostEqual(m["top1_prob"].item(), 1.0, places=5)
        self.assertAlmostEqual(m["prob_balance"].item(), float(E), places=4)
        self.assertAlmostEqual(m["logit_gap"].item(), 20.0, places=5)
        self.assertEqual(m["logit_absmax"].item(), 20.0)
        self.assertAlmostEqual(m["entropy"].item(), 0.0, places=4)
        self.assertAlmostEqual(m["logsumexp_sq"].item(), 400.0, places=3)
        self.assertEqual(m["top1_logit"].item(), 20.0)
        self.assertAlmostEqual(m["mean_prob/e3"].item(), 1.0, places=5)
        self.assertAlmostEqual(m["mean_prob/e0"].item(), 0.0, places=5)

    def test_metrics_skip_masked_tokens_and_count_bias_flips(self):
        logits = (torch.arange(E).float() * 0.1).repeat(4, 1)  # expert e scores 0.1 e
        logits[2:] += 50.0  # tokens 2 and 3 are padding: huge logits that must not count
        mask = torch.tensor([True, True, False, False])
        scores = logits.softmax(dim=-1)
        indices = scores.topk(K, dim=-1).indices
        m = router_metrics(logits, scores, indices, mask)
        torch.testing.assert_close(m["top1_prob"], scores[:2].max(dim=-1).values.mean())
        self.assertAlmostEqual(m["logit_gap"].item(), 0.1, places=5)
        self.assertAlmostEqual(m["logit_absmax"].item(), 0.7, places=5)
        # A bias lifting expert 0 into the top-k changes every kept token's picks.
        bias = torch.zeros(E)
        bias[0] = 1.0
        biased = (scores + bias).topk(K, dim=-1).indices
        m = router_metrics(logits, scores, biased, mask, bias)
        self.assertEqual(m["bias_flip_frac"].item(), 1.0)
        m = router_metrics(logits, scores, indices, mask, bias)  # a bias that changed nothing
        self.assertEqual(m["bias_flip_frac"].item(), 0.0)

    def test_router_keeps_the_metrics_in_training_and_the_hook_pops_them(self):
        torch.manual_seed(0)
        layers = nn.ModuleList([_moe(), _moe(selection_bias=True)])
        x = torch.randn(B, S, H)
        for layer in layers:
            layer(x)
        for layer in layers:
            self.assertEqual(set(layer.gate.metrics), self.NAMES)
            for value in layer.gate.metrics.values():
                self.assertFalse(value.requires_grad)
        recorded = [dict(layer.gate.metrics) for layer in layers]
        out = take_moe_router_metrics(layers)
        torch.testing.assert_close(
            out["moe/top1_prob"],
            (recorded[0]["top1_prob"] + recorded[1]["top1_prob"]) / 2,
        )
        torch.testing.assert_close(out["moe/layer1/logit_gap"], recorded[1]["logit_gap"])
        self.assertEqual(len(out), 3 * len(self.NAMES))  # the mean plus two layers
        for layer in layers:
            self.assertIsNone(layer.gate.metrics)
        self.assertIsNone(take_moe_router_metrics(layers))
        layers[0].eval()(x)
        self.assertIsNone(layers[0].gate.metrics)  # eval records nothing

    def test_step_reports_maxvio_dead_experts_and_the_bias_range(self):
        layers = nn.ModuleList([_moe(selection_bias=True), _moe()])
        with torch.no_grad():
            collapsed = torch.zeros(E)
            collapsed[0] = 8.0  # every slot went to expert 0
            layers[0].gate.tokens_per_expert.copy_(collapsed)
            layers[1].gate.tokens_per_expert.fill_(2.0)  # uniform
        metrics = step_moe_balancing(layers)
        self.assertAlmostEqual(metrics["moe/layer0/step_balance"].item(), float(E))
        self.assertAlmostEqual(metrics["moe/layer0/maxvio"].item(), E - 1)
        self.assertEqual(metrics["moe/layer0/dead_experts"].item(), E - 1)
        self.assertAlmostEqual(metrics["moe/layer1/step_balance"].item(), 1.0)
        self.assertEqual(metrics["moe/layer1/maxvio"].item(), 0.0)
        self.assertEqual(metrics["moe/layer1/dead_experts"].item(), 0.0)
        self.assertAlmostEqual(metrics["moe/maxvio"].item(), (E - 1) / 2)
        bias = layers[0].gate.selection_bias
        torch.testing.assert_close(metrics["moe/layer0/bias_range"], bias.max() - bias.min())
        self.assertGreater(metrics["moe/layer0/bias_range"].item(), 0.0)  # the step moved it
        self.assertEqual(metrics["moe/layer1/bias_range"].item(), 0.0)  # no bias there
        self.assertEqual(metrics["moe/layer0/share/e0"].item(), 1.0)
        self.assertEqual(metrics["moe/layer0/share/e1"].item(), 0.0)
        self.assertAlmostEqual(metrics["moe/layer1/share/e5"].item(), 1 / E, places=6)
        torch.testing.assert_close(metrics["moe/layer0/bias/e0"], bias[0])
        self.assertNotIn("moe/layer1/bias/e0", metrics)  # no bias there
        for layer in layers:
            self.assertTrue(torch.all(layer.gate.tokens_per_expert == 0))

    def test_objective_logs_the_router_metrics(self):
        torch.manual_seed(0)
        model = _online(_tiny_moe_draft())
        _, _, metrics = model(
            input_ids=torch.randint(0, V - 1, (1, 8)),
            hidden_states=torch.randn(1, 8, H),
            loss_mask=torch.ones(1, 8),
        )
        ratio = metrics["ratio_metrics"]
        for name in self.BASE | {"mean_prob/e0", "mean_prob/e3"}:  # the tiny draft has 4 experts
            value, weight = ratio[f"moe/{name}"]
            self.assertEqual(weight.item(), 1.0)
            self.assertFalse(value.requires_grad)
            self.assertIn(f"moe/layer0/{name}", ratio)
        self.assertGreaterEqual(ratio["moe/top1_prob"][0].item(), 1 / 4)  # 4 experts
        self.assertIsNone(model.draft_model.layers[0].mlp.gate.metrics)


class BalancingHooksTest(unittest.TestCase):
    """Two micro-batches and one optimizer step through the hooks in balance.py."""

    def test_micro_batches_feed_the_loss_and_the_step_moves_the_bias(self):
        torch.manual_seed(0)
        layers = nn.ModuleList(
            [
                _moe(coef=0.5, selection_bias=True),
                _moe(
                    coef=0.25,
                    selection_bias=True,
                    moe_router_aux_loss_type="batch_wise",
                ),
            ]
        )
        gates = [layer.gate for layer in layers]
        x, mask = torch.randn(B, S, H), _real_token_mask()

        with self.subTest(phase="first micro-batch, masked"):
            set_moe_token_mask(layers, mask)
            for layer in layers:
                self.assertIs(layer.token_mask, mask)
                layer(x)
            losses = [gate.aux_loss for gate in gates]
            first = [gate.tokens_per_expert.clone() for gate in gates]
            aux, raw = take_moe_aux_loss(layers)
            torch.testing.assert_close(aux, 0.5 * losses[0] + 0.25 * losses[1])
            torch.testing.assert_close(raw, (losses[0] + losses[1]) / 2)
            for layer in layers:
                self.assertIsNone(layer.gate.aux_loss)
                self.assertIsNone(layer.token_mask)
            # On the step's first forward both loss types read this micro-batch alone.
            for gate, loss in zip(gates, losses):
                scores, indices = _scores_and_indices(gate, x)
                torch.testing.assert_close(
                    loss, _loss(scores, indices, mask.reshape(-1))
                )
                (grad,) = torch.autograd.grad(loss, gate.weight)
                self.assertTrue(torch.any(grad != 0))

        with self.subTest(phase="second micro-batch, batch-wise load"):
            x2 = torch.randn(B, S, H)
            for layer in layers:
                layer(x2)
            got = gates[1].aux_loss
            scores, indices = _scores_and_indices(gates[1], x2)
            pooled = first[1] + count_tokens_per_expert(indices, None, E)
            torch.testing.assert_close(got, load_balance_loss(scores, pooled))
            take_moe_aux_loss(layers)

        with self.subTest(phase="optimizer step"):
            totals = [gate.tokens_per_expert.clone() for gate in gates]
            other = torch.tensor([9.0, 0, 0, 0, 0, 0, 0, 3.0])  # another rank's counts
            with mock.patch.object(dist, "is_initialized", return_value=True):
                with mock.patch.object(dist, "all_reduce", lambda t: t.add_(other)):
                    metrics = step_moe_balancing(layers)
            for gate, total in zip(gates, totals):
                want = _zero_mean_sign_step(total + other)
                torch.testing.assert_close(gate.selection_bias, want)
                self.assertTrue(torch.all(gate.tokens_per_expert == 0))
            want = torch.stack([dispatch_balance(t + other) for t in totals]).mean()
            torch.testing.assert_close(metrics["moe/step_balance"], want)
            uniform, collapsed = torch.full((E,), 3.0), torch.tensor([24.0] + [0.0] * 7)
            torch.testing.assert_close(dispatch_balance(uniform), torch.tensor(1.0))
            torch.testing.assert_close(dispatch_balance(collapsed), torch.tensor(8.0))

        with self.subTest(phase="the bias steers the choice, not the weights"):
            scores, indices, weights = gates[0](x.reshape(-1, H))
            biased = (scores + gates[0].selection_bias).topk(K, dim=-1).indices
            self.assertTrue(torch.equal(indices, biased))
            picked = scores.gather(-1, indices)
            torch.testing.assert_close(weights, picked / picked.sum(-1, keepdim=True))

        with self.subTest(phase="eval and dense models"):
            set_moe_token_mask(layers, mask)
            layers[0].eval()(x)
            self.assertIsNone(gates[0].aux_loss)
            self.assertEqual(step_moe_balancing(nn.Linear(2, 2)), {})


class ObjectiveTest(unittest.TestCase):
    """Each objective adds coef x the raw loss; the draft blocks hand over the mask."""

    def _inputs(self, **extra):
        torch.manual_seed(0)
        return dict(
            input_ids=torch.randint(0, V - 1, (1, 8)),
            hidden_states=torch.randn(1, 8, H),
            loss_mask=torch.ones(1, 8),
            **extra,
        )

    def _check_fold(self, with_aux, inputs, has_loss_terms):
        plain = copy.deepcopy(with_aux)
        plain.draft_model.layers[0].mlp.gate.aux_loss_coef = 0.0
        torch.manual_seed(1)
        loss, _, metrics = with_aux(**inputs)
        torch.manual_seed(1)
        plain_loss, _, plain_metrics = plain(**inputs)
        raw, _ = metrics["ratio_metrics"]["moe/aux_loss"]
        # The metric is the unweighted loss, reported at any coefficient.
        plain_raw = plain_metrics["ratio_metrics"]["moe/aux_loss"][0]
        torch.testing.assert_close(plain_raw, raw)
        torch.testing.assert_close(loss - plain_loss, 0.5 * raw)
        if has_loss_terms:
            numerator, denominator = metrics["loss_terms"]
            plain_numerator, _ = plain_metrics["loss_terms"]
            torch.testing.assert_close(
                numerator - plain_numerator, 0.5 * raw * denominator
            )
        else:
            self.assertNotIn("loss_terms", metrics)
        # The per-forward state is consumed.
        self.assertIsNone(with_aux.draft_model.layers[0].mlp.gate.aux_loss)
        self.assertIsNone(with_aux.draft_model.layers[0].mlp.token_mask)

    def test_each_objective_folds_the_loss_and_hands_the_block_mask(self):
        with self.subTest(family="dflash"):
            self._check_fold(_online(_tiny_moe_draft()), self._inputs(), True)
        with self.subTest(family="domino"):
            draft = _tiny_moe_draft(cls=DominoDraftModel, emb_dim=8, gru_hidden_dim=8)
            model = _online(draft, cls=OnlineDominoModel)
            self._check_fold(model, self._inputs(lambda_base=0.3), True)
        with self.subTest(family="dspark"):
            draft = _tiny_moe_draft(cls=DSparkDraftModel, markov_rank=0)
            model = _online(draft, cls=OnlineDSparkModel)
            inputs = self._inputs(target_last_hidden_states=torch.randn(1, 8, H))
            self._check_fold(model, inputs, False)
        with self.subTest(family="dense"):
            config = copy.deepcopy(_tiny_moe_draft().config)
            config.moe_preset = None
            _, _, metrics = _online(DFlashDraftModel(config))(**self._inputs())
            self.assertNotIn("moe/aux_loss", metrics["ratio_metrics"])
        with self.subTest(phase="the draft blocks hand the layers the kept-block mask"):
            block_size = 3
            draft = _MaskRecordingDraft(_moe())
            model = _online(draft, block_size=block_size)
            keep = torch.tensor([[True, True], [True, False]])
            anchors = torch.tensor([[1, 3], [2, 0]])
            model._sample_anchor_positions = lambda *args, **kwargs: (anchors, keep)
            input_ids = torch.randint(1, V, (2, 6))
            model._forward_draft_blocks(
                input_ids, torch.randn(2, 6, H), torch.ones(2, 6)
            )
            want = keep.repeat_interleave(block_size, dim=1)
            torch.testing.assert_close(draft.seen_mask, want)


class _Backend(TrainingBackend):
    name = "test"

    def prepare_model(self, model):
        return model

    def backward(self, loss, *, is_boundary=True):
        loss.backward()

    def step(self):
        return torch.tensor(1.0)

    def optimizer_breakdown(self):
        return {"grad_norm/layer0/router": torch.tensor(0.5)}

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass


class _Wrapped(nn.Module):
    """Stands in for DDP and FSDP, which keep the model under ``.module``."""

    def __init__(self, module):
        super().__init__()
        self.module = module


class _MoEStrategy(DraftTrainStrategy):
    name = "moe-test"
    required_features = {"hidden_states"}

    def __init__(self, model):
        self.model = model

    def trainable_module(self):
        return self.model

    def forward_loss(self, batch, ctx=None):
        draft = getattr(self.model, "module", self.model).draft_model
        out = draft.moe(batch.tensors["hidden_states"])
        return StepOutput(loss=out.square().mean(), metrics={})


def _batch(x):
    tensors = {"hidden_states": x}
    return TrainBatch(sample_ids=["s0"], strategy="moe-test", tensors=tensors)


class StepHookTest(unittest.TestCase):
    def test_trainer_runs_the_hook_at_the_optimizer_step_boundary(self):
        torch.manual_seed(0)
        draft, x = _MaskRecordingDraft(_moe(selection_bias=True)), torch.randn(B, S, H)
        # DDP and FSDP keep the model under .module; the hook is found through it.
        strategy = _MoEStrategy(_Wrapped(_online(draft)))
        core = TrainerCore(strategy, _Backend(), accumulation_steps=2)
        first = core.train_step(_batch(x))
        counts = draft.moe.gate.tokens_per_expert.clone()
        self.assertTrue(torch.all(draft.moe.gate.selection_bias == 0))
        self.assertNotIn("moe/step_balance", first.metrics)
        second = core.train_step(_batch(x))  # the same routing again: counts double
        want = _zero_mean_sign_step(2 * counts)
        torch.testing.assert_close(draft.moe.gate.selection_bias, want)
        self.assertTrue(torch.all(draft.moe.gate.tokens_per_expert == 0))
        # The step's pooled balance is reported with the boundary step's metrics.
        self.assertAlmostEqual(
            second.metrics["moe/step_balance"],
            dispatch_balance(2 * counts).item(),
            places=5,
        )
        # The backend's per-module grad norms join the boundary step's metrics too.
        self.assertAlmostEqual(second.metrics["grad_norm/layer0/router"], 0.5)
        self.assertNotIn("grad_norm/layer0/router", first.metrics)
        self.assertIsNone(_MoEStrategy(nn.Linear(2, 2)).on_optimizer_step())


class _DraftLike(nn.Module):
    """Parameter names shaped like a DSpark MoE draft: fc, layers.{i}.{self_attn,mlp,norms}, norm."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4, bias=False)
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj = nn.Linear(4, 4, bias=False)
        layer.self_attn.q_norm = nn.Linear(4, 1, bias=False)
        layer.mlp = nn.Module()
        layer.mlp.gate = nn.Linear(4, 2, bias=False)
        layer.mlp.experts = nn.Module()
        layer.mlp.experts.gate_up_proj = nn.Parameter(torch.randn(2, 4, 4))
        layer.input_layernorm = nn.Linear(4, 1, bias=False)
        self.layers = nn.ModuleList([layer])
        self.norm = nn.Linear(4, 1, bias=False)

    def forward(self, x):
        h = self.fc(x)
        layer = self.layers[0]
        h = layer.self_attn.q_proj(h) + layer.self_attn.q_norm(h) + layer.input_layernorm(h)
        h = h + layer.mlp.gate(h).sum(-1, keepdim=True) + (h @ layer.mlp.experts.gate_up_proj[0]) @ layer.mlp.experts.gate_up_proj[1]
        return self.norm(h)


class GradNormBreakdownTest(unittest.TestCase):
    """Per-module pre-clip grad norms next to the global one."""

    def test_group_names_follow_the_draft_layout_and_skip_wrappers(self):
        self.assertEqual(grad_norm_group("layers.3.mlp.gate.weight"), "layer3/router")
        self.assertEqual(grad_norm_group("layers.3.mlp.experts.down_proj"), "layer3/experts")
        self.assertEqual(grad_norm_group("layers.0.mlp.gate_proj.weight"), "layer0/mlp")
        self.assertEqual(grad_norm_group("layers.0.self_attn.q_proj.weight"), "layer0/attn")
        self.assertEqual(grad_norm_group("layers.0.post_attention_layernorm.weight"), "layer0/norm")
        self.assertEqual(
            grad_norm_group("_fsdp_wrapped_module.layers.1._fsdp_wrapped_module.self_attn.k_norm.weight"),
            "layer1/attn",
        )
        self.assertEqual(grad_norm_group("module.fc.weight"), "fc")
        self.assertEqual(grad_norm_group("markov_head.markov_w1.weight"), "markov_head")
        self.assertEqual(grad_norm_group("norm.weight"), "norm")

    def test_step_reports_the_breakdown_consistent_with_the_total(self):
        torch.manual_seed(0)
        model = _DraftLike()
        model(torch.randn(3, 4)).square().sum().backward()
        want = {
            grad_norm_group(name): float(p.grad.float().square().sum())
            for name, p in model.named_parameters()
            if p.grad is not None
        }
        expected_leaf = {}
        for name, p in model.named_parameters():
            expected_leaf[grad_norm_group(name)] = expected_leaf.get(grad_norm_group(name), 0.0) + float(p.grad.float().square().sum())
        total = sum(expected_leaf.values()) ** 0.5
        opt = BF16Optimizer(
            model, lr=1e-3, max_grad_norm=1e9, total_steps=10, warmup_ratio=0.0,
            lr_scheduler="constant",
        )
        norm = opt.step()
        groups = opt.last_grad_norm_groups
        self.assertAlmostEqual(norm.item(), total, places=4)
        # The first AdamW step moves every weight with a non-zero grad by ~lr.
        updates = opt.last_update_norm_groups
        fc_numel = model.fc.weight.numel()
        self.assertAlmostEqual(
            updates["update_norm/fc"].item(), 1e-3 * fc_numel**0.5, delta=1e-3 * fc_numel**0.5 * 0.02
        )
        params = opt.last_param_norm_groups
        self.assertAlmostEqual(params["param_norm/fc"].item(), model.fc.weight.float().norm().item(), places=4)
        self.assertEqual({k.split("/", 1)[1] for k in updates}, {k.split("/", 1)[1] for k in groups})
        self.assertEqual(set(opt.last_breakdown()), set(groups) | set(updates) | set(params))
        for leaf, sq in expected_leaf.items():
            self.assertAlmostEqual(groups[f"grad_norm/{leaf}"].item(), sq**0.5, places=4, msg=leaf)
        layer0 = sum(sq for leaf, sq in expected_leaf.items() if leaf.startswith("layer0/")) ** 0.5
        self.assertAlmostEqual(groups["grad_norm/layer0"].item(), layer0, places=4)
        self.assertAlmostEqual(groups["grad_norm/layers/router"].item(), expected_leaf["layer0/router"] ** 0.5, places=4)
        self.assertEqual(
            set(groups),
            {"grad_norm/fc", "grad_norm/norm", "grad_norm/layer0", "grad_norm/layer0/attn", "grad_norm/layer0/router",
             "grad_norm/layer0/experts", "grad_norm/layer0/norm", "grad_norm/layers/attn", "grad_norm/layers/router",
             "grad_norm/layers/experts", "grad_norm/layers/norm"},
        )
        self.assertTrue(all(not v.requires_grad for v in groups.values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
