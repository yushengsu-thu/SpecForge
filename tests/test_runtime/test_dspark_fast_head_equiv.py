"""Reference vs fast DSpark head objective: same terms, same gradients.

GPU-only. Builds the dense-1/8 Qwen3-8B drafter skeleton with a random frozen
head (or the real Qwen3-8B head when ``SPECFORGE_TEST_TARGET_DIR`` is set),
runs both chunk objectives on the same random anchor blocks and compares every
returned term and the gradients of the draft hidden, the Markov embedding and
the Markov projection. ``SPECFORGE_BENCH=1`` also times both paths.
"""

import json
import os
import time
import unittest
from functools import partial
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def _build(device, vocab_size, hidden_size, draft_config):
    from transformers import Qwen3Config

    from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel
    from specforge.modeling.draft.dspark import DSparkDraftModel

    cfg = Qwen3Config(**draft_config)
    draft = DSparkDraftModel(cfg).to(device=device, dtype=torch.bfloat16)
    head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
    target_dir = os.environ.get("SPECFORGE_TEST_TARGET_DIR")
    if target_dir:
        from safetensors import safe_open

        index = json.load(open(Path(target_dir) / "model.safetensors.index.json"))
        shard = index["weight_map"]["lm_head.weight"]
        with safe_open(str(Path(target_dir) / shard), "pt", device="cpu") as f:
            head.weight.data.copy_(f.get_tensor("lm_head.weight"))
    head = head.to(device=device, dtype=torch.bfloat16)
    head.weight.requires_grad_(False)
    embed = torch.nn.Embedding(vocab_size, hidden_size).to(device=device, dtype=torch.bfloat16)
    model = OnlineDSparkModel(
        draft_model=draft,
        target_lm_head=head,
        target_embed_tokens=embed,
        mask_token_id=draft_config["dflash_config"]["mask_token_id"],
        block_size=draft_config["block_size"],
        num_anchors=512,
        loss_decay_gamma=7.0,
        dspark_ce_loss_alpha=0.1,
        dspark_l1_loss_alpha=0.9,
        dspark_confidence_head_alpha=0.0,
        objective_chunk_blocks=128,
    )
    return model


def _inputs(device, batch, seq_len, num_blocks, block_size, hidden_size, vocab_size, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    hidden = (torch.randn(batch, num_blocks, block_size, hidden_size, generator=g) * 0.5).to(
        device=device, dtype=torch.bfloat16
    )
    target_last = (torch.randn(batch, seq_len, hidden_size, generator=g) * 0.5).to(
        device=device, dtype=torch.bfloat16
    )
    anchors = torch.randint(0, seq_len - block_size - 1, (batch, num_blocks), generator=g)
    label_offsets = torch.arange(1, block_size + 1).view(1, 1, -1)
    label_indices = anchors.unsqueeze(-1) + label_offsets
    safe_label_indices = label_indices.clamp(max=seq_len - 1).to(device)
    target_ids = torch.randint(0, vocab_size, (batch, num_blocks, block_size), generator=g).to(device)
    prev_token_ids = torch.randint(0, vocab_size, (batch, num_blocks, block_size), generator=g).to(device)
    eval_mask = (torch.rand(batch, num_blocks, block_size, generator=g) > 0.1).to(device)
    eval_mask = eval_mask.to(torch.int32).cumprod(dim=-1).bool()
    return hidden, target_last, safe_label_indices, target_ids, prev_token_ids, eval_mask


class TestDSparkFastHeadEquivalence(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "GPU only")
    def test_terms_and_gradients_match(self):
        from specforge.core.chunking import checkpointed_chunk_reduce

        device = torch.device("cuda")
        draft_config = json.load(open(ROOT / "configs" / "qwen3-8b-dspark-dense-1-8-noconf-b16.json"))
        vocab_size, hidden_size, block_size = draft_config["vocab_size"], draft_config["hidden_size"], draft_config["block_size"]
        batch, seq_len, num_blocks = 2, 700, int(os.environ.get("SPECFORGE_TEST_BLOCKS", "256"))
        model = _build(device, vocab_size, hidden_size, draft_config)
        hidden, target_last, safe_label_indices, target_ids, prev_token_ids, eval_mask = _inputs(
            device, batch, seq_len, num_blocks, block_size, hidden_size, vocab_size
        )
        loss_weights = model._dspark_loss_weight_mask(eval_mask)
        block_index = torch.arange(num_blocks, device=device).unsqueeze(0)
        markov = model.draft_model.markov_head

        def run(fast: bool):
            h = hidden.clone().requires_grad_(True)
            markov.markov_w1.weight.grad = None
            markov.markov_w2.weight.grad = None
            if fast:
                model.dspark_fast_head = True
                totals = model._dspark_fast_head_totals(
                    hidden_4d=h, prev_token_ids=prev_token_ids, target_ids=target_ids,
                    loss_weights=loss_weights, eval_mask=eval_mask,
                    safe_label_indices=safe_label_indices,
                    target_last_hidden_states=target_last, block_index=block_index,
                    num_blocks=num_blocks,
                )
                self.assertIsNotNone(totals)
            else:
                aligned = model._aligned_target_hidden(target_last, safe_label_indices)
                totals = checkpointed_chunk_reduce(
                    partial(model._dspark_objective_chunk_terms, total_blocks=num_blocks),
                    h, prev_token_ids, target_ids, loss_weights, eval_mask, aligned, block_index,
                    chunk_size=model.objective_chunk_blocks, dim=1,
                )
            loss = 0.1 * totals[0] + 0.9 * totals[1]
            loss.backward()
            grads = (h.grad.float().clone(), markov.markov_w1.weight.grad.float().clone(), markov.markov_w2.weight.grad.float().clone())
            return [x.detach().float() for x in totals], grads

        ref_terms, ref_grads = run(False)
        fast_terms, fast_grads = run(True)
        names = ["ce_num", "l1_num", "confidence_num", "confidence_error_num", "correct_num", "eval_den",
                 "ce_position_num", "correct_position_num", "position_den", "teacher_agreement_num",
                 "teacher_top1_num", "draft_top1_num", "tau_num", "tau_den", "walk_accepted"]
        for name, a, b in zip(names, ref_terms, fast_terms):
            denom = a.abs().max().clamp_min(1e-6)
            rel = ((a - b).abs().max() / denom).item()
            # The reference rounds per-row CE to bf16 (F.cross_entropy on bf16 logits) and its
            # per-position CE metric sums in bf16; the fast path keeps both in fp32.
            tol = {"ce_num": 2e-3, "ce_position_num": 1e-2}.get(name, 1e-4)
            print(f"{name:24s} ref={a.flatten()[:3].tolist()} fast={b.flatten()[:3].tolist()} max_rel={rel:.2e}")
            self.assertLessEqual(rel, tol, name)
        for name, a, b in zip(["grad_hidden", "grad_markov_w1", "grad_markov_w2"], ref_grads, fast_grads):
            rel = ((a - b).norm() / a.norm().clamp_min(1e-12)).item()
            cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
            print(f"{name:24s} rel_l2={rel:.2e} cos={cos:.6f} |ref|={a.norm().item():.4e}")
            self.assertLessEqual(rel, 2e-2, name)
            self.assertGreaterEqual(cos, 0.999, name)

        if os.environ.get("SPECFORGE_BENCH") == "1":
            for fast in (False, True):
                for _ in range(2):
                    run(fast)
                torch.cuda.synchronize(); t0 = time.time()
                for _ in range(3):
                    run(fast)
                torch.cuda.synchronize()
                print(f"{'fast' if fast else 'ref '} objective fwd+bwd ({num_blocks} blocks x {batch}): {(time.time() - t0) / 3:.3f} s/iter, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
                torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    unittest.main()
