"""Backend parity includes grouped heads, unequal value width and empty queries."""

import unittest

import torch

from specforge.modeling.draft.layers.attention.backends import (
    attention_backend,
    execute_attention,
)
from specforge.modeling.draft.layers.attention.masks import (
    AttentionLayout,
    prepare_attention_mask,
)


class AttentionBackendTest(unittest.TestCase):
    def check_backend(self, name, device):
        torch.manual_seed(17)
        for key_heads, value_width in ((2, 16), (4, 8)):
            with self.subTest(backend=name, key_heads=key_heads):
                inputs = [
                    torch.randn(shape, device=device, requires_grad=True)
                    for shape in (
                        (2, 4, 7, 16),
                        (2, key_heads, 11, 16),
                        (2, key_heads, 11, value_width),
                    )
                ]
                mask = torch.rand(2, 1, 7, 11, device=device) > 0.3
                mask[:, :, 0] = False
                options = dict(
                    attention_mask=mask, scale=0.25, dropout_p=0.0, training=True
                )
                expected = attention_backend("eager")(*inputs, **options)
                actual = attention_backend(name)(*inputs, **options)
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
                gradient = torch.randn_like(expected)
                reference = torch.autograd.grad(expected, inputs, gradient)
                result = torch.autograd.grad(actual, inputs, gradient)
                for left, right in zip(result, reference):
                    torch.testing.assert_close(left, right, atol=3e-5, rtol=3e-4)
                if name == "flex_attention":
                    layout = AttentionLayout(
                        query_positions=torch.arange(7, device=device).expand(2, -1),
                        key_positions=torch.arange(11, device=device).expand(2, -1),
                        visibility=lambda b, h, q, k: mask[b, 0, q, k],
                    )
                    options["attention_mask"] = prepare_attention_mask(
                        layout, is_causal=False, sliding_window=None, backend=name
                    )
                    sparse, active = execute_attention(
                        attention_backend(name), *inputs, **options
                    )
                    torch.testing.assert_close(sparse, expected, atol=2e-5, rtol=2e-4)
                    self.assertFalse(active[:, 0].any())
                    for left, right in zip(
                        torch.autograd.grad(sparse, inputs, gradient), reference
                    ):
                        torch.testing.assert_close(left, right, atol=3e-5, rtol=3e-4)

    def test_sdpa(self):
        self.check_backend("sdpa", "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_flex(self):
        self.check_backend("flex_attention", "cuda")


if __name__ == "__main__":
    unittest.main()
