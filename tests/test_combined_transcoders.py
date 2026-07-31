"""Synthetic (CPU) validation of the combined base+adapter transcoder construction.

Verifies that build_combined_*_transcoder forward-equals T_base(x) + T_adapter(x) with a
blockwise activation (JumpReLU base block + ReLU adapter block), plus the base/adapter
feature-index split helpers. Real-weights (GemmaScope + trained adapter) validation runs
on GPU separately.
"""

import unittest

import torch
from torch import nn

from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder, TranscoderSet
from circuit_tracer.transcoder.activation_functions import JumpReLU

from analysis.attribution.combined_transcoders import (
    SOURCE_ADAPTER,
    SOURCE_BASE,
    build_combined_layer_transcoder,
    build_combined_transcoder_set,
    feature_source,
    n_base_features,
    split_combined_feature_index,
    verify_combination,
)

CPU = torch.device("cpu")
DTYPE = torch.float32


def _make_transcoder(d_model, d_transcoder, activation, layer, seed):
    g = torch.Generator().manual_seed(seed)
    tc = SingleLayerTranscoder(d_model, d_transcoder, activation, layer, device=CPU, dtype=DTYPE)
    with torch.no_grad():
        tc.W_enc.copy_(torch.randn(d_transcoder, d_model, generator=g))
        tc.W_dec.copy_(torch.randn(d_transcoder, d_model, generator=g))
        tc.b_enc.copy_(torch.randn(d_transcoder, generator=g))
        tc.b_dec.copy_(torch.randn(d_model, generator=g))
    return tc


class CombinedTranscoderTests(unittest.TestCase):
    def test_layer_forward_equivalence(self):
        d_model, n_base, n_adapter = 16, 8, 5
        base = _make_transcoder(d_model, n_base, JumpReLU(torch.full((n_base,), 0.1)), 0, 1)
        adapter = _make_transcoder(d_model, n_adapter, nn.ReLU(), 0, 2)

        combined = build_combined_layer_transcoder(
            base, adapter, layer_idx=0, device=CPU, dtype=DTYPE
        )
        self.assertEqual(combined.d_transcoder, n_base + n_adapter)
        result = verify_combination(base, adapter, combined, n_samples=16, atol=1e-5, rtol=1e-5)
        self.assertTrue(result["allclose"], result)
        self.assertLess(result["max_abs_diff"], 1e-4, result)

    def test_blockwise_activation_actually_differs_per_block(self):
        # If the adapter block were (wrongly) given JumpReLU too, a negative-ish pre-act in the
        # adapter block would be zeroed differently than ReLU. Confirm the combined activations
        # match concatenated per-block activations on a controlled input.
        d_model, n_base, n_adapter = 4, 3, 3
        base = _make_transcoder(d_model, n_base, JumpReLU(torch.full((n_base,), 0.5)), 0, 7)
        adapter = _make_transcoder(d_model, n_adapter, nn.ReLU(), 0, 8)
        combined = build_combined_layer_transcoder(base, adapter, layer_idx=0, device=CPU, dtype=DTYPE)

        x = torch.randn(5, d_model)
        base_acts = base.encode(x)            # JumpReLU
        adapter_acts = adapter.encode(x)      # ReLU
        combined_acts = combined.encode(x)
        torch.testing.assert_close(combined_acts[:, :n_base], base_acts)
        torch.testing.assert_close(combined_acts[:, n_base:], adapter_acts)

    def test_set_level_equivalence_and_hooks(self):
        d_model, n_base, n_adapter, n_layers = 12, 6, 4, 3
        base_set = TranscoderSet(
            {l: _make_transcoder(d_model, n_base, JumpReLU(torch.full((n_base,), 0.1)), l, 100 + l) for l in range(n_layers)},
            feature_input_hook="ln2.hook_normalized",
            feature_output_hook="hook_mlp_out",
        )
        adapter_set = TranscoderSet(
            {l: _make_transcoder(d_model, n_adapter, nn.ReLU(), l, 200 + l) for l in range(n_layers)},
            feature_input_hook="ln2.hook_normalized",
            feature_output_hook="hook_mlp_out",
        )
        combined_set = build_combined_transcoder_set(base_set, adapter_set, scan_name="base-vs-adapter")
        self.assertEqual(combined_set.n_layers, n_layers)
        self.assertEqual(n_base_features(base_set), n_base)
        for l in range(n_layers):
            result = verify_combination(base_set[l], adapter_set[l], combined_set[l], atol=1e-5, rtol=1e-5)
            self.assertTrue(result["allclose"], (l, result))

    def test_hook_mismatch_rejected(self):
        d_model, n_base, n_adapter = 8, 4, 4
        base_set = TranscoderSet(
            {0: _make_transcoder(d_model, n_base, JumpReLU(torch.full((n_base,), 0.1)), 0, 1)},
            feature_input_hook="ln2.hook_normalized",
            feature_output_hook="hook_mlp_out",
        )
        adapter_set = TranscoderSet(
            {0: _make_transcoder(d_model, n_adapter, nn.ReLU(), 0, 2)},
            feature_input_hook="hook_resid_mid",  # mismatched
            feature_output_hook="hook_mlp_out",
        )
        with self.assertRaisesRegex(ValueError, "feature_input_hook mismatch"):
            build_combined_transcoder_set(base_set, adapter_set)

    def test_feature_index_split(self):
        n_base = 16384
        self.assertEqual(feature_source(0, n_base), SOURCE_BASE)
        self.assertEqual(feature_source(16383, n_base), SOURCE_BASE)
        self.assertEqual(feature_source(16384, n_base), SOURCE_ADAPTER)
        self.assertEqual(split_combined_feature_index(100, n_base), (SOURCE_BASE, 100))
        self.assertEqual(split_combined_feature_index(16384 + 42, n_base), (SOURCE_ADAPTER, 42))


if __name__ == "__main__":
    unittest.main()
