"""Tests for token-conditional specificity (graph-viz metric 2).

Covers the Space-Saving bounded token counter in FeatureCollector.accumulate_batch and the
format_top_tokens() export helper that turns per-feature token counts into the
``token_specificity`` field of each feature JSON.
"""

import unittest

import torch

from analysis.features.collect_feature_activations import (
    FeatureCollector,
    FeatureStats,
    format_top_tokens,
)


class _FakeTokenizer:
    def __init__(self, mapping):
        self._mapping = mapping

    def decode(self, ids):
        return self._mapping.get(int(ids[0]), f"<{int(ids[0])}>")


class FormatTopTokensTests(unittest.TestCase):
    def test_orders_by_count_and_computes_fraction(self):
        stats = FeatureStats(layer_idx=0, feature_idx=1)
        stats.activation_count = 10
        stats.token_counts = {5: 6, 9: 3, 2: 1}
        tok = _FakeTokenizer({5: "\n", 9: " the", 2: " a"})

        result = format_top_tokens(stats, tok, k=2)

        self.assertEqual([r["token"] for r in result], ["\n", " the"])
        self.assertEqual([r["token_id"] for r in result], [5, 9])
        self.assertAlmostEqual(result[0]["fraction"], 0.6)
        self.assertAlmostEqual(result[1]["fraction"], 0.3)

    def test_empty_when_no_token_counts(self):
        stats = FeatureStats(layer_idx=0, feature_idx=0)
        self.assertEqual(format_top_tokens(stats, _FakeTokenizer({})), [])


class TokenCountTrackingTests(unittest.TestCase):
    def _collector(self, **kwargs):
        return FeatureCollector(
            n_layers=1,
            n_features=2,
            top_k=5,
            n_random=0,
            domain_top_k=1,
            context_before=2,
            context_after=2,
            domain_names=["d"],
            activation_example_ranges=[],
            activation_range_examples_per_domain=0,
            **kwargs,
        )

    def _accumulate(self, collector, tokens, active_feature=0):
        # One layer, feature `active_feature` fires (value 1.0) at every position.
        seq = len(tokens)
        acts = torch.zeros(1, seq, 2)
        acts[0, :, active_feature] = 1.0
        markers = {key: None for key in ("bos", "user_marker", "assistant_marker", "think_start", "think_end")}
        collector.accumulate_batch(
            batch_tokens=[tokens],
            batch_domains=["d"],
            batch_markers=[markers],
            batch_seq_idxs=[0],
            batch_source_metadata=[{}],
            layer_activations={0: acts},
        )

    def test_counts_tokens_per_feature(self):
        collector = self._collector()
        # token 7 appears 3x, token 4 appears 1x
        self._accumulate(collector, [7, 7, 4, 7])
        counts = collector.stats[0][0].token_counts
        self.assertEqual(counts[7], 3)
        self.assertEqual(counts[4], 1)

    def test_space_saving_cap_bounds_distinct_tokens(self):
        collector = self._collector(token_count_cap=2)
        # 4 distinct tokens but cap=2 -> at most 2 tracked entries
        self._accumulate(collector, [1, 2, 3, 4, 1, 1])
        counts = collector.stats[0][0].token_counts
        self.assertLessEqual(len(counts), 2)
        # The dominant token (1, appearing 3x) should survive eviction.
        self.assertIn(1, counts)

    def test_tracking_can_be_disabled(self):
        collector = self._collector(track_token_counts=False)
        self._accumulate(collector, [7, 7, 4])
        self.assertEqual(collector.stats[0][0].token_counts, {})


if __name__ == "__main__":
    unittest.main()
