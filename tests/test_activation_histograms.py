import json
import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.features.activation_histograms import (
    DEFAULT_HISTOGRAM_BIN_LOWER_BOUNDS,
    approximate_histogram_quantile,
    build_default_histogram_bin_lower_bounds,
    compute_relative_domain_scores,
    format_activation_range_label,
    histogram_bin_indices,
    max_nonzero_bin_lower_bound,
    parse_activation_example_ranges,
    save_activation_histograms_npz,
)


class ActivationHistogramTests(unittest.TestCase):
    def test_default_bins_cover_dense_interesting_range_and_tail(self):
        bins = build_default_histogram_bin_lower_bounds()
        self.assertEqual(bins.dtype, np.float32)
        self.assertAlmostEqual(float(bins[0]), 0.0)
        self.assertAlmostEqual(float(bins[1]), 0.1, places=6)
        self.assertIn(np.float32(8.0), bins)
        self.assertIn(np.float32(24.0), bins)
        self.assertTrue(np.all(np.diff(bins) > 0))

    def test_histogram_bin_indices_uses_lower_bound_bins(self):
        bins = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float32)
        values = np.array([0.0, 0.49, 0.5, 1.99, 2.0, 9.0], dtype=np.float32)
        got = histogram_bin_indices(values, bins)
        np.testing.assert_array_equal(got, np.array([0, 0, 1, 2, 3, 3], dtype=np.int64))

    def test_approximate_histogram_quantile_uses_midpoint_for_finite_bins(self):
        bins = np.array([0.0, 1.0, 2.0, 4.0], dtype=np.float32)
        counts = np.array([0, 2, 8, 0], dtype=np.uint64)
        self.assertEqual(approximate_histogram_quantile(counts, bins, 0.95), 3.0)

    def test_approximate_histogram_quantile_returns_open_bin_lower_bound(self):
        bins = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        counts = np.array([0, 0, 5], dtype=np.uint64)
        self.assertEqual(approximate_histogram_quantile(counts, bins, 0.5), 2.0)

    def test_max_nonzero_bin_lower_bound(self):
        bins = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        self.assertEqual(
            max_nonzero_bin_lower_bound(np.array([0, 3, 0], dtype=np.uint64), bins),
            1.0,
        )
        self.assertIsNone(
            max_nonzero_bin_lower_bound(np.array([0, 0, 0], dtype=np.uint64), bins)
        )

    def test_approximate_histogram_quantile_returns_none_for_empty_counts(self):
        bins = DEFAULT_HISTOGRAM_BIN_LOWER_BOUNDS
        counts = np.zeros(len(bins), dtype=np.uint64)
        self.assertIsNone(approximate_histogram_quantile(counts, bins, 0.95))

    def test_compute_relative_domain_scores_is_finite_with_zero_baseline(self):
        bins = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        by_domain = np.array(
            [
                [0, 3, 1],
                [0, 0, 0],
            ],
            dtype=np.uint32,
        )
        scores = compute_relative_domain_scores(
            domain_counts={"chat": 4, "fineweb": 0},
            tokens_per_domain={"chat": 100, "fineweb": 200},
            domain_names=["chat", "fineweb"],
            feature_hist_by_domain=by_domain,
            bin_lower_bounds=bins,
            target_domain="chat",
            baseline_domain="fineweb",
        )
        self.assertIsNotNone(scores)
        assert scores is not None
        self.assertEqual(scores["target_count"], 4)
        self.assertEqual(scores["baseline_count"], 0)
        self.assertGreater(scores["log2_density_lift"], 0)
        self.assertIsNone(scores["p95_ratio"])
        self.assertEqual(scores["target_p99"], 2.0)
        self.assertEqual(scores["target_max_nonzero_bin"], 2.0)
        self.assertIsNone(scores["baseline_max_nonzero_bin"])

    def test_compute_relative_domain_scores_returns_none_when_domains_are_absent(self):
        got = compute_relative_domain_scores(
            domain_counts={"chat": 4},
            tokens_per_domain={"chat": 100},
            domain_names=["chat"],
            feature_hist_by_domain=np.zeros((1, 3), dtype=np.uint32),
            bin_lower_bounds=np.array([0.0, 1.0, 2.0], dtype=np.float32),
            target_domain="chat",
            baseline_domain="fineweb",
        )
        self.assertIsNone(got)

    def test_parse_activation_example_ranges(self):
        ranges = parse_activation_example_ranges("2.5:3.0,3.0:4.0")
        self.assertEqual(ranges, [(2.5, 3.0), (3.0, 4.0)])
        self.assertEqual(format_activation_range_label((2.5, 3.0)), "2.5-3.0")

    def test_save_activation_histograms_npz_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "activation_histograms.npz"
            bins = np.array([0.0, 1.0], dtype=np.float32)
            run_total = np.array([5, 7], dtype=np.uint64)
            run_by_domain = np.array([[3, 4], [2, 3]], dtype=np.uint64)
            feature_total_by_layer = [np.array([[1, 2], [3, 4]], dtype=np.uint32)]
            feature_by_domain_by_layer = [
                np.array(
                    [
                        [[1, 0], [0, 1]],
                        [[0, 2], [3, 0]],
                    ],
                    dtype=np.uint32,
                )
            ]

            save_activation_histograms_npz(
                path=path,
                bin_lower_bounds=bins,
                domain_names=["chat", "fineweb"],
                run_hist_total=run_total,
                run_hist_by_domain=run_by_domain,
                feature_hist_total_by_layer=feature_total_by_layer,
                feature_hist_by_domain_by_layer=feature_by_domain_by_layer,
            )

            with np.load(path, allow_pickle=False) as loaded:
                np.testing.assert_array_equal(loaded["bin_lower_bounds"], bins)
                self.assertEqual(loaded["domain_names"].tolist(), ["chat", "fineweb"])
                np.testing.assert_array_equal(loaded["run_hist_total"], run_total)
                np.testing.assert_array_equal(
                    loaded["feature_hist_total_layer_0"],
                    feature_total_by_layer[0],
                )
                np.testing.assert_array_equal(
                    loaded["feature_hist_by_domain_layer_0"],
                    feature_by_domain_by_layer[0],
                )

    def test_collector_histogram_arrays_have_expected_shapes(self):
        from analysis.features.collect_feature_activations import FeatureCollector

        collector = FeatureCollector(
            n_layers=2,
            n_features=3,
            top_k=1,
            n_random=1,
            domain_top_k=1,
            context_before=1,
            context_after=1,
            domain_names=["chat", "fineweb"],
        )
        self.assertEqual(collector.hist_bin_lower_bounds.ndim, 1)
        n_bins = len(collector.hist_bin_lower_bounds)
        self.assertEqual(collector.run_hist_total.shape, (n_bins,))
        self.assertEqual(collector.run_hist_by_domain.shape, (2, n_bins))
        self.assertEqual(collector.feature_hist_total_by_layer[0].shape, (3, n_bins))
        self.assertEqual(collector.feature_hist_by_domain_by_layer[0].shape, (2, 3, n_bins))

    def test_collector_histogram_update_counts_expected_bins(self):
        from analysis.features.collect_feature_activations import FeatureCollector

        collector = FeatureCollector(
            n_layers=1,
            n_features=2,
            top_k=1,
            n_random=1,
            domain_top_k=1,
            context_before=1,
            context_after=1,
            domain_names=["chat", "fineweb"],
            hist_bin_lower_bounds=np.array([0.0, 1.0, 2.0], dtype=np.float32),
        )
        collector._update_activation_histograms(
            layer_idx=0,
            domain="chat",
            active_features=np.array([0, 1, 1], dtype=np.int64),
            active_values=np.array([0.2, 1.2, 2.5], dtype=np.float32),
        )

        np.testing.assert_array_equal(collector.run_hist_total, np.array([1, 1, 1], dtype=np.uint64))
        np.testing.assert_array_equal(
            collector.run_hist_by_domain[0],
            np.array([1, 1, 1], dtype=np.uint64),
        )
        np.testing.assert_array_equal(
            collector.run_hist_by_domain[1],
            np.array([0, 0, 0], dtype=np.uint64),
        )
        np.testing.assert_array_equal(
            collector.feature_hist_total_by_layer[0][0],
            np.array([1, 0, 0], dtype=np.uint32),
        )
        np.testing.assert_array_equal(
            collector.feature_hist_total_by_layer[0][1],
            np.array([0, 1, 1], dtype=np.uint32),
        )
        np.testing.assert_array_equal(
            collector.feature_hist_by_domain_by_layer[0][0, 1],
            np.array([0, 1, 1], dtype=np.uint32),
        )

    def test_collector_keeps_bounded_activation_range_examples(self):
        import torch

        from analysis.features.collect_feature_activations import FeatureCollector

        collector = FeatureCollector(
            n_layers=1,
            n_features=1,
            top_k=1,
            n_random=1,
            domain_top_k=1,
            context_before=1,
            context_after=1,
            domain_names=["chat"],
            activation_example_ranges=[(2.5, 3.0)],
            activation_range_examples_per_domain=1,
        )
        stats = collector.stats[0][0]
        collector._maybe_add_activation_range_example(
            stats,
            activation=2.7,
            tokens=[10, 11, 12],
            position=1,
            features_gpu=torch.tensor([[0.1], [2.7], [0.2]]),
            feature_idx=0,
            domain="chat",
            region="all",
            thinking_position=None,
            sequence_idx=0,
        )
        kept = stats.activation_range_examples["chat|2.5-3.0"]
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].activation, 2.7)

    def test_build_examples_quantiles_includes_activation_ranges_in_scale(self):
        import torch

        from analysis.features.collect_feature_activations import (
            FeatureCollector,
            _build_examples_quantiles,
        )

        class TinyTokenizer:
            def decode(self, token_ids):
                return f"tok{token_ids[0]}"

        collector = FeatureCollector(
            n_layers=1,
            n_features=1,
            top_k=1,
            n_random=0,
            domain_top_k=1,
            context_before=1,
            context_after=1,
            domain_names=["chat"],
            activation_example_ranges=[(2.5, 3.0)],
            activation_range_examples_per_domain=1,
        )
        stats = collector.stats[0][0]
        collector._maybe_add_activation_range_example(
            stats,
            activation=2.7,
            tokens=[10, 11, 12],
            position=1,
            features_gpu=torch.tensor([[0.1], [2.7], [0.2]]),
            feature_idx=0,
            domain="chat",
            region="all",
            thinking_position=None,
            sequence_idx=0,
        )

        quantiles, act_min, act_max = _build_examples_quantiles(stats, TinyTokenizer())
        names = [q["quantile_name"] for q in quantiles]
        self.assertIn("Activation range 2.5-3.0 (chat)", names)
        self.assertLessEqual(act_min, 2.7)
        self.assertGreaterEqual(act_max, 2.7)

    def test_metadata_summary_and_relative_scores_are_json_safe(self):
        from analysis.features.collect_feature_activations import (
            FeatureCollector,
            build_feature_metadata_entry,
            build_metadata_payload,
        )

        collector = FeatureCollector(
            n_layers=1,
            n_features=2,
            top_k=1,
            n_random=1,
            domain_top_k=1,
            context_before=1,
            context_after=1,
            domain_names=["chat", "fineweb"],
        )
        collector.total_tokens = 300
        collector.tokens_per_domain["chat"] = 100
        collector.tokens_per_domain["fineweb"] = 200

        stats = collector.stats[0][0]
        stats.activation_count = 4
        stats.domain_counts["chat"] = 4
        collector.feature_hist_by_domain_by_layer[0][0, 0, 10] = 4

        feature_meta = build_feature_metadata_entry(
            collector=collector,
            stats=stats,
            layer_idx=0,
            feature_idx=0,
            target_domain="chat",
            baseline_domain="fineweb",
        )
        self.assertIn("relative_domain_scores", feature_meta)
        self.assertGreater(feature_meta["relative_domain_scores"]["log2_density_lift"], 0)
        self.assertIn("target_p99", feature_meta["relative_domain_scores"])
        self.assertIn("target_max_nonzero_bin", feature_meta["relative_domain_scores"])

        payload = build_metadata_payload(
            collector=collector,
            target_domain="chat",
            baseline_domain="fineweb",
        )
        encoded = json.dumps(payload)
        self.assertIn("activation_histograms.npz", encoded)
        self.assertIn("activation_histogram_summary", payload)
        self.assertIn("feature_frequency_summary", payload)
        frequency_summary = payload["feature_frequency_summary"]
        self.assertEqual(frequency_summary["all_features_total"], 2)
        self.assertEqual(frequency_summary["active_features_total"], 1)
        self.assertIn("chat", frequency_summary["feature_frequency_hist_by_domain"])
        self.assertIn("fineweb", frequency_summary["feature_frequency_hist_by_domain"])
        self.assertGreater(frequency_summary["global_nonzero_density_by_domain"]["chat"], 0)

    def test_export_run_arguments_writes_replay_command_and_full_json(self):
        from analysis.features.collect_feature_activations import export_run_arguments

        parser = argparse.ArgumentParser()
        parser.add_argument("--model_path", required=True)
        parser.add_argument("--val_data", nargs="+", required=True)
        parser.add_argument("--output_dir", default=None)
        parser.add_argument("--max_samples", type=int, default=None)
        parser.add_argument("--top_k", type=int, default=20)
        parser.add_argument("--shuffle", action="store_true")
        parser.add_argument(
            "--shuffle_batches",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        parser.add_argument("--tokenizer", default=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            args = parser.parse_args([
                "--model_path",
                "org/model",
                "--val_data",
                "chat:org/chat",
                "web:org/web",
                "--shuffle",
                "--no-shuffle_batches",
            ])
            args.output_dir = str(output_dir)

            export_run_arguments(parser, args, output_dir)

            command = (output_dir / "collect_feature_activations_command.sh").read_text()
            self.assertIn("--model_path=org/model", command)
            self.assertIn("--val_data chat:org/chat web:org/web", command)
            self.assertIn(f"--output_dir={output_dir}", command)
            self.assertIn("--top_k=20", command)
            self.assertIn("--shuffle", command)
            self.assertIn("--no-shuffle_batches", command)
            self.assertNotIn("--max_samples", command)
            self.assertNotIn("--tokenizer", command)

            payload = json.loads(
                (output_dir / "collect_feature_activations_args.json").read_text()
            )
            self.assertEqual(payload["top_k"], 20)
            self.assertEqual(payload["max_samples"], None)
            self.assertEqual(payload["shuffle"], True)
            self.assertEqual(payload["shuffle_batches"], False)


if __name__ == "__main__":
    unittest.main()
