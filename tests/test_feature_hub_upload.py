import argparse
import gzip
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analysis.features.hub_upload import (
    build_feature_collection_repo_id,
    dataset_ids_from_val_data,
    feature_collection_config_from_args,
    load_circuit_tracer_feature_from_hub,
)


class FeatureHubUploadTest(unittest.TestCase):
    def _args(self, **overrides):
        values = dict(
            model_path="org/model-name",
            val_data=["chat:hf://data-org/chat-dataset/data/val.jsonl", "science/fineweb"],
            output_dir="/tmp/run-a",
            export_circuit_tracer_features=True,
            upload_circuit_tracer_features_to_hub=True,
            hf_feature_repo_id=None,
            hub_org="me",
            max_samples=10,
            shuffle=False,
            shuffle_seed=None,
            top_k=20,
            domain_top_k=10,
            n_random=10,
            activation_example_ranges="0.5:1.0",
            activation_range_examples_per_domain=4,
            relative_target_domain="chat",
            relative_baseline_domain="fineweb",
            context_before=75,
            context_after=20,
            batch_size=16,
            shuffle_batches=True,
            shuffle_batches_seed=123,
            tokenizer=None,
            max_length=4096,
        )
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_repo_id_is_deterministic_and_ignores_local_output_dir(self):
        args_a = self._args(output_dir="/tmp/a")
        args_b = self._args(output_dir="/tmp/b")

        with patch("analysis.features.hub_upload.HfApi") as api_cls:
            repo_a, config_a = build_feature_collection_repo_id(args_a)
            repo_b, config_b = build_feature_collection_repo_id(args_b)

        api_cls.assert_not_called()
        self.assertEqual(repo_a, repo_b)
        self.assertEqual(config_a, config_b)
        self.assertTrue(repo_a.startswith("me/2026.TA.features_model-name_"))
        self.assertNotIn("output_dir", config_a)

    def test_dataset_ids_from_val_data_handles_domains_and_hf_uris(self):
        self.assertEqual(
            dataset_ids_from_val_data([
                "chat:hf://data-org/chat-dataset/data/val.jsonl",
                "hf://other-org/other-dataset/split.jsonl",
                "science/fineweb",
                "/local/file.jsonl",
            ]),
            ["data-org/chat-dataset", "other-org/other-dataset", "science/fineweb"],
        )

    def test_load_circuit_tracer_feature_from_hub_reads_uploaded_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            features = root / "features"
            features.mkdir()
            feature = {
                "layer": 0,
                "feature": 1,
                "top_logits": [" yes"],
                "bottom_logits": [" no"],
                "examples_quantiles": [],
                "activation_frequency": 0.5,
            }
            compressed = gzip.compress(json.dumps(feature).encode("utf-8"))
            payload = struct.pack("<I", len(compressed)) + compressed
            (features / "layer_0.bin").write_bytes(payload)
            with gzip.open(features / "index.json.gz", "wt") as f:
                json.dump({"version": "1.0", "format": "variable_chunks", "0": {"filename": "layer_0.bin", "offsets": [0, 0, len(payload)]}}, f)

            def fake_download(repo_id, filename, repo_type):
                self.assertEqual(repo_id, "me/features")
                self.assertEqual(repo_type, "model")
                return str(root / filename)

            with patch("analysis.features.hub_upload.hf_hub_download", side_effect=fake_download):
                loaded = load_circuit_tracer_feature_from_hub("me/features", 2)

        self.assertEqual(loaded["top_logits"], [" yes"])
        self.assertEqual(loaded["layer"], 0)
        self.assertEqual(loaded["feature"], 1)

    def test_config_marks_hub_upload_layout(self):
        config = feature_collection_config_from_args(self._args())
        self.assertEqual(config["upload_format"], "circuit_tracer_packed_features_v1")
        self.assertEqual(config["features_path_in_repo"], "features/")


if __name__ == "__main__":
    unittest.main()
