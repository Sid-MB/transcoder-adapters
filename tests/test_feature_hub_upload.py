import argparse
import gzip
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analysis.features.hub_upload import (
    FEATURE_REPO_NAME_MAX_LEN,
    FEATURE_SIDECAR_FILENAMES,
    build_feature_collection_repo_id,
    check_feature_collection_exists,
    dataset_ids_from_val_data,
    feature_collection_fingerprint,
    feature_collection_config_from_args,
    load_circuit_tracer_feature_from_hub,
    upload_circuit_tracer_features_to_hub,
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
        self.assertTrue(repo_a.endswith(f"_h{feature_collection_fingerprint(config_a)}"))
        self.assertNotIn("output_dir", config_a)

    def test_repo_id_ignores_sharding_and_merge_flags(self):
        # A collection produced as N shards + a merge is identical to a single-process
        # run, so it must map to the SAME deterministic repo. Excluding these keys also
        # keeps configs reserved before the sharding CLI existed (no such keys) equal.
        single = self._args()  # no sharding keys at all (pre-sharding-CLI shape)
        shard_worker = self._args(num_shards=64, shard_index=7)
        merge_job = self._args(num_shards=1, shard_index=0, merge_shards="/d/shard_*.pkl")

        cfg_single = feature_collection_config_from_args(single)
        cfg_shard = feature_collection_config_from_args(shard_worker)
        cfg_merge = feature_collection_config_from_args(merge_job)

        self.assertEqual(cfg_single, cfg_shard)
        self.assertEqual(cfg_single, cfg_merge)
        self.assertEqual(
            feature_collection_fingerprint(cfg_single),
            feature_collection_fingerprint(cfg_merge),
        )
        for key in ("num_shards", "shard_index", "merge_shards"):
            self.assertNotIn(key, cfg_merge)

    def test_repo_id_truncation_preserves_config_hash(self):
        args = self._args(model_path=f"org/{'very-long-model-name-' * 8}")

        repo_id, config = build_feature_collection_repo_id(args)
        _, repo_name = repo_id.split("/", 1)

        self.assertLessEqual(len(repo_name), FEATURE_REPO_NAME_MAX_LEN)
        self.assertTrue(repo_name.endswith(f"_h{feature_collection_fingerprint(config)}"))

    def test_repo_id_is_deterministic_for_val_data_order(self):
        val_data = ["chat:hf://data-org/chat-dataset/data/val.jsonl", "science/fineweb"]
        args_a = self._args(val_data=val_data)
        args_b = self._args(val_data=list(reversed(val_data)))

        repo_a, config_a = build_feature_collection_repo_id(args_a)
        repo_b, config_b = build_feature_collection_repo_id(args_b)

        self.assertEqual(repo_a, repo_b)
        self.assertEqual(config_a, config_b)
        self.assertEqual(config_a["val_data"], sorted(val_data))

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

    def test_check_feature_collection_exists_returns_none_when_missing(self):
        config = feature_collection_config_from_args(self._args())

        with patch("analysis.features.hub_upload.HfApi"), \
             patch("analysis.features.hub_upload._repo_exists", return_value=False):
            self.assertIsNone(check_feature_collection_exists("me/features", config))

    def test_check_feature_collection_exists_returns_repo_id_for_matching_config(self):
        config = feature_collection_config_from_args(self._args())

        with patch("analysis.features.hub_upload.HfApi"), \
             patch("analysis.features.hub_upload._repo_exists", return_value=True), \
             patch("analysis.features.hub_upload._download_existing_config", return_value=config):
            self.assertEqual(
                check_feature_collection_exists("me/features", config),
                "me/features",
            )

    def test_check_feature_collection_exists_errors_for_config_mismatch(self):
        config = feature_collection_config_from_args(self._args())
        existing_config = dict(config)
        existing_config["top_k"] = 99

        with patch("analysis.features.hub_upload.HfApi"), \
             patch("analysis.features.hub_upload._repo_exists", return_value=True), \
             patch("analysis.features.hub_upload._download_existing_config", return_value=existing_config):
            with self.assertRaisesRegex(RuntimeError, "different collection config"):
                check_feature_collection_exists("me/features", config)

    def test_upload_circuit_tracer_features_uploads_sidecars(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            packed_dir = output_dir / "circuit_tracer_features"
            packed_dir.mkdir()
            (packed_dir / "index.json.gz").write_bytes(b"index")
            (packed_dir / "layer_0.bin").write_bytes(b"layer")
            (output_dir / "feature_metadata.json").write_text(json.dumps({
                "total_tokens": 1,
                "tokens_per_domain": {},
                "feature_frequency_summary": {},
            }))
            (output_dir / "activation_histograms.npz").write_bytes(b"npz")
            (output_dir / "feature_annotations.json").write_text("{}")
            config = feature_collection_config_from_args(self._args())

            with patch("analysis.features.hub_upload.HfApi") as api_cls, \
                 patch("analysis.features.hub_upload._upload_json"), \
                 patch("analysis.features.hub_upload._push_feature_model_card"):
                api = api_cls.return_value
                upload_circuit_tracer_features_to_hub(
                    repo_id="me/features",
                    output_dir=output_dir,
                    config=config,
                )

            uploaded_paths = {
                call.kwargs["path_in_repo"]
                for call in api.upload_file.call_args_list
            }
            self.assertEqual(uploaded_paths, set(FEATURE_SIDECAR_FILENAMES))
            api.upload_folder.assert_called_once()

    def test_upload_circuit_tracer_features_requires_sidecars(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            packed_dir = output_dir / "circuit_tracer_features"
            packed_dir.mkdir()
            (packed_dir / "index.json.gz").write_bytes(b"index")
            config = feature_collection_config_from_args(self._args())

            with patch("analysis.features.hub_upload.HfApi"):
                with self.assertRaisesRegex(FileNotFoundError, "feature_metadata.json"):
                    upload_circuit_tracer_features_to_hub(
                        repo_id="me/features",
                        output_dir=output_dir,
                        config=config,
                    )


if __name__ == "__main__":
    unittest.main()
