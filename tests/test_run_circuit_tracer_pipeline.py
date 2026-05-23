import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analysis.attribution.prepare_circuit_tracer_assets import prepare_assets
from analysis.attribution.run_circuit_tracer_pipeline import (
    LOCAL_FEATURE_SCAN,
    ensure_feature_data_conversion,
    is_hf_feature_ref,
    normalize_hf_feature_ref,
    run_pipeline,
)


class RunCircuitTracerPipelineTest(unittest.TestCase):
    def test_skips_existing_conversions_and_runs_attribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcoder_output_dir = root / "transcoders"
            feature_output_dir = root / "features"
            graph_output_dir = root / "graphs"
            prompts = root / "prompts"
            transcoder_output_dir.mkdir()
            feature_output_dir.mkdir()
            (transcoder_output_dir / "config.yaml").write_text("model_name: base\n")
            (feature_output_dir / "index.json.gz").write_bytes(b"")
            prompts.mkdir()

            args = argparse.Namespace(
                transcoder_model_path="org/model",
                base_model="google/gemma-2-2b",
                prompts=prompts,
                feature_data_path=str(root / "feature_data"),
                run_name="run",
                transcoder_output_dir=transcoder_output_dir,
                feature_output_dir=feature_output_dir,
                graph_output_dir=graph_output_dir,
                prompt_format="raw",
                max_n_logits=2,
                batch_size=4,
                max_feature_nodes=8,
                node_threshold=0.7,
                edge_threshold=0.9,
                device="cuda",
                device_map=None,
                auto_shard_gpus=False,
                n_layers=None,
                n_features=None,
                feature_input_hook="ln2.hook_normalized",
                feature_output_hook="hook_mlp_out",
                activation="relu",
                port=8123,
                serve=False,
            )

            with (
                patch("analysis.attribution.run_circuit_tracer_pipeline.export_circuit_tracer_transcoders") as export_transcoders,
                patch("analysis.attribution.run_circuit_tracer_pipeline.export_circuit_tracer_feature_data") as export_features,
                patch("analysis.attribution.run_circuit_tracer_pipeline.run_attribution", return_value={"p": "skipped"}) as run_attribution,
            ):
                result = run_pipeline(args)

            self.assertEqual(result, {"p": "skipped"})
            export_transcoders.assert_not_called()
            export_features.assert_not_called()
            attribution_args = run_attribution.call_args.args[0]
            self.assertEqual(attribution_args.scan, LOCAL_FEATURE_SCAN)
            self.assertEqual(attribution_args.features_dir, str(feature_output_dir))
            self.assertEqual(attribution_args.output_dir, graph_output_dir)
            self.assertFalse(attribution_args.serve)

    def test_existing_incomplete_feature_conversion_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "features"
            output_dir.mkdir()

            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                ensure_feature_data_conversion(
                    str(Path(tmp) / "feature_data"),
                    output_dir,
                    n_layers=1,
                    n_features=1,
                )

    def test_auto_detects_collected_packed_feature_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            feature_data_path = Path(tmp) / "feature_data"
            packed_dir = feature_data_path / "circuit_tracer_features"
            packed_dir.mkdir(parents=True)
            (packed_dir / "index.json.gz").write_bytes(b"")

            with patch("analysis.attribution.run_circuit_tracer_pipeline.export_circuit_tracer_feature_data") as export_features:
                result = ensure_feature_data_conversion(
                    str(feature_data_path),
                    None,
                    n_layers=1,
                    n_features=1,
                )

            self.assertEqual(result, packed_dir)
            export_features.assert_not_called()

    def test_explicit_feature_output_dir_overrides_auto_detected_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_data_path = root / "feature_data"
            auto_packed_dir = feature_data_path / "circuit_tracer_features"
            explicit_output_dir = root / "explicit_features"
            auto_packed_dir.mkdir(parents=True)
            (auto_packed_dir / "index.json.gz").write_bytes(b"")
            explicit_output_dir.mkdir()
            (explicit_output_dir / "index.json.gz").write_bytes(b"")

            with patch("analysis.attribution.run_circuit_tracer_pipeline.export_circuit_tracer_feature_data") as export_features:
                result = ensure_feature_data_conversion(
                    str(feature_data_path),
                    explicit_output_dir,
                    n_layers=1,
                    n_features=1,
                )

            self.assertEqual(result, explicit_output_dir)
            export_features.assert_not_called()

    def test_pipeline_passes_auto_detected_feature_cache_to_attribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcoder_output_dir = root / "transcoders"
            graph_output_dir = root / "graphs"
            feature_data_path = root / "feature_data"
            packed_dir = feature_data_path / "circuit_tracer_features"
            prompts = root / "prompts"
            transcoder_output_dir.mkdir()
            packed_dir.mkdir(parents=True)
            prompts.mkdir()
            (transcoder_output_dir / "config.yaml").write_text("model_name: base\n")
            (packed_dir / "index.json.gz").write_bytes(b"")

            args = argparse.Namespace(
                transcoder_model_path="org/model",
                base_model="google/gemma-2-2b",
                prompts=prompts,
                feature_data_path=str(feature_data_path),
                run_name="run",
                transcoder_output_dir=transcoder_output_dir,
                feature_output_dir=None,
                graph_output_dir=graph_output_dir,
                prompt_format="raw",
                max_n_logits=2,
                batch_size=4,
                max_feature_nodes=8,
                node_threshold=0.7,
                edge_threshold=0.9,
                device="cuda",
                device_map=None,
                auto_shard_gpus=False,
                n_layers=None,
                n_features=None,
                feature_input_hook="ln2.hook_normalized",
                feature_output_hook="hook_mlp_out",
                activation="relu",
                port=8123,
                serve=False,
            )

            with (
                patch("analysis.attribution.run_circuit_tracer_pipeline.export_circuit_tracer_transcoders") as export_transcoders,
                patch("analysis.attribution.run_circuit_tracer_pipeline.export_circuit_tracer_feature_data") as export_features,
                patch("analysis.attribution.run_circuit_tracer_pipeline.run_attribution", return_value={"p": "skipped"}) as run_attribution,
            ):
                result = run_pipeline(args)

            self.assertEqual(result, {"p": "skipped"})
            export_transcoders.assert_not_called()
            export_features.assert_not_called()
            attribution_args = run_attribution.call_args.args[0]
            self.assertEqual(attribution_args.scan, LOCAL_FEATURE_SCAN)
            self.assertEqual(attribution_args.features_dir, str(packed_dir))

    def test_hf_feature_repo_skips_conversion_and_uses_repo_as_scan(self):
        with patch("analysis.attribution.run_circuit_tracer_pipeline.export_circuit_tracer_feature_data") as export_features:
            result = ensure_feature_data_conversion(
                "org/feature-repo",
                None,
                n_layers=1,
                n_features=1,
            )

        self.assertEqual(result, "org/feature-repo")
        export_features.assert_not_called()

    def test_hf_feature_repo_rejects_local_feature_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "feature_output_dir"):
                ensure_feature_data_conversion(
                    "org/feature-repo",
                    Path(tmp) / "features",
                    n_layers=1,
                    n_features=1,
                )

    def test_hf_feature_refs_are_normalized_for_circuit_tracer_scan(self):
        self.assertTrue(is_hf_feature_ref("org/feature-repo"))
        self.assertTrue(is_hf_feature_ref("hf://org/feature-repo"))
        self.assertTrue(is_hf_feature_ref("https://huggingface.co/org/feature-repo"))
        self.assertFalse(is_hf_feature_ref("/tmp/feature_data"))
        self.assertEqual(normalize_hf_feature_ref("hf://org/feature-repo"), "org/feature-repo")
        self.assertEqual(
            normalize_hf_feature_ref("https://huggingface.co/org/feature-repo"),
            "org/feature-repo",
        )

    def test_pipeline_passes_hf_feature_repo_to_attribution_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcoder_output_dir = root / "transcoders"
            graph_output_dir = root / "graphs"
            prompts = root / "prompts"
            transcoder_output_dir.mkdir()
            prompts.mkdir()
            (transcoder_output_dir / "config.yaml").write_text("model_name: base\n")

            args = argparse.Namespace(
                transcoder_model_path="org/model",
                base_model="google/gemma-2-2b",
                prompts=prompts,
                feature_data_path="org/feature-repo",
                run_name="run",
                transcoder_output_dir=transcoder_output_dir,
                feature_output_dir=None,
                graph_output_dir=graph_output_dir,
                prompt_format="raw",
                max_n_logits=2,
                batch_size=4,
                max_feature_nodes=8,
                node_threshold=0.7,
                edge_threshold=0.9,
                device="cuda",
                device_map=None,
                auto_shard_gpus=False,
                n_layers=None,
                n_features=None,
                feature_input_hook="ln2.hook_normalized",
                feature_output_hook="hook_mlp_out",
                activation="relu",
                port=8123,
                serve=False,
            )

            with (
                patch("analysis.attribution.run_circuit_tracer_pipeline.export_circuit_tracer_transcoders") as export_transcoders,
                patch("analysis.attribution.run_circuit_tracer_pipeline.export_circuit_tracer_feature_data") as export_features,
                patch("analysis.attribution.run_circuit_tracer_pipeline.run_attribution", return_value={"p": "skipped"}) as run_attribution,
            ):
                result = run_pipeline(args)

            self.assertEqual(result, {"p": "skipped"})
            export_transcoders.assert_not_called()
            export_features.assert_not_called()
            attribution_args = run_attribution.call_args.args[0]
            self.assertEqual(attribution_args.scan, "org/feature-repo")
            self.assertIsNone(attribution_args.features_dir)

    def test_prepare_manifest_uses_short_scan_and_real_features_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcoder_output_dir = root / "transcoders"
            graph_output_dir = root / "graphs"
            feature_data_path = root / "feature_data"
            packed_dir = feature_data_path / "circuit_tracer_features"
            prompts = root / "prompts"
            manifest_path = root / "manifest.json"
            transcoder_output_dir.mkdir()
            packed_dir.mkdir(parents=True)
            prompts.mkdir()
            (transcoder_output_dir / "config.yaml").write_text("model_name: base\n")
            (packed_dir / "index.json.gz").write_bytes(b"")

            args = argparse.Namespace(
                transcoder_model_path="org/model",
                base_model="google/gemma-2-2b",
                prompts=prompts,
                feature_data_path=str(feature_data_path),
                run_name="run",
                transcoder_output_dir=transcoder_output_dir,
                feature_output_dir=None,
                graph_output_dir=graph_output_dir,
                manifest_path=manifest_path,
                n_layers=None,
                n_features=None,
                feature_input_hook="ln2.hook_normalized",
                feature_output_hook="hook_mlp_out",
                activation="relu",
            )

            with (
                patch("analysis.attribution.run_circuit_tracer_pipeline.export_circuit_tracer_transcoders") as export_transcoders,
                patch("analysis.attribution.run_circuit_tracer_pipeline.export_circuit_tracer_feature_data") as export_features,
            ):
                manifest = prepare_assets(args)

            export_transcoders.assert_not_called()
            export_features.assert_not_called()
            self.assertEqual(manifest["scan"], LOCAL_FEATURE_SCAN)
            self.assertEqual(manifest["features_dir"], str(packed_dir))
            written_manifest = json.loads(manifest_path.read_text())
            self.assertEqual(written_manifest["scan"], LOCAL_FEATURE_SCAN)
            self.assertEqual(written_manifest["features_dir"], str(packed_dir))

    def test_prepare_manifest_uses_hf_feature_repo_as_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcoder_output_dir = root / "transcoders"
            graph_output_dir = root / "graphs"
            prompts = root / "prompts"
            manifest_path = root / "manifest.json"
            transcoder_output_dir.mkdir()
            prompts.mkdir()
            (transcoder_output_dir / "config.yaml").write_text("model_name: base\n")

            args = argparse.Namespace(
                transcoder_model_path="org/model",
                base_model="google/gemma-2-2b",
                prompts=prompts,
                feature_data_path="hf://org/feature-repo",
                run_name="run",
                transcoder_output_dir=transcoder_output_dir,
                feature_output_dir=None,
                graph_output_dir=graph_output_dir,
                manifest_path=manifest_path,
                n_layers=None,
                n_features=None,
                feature_input_hook="ln2.hook_normalized",
                feature_output_hook="hook_mlp_out",
                activation="relu",
            )

            with (
                patch("analysis.attribution.run_circuit_tracer_pipeline.export_circuit_tracer_transcoders") as export_transcoders,
                patch("analysis.attribution.run_circuit_tracer_pipeline.export_circuit_tracer_feature_data") as export_features,
            ):
                manifest = prepare_assets(args)

            export_transcoders.assert_not_called()
            export_features.assert_not_called()
            self.assertEqual(manifest["scan"], "org/feature-repo")
            self.assertIsNone(manifest["features_dir"])
            self.assertIsNone(manifest["feature_output_dir"])
            written_manifest = json.loads(manifest_path.read_text())
            self.assertEqual(written_manifest["scan"], "org/feature-repo")
            self.assertIsNone(written_manifest["features_dir"])


if __name__ == "__main__":
    unittest.main()
