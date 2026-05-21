import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analysis.attribution.run_circuit_tracer_pipeline import (
    ensure_feature_data_conversion,
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
                feature_data_dir=str(root / "feature_data"),
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
            self.assertEqual(attribution_args.scan, str(feature_output_dir))
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


if __name__ == "__main__":
    unittest.main()
