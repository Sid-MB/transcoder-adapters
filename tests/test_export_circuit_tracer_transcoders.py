import json
import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file

from analysis.attribution.export_circuit_tracer_transcoders import (
    export_circuit_tracer_transcoders,
)


class ExportCircuitTracerTranscodersTest(unittest.TestCase):
    def test_exports_sharded_checkpoint_to_circuit_tracer_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            checkpoint = tmp_path / "checkpoint"
            output_dir = tmp_path / "exported"
            checkpoint.mkdir()

            tensors = {}
            weight_map = {}
            for layer in range(2):
                prefix = f"model.layers.{layer}.mlp"
                tensors[f"{prefix}.transcoder_enc.weight"] = torch.arange(
                    12, dtype=torch.float32
                ).reshape(3, 4) + layer
                tensors[f"{prefix}.transcoder_enc.bias"] = torch.arange(3, dtype=torch.float32) + layer
                tensors[f"{prefix}.transcoder_dec.weight"] = torch.arange(
                    12, dtype=torch.float32
                ).reshape(4, 3) + 10 + layer
                tensors[f"{prefix}.transcoder_dec.bias"] = torch.arange(4, dtype=torch.float32) + 20 + layer
                for name in tensors:
                    weight_map[name] = "model-00001-of-00001.safetensors"

            save_file(tensors, checkpoint / "model-00001-of-00001.safetensors")
            (checkpoint / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": weight_map, "metadata": {}})
            )
            (checkpoint / "config.json").write_text(json.dumps({"model_type": "gemma2"}))
            (checkpoint / "training-config.yaml").write_text("model_name: google/gemma-2-2b\n")

            export_circuit_tracer_transcoders(str(checkpoint), output_dir)

            config = yaml.safe_load((output_dir / "config.yaml").read_text())
            self.assertEqual(config["model_name"], "google/gemma-2-2b")
            self.assertEqual(config["model_kind"], "transcoder_set")
            self.assertEqual(config["activation"], "relu")

            with safe_open(output_dir / "layer_1.safetensors", framework="pt", device="cpu") as f:
                self.assertEqual(set(f.keys()), {"W_enc", "W_dec", "b_enc", "b_dec"})
                self.assertTrue(
                    torch.equal(f.get_tensor("W_enc"), tensors["model.layers.1.mlp.transcoder_enc.weight"])
                )
                self.assertTrue(
                    torch.equal(
                        f.get_tensor("W_dec"),
                        tensors["model.layers.1.mlp.transcoder_dec.weight"].T.contiguous(),
                    )
                )
                self.assertTrue(
                    torch.equal(f.get_tensor("b_enc"), tensors["model.layers.1.mlp.transcoder_enc.bias"])
                )
                self.assertTrue(
                    torch.equal(f.get_tensor("b_dec"), tensors["model.layers.1.mlp.transcoder_dec.bias"])
                )

    def test_existing_output_dir_raises_already_converted_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "exported"
            output_dir.mkdir()

            with self.assertRaisesRegex(FileExistsError, "already done the conversion"):
                export_circuit_tracer_transcoders("checkpoint", output_dir)


if __name__ == "__main__":
    unittest.main()
