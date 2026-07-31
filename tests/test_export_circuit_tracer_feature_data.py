import gzip
import json
import struct
import tempfile
import unittest
from pathlib import Path

from analysis.attribution.export_circuit_tracer_feature_data import (
    export_circuit_tracer_feature_data,
    infer_feature_shape,
)


class ExportCircuitTracerFeatureDataTest(unittest.TestCase):
    def test_exports_feature_data_to_binary_index_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_data_dir = root / "feature_data_run"
            features_dir = feature_data_dir / "features"
            output_dir = root / "packed"
            features_dir.mkdir(parents=True)

            metadata_features = []
            for layer in range(2):
                for feature in range(3):
                    cantor_id = (layer + feature) * (layer + feature + 1) // 2 + feature
                    metadata_features.append({
                        "layer": layer,
                        "feature": feature,
                        "cantor_id": cantor_id,
                    })
            (feature_data_dir / "feature_metadata.json").write_text(
                json.dumps({"features": metadata_features})
            )

            feature_payload = {
                "top_logits": [" yes"],
                "bottom_logits": [" no"],
                "act_min": 0.0,
                "act_max": 1.0,
                "examples_quantiles": [],
                "activation_frequency": 0.25,
                "layer": 1,
                "feature": 2,
            }
            cantor_id = (1 + 2) * (1 + 2 + 1) // 2 + 2
            (features_dir / f"{cantor_id}.json").write_text(json.dumps(feature_payload))

            export_circuit_tracer_feature_data(feature_data_dir, output_dir)

            with gzip.open(output_dir / "index.json.gz", "rt") as f:
                index = json.load(f)
            self.assertEqual(index["version"], "1.0")
            self.assertEqual(index["1"]["filename"], "layer_1.bin")

            start = index["1"]["offsets"][2]
            end = index["1"]["offsets"][3]
            raw = (output_dir / "layer_1.bin").read_bytes()[start:end]
            compressed_len = struct.unpack("<I", raw[:4])[0]
            packed_feature = json.loads(gzip.decompress(raw[4:4 + compressed_len]).decode("utf-8"))
            self.assertEqual(packed_feature["top_logits"], [" yes"])
            self.assertEqual(packed_feature["layer"], 1)
            self.assertEqual(packed_feature["feature"], 2)

    def test_existing_output_dir_raises_already_converted_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            feature_data_dir = Path(tmp) / "feature_data_run"
            (feature_data_dir / "features").mkdir(parents=True)
            output_dir = Path(tmp) / "packed"
            output_dir.mkdir()

            with self.assertRaisesRegex(FileExistsError, "already done the conversion"):
                export_circuit_tracer_feature_data(
                    feature_data_dir,
                    output_dir,
                    n_layers=1,
                    n_features=1,
                )

    def test_infers_shape_from_feature_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            feature_data_dir = Path(tmp) / "feature_data_run"
            feature_data_dir.mkdir()
            (feature_data_dir / "feature_metadata.json").write_text(json.dumps({
                "features": [
                    {"layer": 0, "feature": 0},
                    {"layer": 2, "feature": 4},
                ]
            }))

            self.assertEqual(infer_feature_shape(feature_data_dir), (3, 5))


if __name__ == "__main__":
    unittest.main()
