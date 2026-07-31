import gzip
import json
import struct
import tempfile
import unittest
from pathlib import Path

from analysis.features.collect_feature_activations import (
    ActivatingExample,
    FeatureCollector,
    export_circuit_tracer_json,
)


class TinyTokenizer:
    def decode(self, token_ids):
        return f"tok{token_ids[0]}"


class CollectFeatureDirectPackTest(unittest.TestCase):
    def test_exports_packed_circuit_tracer_features_during_json_export(self):
        collector = FeatureCollector(n_layers=1, n_features=3)
        collector.total_tokens = 10
        stats = collector.stats[0][1]
        stats.activation_count = 2
        stats.top_k_examples.append(
            ActivatingExample(
                activation=1.5,
                token_id=11,
                position=1,
                context_tokens=[10, 11, 12],
                context_activations=[0.0, 1.5, 0.2],
                position_in_context=1,
                domain="chat",
                region="answer",
                thinking_position=None,
                sequence_idx=0,
            )
        )

        logit_lens_data = [{
            "top_ids": [[101], [102], [103]],
            "bot_ids": [[201], [202], [203]],
        }]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            export_circuit_tracer_json(
                collector,
                logit_lens_data,
                TinyTokenizer(),
                output_dir,
                n_workers=1,
                export_circuit_tracer_features=True,
            )

            self.assertTrue((output_dir / "features" / "2.json").exists())
            packed_dir = output_dir / "circuit_tracer_features"
            with gzip.open(packed_dir / "index.json.gz", "rt") as f:
                index = json.load(f)

            offsets = index["0"]["offsets"]
            self.assertEqual(index["0"]["filename"], "layer_0.bin")
            self.assertEqual(len(offsets), 4)
            self.assertEqual(offsets[0], offsets[1])
            self.assertLess(offsets[1], offsets[2])
            self.assertEqual(offsets[2], offsets[3])

            raw = (packed_dir / "layer_0.bin").read_bytes()[offsets[1]:offsets[2]]
            compressed_len = struct.unpack("<I", raw[:4])[0]
            feature = json.loads(gzip.decompress(raw[4:4 + compressed_len]).decode("utf-8"))
            self.assertEqual(feature["layer"], 0)
            self.assertEqual(feature["feature"], 1)
            self.assertEqual(feature["top_logits"], ["tok102"])
            self.assertEqual(feature["activation_frequency"], 0.2)


if __name__ == "__main__":
    unittest.main()
