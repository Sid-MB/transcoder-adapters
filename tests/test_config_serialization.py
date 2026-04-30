import os
import tempfile
import unittest

import yaml

from training.config import (
    BridgingConfig,
    DatasetEntryConfig,
    ExperimentConfig,
    LengthExcessionBehavior,
    TranscoderConfig,
    load_config,
    save_config,
)


class ConfigSerializationTests(unittest.TestCase):
    def test_save_config_writes_yaml_safe_enum_values(self):
        config = ExperimentConfig(
            model_name="google/gemma-4-E2B",
            model_arch="gemma4",
            transcoder=TranscoderConfig(),
            bridging=BridgingConfig(reference_model_path="google/gemma-4-E2B-it"),
            datasets=[
                DatasetEntryConfig(
                    type="fineweb",
                    datapath="science-of-finetuning/fineweb-1m-sample",
                    length_excession_behavior=LengthExcessionBehavior.TRUNCATE,
                )
            ],
            output_dir="/tmp/transcoder-adapters-test",
            wandb_run_name="serialization-test",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "train_config.yaml")
            save_config(config, path)
            with open(path) as f:
                text = f.read()
            reloaded = load_config(path)

        self.assertNotIn("!!python", text)
        parsed = yaml.safe_load(text)
        self.assertEqual(parsed["datasets"][0]["length_excession_behavior"], "truncate")
        self.assertEqual(reloaded.datasets[0].length_excession_behavior, LengthExcessionBehavior.TRUNCATE)

    def test_load_config_accepts_legacy_enum_tag(self):
        legacy_yaml = """
model_arch: gemma4
output_dir: /tmp/transcoder-adapters-test
wandb_run_name: legacy-config
datasets:
- type: fineweb
  datapath: science-of-finetuning/fineweb-1m-sample
  length_excession_behavior: !!python/object/apply:training.config.LengthExcessionBehavior
  - truncate
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "legacy_config.yaml")
            with open(path, "w") as f:
                f.write(legacy_yaml)

            config = load_config(path)

        self.assertEqual(config.datasets[0].length_excession_behavior, LengthExcessionBehavior.TRUNCATE)


if __name__ == "__main__":
    unittest.main()
