import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from analysis.simple_load import simple_load


MODEL_PATH = "siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860"


class FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, return_tensors):
        return {"input_ids": torch.tensor([[1, 2]])}

    def decode(self, ids, skip_special_tokens=True):
        return "ok"


class FakeModel:
    device = "cpu"

    def generate(self, input_ids, **kwargs):
        return torch.tensor([[1, 2, 3]])

    def set_feature_steering(self, specs, mode):
        pass


class RejectingSteeringModel(FakeModel):
    def set_feature_steering(self, specs, mode):
        raise ValueError("Feature steering feature 999 out of range for layer 0 with 512 features")


class SimpleLoadCliTests(unittest.TestCase):
    def run_cli(self, argv, model=None):
        with (
            patch.object(simple_load.sys, "argv", argv),
            patch.object(simple_load.sys, "stdout", io.StringIO()),
            patch.object(simple_load, "setup_logging"),
            patch.object(simple_load, "load_model", return_value=(model or FakeModel(), FakeTokenizer())),
            patch.object(simple_load.logger, "info"),
        ):
            simple_load.main()

    def test_prompt_command_succeeds(self):
        self.run_cli([
            "analysis.simple_load.simple_load",
            MODEL_PATH,
            "--prompt=Hi",
        ])

    def test_load_model_applies_gemma4_checkpoint_load_kwargs(self):
        class FakeConfig:
            _name_or_path = "google/gemma-4-E2B-it"

        class FakeLoadedModel:
            def eval(self):
                pass

        class FakeModelCls:
            @classmethod
            def from_pretrained(cls, model_path, **kwargs):
                FakeModelCls.kwargs = kwargs
                return FakeLoadedModel()

        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=FakeConfig()),
            patch.object(simple_load, "get_transcoder_classes", return_value=(object, FakeModelCls)),
            patch.object(simple_load, "load_tokenizer", return_value=FakeTokenizer()),
        ):
            simple_load.load_model("checkpoint")

        self.assertEqual(
            FakeModelCls.kwargs["key_mapping"],
            {r"^model\.language_model\.": "model."},
        )

    def test_prompt_command_with_steering_succeeds(self):
        self.run_cli([
            "analysis.simple_load.simple_load",
            MODEL_PATH,
            "--prompt=Hi",
            "--steer",
            "11551209:4",
        ])

    def test_prompt_command_with_feature_data_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            features_dir = Path(tmp) / "features"
            features_dir.mkdir()
            with open(features_dir / "11551209.json", "w") as f:
                json.dump(
                    {
                        "examples_quantiles": [
                            {
                                "quantile_name": "Top activations",
                                "examples": [
                                    {
                                        "tokens": ["Hi", " there"],
                                        "tokens_acts_list": [0.0, 2.5],
                                        "train_token_ind": 1,
                                    }
                                ],
                            }
                        ]
                    },
                    f,
                )

            self.run_cli([
                "analysis.simple_load.simple_load",
                MODEL_PATH,
                "--prompt=Hi",
                "--steer",
                "11551209:4",
                "--feature-data",
                tmp,
            ])

    def test_prompt_command_with_activation_highlights_succeeds(self):
        with patch(
            "analysis.simple_load.activation_highlights.run_prompt_with_activation_highlights"
        ) as run_with_highlights:
            self.run_cli([
                "analysis.simple_load.simple_load",
                MODEL_PATH,
                "--prompt=Hi",
                "--steer",
                "11551209:4",
                "--show-steered-activations",
            ])

        run_with_highlights.assert_called_once()

    def test_invalid_steering_target_exits_with_error(self):
        argv = [
            "analysis.simple_load.simple_load",
            MODEL_PATH,
            "--prompt=Hi",
            "--steer",
            "999:4",
        ]

        with self.assertRaises(SystemExit) as exc:
            self.run_cli(argv, model=RejectingSteeringModel())

        self.assertEqual(exc.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
