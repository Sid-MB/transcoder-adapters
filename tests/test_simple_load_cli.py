import io
import unittest
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

    def test_prompt_command_with_steering_succeeds(self):
        self.run_cli([
            "analysis.simple_load.simple_load",
            MODEL_PATH,
            "--prompt=Hi",
            "--steer",
            "11551209:4",
        ])

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
