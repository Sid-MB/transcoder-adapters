import unittest
from types import SimpleNamespace

from training.upload_models.hub import _build_model_card


class TrainingHubModelCardTest(unittest.TestCase):
    def test_model_card_includes_evaluation_stats(self):
        config = SimpleNamespace(
            model_name="org/base",
            model_arch="gemma",
            wandb_run_name="run",
            direct=None,
            bridging=SimpleNamespace(
                reference_model_path="org/ref",
                loss_type="kl",
                lambda_adapt=1.0,
                lambda_bridge=1.0,
                lambda_nmse=0.0,
                n_cutoffs=2,
                backbone="base",
            ),
            transcoder=SimpleNamespace(n_features=16, dec_bias=True, l1_weight=0.001),
            learning_rate=0.001,
            batch_size=2,
            num_epochs=1,
            warmup_ratio=0.1,
            datasets=[],
        )

        card = _build_model_card(
            config,
            "me/model",
            evaluation_stats={
                "token_metrics_final": {
                    "kl_mean": 0.123456789,
                    "top1_agreement": 0.75,
                }
            },
        )

        content = str(card)
        self.assertIn("## Evaluation", content)
        self.assertIn("token_metrics_final/kl_mean", content)
        self.assertIn("0.123457", content)
        self.assertIn("token_metrics_final/top1_agreement", content)


if __name__ == "__main__":
    unittest.main()
