import json
import unittest
from dataclasses import asdict

import numpy as np
import torch

from analysis.evals import compute_token_metrics_onpolicy as token_metrics


class TokenMetricsOnPolicyTests(unittest.TestCase):
    def test_model_input_device_uses_input_embedding_device(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(4, 2, device="meta")

            def get_input_embeddings(self):
                return self.embedding

        self.assertEqual(token_metrics._model_input_device(TinyModel()), torch.device("meta"))

    def test_json_safe_converts_numpy_scalars_in_per_benchmark_metrics(self):
        output_data = {
            "per_benchmark": {
                "lmsys_chat": asdict(
                    token_metrics.BenchmarkMetrics(
                        n_samples=np.int64(1),
                        n_tokens=np.int64(263),
                        kl_mean=np.float32(1.1334),
                        top1_agreement=np.float32(0.5817),
                        n_interesting=np.int64(76),
                        kl_mean_interesting=np.float32(1.9861),
                        top1_agreement_interesting=np.float32(0.2237),
                    )
                )
            }
        }

        json.dumps(token_metrics._json_safe(output_data))


if __name__ == "__main__":
    unittest.main()
