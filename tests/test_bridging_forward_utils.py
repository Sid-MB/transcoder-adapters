import unittest

import torch
from transformers import Gemma2Config

from models.gemma2_transcoder import Gemma2ForCausalLMWithTranscoder
from training.forward_utils import forward_mixed
from training.losses import compute_nmse_loss


def tiny_gemma2_config() -> Gemma2Config:
    config = Gemma2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        sliding_window=8,
        final_logit_softcapping=None,
        attn_logit_softcapping=None,
    )
    config.transcoder_n_features = 8
    config.transcoder_dec_bias = False
    config.architectures = ["Gemma2ForCausalLMWithTranscoder"]
    return config


def tiny_gemma2_model() -> Gemma2ForCausalLMWithTranscoder:
    return Gemma2ForCausalLMWithTranscoder(tiny_gemma2_config())


class BridgingForwardUtilsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        self.attention_mask = torch.ones_like(self.input_ids)

    def test_forward_mixed_uses_real_gemma2_components(self):
        model1 = tiny_gemma2_model()
        model2 = tiny_gemma2_model()

        for switch_layer in range(len(model1.model.layers) + 1):
            with self.subTest(switch_layer=switch_layer):
                logits = forward_mixed(model1, model2, self.input_ids, self.attention_mask, switch_layer=switch_layer)

                self.assertEqual(logits.shape, (1, self.input_ids.shape[1], model2.config.vocab_size))
                self.assertTrue(torch.isfinite(logits).all())

    def test_compute_nmse_loss_uses_real_gemma2_components(self):
        model = tiny_gemma2_model()
        ref_model = tiny_gemma2_model()
        for parameter in ref_model.parameters():
            parameter.requires_grad_(False)

        loss, layerwise = compute_nmse_loss(
            model,
            ref_model,
            self.input_ids,
            self.attention_mask,
            return_layerwise=True,
        )

        self.assertEqual(set(layerwise), {0, 1, 2})
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.model.embed_tokens.weight.grad)
        self.assertTrue(all(parameter.grad is None for parameter in ref_model.parameters()))


if __name__ == "__main__":
    unittest.main()
