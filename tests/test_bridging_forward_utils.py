import unittest

import torch
from transformers import Gemma2Config

from training.forward_utils import forward_mixed
from training.losses import compute_nmse_loss


class TinyLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, offset: float):
        super().__init__()
        self.proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(hidden_size))
        self.offset = offset

    def forward(self, hidden_states, **kwargs):
        return self.proj(hidden_states) + self.offset


class TinyBackbone(torch.nn.Module):
    def __init__(self, config: Gemma2Config, offset: float):
        super().__init__()
        self.config = config
        self.hidden_size_per_layer_input = 0
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = torch.nn.ModuleList([TinyLayer(config.hidden_size, offset)])
        self.norm = torch.nn.Identity()

    def rotary_emb(self, hidden_states, position_ids, layer_type=None):
        cos = torch.ones_like(hidden_states)
        sin = torch.zeros_like(hidden_states)
        return cos, sin


class TinyCausalLM(torch.nn.Module):
    def __init__(self, offset: float):
        super().__init__()
        self.config = Gemma2Config(
            vocab_size=17,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=16,
            final_logit_softcapping=None,
        )
        self.model = TinyBackbone(self.config, offset)
        self.lm_head = torch.nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens


class BridgingForwardUtilsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.input_ids = torch.tensor([[1, 2, 3, 4]])
        self.attention_mask = torch.ones_like(self.input_ids)

    def test_forward_mixed_uses_current_transformers_mask_api(self):
        model1 = TinyCausalLM(offset=0.1)
        model2 = TinyCausalLM(offset=0.2)

        logits = forward_mixed(model1, model2, self.input_ids, self.attention_mask, switch_layer=1)

        self.assertEqual(logits.shape, (1, 4, model2.config.vocab_size))
        self.assertTrue(torch.isfinite(logits).all())

    def test_compute_nmse_loss_uses_current_transformers_mask_api(self):
        model = TinyCausalLM(offset=0.1)
        ref_model = TinyCausalLM(offset=0.2)

        loss, layerwise = compute_nmse_loss(
            model,
            ref_model,
            self.input_ids,
            self.attention_mask,
            return_layerwise=True,
        )

        self.assertEqual(set(layerwise), {0, 1})
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.model.embed_tokens.weight.grad)


if __name__ == "__main__":
    unittest.main()
