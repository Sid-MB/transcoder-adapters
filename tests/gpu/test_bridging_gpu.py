import unittest
from types import SimpleNamespace

import torch
from transformers import Gemma2Config

from training.config import BridgingConfig, ExperimentConfig, TranscoderConfig
from training.train import train_step_bridging


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


class TinyTrainCausalLM(torch.nn.Module):
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

    def set_cache_features(self, enabled: bool):
        self.cache_features = enabled

    def collect_sparsity_loss(self):
        return torch.zeros((), device=self.get_input_embeddings().weight.device)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.model.embed_tokens(input_ids)
        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        for layer in self.model.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                cache_position=position_ids.squeeze(0),
                shared_kv_states={},
                past_key_values=None,
            )
        logits = self.lm_head(self.model.norm(hidden_states))
        return SimpleNamespace(logits=logits)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
class BridgingGpuTests(unittest.TestCase):
    def test_train_step_bridging_runs_on_cuda(self):
        torch.manual_seed(0)
        device = torch.device("cuda")
        model = TinyTrainCausalLM(offset=0.1).to(device)
        ref_model = TinyTrainCausalLM(offset=0.2).to(device)
        for parameter in ref_model.parameters():
            parameter.requires_grad_(False)

        config = ExperimentConfig(
            transcoder=TranscoderConfig(l1_weight=0.0),
            bridging=BridgingConfig(
                loss_type="kl",
                n_cutoffs=2,
                sampling=[0, 1],
                always_include_adapt_only=True,
                lambda_adapt=1.0,
                lambda_bridge=1.0,
                lambda_nmse=1.0,
            ),
            batch_size=1,
            micro_batch_size=1,
            use_wandb=False,
        )
        input_ids = torch.tensor([[1, 2, 3, 4]], device=device)
        batch = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone(),
        }

        metrics = train_step_bridging(
            model,
            ref_model,
            batch,
            config,
            global_step=0,
            total_steps=1,
            gradient_accumulation_steps=1,
        )

        self.assertIn("bridging/nmse", metrics)
        self.assertIn("bridging/bridge", metrics)
        grads = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        self.assertTrue(any(grad is not None and torch.isfinite(grad).all() for grad in grads))


if __name__ == "__main__":
    unittest.main()
