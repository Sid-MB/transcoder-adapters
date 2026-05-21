import inspect
import unittest

import torch
from transformers import Gemma2Config

from models.gemma2_transcoder import Gemma2ForCausalLMWithTranscoder
from training.config import BridgingConfig, ExperimentConfig, TranscoderConfig
from training.forward_utils import forward_mixed
from training.losses import compute_nmse_loss
from training.train import train_epoch, train_step_bridging


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


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
class Gemma2RealComponentGpuTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.device = torch.device("cuda")
        self.input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=self.device)
        self.attention_mask = torch.ones_like(self.input_ids)
        self.labels = self.input_ids.clone()

    def _model_pair(self):
        model = tiny_gemma2_model().to(self.device)
        ref_model = tiny_gemma2_model().to(self.device)
        for parameter in ref_model.parameters():
            parameter.requires_grad_(False)
        return model, ref_model

    def test_real_gemma2_rotary_signature_matches_upstream_api(self):
        model = tiny_gemma2_model().to(self.device)

        self.assertNotIn("layer_type", inspect.signature(model.model.rotary_emb.forward).parameters)
        embeddings = model.model.embed_tokens(self.input_ids)
        position_ids = torch.arange(self.input_ids.shape[1], device=self.device).unsqueeze(0)
        cos, sin = model.model.rotary_emb(embeddings, position_ids)

        self.assertEqual(cos.shape[:2], embeddings.shape[:2])
        self.assertEqual(sin.shape[:2], embeddings.shape[:2])
        self.assertEqual(cos.shape[-1], model.config.head_dim)
        self.assertEqual(sin.shape[-1], model.config.head_dim)
        self.assertTrue(torch.isfinite(cos).all())
        self.assertTrue(torch.isfinite(sin).all())

    def test_forward_mixed_real_gemma2_all_cutoffs(self):
        model, ref_model = self._model_pair()

        for switch_layer in range(len(model.model.layers) + 1):
            with self.subTest(switch_layer=switch_layer):
                logits = forward_mixed(
                    model,
                    ref_model,
                    self.input_ids,
                    self.attention_mask,
                    switch_layer=switch_layer,
                )

                self.assertEqual(logits.shape, (1, self.input_ids.shape[1], model.config.vocab_size))
                self.assertEqual(logits.device.type, "cuda")
                self.assertTrue(torch.isfinite(logits).all())

    def test_compute_nmse_loss_real_gemma2_has_layerwise_cuda_gradients(self):
        model, ref_model = self._model_pair()

        loss, layerwise = compute_nmse_loss(
            model,
            ref_model,
            self.input_ids,
            self.attention_mask,
            return_layerwise=True,
        )
        loss.backward()

        self.assertEqual(set(layerwise), {0, 1, 2})
        self.assertEqual(loss.device.type, "cuda")
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)
        model_grads = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        ref_grads = [parameter.grad for parameter in ref_model.parameters()]
        self.assertTrue(any(grad is not None and torch.isfinite(grad).all() for grad in model_grads))
        self.assertTrue(all(grad is None for grad in ref_grads))

    def test_train_step_bridging_real_gemma2_runs_complete_cuda_step(self):
        model, ref_model = self._model_pair()
        config = ExperimentConfig(
            transcoder=TranscoderConfig(n_features=8, dec_bias=False, l1_weight=0.0),
            bridging=BridgingConfig(
                loss_type="kl",
                n_cutoffs=3,
                sampling=[0, 1, 2],
                always_include_adapt_only=True,
                lambda_adapt=1.0,
                lambda_bridge=1.0,
                lambda_nmse=1.0,
            ),
            batch_size=1,
            micro_batch_size=1,
            use_wandb=False,
        )
        batch = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "labels": self.labels,
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

        expected_keys = {
            "train/kl_to_ref",
            "train/lm_loss",
            "bridging/adapt_only",
            "train/sparsity",
            "bridging/nmse",
            "bridging_layerwise/nmse_layer_0",
            "bridging_layerwise/nmse_layer_1",
            "bridging_layerwise/nmse_layer_2",
            "bridging_layerwise/r2a_kl_layer_0",
            "bridging_layerwise/a2r_kl_layer_1",
            "bridging_layerwise/r2a_kl_layer_1",
            "bridging_layerwise/a2r_kl_layer_2",
            "bridging/bridge",
            "bridging/total",
            "train/total_loss",
        }
        self.assertTrue(expected_keys.issubset(metrics))
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))

        model_grads = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        ref_grads = [parameter.grad for parameter in ref_model.parameters()]
        self.assertTrue(any(grad is not None and torch.isfinite(grad).all() for grad in model_grads))
        self.assertTrue(all(grad is None for grad in ref_grads))

    def test_train_epoch_real_gemma2_one_batch_without_data_loading(self):
        model, ref_model = self._model_pair()
        config = ExperimentConfig(
            transcoder=TranscoderConfig(n_features=8, dec_bias=False, l1_weight=0.0),
            bridging=BridgingConfig(
                loss_type="kl",
                n_cutoffs=3,
                sampling=[0, 1, 2],
                always_include_adapt_only=True,
                lambda_adapt=1.0,
                lambda_bridge=1.0,
                lambda_nmse=1.0,
            ),
            batch_size=1,
            micro_batch_size=1,
            gradient_clip_norm=1.0,
            use_wandb=False,
            debug_mode=True,
            save_checkpoints=False,
        )
        batch = {
            "input_ids": self.input_ids.detach().cpu(),
            "attention_mask": self.attention_mask.detach().cpu(),
            "labels": self.labels.detach().cpu(),
        }
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

        epoch_loss, current_step, total_samples_seen = train_epoch(
            model,
            ref_model,
            tokenizer=None,
            dataloader=[batch],
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            epoch=0,
            starting_step=0,
            total_steps=1,
            total_samples_seen=0,
            val_dataloader=None,
            token_metrics_examples=None,
        )

        self.assertEqual(current_step, 1)
        self.assertEqual(total_samples_seen, 1)
        self.assertTrue(torch.isfinite(torch.tensor(epoch_loss)))
        self.assertGreaterEqual(epoch_loss, 0.0)


if __name__ == "__main__":
    unittest.main()
