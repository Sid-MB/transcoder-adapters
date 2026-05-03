import tempfile
import unittest
from unittest.mock import patch

from safetensors import safe_open
from transformers.modeling_utils import LoadStateDictInfo

import models
from models.auto import AutoModelForCausalLMWithTranscoder
from models.auto import _base_tokenizer_for


class ModelRegistryTests(unittest.TestCase):
    def tearDown(self):
        models._REGISTRY.clear()

    def test_detect_gemma4_validates_backend(self):
        self.assertEqual(models.detect_architecture("google/gemma-4-E2B-it"), "gemma4")
        self.assertIn("gemma4", models._REGISTRY)

    def test_gemma4_text_alias_loads_gemma4_classes_from_checkpoint_model_type(self):
        self.assertEqual(models.canonical_architecture("gemma4_text"), "gemma4")
        self.assertIs(
            models.get_transcoder_classes_for_model_type("gemma4_text"),
            models.get_transcoder_classes("gemma4"),
        )

    def test_gemma4_text_is_rejected_as_user_architecture(self):
        with self.assertRaisesRegex(ValueError, "Use model_arch: 'gemma4'"):
            models.get_transcoder_classes("gemma4_text")

    def test_available_architectures_only_lists_user_architectures(self):
        available = models.available_architectures()
        self.assertIn("gemma4", available)
        self.assertNotIn("gemma4_text", available)

    def test_gemma4_text_tokenizer_fallback_uses_gemma4_base(self):
        self.assertEqual(_base_tokenizer_for("gemma4_text"), _base_tokenizer_for("gemma4"))

    def test_gemma4_checkpoint_load_kwargs_remap_language_model_keys(self):
        self.assertEqual(
            models.checkpoint_load_kwargs_for_model_type("gemma4"),
            {"key_mapping": {r"^model\.language_model\.": "model."}},
        )
        self.assertEqual(
            models.checkpoint_load_kwargs_for_model_type("gemma4_text"),
            {"key_mapping": {r"^model\.language_model\.": "model."}},
        )

    def test_non_gemma4_checkpoint_load_kwargs_are_empty(self):
        self.assertEqual(models.checkpoint_load_kwargs_for_model_type("gemma2"), {})
        self.assertEqual(models.checkpoint_load_kwargs_for_model_type("qwen2"), {})

    def test_auto_loader_applies_gemma4_checkpoint_load_kwargs(self):
        class FakeConfig:
            model_type = "gemma4_text"

        class FakeModel:
            @classmethod
            def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
                return kwargs

        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=FakeConfig()),
            patch("models.get_transcoder_classes_for_model_type", return_value=(object, FakeModel)),
        ):
            kwargs = AutoModelForCausalLMWithTranscoder.from_pretrained("checkpoint")

        self.assertEqual(kwargs["key_mapping"], {r"^model\.language_model\.": "model."})

    def test_gemma4_transcoder_uses_text_model_backbone(self):
        ConfigCls, ModelCls = models.get_transcoder_classes("gemma4")
        config = ConfigCls(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            transcoder_n_features=4,
        )
        model = ModelCls(config)

        self.assertTrue(hasattr(model.model, "layers"))
        self.assertFalse(hasattr(model.model, "language_model"))

    def test_saved_checkpoint_key_format_for_all_transcoder_models(self):
        configs = {
            "gemma2": dict(
                vocab_size=32,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=4,
                transcoder_n_features=4,
            ),
            "gemma4": dict(
                vocab_size=32,
                vocab_size_per_layer_input=32,
                hidden_size=8,
                hidden_size_per_layer_input=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=4,
                global_head_dim=4,
                transcoder_n_features=4,
            ),
            "qwen2": dict(
                vocab_size=32,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                transcoder_n_features=4,
            ),
        }

        for arch, kwargs in configs.items():
            with self.subTest(arch=arch):
                ConfigCls, ModelCls = models.get_transcoder_classes(arch)
                model = ModelCls(ConfigCls(**kwargs))

                with tempfile.TemporaryDirectory() as tmp:
                    model.save_pretrained(tmp)
                    with safe_open(f"{tmp}/model.safetensors", framework="pt") as f:
                        keys = set(f.keys())

                self.assertIn("model.layers.0.mlp.transcoder_enc.weight", keys)
                self.assertIn("model.layers.0.mlp.transcoder_dec.weight", keys)
                self.assertFalse(
                    any(key.startswith("model.language_model.") for key in keys),
                    f"{arch} saved nested language_model keys",
                )

    def test_gemma4_loader_ignores_unused_multimodal_checkpoint_keys(self):
        ConfigCls, ModelCls = models.get_transcoder_classes("gemma4")
        config = ConfigCls(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            transcoder_n_features=4,
        )
        model = ModelCls(config)
        loading_info = LoadStateDictInfo(
            missing_keys={
                "model.layers.0.mlp.transcoder_enc.weight",
                "model.layers.0.mlp.transcoder_dec.weight",
            },
            unexpected_keys={
                "model.audio_tower.layers.0.self_attn.q_proj.weight",
                "model.embed_audio.embedding_projection.weight",
                "model.embed_vision.embedding_projection.weight",
                "model.vision_tower.encoder.layers.0.mlp.up_proj.weight",
            },
            mismatched_keys=set(),
            error_msgs=[],
            conversion_errors={},
        )

        model._adjust_missing_and_unexpected_keys(loading_info)

        self.assertEqual(loading_info.unexpected_keys, set())
        self.assertEqual(
            loading_info.missing_keys,
            {
                "model.layers.0.mlp.transcoder_enc.weight",
                "model.layers.0.mlp.transcoder_dec.weight",
            },
        )

    def test_unknown_architecture_lists_available_architectures(self):
        with self.assertRaisesRegex(ValueError, "gemma4"):
            models.get_transcoder_classes("not-real")


if __name__ == "__main__":
    unittest.main()
