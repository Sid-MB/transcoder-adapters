import math
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from models.qwen2_transcoder import Qwen2ConfigWithTranscoder, Qwen2ForCausalLMWithTranscoder
from models.qwen2_with_transcoder_relp import (
    Qwen2ForCausalLMWithTranscoderRelP,
    Qwen2MLPWithTranscoderRelP,
)
from models.relp_common import RelPMLPWithTranscoder
from models.steering import (
    FeatureSteeringSpec,
    apply_feature_steering,
    cantor_pair,
    cantor_unpair,
)


def tiny_qwen_config(num_hidden_layers: int = 1) -> Qwen2ConfigWithTranscoder:
    return Qwen2ConfigWithTranscoder(
        vocab_size=32,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
        transcoder_n_features=4,
        transcoder_dec_bias=False,
        tie_word_embeddings=False,
    )


def zero_qwen_mlp(mlp) -> None:
    with torch.no_grad():
        for module in (
            mlp.gate_proj,
            mlp.up_proj,
            mlp.down_proj,
            mlp.transcoder_enc,
            mlp.transcoder_dec,
        ):
            module.weight.zero_()
            if module.bias is not None:
                module.bias.zero_()


class FeatureSteeringTests(unittest.TestCase):
    def test_cantor_round_trips_exactly(self):
        pairs = [
            (0, 0),
            (0, 1),
            (1, 0),
            (3, 5),
            (10_000, 7),
            (1_000_000, 1_000_001),
        ]

        for layer, feature in pairs:
            with self.subTest(layer=layer, feature=feature):
                cantor_id = cantor_pair(layer, feature)
                self.assertEqual(cantor_unpair(cantor_id), (layer, feature))

        self.assertEqual(cantor_pair(3, 5), 41)

    def test_cantor_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            cantor_pair(-1, 0)
        with self.assertRaises(ValueError):
            cantor_unpair(-1)
        with self.assertRaises(TypeError):
            cantor_pair(True, 0)

    def test_apply_feature_steering_modes_do_not_mutate_input(self):
        features = torch.tensor([[[0.0, 2.0, 4.0]]])
        original = features.clone()
        targets = ((0, 1.5), (1, 1.0))

        min_result = apply_feature_steering(features, targets, "min")
        add_result = apply_feature_steering(features, targets, "add")
        set_result = apply_feature_steering(features, targets, "set")

        torch.testing.assert_close(features, original)
        torch.testing.assert_close(min_result, torch.tensor([[[1.5, 2.0, 4.0]]]))
        torch.testing.assert_close(add_result, torch.tensor([[[1.5, 3.0, 4.0]]]))
        torch.testing.assert_close(set_result, torch.tensor([[[1.5, 1.0, 4.0]]]))

    def test_apply_feature_steering_supports_backward(self):
        features = torch.tensor([[[0.25, 2.0, 3.0]]], requires_grad=True)
        min_result = apply_feature_steering(features, ((0, 0.5), (1, 1.0)), "min")
        min_result.sum().backward()
        torch.testing.assert_close(features.grad, torch.tensor([[[0.0, 1.0, 1.0]]]))

        features = torch.tensor([[[0.25, 2.0, 3.0]]], requires_grad=True)
        add_result = apply_feature_steering(features, ((0, 0.5), (1, 1.0)), "add")
        add_result.sum().backward()
        torch.testing.assert_close(features.grad, torch.ones_like(features))

        features = torch.tensor([[[0.25, 2.0, 3.0]]], requires_grad=True)
        set_result = apply_feature_steering(features, ((0, 0.5), (1, 1.0)), "set")
        set_result.sum().backward()
        torch.testing.assert_close(features.grad, torch.tensor([[[0.0, 0.0, 1.0]]]))

    def test_invalid_specs_and_targets_are_rejected(self):
        with self.assertRaises(ValueError):
            FeatureSteeringSpec(0, -1.0)
        with self.assertRaises(ValueError):
            FeatureSteeringSpec(0, math.nan)
        with self.assertRaises(ValueError):
            apply_feature_steering(torch.zeros(1, 1, 2), ((2, 1.0),), "min")
        with self.assertRaises(ValueError):
            apply_feature_steering(torch.zeros(1, 1, 2), ((1, 1.0), (1, 2.0)), "min")
        with self.assertRaises(ValueError):
            apply_feature_steering(torch.zeros(1, 1, 2), ((0, 1.0),), "bad")  # type: ignore[arg-type]

    def test_standard_model_api_propagates_to_layer_local_targets(self):
        model = Qwen2ForCausalLMWithTranscoder(tiny_qwen_config(num_hidden_layers=2))
        spec = FeatureSteeringSpec(cantor_pair(1, 2), 0.75)

        model.set_feature_steering([spec], mode="add")

        self.assertEqual(model.model.layers[0].mlp.feature_steering_targets, ())
        self.assertEqual(model.model.layers[1].mlp.feature_steering_targets, ((2, 0.75),))
        self.assertEqual(model.model.layers[1].mlp.feature_steering_mode, "add")
        self.assertEqual(model.get_feature_steering(), ((spec,), "add"))

        model.clear_feature_steering()
        self.assertEqual(model.get_feature_steering(), ((), "min"))
        self.assertEqual(model.model.layers[1].mlp.feature_steering_targets, ())

    def test_model_api_rejects_duplicates_and_out_of_range_targets(self):
        model = Qwen2ForCausalLMWithTranscoder(tiny_qwen_config(num_hidden_layers=1))
        spec = FeatureSteeringSpec(cantor_pair(0, 1), 1.0)

        with self.assertRaises(ValueError):
            model.set_feature_steering([spec, spec])
        with self.assertRaises(ValueError):
            model.set_feature_steering([FeatureSteeringSpec(cantor_pair(1, 0), 1.0)])
        with self.assertRaises(ValueError):
            model.set_feature_steering([FeatureSteeringSpec(cantor_pair(0, 4), 1.0)])

    def test_standard_forward_delta_matches_decoder_column(self):
        model = Qwen2ForCausalLMWithTranscoder(tiny_qwen_config(num_hidden_layers=1))
        mlp = model.model.layers[0].mlp
        zero_qwen_mlp(mlp)

        decoder_column = torch.tensor([1.0, -2.0, 0.5, 3.0])
        with torch.no_grad():
            mlp.transcoder_dec.weight[:, 2] = decoder_column

        strength = 1.25
        model.set_feature_steering([FeatureSteeringSpec(cantor_pair(0, 2), strength)], mode="min")

        hidden_states = torch.zeros(1, 2, 4)
        output = mlp(hidden_states)
        expected = decoder_column.mul(strength).view(1, 1, 4).expand(1, 2, 4)
        torch.testing.assert_close(output, expected)

    def test_standard_forward_delta_supports_backward(self):
        model = Qwen2ForCausalLMWithTranscoder(tiny_qwen_config(num_hidden_layers=1))
        mlp = model.model.layers[0].mlp
        zero_qwen_mlp(mlp)

        with torch.no_grad():
            mlp.transcoder_dec.weight[:, 2] = torch.tensor([1.0, -2.0, 0.5, 3.0])
        model.set_feature_steering([FeatureSteeringSpec(cantor_pair(0, 2), 1.25)], mode="min")

        hidden_states = torch.zeros(1, 2, 4, requires_grad=True)
        mlp(hidden_states).sum().backward()
        self.assertIsNotNone(hidden_states.grad)
        self.assertTrue(torch.isfinite(hidden_states.grad).all())

    def test_relp_model_api_propagates_to_qwen_layers(self):
        model = Qwen2ForCausalLMWithTranscoderRelP(tiny_qwen_config(num_hidden_layers=1))
        spec = FeatureSteeringSpec(cantor_pair(0, 3), 2.0)

        model.set_feature_steering([spec], mode="set")

        self.assertEqual(model.model.layers[0].mlp.feature_steering_targets, ((3, 2.0),))
        self.assertEqual(model.get_feature_steering(), ((spec,), "set"))

    def test_qwen_relp_masks_run_after_steering(self):
        config = tiny_qwen_config(num_hidden_layers=1)
        mlp = Qwen2MLPWithTranscoderRelP(config)
        zero_qwen_mlp(mlp)

        with torch.no_grad():
            mlp.transcoder_dec.weight[:, 1] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mlp.feature_steering_targets = ((1, 2.0),)
        mlp.feature_steering_mode = "min"
        mlp.feature_mask = torch.tensor([1.0, 0.0, 1.0, 1.0])

        output = mlp(torch.zeros(1, 1, 4))
        torch.testing.assert_close(output, torch.zeros_like(output))

    def test_shared_relp_mlp_applies_steering_before_decoder(self):
        config = SimpleNamespace(
            hidden_size=4,
            intermediate_size=8,
            transcoder_n_features=4,
            transcoder_dec_bias=False,
        )
        mlp = RelPMLPWithTranscoder(config, act_fn=F.gelu)
        zero_qwen_mlp(mlp)

        with torch.no_grad():
            mlp.transcoder_dec.weight[:, 0] = torch.tensor([0.5, 1.0, -1.0, 2.0])
        mlp.feature_steering_targets = ((0, 3.0),)
        mlp.feature_steering_mode = "set"

        output = mlp(torch.zeros(1, 1, 4))
        expected = torch.tensor([[[1.5, 3.0, -3.0, 6.0]]])
        torch.testing.assert_close(output, expected)


if __name__ == "__main__":
    unittest.main()
