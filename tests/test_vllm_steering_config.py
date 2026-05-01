import unittest
from types import SimpleNamespace

from models.steering import FeatureSteeringSpec, cantor_pair
from models.vllm_steering import (
    STEERING_MODE_CONFIG_KEY,
    STEERING_SPECS_CONFIG_KEY,
    feature_steering_hf_overrides,
    get_feature_steering_from_config,
)


class VLLMSteeringConfigTests(unittest.TestCase):
    def test_hf_overrides_round_trip_feature_steering_specs(self):
        specs = [
            FeatureSteeringSpec(cantor_pair(0, 2), 1.5),
            FeatureSteeringSpec(cantor_pair(3, 4), 2.5),
        ]

        overrides = feature_steering_hf_overrides(specs, mode="add")
        config = SimpleNamespace(**overrides)

        parsed_specs, parsed_mode = get_feature_steering_from_config(config)

        self.assertEqual(parsed_specs, tuple(specs))
        self.assertEqual(parsed_mode, "add")
        self.assertEqual(overrides[STEERING_MODE_CONFIG_KEY], "add")
        self.assertEqual(
            overrides[STEERING_SPECS_CONFIG_KEY],
            [
                {"cantor_id": specs[0].cantor_id, "strength": 1.5},
                {"cantor_id": specs[1].cantor_id, "strength": 2.5},
            ],
        )

    def test_config_parser_accepts_legacy_pair_shape(self):
        spec = FeatureSteeringSpec(cantor_pair(1, 0), 3.0)
        config = SimpleNamespace(
            **{
                STEERING_SPECS_CONFIG_KEY: [(spec.cantor_id, spec.strength)],
                STEERING_MODE_CONFIG_KEY: "set",
            }
        )

        self.assertEqual(get_feature_steering_from_config(config), ((spec,), "set"))

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            feature_steering_hf_overrides([], mode="bad")  # type: ignore[arg-type]

        config = SimpleNamespace(**{STEERING_MODE_CONFIG_KEY: "bad"})
        with self.assertRaises(ValueError):
            get_feature_steering_from_config(config)


if __name__ == "__main__":
    unittest.main()
