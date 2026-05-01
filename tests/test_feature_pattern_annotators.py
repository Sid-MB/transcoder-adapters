import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from analysis.features.annotate.annotate_feature_patterns import TokenSurfaceAnnotator
from analysis.features.annotate.annotation_framework import run_annotation

REPO_ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f)


def example(tokens, highlight_idx, acts=None):
    return {
        "tokens": tokens,
        "tokens_acts_list": acts or [0.0 for _ in tokens],
        "train_token_ind": highlight_idx,
        "peak_activation": max(acts or [0.0]),
        "is_repeated_datapoint": False,
    }


class FeaturePatternAnnotatorTests(unittest.TestCase):
    def make_pattern_fixture(self, root: Path) -> None:
        metadata = {
            "tokens_per_domain": {"math": 100, "chat": 100, "code": 100},
            "tokens_per_region": {"question": 100, "answer": 100, "thinking": 100},
            "features": [
                {
                    "layer": 0,
                    "feature": 0,
                    "cantor_id": 0,
                    "activation_count": 10,
                    "activation_freq": 0.001,
                    "domain_fraction": {"math": 0.9, "chat": 0.1},
                    "region_fraction": {"answer": 0.8, "question": 0.2},
                },
                {
                    "layer": 0,
                    "feature": 1,
                    "cantor_id": 1,
                    "activation_count": 12,
                    "activation_freq": 0.002,
                    "domain_fraction": {"math": 0.34, "chat": 0.33, "code": 0.33},
                    "region_fraction": {
                        "question": 0.34,
                        "thinking": 0.33,
                        "answer": 0.33,
                    },
                },
            ],
        }
        write_json(root / "feature_metadata.json", metadata)
        write_json(
            root / "feature_annotations.json",
            {
                "0": {
                    "tags": ["manual_keep", "other_tag", "promotes_refusal"],
                    "notes": "keep this note",
                    "auto_tags": {
                        "logit_effect": ["promotes_refusal"],
                        "other_annotator": ["other_tag"],
                    },
                    "auto_scores": {
                        "logit_effect": {"old": 1},
                        "other_annotator": {"score": 1},
                    },
                }
            },
        )

        repeated_digit_examples = [
            example(
                ["The", " answer", " is", " 42", "."],
                3,
                [0.0, 0.1, 0.2, 5.0, 0.1],
            )
            for _ in range(4)
        ]
        write_json(
            root / "features" / "0.json",
            {
                "top_logits": [
                    " answer",
                    " 123",
                    "\n",
                    " def",
                    "<eos>",
                ],
                "bottom_logits": [" answer", " therefore"],
                "examples_quantiles": [
                    {
                        "quantile_name": "Top activations",
                        "examples": repeated_digit_examples,
                    }
                ],
            },
        )

        reasoning_examples = [
            example(
                [
                    "Wait",
                    ",",
                    " maybe",
                    " we",
                    " check",
                    " then",
                    " final",
                    " answer",
                ],
                0,
                [1.0] * 8,
            )
            for _ in range(3)
        ]
        write_json(
            root / "features" / "1.json",
            {
                "top_logits": [" maybe", " could", " sorry"],
                "bottom_logits": [],
                "examples_quantiles": [
                    {
                        "quantile_name": "Top activations",
                        "examples": reasoning_examples,
                    }
                ],
            },
        )

    def test_feature_pattern_cli_emits_tags_and_preserves_other_annotations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.make_pattern_fixture(root)

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "analysis.features.annotate.annotate_feature_patterns",
                    "--data_dir",
                    str(root),
                    "--top_k",
                    "5",
                ],
                check=True,
            )

            annotations = json.loads((root / "feature_annotations.json").read_text())
            entry0 = annotations["0"]
            entry1 = annotations["1"]

        self.assertEqual(entry0["notes"], "keep this note")
        self.assertIn("manual_keep", entry0["tags"])
        self.assertIn("other_tag", entry0["tags"])
        self.assertNotIn("promotes_refusal", entry0["tags"])
        self.assertEqual(entry0["auto_tags"]["other_annotator"], ["other_tag"])
        self.assertNotEqual(entry0["auto_scores"]["logit_effect"], {"old": 1})

        self.assertIn("promotes_numbers", entry0["auto_tags"]["logit_effect"])
        self.assertIn("promotes_newline", entry0["auto_tags"]["logit_effect"])
        self.assertIn("promotes_code", entry0["auto_tags"]["logit_effect"])
        self.assertIn("promotes_special_tokens", entry0["auto_tags"]["logit_effect"])
        self.assertIn("promotes_answer_tokens", entry0["auto_tags"]["logit_effect"])
        self.assertIn("suppresses_answer_tokens", entry0["auto_tags"]["logit_effect"])
        self.assertIn("single_token_spike", entry0["auto_tags"]["activation_shape"])
        self.assertIn("domain_specialist", entry0["auto_tags"]["feature_specificity"])
        self.assertIn("region_specialist", entry0["auto_tags"]["feature_specificity"])
        self.assertIn("token_specialist", entry0["auto_tags"]["feature_specificity"])
        self.assertIn("digit_feature", entry0["auto_tags"]["token_surface"])
        self.assertIn("consistent_token", entry0["auto_tags"]["cross_example_consistency"])

        self.assertIn("sustained_context", entry1["auto_tags"]["activation_shape"])
        self.assertIn("broad_context", entry1["auto_tags"]["feature_specificity"])
        self.assertIn("self_correction", entry1["auto_tags"]["reasoning_move"])
        self.assertIn("uncertainty", entry1["auto_tags"]["reasoning_move"])
        self.assertIn("verification", entry1["auto_tags"]["reasoning_move"])
        self.assertIn("answer_commitment", entry1["auto_tags"]["reasoning_move"])

    def test_missing_feature_json_clears_only_that_annotator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_json(
                root / "feature_metadata.json",
                {
                    "features": [
                        {
                            "layer": 0,
                            "feature": 9,
                            "cantor_id": 9,
                            "activation_count": 1,
                            "activation_freq": 0.001,
                        }
                    ]
                },
            )
            write_json(
                root / "feature_annotations.json",
                {
                    "9": {
                        "tags": ["manual", "digit_feature", "other"],
                        "notes": "manual note",
                        "auto_tags": {
                            "token_surface": ["digit_feature"],
                            "other": ["other"],
                        },
                        "auto_scores": {
                            "token_surface": {"digit_fraction": 1.0},
                            "other": {"score": 1.0},
                        },
                    }
                },
            )

            run_annotation(
                data_dir=root,
                annotations_file=None,
                annotator=TokenSurfaceAnnotator(data_dir=root, top_k=5),
            )
            annotations = json.loads((root / "feature_annotations.json").read_text())
            entry = annotations["9"]

        self.assertEqual(entry["notes"], "manual note")
        self.assertEqual(entry["auto_tags"], {"other": ["other"]})
        self.assertEqual(entry["auto_scores"], {"other": {"score": 1.0}})
        self.assertIn("manual", entry["tags"])
        self.assertIn("other", entry["tags"])
        self.assertNotIn("digit_feature", entry["tags"])

    def test_contrastive_cli_compares_shared_features_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            write_json(
                source / "feature_metadata.json",
                {
                    "features": [
                        {
                            "layer": 0,
                            "feature": 0,
                            "cantor_id": 0,
                            "activation_count": 10,
                            "activation_freq": 0.001,
                            "domain_fraction": {"math": 0.8, "chat": 0.2},
                            "region_fraction": {"answer": 0.8, "question": 0.2},
                        },
                        {
                            "layer": 0,
                            "feature": 1,
                            "cantor_id": 1,
                            "activation_count": 40,
                            "activation_freq": 0.004,
                            "domain_fraction": {"math": 0.5, "chat": 0.5},
                            "region_fraction": {"answer": 0.5, "question": 0.5},
                        },
                        {
                            "layer": 0,
                            "feature": 2,
                            "cantor_id": 2,
                            "activation_count": 10,
                            "activation_freq": 0.001,
                            "domain_fraction": {"math": 0.5, "chat": 0.5},
                            "region_fraction": {"answer": 0.5, "question": 0.5},
                        },
                    ]
                },
            )
            write_json(
                target / "feature_metadata.json",
                {
                    "features": [
                        {
                            "layer": 0,
                            "feature": 0,
                            "cantor_id": 0,
                            "activation_count": 30,
                            "activation_freq": 0.003,
                            "domain_fraction": {"math": 0.75, "chat": 0.25},
                            "region_fraction": {"answer": 0.75, "question": 0.25},
                        },
                        {
                            "layer": 0,
                            "feature": 1,
                            "cantor_id": 1,
                            "activation_count": 10,
                            "activation_freq": 0.001,
                            "domain_fraction": {"math": 0.5, "chat": 0.5},
                            "region_fraction": {"answer": 0.5, "question": 0.5},
                        },
                        {
                            "layer": 0,
                            "feature": 2,
                            "cantor_id": 2,
                            "activation_count": 11,
                            "activation_freq": 0.0011,
                            "domain_fraction": {"math": 0.52, "chat": 0.48},
                            "region_fraction": {"answer": 0.52, "question": 0.48},
                        },
                        {
                            "layer": 0,
                            "feature": 99,
                            "cantor_id": 99,
                            "activation_count": 99,
                            "activation_freq": 0.01,
                            "domain_fraction": {"math": 1.0},
                            "region_fraction": {"answer": 1.0},
                        },
                    ]
                },
            )

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "analysis.features.annotate.annotate_contrastive_runs",
                    "--source_data_dir",
                    str(source),
                    "--target_data_dir",
                    str(target),
                    "--top_k",
                    "5",
                ],
                check=True,
            )
            annotations = json.loads((target / "feature_annotations.json").read_text())

        self.assertIn("it_amplified", annotations["0"]["auto_tags"]["contrastive_run"])
        self.assertIn("base_amplified", annotations["1"]["auto_tags"]["contrastive_run"])
        self.assertIn("stable_feature", annotations["2"]["auto_tags"]["contrastive_run"])
        self.assertNotIn("99", annotations)

    def test_feature_pattern_wrapper_runs_with_uv_cache_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.make_pattern_fixture(root)
            env = {**os.environ, "UV_CACHE_DIR": "/dev/null"}

            subprocess.run(
                [
                    "./sh/annotate/annotate_feature_patterns.sh",
                    "--data_dir",
                    str(root),
                    "--annotators",
                    "token_surface,reasoning_move",
                    "--top_k",
                    "5",
                ],
                cwd=REPO_ROOT,
                env=env,
                check=True,
            )
            annotations = json.loads((root / "feature_annotations.json").read_text())

        self.assertIn("digit_feature", annotations["0"]["auto_tags"]["token_surface"])
        self.assertIn("self_correction", annotations["1"]["auto_tags"]["reasoning_move"])

    def test_contrastive_wrapper_runs_with_uv_cache_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            write_json(
                source / "feature_metadata.json",
                {
                    "features": [
                        {
                            "layer": 0,
                            "feature": 0,
                            "cantor_id": 0,
                            "activation_count": 10,
                            "activation_freq": 0.001,
                            "domain_fraction": {"math": 1.0},
                            "region_fraction": {"answer": 1.0},
                        }
                    ]
                },
            )
            write_json(
                target / "feature_metadata.json",
                {
                    "features": [
                        {
                            "layer": 0,
                            "feature": 0,
                            "cantor_id": 0,
                            "activation_count": 30,
                            "activation_freq": 0.003,
                            "domain_fraction": {"math": 1.0},
                            "region_fraction": {"answer": 1.0},
                        }
                    ]
                },
            )
            env = {**os.environ, "UV_CACHE_DIR": "/dev/null"}

            subprocess.run(
                [
                    "./sh/annotate/annotate_contrastive_runs.sh",
                    "--source_data_dir",
                    str(source),
                    "--target_data_dir",
                    str(target),
                    "--top_k",
                    "5",
                ],
                cwd=REPO_ROOT,
                env=env,
                check=True,
            )
            annotations = json.loads((target / "feature_annotations.json").read_text())

        self.assertIn("it_amplified", annotations["0"]["auto_tags"]["contrastive_run"])


if __name__ == "__main__":
    unittest.main()
