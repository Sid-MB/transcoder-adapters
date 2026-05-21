import tempfile
import unittest
from pathlib import Path

import torch
from transformers import Gemma2Config
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from analysis.attribution.relp_model import RelPReplacementModel
from analysis.attribution.run_attribution import load_prompt_file, run_attribution_for_prompt
from analysis.features.visualize.feature_dashboard import _save_prompt_example
from models.gemma2_with_transcoder_relp import Gemma2ForCausalLMWithTranscoderRelP


class TinyGemmaRawTokenizer:
    bos_token_id = 2
    eos_token_id = 1
    pad_token_id = 0
    start_of_turn_id = 10
    user_id = 11
    model_id = 12
    end_of_turn_id = 13
    newline_id = 14
    char_offset = 100

    def __init__(self, name_or_path: str | None = None):
        self.name_or_path = name_or_path
        self._pieces = {
            "<bos>": [self.bos_token_id],
            "<start_of_turn>user\n": [self.start_of_turn_id, self.user_id, self.newline_id],
            "<start_of_turn>model\n": [self.start_of_turn_id, self.model_id, self.newline_id],
            "<end_of_turn>": [self.end_of_turn_id],
        }

    def _char_id(self, char: str) -> int:
        return self.char_offset + ord(char)

    def encode(self, text, add_special_tokens=False):
        ids = []
        i = 0
        while i < len(text):
            for piece, piece_ids in self._pieces.items():
                if text.startswith(piece, i):
                    ids.extend(piece_ids)
                    i += len(piece)
                    break
            else:
                ids.append(self._char_id(text[i]))
                i += 1
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        return ids

    def decode(self, token_ids):
        pieces = []
        i = 0
        token_ids = list(token_ids)
        while i < len(token_ids):
            if token_ids[i] == self.bos_token_id:
                pieces.append("<bos>")
                i += 1
            elif token_ids[i:i + 3] == [self.start_of_turn_id, self.user_id, self.newline_id]:
                pieces.append("<start_of_turn>user\n")
                i += 3
            elif token_ids[i:i + 3] == [self.start_of_turn_id, self.model_id, self.newline_id]:
                pieces.append("<start_of_turn>model\n")
                i += 3
            elif token_ids[i] == self.end_of_turn_id:
                pieces.append("<end_of_turn>")
                i += 1
            elif token_ids[i] >= self.char_offset:
                pieces.append(chr(token_ids[i] - self.char_offset))
                i += 1
            else:
                pieces.append(f"<unk{token_ids[i]}>")
                i += 1
        return "".join(pieces)


def save_graph_export_tokenizer(path: Path) -> None:
    id_to_token = {idx: f"<tok_{idx}>" for idx in range(512)}
    id_to_token[0] = "<pad>"
    id_to_token[1] = "<eos>"
    id_to_token[2] = "<bos>"
    id_to_token[10] = "<start_of_turn>"
    id_to_token[11] = "user"
    id_to_token[12] = "model"
    id_to_token[13] = "<end_of_turn>"
    id_to_token[14] = "<newline>"
    for char_code in range(128):
        id_to_token[TinyGemmaRawTokenizer.char_offset + char_code] = chr(char_code)
    vocab = {token: idx for idx, token in id_to_token.items()}
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
        additional_special_tokens=["<start_of_turn>", "<end_of_turn>"],
    )
    tokenizer.save_pretrained(path)


def tiny_gemma2_relp_config() -> Gemma2Config:
    config = Gemma2Config(
        vocab_size=512,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        sliding_window=16,
        final_logit_softcapping=None,
        attn_logit_softcapping=None,
    )
    config.transcoder_n_features = 8
    config.transcoder_dec_bias = False
    config.architectures = ["Gemma2ForCausalLMWithTranscoderRelP"]
    return config


def tiny_relp_model(tokenizer: TinyGemmaRawTokenizer) -> RelPReplacementModel:
    model = Gemma2ForCausalLMWithTranscoderRelP(tiny_gemma2_relp_config()).to("cuda")
    model.eval()
    for layer in model.model.layers:
        torch.nn.init.constant_(layer.mlp.transcoder_enc.bias, 0.1)
    return RelPReplacementModel(model, tokenizer)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
class DashboardPromptAttributionE2ETests(unittest.TestCase):
    def test_saved_dashboard_prompt_runs_raw_attribution_with_gemma_special_tokens(self):
        torch.manual_seed(0)
        tokenizer = TinyGemmaRawTokenizer()
        transcript = (
            "<bos><start_of_turn>user\n"
            "Hi?<end_of_turn>"
            "<start_of_turn>model\n"
            "OK<end_of_turn>\n"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tokenizer_dir = root / "tokenizer"
            save_graph_export_tokenizer(tokenizer_dir)
            tokenizer.name_or_path = str(tokenizer_dir)
            data_dir = root / "feature_run"
            data_dir.mkdir()
            save_result = _save_prompt_example(
                prompt_output_dir=root / "prompts",
                data_dir=data_dir,
                payload={
                    "transcript": transcript,
                    "cantor_id": 123,
                    "layer": 1,
                    "feature": 2,
                    "quantile_name": "Top activations",
                    "example_index": 0,
                },
            )
            prompt_path = Path(save_result["path"])
            saved_text = prompt_path.read_text()

            self.assertEqual(save_result["prompt_format"], "raw")
            self.assertFalse(saved_text.endswith("\n"))
            self.assertIn("<bos><start_of_turn>user\n", saved_text)
            self.assertIn("<start_of_turn>model\n", saved_text)
            self.assertTrue(saved_text.endswith("<end_of_turn>"))

            prompt_tokens, target, decoded_prompt = load_prompt_file(
                prompt_path,
                tokenizer,
                prompt_format="raw",
                model_type="gemma2",
            )

            self.assertEqual(prompt_tokens[0], tokenizer.bos_token_id)
            self.assertIn(tokenizer.start_of_turn_id, prompt_tokens)
            self.assertIn(tokenizer.user_id, prompt_tokens)
            self.assertIn(tokenizer.model_id, prompt_tokens)
            self.assertEqual(target, tokenizer.end_of_turn_id)
            self.assertEqual(tokenizer.decode([target]), "<end_of_turn>")
            self.assertTrue(decoded_prompt.endswith("OK"))

            output_dir = root / "graphs"
            output_dir.mkdir()
            graph = run_attribution_for_prompt(
                prompt_tokens=prompt_tokens,
                slug="dashboard_prompt_e2e",
                model=tiny_relp_model(tokenizer),
                scan="dashboard_prompt_e2e",
                output_dir=str(output_dir),
                max_n_logits=2,
                batch_size=4,
                max_feature_nodes=8,
                node_threshold=0.8,
                edge_threshold=0.98,
            )

            self.assertGreater(graph.active_features.shape[0], 0)
            self.assertTrue((output_dir / "dashboard_prompt_e2e.json").exists())


if __name__ == "__main__":
    unittest.main()
