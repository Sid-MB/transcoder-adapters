# Adding a new model

## Create models/<model_name>_transcoder.py
- [ ] Add `models/<model_name>_transcoder.py` with:
  - `<ModelName>ConfigWithTranscoder`
  - `<ModelName>ForCausalLMWithTranscoder`
  - transcoder-wrapped MLP/module replacement
- [ ] Register the architecture in `models/__init__.py`.
- [ ] Add any HF `config.model_type` alias in `models/__init__.py` only if it differs from our canonical `model_arch`.
  `model_arch` is the user-facing architecture name used in training configs and should stay simple and canonical (for example, `gemma4`). HF-specific names such as `gemma4_text` should only be accepted as checkpoint
  compatibility aliases, not as config values.
- [ ] Add a base tokenizer fallback in `models/auto.py` only for legacy checkpoints without tokenizer files.

### Add tests
- [ ] Add model registry tests for:
  - auto-detecting the canonical `model_arch`
  - rejecting HF-only aliases as user config values
  - accepting HF `config.model_type` aliases for checkpoint loading, if needed
- [ ] Add config-loading tests if the new model adds aliases or validation behavior.
- [ ] Add a small model-construction or load smoke test when feasible.

## tokens.py

Usually no code change is needed here for a new model. `models/tokens.py`
detects chat markers from the tokenizer's `apply_chat_template()` output.

When adding a model:

- [ ] Make sure the tokenizer has a chat template.
- [ ] Run the real tokenizer smoke test:

```bash
uv run python -m unittest tests.test_real_tokenizers
```

- [ ] If the tokenizer is public, add it to `REAL_TOKENIZER_CASES` in
  `tests/test_real_tokenizers.py`.
- [ ] If the model uses a new thinking-token format, update `models/tokens.py` and
  add a fixture in `tests/test_tokens.py`.

Supported thinking formats today:

- `<think>` ... `</think>`
- `<|channel>thought\n` ... `<channel|>`

## Add a config in training/configs to train your model
- [ ] Likely want to start with a very simple and fast training config to make sure everything is working

Once you've made all changes, start with a very simple and fast debug training config and launch runs until there are no more discernible errors or issues in the out and err logs. This will likely require a lot of iteration. 
