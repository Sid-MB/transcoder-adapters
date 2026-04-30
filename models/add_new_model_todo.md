# Adding a new model

## Create models/<model_name>_transcoder.py
<!-- Todo, instructions -->
- [ ]

### Add tests
- [ ]

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
- [ ]If the model uses a new thinking-token format, update `models/tokens.py` and
  add a fixture in `tests/test_tokens.py`.

Supported thinking formats today:

- `<think>` ... `</think>`
- `<|channel>thought\n` ... `<channel|>`

## Add a config in training/configs to train your model
- [ ] Likely want to start with a very simple and fast training config to make sure everything is working

