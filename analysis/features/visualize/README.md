# Feature activation dashboard

Local browser UI for outputs of [`collect_feature_activations`](../collect_feature_activations.py): it reads `feature_metadata.json` and per-feature JSON files under `features/` (the same artifacts used by `pack_features.py` — **you do not need to pack** to use this dashboard).

## Prerequisites

Run `collect_feature_activations` first, pointing `--output_dir` at a directory that will contain:

- `feature_metadata.json` — global token counts and per-feature stats (including per-domain density/fraction when multiple `val_data` sources are used)
- `activation_histograms.npz` — exact all-nonzero activation magnitude histograms for new runs
- `features/{cantor_id}.json` — circuit-tracer-style feature records (examples, logit lens, etc.)

<!-- Example:

```bash
uv run python -m analysis.features.collect_feature_activations \
  --model_path … \
  --val_data chat:path/or/dataset fineweb:path/or/dataset \
``` -->

## Start the dashboard

From the repo root (with your environment activated):

```sh
uv run python -m analysis.features.visualize.feature_dashboard --data_dir /path/to/run
```

A browser tab should open to `http://127.0.0.1:8765/` by default.

Example:
```sh
uv run python -m analysis.features.visualize.feature_dashboard --data_dir $LARGE_ARTIFACTS_DIR/transcoder-adapters/feature_data/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl147_20260416_151353_15175430
```

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--data_dir` | (required) | Directory that contains `feature_metadata.json` and `features/` |
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8765` | Port |
| `--annotations_file` | `<data_dir>/feature_annotations.json` | Persistent feature tags and notes |
| `--prompt_output_dir` | `analysis/attribution/prompts` | Root directory for saved raw attribution prompts |
| `--no-open` | off | Do not open a browser automatically |

Stop the server with **Ctrl+C**.

## Annotate assistant-response features

After collecting activations, run the metadata-only scanner:

```sh
./sh/annotate/annotate_assistant_response_features.sh --data_dir /path/to/run
```

It tags features that concentrate on `assistant_marker` or `answer` regions and
stores them in `feature_annotations.json`. By default, the annotator updates only
its own tags and scores in place, so multiple annotators can share one
annotations file. Pass `--replace_all` to archive the existing file under
`<data_dir>/archive/` and write fresh annotations. The dashboard loads that file,
lets you filter by tag, and lets you edit tags and notes from the feature detail
pane. Manual edits are saved back to the same JSON file.

## Annotate feature patterns

For broader dashboard tags from per-feature JSONs, run:

```sh
./sh/annotate/annotate_feature_patterns.sh --data_dir /path/to/run
```

This runs the `logit_effect`, `activation_shape`, `feature_specificity`,
`token_surface`, `cross_example_consistency`, and `reasoning_move` annotators by
default. To run a subset:

```sh
./sh/annotate/annotate_feature_patterns.sh \
  --data_dir /path/to/run \
  --annotators logit_effect,token_surface,reasoning_move
```

Use `--annotations_file /tmp/some_file.json` for smoke tests that should not
modify the run directory.

## Annotate contrastive runs

To compare matching `cantor_id` entries across two feature runs:

```sh
./sh/annotate/annotate_contrastive_runs.sh \
  --source_data_dir /path/to/source/run \
  --target_data_dir /path/to/target/run
```

The target run receives `contrastive_run` tags and scores. By default the output
is `<target_data_dir>/feature_annotations.json`; pass `--annotations_file` to
write somewhere else.

## What you’ll see

- **Overview:** validation mix by domain, regions, and (when applicable) thinking-position bins.
- **Histograms and activation-range examples:** new collection runs show global and per-domain activation magnitude distributions, `feature_frequency_summary` firing-frequency distributions across all features, normalized per-domain activation densities, conditional per-feature magnitude distributions, run-level nonzero token-feature density by domain, and a joint target-vs-baseline feature-density scatter. Feature detail pages also expose bounded `Activation range ...` example tabs for configured bands such as `2.5:3.0`. Older runs without `activation_histograms.npz` still load, but histogram sections are hidden.
- **Table:** browse features with frequency, annotation tags, domain skew, and per-domain activation density; sort and filter by layer or tag.
- **Detail:** click a row to load that feature’s JSON — editable annotations, per-domain bars, regions, thinking bins, logit lens, and example tabs (global top, per-domain top quantiles, random samples). Each example includes a **scale bar**: `act_min` and `act_max` (from the feature JSON) at the ends, **peak** (highlighted token) as a dot with its numeric value; per-token activations still appear in hover tooltips. Re-run collection to get an explicit `peak_activation` field in each example; older runs still derive the peak from `tokens_acts_list`. Example controls can show, copy, and save the full model-native decoded token transcript when `feature_metadata.json` includes tokenization settings, including tokenizer/chat-template special tokens, with the collected window boxed and activation-highlighted. Saved dashboard prompts go under `analysis/attribution/prompts/<run-name>/`, preserve the model-native transcript, and should be used with `--prompt_format raw`.

## Troubleshooting

- **404 on metadata:** check that `--data_dir` is the directory that directly contains `feature_metadata.json`, not a parent path.
- **Empty table:** no features had nonzero activations in that run.
- **Browser can’t load the page:** ensure nothing else is bound to the chosen host/port, or pass `--port` with a free port.
