# Feature activation dashboard

Local browser UI for outputs of [`collect_feature_activations`](../collect_feature_activations.py): it reads `feature_metadata.json` and per-feature JSON files under `features/` (the same artifacts used by `pack_features.py` — **you do not need to pack** to use this dashboard).

## Prerequisites

Run `collect_feature_activations` first, pointing `--output_dir` at a directory that will contain:

- `feature_metadata.json` — global token counts and per-feature stats (including per-domain density/fraction when multiple `val_data` sources are used)
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
uv run python -m analysis.features.visualize.feature_dashboard --data_dir /nlp/scr/siddharth/sparse-adaptation/feature_data/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl147_20260416_151353_15175430
```

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--data_dir` | (required) | Directory that contains `feature_metadata.json` and `features/` |
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8765` | Port |
| `--no-open` | off | Do not open a browser automatically |

Stop the server with **Ctrl+C**.

## What you’ll see

- **Overview:** validation mix by domain, regions, and (when applicable) thinking-position bins.
- **Table:** browse features with frequency, domain skew, and per-domain activation density; sort and filter by layer.
- **Detail:** click a row to load that feature’s JSON — per-domain bars, regions, thinking bins, logit lens, and example tabs (global top, per-domain top quantiles, random samples). Each example includes a **scale bar**: `act_min` and `act_max` (from the feature JSON) at the ends, **peak** (highlighted token) as a dot with its numeric value; per-token activations still appear in hover tooltips. Re-run collection to get an explicit `peak_activation` field in each example; older runs still derive the peak from `tokens_acts_list`.

## Troubleshooting

- **404 on metadata:** check that `--data_dir` is the directory that directly contains `feature_metadata.json`, not a parent path.
- **Empty table:** no features had nonzero activations in that run.
- **Browser can’t load the page:** ensure nothing else is bound to the chosen host/port, or pass `--port` with a free port.
