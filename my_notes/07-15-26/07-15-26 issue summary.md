Here's everything. First the two commands, then the code diffs.

## Old command (produced base completions)

Built with `run_combined_attribution` — one base-backbone `ReplacementModel` whose forward collapses to base:

```bash
uv run --extra viz python -m analysis.attribution.run_combined_attribution \
  --adapter_checkpoint siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860 \
  --base_model google/gemma-2-2b --gemmascope_width width_16k --gemmascope_l0 average_l0_76 \
  --finetuned_transcoder_dir siddharthmb/2026.TA.gemma2_2b_gemmascope_transcoders_instruct_ft_L0-24-25 --finetuned_layers 0 24 25 \
  --prompts analysis/attribution/prompts/interesting_small --prompt_format chat \
  --run_name hybrid_finetuned_ftL0-24-25 \
  --output_dir $LARGE_ARTIFACTS_DIR/transcoder-adapters/base_adapter_comparisons/hybrid_finetuned_ftL0-24-25
```

## New command (instruct completions on the adapter side)

Built with `run_base_adapter_comparison` — runs the real adapter model + overlays a separate base run:

```bash
uv run --extra viz python -m analysis.attribution.run_base_adapter_comparison \
  --adapter_checkpoint siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860 \
  --base_model google/gemma-2-2b --gemmascope_width width_16k --gemmascope_l0 average_l0_76 \
  --finetuned_transcoder_dir siddharthmb/2026.TA.gemma2_2b_gemmascope_transcoders_instruct_ft_L0-24-25 --finetuned_layers 0 24 25 \
  --base_feature_data_path siddharthmb/2026.TA.features_gemma-2-2b_gemmascope_width_16k_average_l0_76_ms100000_ml1024_tk2_h12ad59325ffd \
  --feature_data_path siddharthmb/2026.TA.features_2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr_he94e9602bafa \
  --prompts analysis/attribution/prompts/interesting_small --prompt_format chat \
  --max_feature_nodes 2048 --max_n_logits 10 \
  --run_name hybrid_ft_overlay_ms100k \
  --output_dir $LARGE_ARTIFACTS_DIR/transcoder-adapters/base_adapter_comparisons/hybrid_ft_overlay_ms100k
```

The key differences: module `run_combined_attribution` → `run_base_adapter_comparison`, and the new one takes **two** feature repos (`--base_feature_data_path` for the base/GemmaScope side, `--feature_data_path` for the adapter side, both the `ms100000` 100k-corpus collections).

## The passthrough diff (commit `d7187f2`)

This is what lets the base/GemmaScope side use the *fine-tuned* transcoders — it patches the `TranscoderSet` in place right after loading:

```diff
--- a/analysis/attribution/run_base_adapter_comparison.py
+++ b/analysis/attribution/run_base_adapter_comparison.py
@@ def run_base_attribution(
     prompt_tokenizer_model: str | None,
+    finetuned_transcoder_dir: str | None = None,
+    finetuned_layers: list[int] | None = None,
 ) -> dict[str, str]:
@@   (after _load_gemmascope_transcoders → transcoders)
+    if finetuned_transcoder_dir and finetuned_layers:
+        from analysis.attribution.gemmascope_finetune import patch_finetuned_layers
+
+        patch_finetuned_layers(transcoders, finetuned_transcoder_dir, finetuned_layers, device_obj, dtype_obj)
+        logger.info("Applied fine-tuned transcoder weights (layers %s) from %s", finetuned_layers, finetuned_transcoder_dir)
@@ def build_parser()
+    parser.add_argument("--finetuned_transcoder_dir", type=str, default=None, help="Optional dir with finetuned_layer_*.safetensors: apply those layers' fine-tuned weights to the base GemmaScope transcoders before attribution/serving. Use with --finetuned_layers.")
+    parser.add_argument("--finetuned_layers", nargs="+", type=int, default=None, help="Layers to patch from --finetuned_transcoder_dir (e.g. 0 24 25).")
@@ def run_comparison(args)  (call to run_base_attribution)
+        finetuned_transcoder_dir=args.finetuned_transcoder_dir,
+        finetuned_layers=args.finetuned_layers,
```

The actual weight-copy logic lives in the **shared** `patch_finetuned_layers` (in `gemmascope_finetune.py`, also added in `d7187f2`) so `run_combined_attribution`, `transcoder_input_shift`, and this overlay path all reuse one implementation.

## The two import fixes (commits `7f74229`, `2a73aa7`)

These are the crashes that killed the first two relaunches — one-line each:

```diff
 from analysis.attribution.run_circuit_tracer_pipeline import (
     ensure_feature_data_conversion,
+    is_hf_feature_ref,
+    normalize_hf_feature_ref,
     run_pipeline,
     scan_name_for_feature_output,
 )
```

