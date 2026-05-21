
To add more prompts to attribute:
1. Add new txt files to [the prompts folder you want to use](../../analysis/attribution/prompts/interesting_small)
2. Run the attribution pipeline:
```sh
./sh/attribution/slurm_circuit_tracer_pipeline.sh --transcoder_model_path siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860 --base_model google/gemma-2-2b --feature_data_dir /nlp/scr/siddharth/sparse-adaptation/feature_data/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860_20260521_011754_15515871 --prompts analysis/attribution/prompts/interesting_small --run_name interesting_small_sl14793860_features15515871 --prompt_format chat --max_feature_nodes 256 --batch_size 4 --max_n_logits 5
```
3. Visualize with:
```sh
uv run --extra viz circuit-tracer start-server --graph_file_dir /nlp/scr/siddharth/sparse-adaptation/attribution_graphs/interesting_small_sl14793860_features15515871_2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860 --features_dir /nlp/scr/siddharth/sparse-adaptation/feature_data/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860_20260521_011754_15515871/circuit_tracer_features
```


To refresh changed txt files:
```sh
GRAPH_DIR="/nlp/scr/siddharth/sparse-adaptation/attribution_graphs/interesting_small_sl14793860_features15515871_2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860"

rm -f \
  "$GRAPH_DIR/interesting_small_sl14793860_features15515871__bomb_refusal_help.json" \
  "$GRAPH_DIR/interesting_small_sl14793860_features15515871__hacking_refusal_assist.json" \
  "$GRAPH_DIR/interesting_small_sl14793860_features15515871__poison_refusal_help.json"
```
