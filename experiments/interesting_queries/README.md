<!-- Created by Claude Code session "explore: queries to analyze". -->
# Interesting queries for circuit-tracing analysis

Finds queries (and the exact token position to trace) that are interesting for base-vs-instruct circuit tracing on the `google/gemma-2-2b` (base) vs `google/gemma-2-2b-it` (instruct) pair. Three finders each write JSON to `results/` and emit DeepSeek-marker `.txt` prompts to `prompts/` ready for [`analysis/attribution/run_combined_attribution.py`](../../analysis/attribution/run_combined_attribution.py).

Reuses the KL + chat-rendering machinery from [`experiments/base_vs_adapter_circuit_trace/scripts/measure_agreement.py`](../base_vs_adapter_circuit_trace/scripts/measure_agreement.py) (KL(instruct‖base) over full-vocab softmax) via [`scripts/common.py`](scripts/common.py).

## Finders
- **#1 harmful** (`scripts/find_harmful.py`) — [`walledai/HarmBench`](https://huggingface.co/datasets/walledai/HarmBench) `standard` (200 prompts, `prompt`+`category`). Keeps prompts where **instruct refuses but base complies** (no refusal markers = non-refusal). Target = the **first assistant-response token** (refuse "I…" vs comply "Sure…").
- **#2 divergent** (`scripts/find_divergent.py`) — corpus of lmsys val first-user prompts ([`siddharthmb/2026.transcoder-adapters.lmsys-chat-1m-splits`](https://huggingface.co/datasets/siddharthmb/2026.transcoder-adapters.lmsys-chat-1m-splits)) + Alpaca-style instructions + HarmBench. Greedy instruct rollout of first ~8 response tokens, KL(instruct‖base) per position, score = **max KL**. Target = the **argmax-KL position**.
- **#3 adv_suffix** (`scripts/find_adv_suffix.py`) — HarmBench prompts instruct refuses; nanoGCG (target "Sure, here is", ~200 steps) + published universal GCG suffixes as fallback. Target = **first response token, with vs without the suffix**. Verified with the refusal detector.

Aggregate: `scripts/aggregate.py` → `results/interesting_queries.json` + `results/interesting_queries.md`.

## Outputs
- `results/find_harmful.json`, `results/find_divergent.json`, `results/find_adv_suffix.json`
- `results/interesting_queries.json`, `results/interesting_queries.md` (curated table)
- `prompts/{harmful,divergent,adv_suffix}/*.txt` (attribution-ready)
- Logs: `logs/interesting_queries/` (each run logs its invocation + artifact paths as first lines)

## Reproduce
```bash
export LARGE_ARTIFACTS_DIR=/nlp/scr/siddharth
export HF_TOKEN=$(cat ~/.shell/secrets/hf_token_write)

# GPU (jagupard) via slurm:
./sh/sbatch --partition=jag-standard --gres=gpu:1 --constraint=48G --mem=64G --time=0-04:00:00 --job-name=iq_harmful \
  ./run_on_gpu/run_interesting_queries.sh experiments/interesting_queries/scripts/find_harmful.py --n_select 18
./sh/sbatch --partition=jag-standard --gres=gpu:1 --constraint=48G --mem=64G --time=0-04:00:00 --job-name=iq_divergent \
  ./run_on_gpu/run_interesting_queries.sh experiments/interesting_queries/scripts/find_divergent.py --n_corpus 300 --n_select 18
# #3 depends on #1's find_harmful.json:
./sh/sbatch --partition=jag-standard --gres=gpu:1 --constraint=48G --mem=64G --time=0-08:00:00 --job-name=iq_advsuffix \
  --dependency=afterok:<harmful_jobid> \
  ./run_on_gpu/run_interesting_queries.sh experiments/interesting_queries/scripts/find_adv_suffix.py --n_prompts 8 --gcg_steps 200

# Consolidate (CPU/login node ok):
uv run --no-sync python experiments/interesting_queries/scripts/aggregate.py
```

## Trace a selected query
```bash
uv run --extra viz python -m analysis.attribution.run_combined_attribution \
  --adapter_checkpoint <adapter> --base_model google/gemma-2-2b \
  --prompts experiments/interesting_queries/prompts/harmful/harm_000.txt \
  --prompt_format chat --gemmascope_width width_16k --gemmascope_l0 average_l0_76 \
  --base_feature_data_path mntss/gemma-scope-transcoders --max_feature_nodes 256
```
