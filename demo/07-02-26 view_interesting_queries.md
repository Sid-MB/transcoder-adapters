<!-- Created by session "explore: queries to analyze" (id 44e3f52a-7faa-4243-9955-55e3297aa873) -->

# Viewing the "interesting queries" overlays — 2026-07-02

Four base-vs-adapter circuit-tracing overlays over **curated, interesting prompts**, traced through the **new 16384-feature `gemma2_2b_huge` adapter** (2× the deployed adapter's capacity, freshly trained). Each overlay is one full-replacement attribution graph per prompt, with every node tagged **base GemmaScope ⬢ / huge-adapter ● / reconstruction-error ▲**. The prompts were *found* by three finders (harmful, base-vs-instruct-divergent, adversarial-suffix) — see [`experiments/interesting_queries/`](../experiments/interesting_queries/) and its [`results/interesting_queries.md`](../experiments/interesting_queries/results/interesting_queries.md) for the per-query **target token** and the metric that selected it.

Shared identifiers:
- Huge adapter: `siddharthmb/2026.TA.gemma2_2b_huge_tc16384_decb_l1w0.0003_norm_sch_tarbb_lb2.0_ln1.0_dr500000_lr2e-04_bs8_sl` (KL 0.142 / top-1 86.35% at end of training)
- Base features: full-corpus [`…ms100000…hf83e96d574d2`](https://huggingface.co/siddharthmb/2026.TA.features_gemma-2-2b_gemmascope_width_16k_average_l0_76_ms100000_ml1024_tk1_hf83e96d574d2) · Base model: `google/gemma-2-2b`
- Prompts (shared library): [`analysis/attribution/prompts/interesting_queries/{harmful,divergent,adv_suffix}`](../analysis/attribution/prompts/interesting_queries) · comprehensive: [`analysis/attribution/prompts/comprehensive`](../analysis/attribution/prompts/comprehensive)

## How to view
Serve any set, then open `http://localhost:<port>` (add an SSH forward per port if remote, e.g. `ssh -L 8049:localhost:8049 <host>`). Distinct ports so all four run at once. In the viewer: **click a node** to see its per-feature proportions ("fires on X% of tokens" = activation_frequency; top tokens with "NN%" = token_specificity) and **activation examples** collected on our chat+web data; the top-left banner shows `Base ⬢ N · Adapter ● N · Error ▲ N` (recounts live as you drag the pruning slider). Every graph is traced **at one target token** (the last token of each prompt file) — that's the token whose circuit you're reading.

## The four overlays

| Set | What it is / what to look at | Target token | Serve command |
|---|---|---|---|
| **base vs huge adapter** (baseline) | All 40 `comprehensive` prompts (factual + behavioral). The reference overlay against the huge adapter — use it to get a feel for the huge adapter's typical base/adapter/error composition before diving into the curated sets. | first response / mid-response token per prompt | `uv run --extra viz python -m analysis.attribution.serve_comparison_graphs --graph_file_dir /nlp/scr/siddharth/transcoder-adapters/overlay_huge/graph --port 8048` |
| **#1 harmful** | 18 HarmBench prompts where **the instruct model refuses but the base model complies** — i.e. the refusal is an instruction-tuning behavior. Look at which **adapter ● features** (and error ▲ nodes) drive the model to commit to a refusal, vs the base features that would have complied. | **first response token** — the refuse-vs-comply decision (`I…` refuse vs `Sure/Give/List…` comply) | `uv run --extra viz python -m analysis.attribution.serve_comparison_graphs --graph_file_dir /nlp/scr/siddharth/transcoder-adapters/overlay_huge/iq_harmful --port 8049` |
| **#2 divergent** | 18 prompts with the **highest `KL(instruct‖base)`** (mined from a corpus). At the token where the two models most disagree, look at what the adapter recruits that the base doesn't — the circuit of the instruction-tuning delta. | **argmax-KL position** (the base-top1 vs instruct-top1 tokens differ most here) | `uv run --extra viz python -m analysis.attribution.serve_comparison_graphs --graph_file_dir /nlp/scr/siddharth/transcoder-adapters/overlay_huge/iq_divergent --port 8051` |
| **#3 adversarial suffix** | 12 graphs = **6 with/without-suffix pairs**. A GCG-optimized suffix flips gemma-2-2b-it from refusing to complying. Compare the `*_no_suffix` vs `*_with_suffix` graph for the same prompt to see **which features the suffix suppresses/hijacks** to flip the refusal decision. | **first response token**, compared with vs without the suffix (`I`→`Sure`/`It`) | `uv run --extra viz python -m analysis.attribution.serve_comparison_graphs --graph_file_dir /nlp/scr/siddharth/transcoder-adapters/overlay_huge/iq_adv_suffix --port 8050` |

## Caveats worth knowing
- **Node budget is lower than the deployed-adapter overlays.** The huge adapter fires extremely densely (up to **1.67M active features** on long prompts), and attribution memory scales as `nodes × active_features`. So these run at **4096 nodes** (baseline / harmful / adv_suffix) and **2048** (divergent, the densest), vs the deployed-adapter overlays' 8192. That's the max feasible on 48–80 GB GPUs; it still captures the top-influence circuit, but the graphs are sparser than the `dtk20` overlays in [products.md](06-18-26%20to%2006-25-26%20products.md). Not directly node-count-comparable to those.
- **#3 "compliance" is shallow.** The GCG suffix genuinely flips the **refusal-decision token** to `Sure` (exactly what you want to trace), but the continuation is often off-topic — it's a decision-token flip, not real harmful content. Published *universal* GCG suffixes (Zou et al.) transferred 0/8 to gemma-2-2b-it; only per-prompt nanoGCG worked (6/8 flipped).
- **These trace the huge adapter, not the instruct model.** The queries were *found* using gemma-2-2b-it (the ground-truth refuser/diverger); the overlay traces the **adapter's** circuit (the adapter approximates the instruct delta), so the traced behavior is the adapter's approximation.

## Reproduce
Finders (GPU): see [`experiments/interesting_queries/README.md`](../experiments/interesting_queries/README.md). Overlays were built with `run_combined_attribution` (huge adapter + `ms100000` base + the huge-adapter feature collection at `/nlp/scr/siddharth/transcoder-adapters/overlay_huge/adapter_features`), `--max_feature_nodes {4096|2048} --batch_size 1 --max_n_logits 10 --max_error_nodes 64`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; dense sets on sphinx 80 GB. Logs: `logs/combined_attribution/`.

---

<!-- Appended by session "implement: transcoder adapters larger feature collection" (id 57bad3cb-3830-47fe-8b32-b06a0348779a) -->

## Deployed-adapter counterpart (`tc8192` + `dtk20` examples) — all 48 in one overlay

Everything above traces the **huge (`tc16384`) adapter**. This overlay traces the **deployed adapter** (`sl14793860`, `tc8192` — the one every [products.md](06-18-26%20to%2006-25-26%20products.md) overlay uses) over the **same** interesting_queries prompts, so you can compare the two adapters on identical inputs. What's different from the huge overlays:

- **One combined overlay, all 48 prompts** (18 harmful + 18 divergent + 12 adv_suffix) in a single dropdown — pick a category by filename prefix: `harm_###` = harmful (**base complies, instruct refuses** — the base-allowed/instruct-refused case), `div_####` = divergent (max `KL(instruct‖base)`), `harm_###_no_suffix`/`_with_suffix` = adversarial-suffix pairs.
- **`dtk20` feature scans** baked in (base [`…tk2_h12ad59325ffd`](https://huggingface.co/siddharthmb/2026.TA.features_gemma-2-2b_gemmascope_width_16k_average_l0_76_ms100000_ml1024_tk2_h12ad59325ffd) + adapter [`…lr_he94e9602bafa`](https://huggingface.co/siddharthmb/2026.TA.features_2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr_he94e9602bafa)) → a node-click shows **~20 activation examples/dataset**.
- **Standard budget 4096 / 5 / 32.** Because `tc8192` is far sparser than the huge adapter, this is **directly node-count-comparable** to the `dtk20` overlays in [products.md](06-18-26%20to%2006-25-26%20products.md) — unlike the huge overlays (see the node-budget caveat above).

Serve (port **8052**, free of the 8048–8051 the huge overlays use), then open `http://localhost:8052` (remote: `ssh -L 8052:localhost:8052 <host>`):

```sh
uv run --extra viz python -m analysis.attribution.serve_comparison_graphs --graph_file_dir /nlp/scr/siddharth/transcoder-adapters/graph_interesting_queries_dtk20 --port 8052
```

**Incremental by construction.** Built by [`run_on_gpu/run_trace_interesting_queries.sh`](../run_on_gpu/run_trace_interesting_queries.sh), which loops every `interesting_queries/<category>/` subdir into the shared `graph_interesting_queries_dtk20` dir with run_name `iq`. `run_combined_attribution` names each graph `iq__{stem}__h{content_hash}.json` and **skips any prompt whose graph already exists**, while `graph-metadata.json` (the dropdown index) grows **cumulatively** — so adding prompts (or a whole new category subdir) and re-submitting the same job only traces the additions; nothing is recomputed:

```sh
export LARGE_ARTIFACTS_DIR=/nlp/scr/siddharth HF_TOKEN=$(cat ~/.shell/secrets/hf_token_write)
./sh/sbatch --gres=gpu:1 --constraint=48G --mem=128G --cpus-per-task=8 --partition=jag-standard --job-name=trace_iq ./run_on_gpu/run_trace_interesting_queries.sh
```

Built by job `16058528` (48/48 graphs; deployed adapter `sl14793860`; standard budget). Logs: `logs/combined_attribution/trace_iq_16058528.{out,err}`.
