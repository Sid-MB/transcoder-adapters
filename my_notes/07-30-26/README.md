<!-- Claude Code session "explore: weird circuit tracer issues". 2026-07-30. -->
# Huge-adapter overlay — fixed rebuild (instruct completions)

**The problem.** The huge-adapter overlay at `/nlp/scr/siddharth/transcoder-adapters/overlay_huge/graph` (served as #5 in `sh/visualize graphs/visualize_base_adapter*.sh`) showed **base-model completions**, not the instruct-bridging adapter's. Root cause: it was built with `run_combined_attribution`, which runs ONE circuit-tracer model on the **base backbone** (`MLP = T_base + T_adapter + Err`, reconstructing the base MLP), so its forward collapses to base and its logit/completion nodes are the base model's next-token predictions (e.g. `Answer`/`\n` — base echoing/continuing the prompt). Same trap documented in [`my_notes/07-09-26/07-15-26 hybrid graph instruct-completions bug.md`](../07-09-26/07-15-26%20hybrid%20graph%20instruct-completions%20bug.md).

**The fix.** Rebuilt with **`run_base_adapter_comparison`** (runs the REAL huge-adapter model and overlays a separate base GemmaScope run), on a RunPod **B200**, over the 40-prompt `comprehensive` set. The adapter side now shows instruction-tuned completions:

| prompt (one-word capital of France) | base side | adapter side |
|---|---|---|
| top logit | `Answer` (p=0.55) — base echoes the instruction | **`Paris` (p=0.99)** — instruct-like ✅ |

**Artifacts.**
- **Graphs on HF:** [`siddharthmb/2026.TA.overlay_huge_graphs`](https://huggingface.co/datasets/siddharthmb/2026.TA.overlay_huge_graphs) — `overlay/` (full) + `overlay_compact/`, 40 comprehensive prompts.
- **Huge adapter model:** [`…gemma2_2b_huge_tc16384…bs8_sl`](https://huggingface.co/siddharthmb/2026.TA.gemma2_2b_huge_tc16384_decb_l1w0.0003_norm_sch_tarbb_lb2.0_ln1.0_dr500000_lr2e-04_bs8_sl) (16384 features; final eval KL 0.142 / top-1 86.35%).
- **Adapter training wandb:** [`siddharth-stanford/sparse-adaptation/runs/c7vfz8o0`](https://wandb.ai/siddharth-stanford/sparse-adaptation/runs/c7vfz8o0).
- **Superseded original** (base completions): local `/nlp/scr/siddharth/transcoder-adapters/overlay_huge/graph`.

**Serve (from HF, reproducible — no local copy needed):**
```bash
uv run --extra viz python -m analysis.attribution.serve_comparison_graphs \
  --graph_file_dir siddharthmb/2026.TA.overlay_huge_graphs:overlay --port 8049
```
See #6 in `sh/visualize graphs/visualize_base_adapter*.sh`.

**Permanent fixes (so this doesn't recur).**
- `run_combined_attribution` now emits a loud runtime warning + a top-of-file CAVEAT that its completions are base-model, pointing to `run_base_adapter_comparison`.
- `sh/autoqueue_overlay_rebuild.sh` (the auto-rebuild trigger) was swapped from `run_combined_attribution` to `run_base_adapter_comparison`, so a re-fire won't reproduce the base-collapse.
- Viewer: base-vs-adapter logit completions are now tagged + colored by side (base `·base` dimmed red / adapter `·IT` blue); wide-graph logit labels adaptively steepen/shrink so they stay readable.
