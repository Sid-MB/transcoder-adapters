#!/usr/bin/env python
"""Causal test: is the adapter feature L6/F4241 necessary for the refusal opening?

Session "[refusal-token-tracing]" (fd1f0d19), 2026-08-13.

The compliance contrast (compare_refusal_compliance.md) found that of the 13 adapter features that drive
the "I cannot..." opening, all but one also fire when the adapter complies -- the sole refusal-SPECIFIC
candidate is L6/F4241. Correlation, though: this script tests CAUSATION by ablating (clamping to 0) that
transcoder feature and measuring whether the adapter still refuses.

Method. Load the tc8192 adapter (models.auto). For each refusal prompt: greedily decode the adapter's own
opening (the refusal), then TEACHER-FORCE that token sequence and read p(opening_token_k | prefix) at each
position -- once with no ablation, once with L6/F4241 clamped to 0, once with a generic high-influence
control feature (L0/F5489) clamped to 0. If L6/F4241 is causally necessary, clamping it drops p(refusal
opener) far more than the generic control does; if it is merely correlational, the drop is small. We also
greedily regenerate under each ablation to see whether the opening flips (refuse -> comply). As controls we
run the same ablations on prompts the adapter COMPLIES with (its opener should be unchanged, since L6/F4241
does not fire there).

Ablation uses the model's built-in feature steering (models.steering: mode="set", strength 0 == clamp the
post-encoder feature activation to zero at every position of every layer-6 forward).

Outputs (--output_dir): results.json, summary.md, ablation_refusal.png, transcripts/<id>__<cond>.md.

Example:
    PYTHONPATH=. uv run --extra viz python experiments/interesting_queries/scripts/ablate_refusal_feature.py \\
        --ablate_feature 6:4241 --control_feature 0:5489 \\
        --refusal_ids harm_125 harm_139 harm_116
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from helpers.log import logger, setup_logging  # noqa: E402
from models.steering import FeatureSteeringSpec, cantor_pair  # noqa: E402

INSTRUCT_MODEL = "google/gemma-2-2b-it"
DEPLOYED_ADAPTER = "siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860"
STRICT_RESULTS = "experiments/interesting_queries/results/strict_compliance_refusal/results.json"
# The 8 prompts the tc8192 adapter complies with under chat (compliance contrast), for the control panel.
DEFAULT_COMPLY_REQUESTS = "my_notes/08-10-26/refusal_token_tracing/contrast_requests.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter_checkpoint", default=DEPLOYED_ADAPTER, help="Transcoder adapter to ablate.")
    p.add_argument("--ablate_feature", default="6:4241", help="'layer:feature' of the candidate refusal-specific feature to clamp to 0.")
    p.add_argument("--control_feature", default="0:5489", help="'layer:feature' of a generic high-influence feature, clamped to 0 as a comparison (expected: small effect on refusal-vs-compliance selectivity).")
    p.add_argument("--refusal_ids", nargs="+", default=["harm_125", "harm_139", "harm_116"], help="Strict-flip ids the adapter refuses (from --strict_results).")
    p.add_argument("--comply_requests", default=DEFAULT_COMPLY_REQUESTS, help="JSON {id: request} of prompts the adapter complies with (control panel). Set '' to skip.")
    p.add_argument("--strict_results", default=STRICT_RESULTS, help="results.json supplying the refusal prompts' request text.")
    p.add_argument("--tokenizer_model", default=INSTRUCT_MODEL, help="Tokenizer / chat template.")
    p.add_argument("--n_positions", type=int, default=10, help="How many leading opening-token positions to score p(token) at.")
    p.add_argument("--gen_tokens", type=int, default=60, help="Greedy tokens to regenerate under each condition (to see if the opening flips).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--output_dir", default="experiments/interesting_queries/results/ablate_refusal_feature")
    return p.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _spec(feat: str):
    layer, feature = (int(x) for x in feat.split(":"))
    return FeatureSteeringSpec(cantor_pair(layer, feature), 0.0)


def set_ablation(model, feat: str | None) -> None:
    """Clamp one 'layer:feature' to 0 (mode='set'), or clear all steering when feat is None."""
    if feat is None:
        model.clear_feature_steering()
    else:
        model.set_feature_steering([_spec(feat)], mode="set")


def load_requests(args) -> tuple[dict[str, str], dict[str, str]]:
    data = json.loads(Path(args.strict_results).read_text())
    by_id = {r["id"]: r for r in data["records"] if r.get("selected")}
    refusal = {pid: by_id[pid]["prompt"] for pid in args.refusal_ids if pid in by_id}
    comply: dict[str, str] = {}
    if args.comply_requests:
        comply = json.loads(Path(args.comply_requests).read_text())
    return refusal, comply


@torch.no_grad()
def greedy_opening(model, tok, prompt_ids: list[int], n: int, device: str) -> list[int]:
    enc = torch.tensor([prompt_ids], device=device)
    gen = model.generate(enc, max_new_tokens=n, do_sample=False, use_cache=True, pad_token_id=tok.pad_token_id)
    return gen[0][enc.shape[1]:].tolist()


@torch.no_grad()
def teacher_forced_probs(model, prompt_ids: list[int], opening_ids: list[int], device: str) -> list[float]:
    """p(opening_ids[k] | prompt + opening_ids[:k]) for each k, in one forward pass."""
    seq = prompt_ids + opening_ids
    logits = model(torch.tensor([seq], device=device)).logits[0]  # [T, V]
    probs = []
    start = len(prompt_ids)
    for k, tid in enumerate(opening_ids):
        p = F.softmax(logits[start + k - 1].float(), dim=-1)[tid].item()
        probs.append(p)
    return probs


def main() -> None:
    setup_logging()
    args = parse_args()
    logging.getLogger().setLevel(logging.INFO)
    out = Path(args.output_dir)
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    logger.info("Invocation: ablate_refusal_feature.py %s", " ".join(f"--{k} {v}" for k, v in vars(args).items()))

    tok = AutoTokenizer.from_pretrained(args.tokenizer_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    from models.auto import AutoModelForCausalLMWithTranscoder

    model = AutoModelForCausalLMWithTranscoder.from_pretrained(args.adapter_checkpoint, dtype=_dtype(args.dtype))
    model = model.to(args.device).eval()

    refusal, comply = load_requests(args)
    conditions = [("none", None), (f"ablate_{args.ablate_feature}", args.ablate_feature),
                  (f"control_{args.control_feature}", args.control_feature)]

    records = []
    for panel, reqs in (("refusal", refusal), ("comply", comply)):
        for pid, request in reqs.items():
            prompt_ids = list(tok.apply_chat_template([{"role": "user", "content": request}], tokenize=True, add_generation_prompt=True))
            # Opening = the UN-ablated adapter's greedy opener (the sequence we score p() on).
            set_ablation(model, None)
            opening_ids = greedy_opening(model, tok, prompt_ids, args.n_positions, args.device)
            opening_text = tok.decode(opening_ids, skip_special_tokens=True)
            rec: dict[str, Any] = {"id": pid, "panel": panel, "request": request,
                                   "opening_ids": opening_ids, "opening_text": opening_text,
                                   "opening_tokens": [tok.decode([t]) for t in opening_ids], "conditions": {}}
            for cond, feat in conditions:
                set_ablation(model, feat)
                probs = teacher_forced_probs(model, prompt_ids, opening_ids, args.device)
                gen_ids = greedy_opening(model, tok, prompt_ids, args.gen_tokens, args.device)
                gen_text = tok.decode(gen_ids, skip_special_tokens=True)
                rec["conditions"][cond] = {"token_probs": probs, "gen_text": gen_text}
                logger.info("%-7s %-14s %-18s p0=%.3f meanp=%.3f gen=%r", panel, pid, cond, probs[0],
                            sum(probs) / len(probs), gen_text[:60])
            set_ablation(model, None)
            records.append(rec)

    payload = {
        "configuration": {"adapter": args.adapter_checkpoint, "ablate_feature": args.ablate_feature,
                          "control_feature": args.control_feature, "conditions": [c for c, _ in conditions],
                          "n_positions": args.n_positions, "gen_tokens": args.gen_tokens},
        "records": records,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_summary_and_plot(args, out, records, conditions)
    logger.info("Wrote %s", out / "results.json")


def write_summary_and_plot(args, out: Path, records, conditions) -> None:
    conds = [c for c, _ in conditions]
    ablate_cond = conds[1]

    # transcripts
    for rec in records:
        lines = [f"# {rec['id']} ({rec['panel']})", "", "## Request", "", "```", rec["request"], "```", "",
                 f"Un-ablated opener: `{rec['opening_text']!r}` tokens={rec['opening_tokens']}", ""]
        for cond in conds:
            e = rec["conditions"][cond]
            pr = ", ".join(f"{t}:{p:.2f}" for t, p in zip(rec["opening_tokens"], e["token_probs"]))
            lines += [f"## {cond}", "", f"p(opener token | prefix): {pr}", "", "greedy gen:", "```", e["gen_text"] or "(empty)", "```", ""]
        (out / "transcripts" / f"{rec['id']}__{rec['panel']}.md").write_text("\n".join(lines) + "\n")

    def mean_p(rec, cond):
        p = rec["conditions"][cond]["token_probs"]
        return sum(p) / len(p) if p else 0.0

    lines = ["# Ablating L6/F4241: is it causally necessary for the refusal opening?", "",
             f"Adapter `{args.adapter_checkpoint.split('/')[-1]}`. Clamp a feature to 0 (mode=set) and measure "
             f"p(the adapter's own opening token) at the first {args.n_positions} positions, teacher-forced. "
             f"Candidate = **{args.ablate_feature}** (the one refusal-specific feature); control = "
             f"**{args.control_feature}** (a generic high-influence feature). If the candidate is causally "
             "necessary, ablating it drops the refusal opener's probability far more than the control does, and "
             "flips the greedy opening from refusal to compliance.", ""]
    for panel in ("refusal", "comply"):
        prs = [r for r in records if r["panel"] == panel]
        if not prs:
            continue
        reading = "refusal should DROP if L6/F4241 is necessary" if panel == "refusal" else "opener should be UNCHANGED (feature doesn't fire on compliance)"
        lines += [f"## {panel.title()} prompts (n={len(prs)}) — {reading}", "",
                  "| id | opener | mean p(opener): none | " + f"{ablate_cond} | {conds[2]} | none→ablate Δ |",
                  "|---|---|---:|---:|---:|---:|"]
        for r in prs:
            m_none, m_abl, m_ctl = mean_p(r, "none"), mean_p(r, ablate_cond), mean_p(r, conds[2])
            opener = r["opening_text"].replace("\n", " ")[:24]
            lines.append(f"| `{r['id']}` | {opener!r} | {m_none:.3f} | {m_abl:.3f} | {m_ctl:.3f} | {m_abl-m_none:+.3f} |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines) + "\n")

    # Plot: p(opener token) vs position, none vs ablate vs control, one subplot per refusal prompt.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    refusal_recs = [r for r in records if r["panel"] == "refusal"]
    if not refusal_recs:
        return
    fig, axes = plt.subplots(1, len(refusal_recs), figsize=(4.6 * len(refusal_recs), 4.2), squeeze=False)
    colours = {conds[0]: "#718096", ablate_cond: "#c53030", conds[2]: "#2b6cb0"}
    for ax, r in zip(axes[0], refusal_recs):
        ks = list(range(len(r["opening_ids"])))
        for cond in conds:
            ax.plot(ks, r["conditions"][cond]["token_probs"], marker="o", label=cond, color=colours.get(cond))
        ax.set_title(f"{r['id']}\n{r['opening_text'][:22]!r}", fontsize=9)
        ax.set_xlabel("opener position"); ax.set_ylim(-.03, 1.03); ax.set_xticks(ks); ax.grid(alpha=.3)
        ax.set_xticklabels([t.strip()[:6] for t in r["opening_tokens"]], rotation=45, fontsize=7)
    axes[0][0].set_ylabel("p(opener token | prefix)")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"Ablating {args.ablate_feature} (red) vs generic control {args.control_feature} (blue) vs none (grey)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "ablation_refusal.pdf"); fig.savefig(out / "ablation_refusal.png", dpi=150)
    logger.info("wrote %s", out / "ablation_refusal.pdf")


if __name__ == "__main__":
    main()
