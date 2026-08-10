#!/usr/bin/env python
"""Refusal-prefill flip test: is it the opening tokens ("I", "I cannot", ...) that *load* the refusal?

Session "[gemma-attribution: refusal token tracing]" (fd1f0d19-f5d1-48cc-9002-3f06f2abe2fd), 2026-08-10.

Motivating question (Anthropic-circuits style): the instruct model refuses a harmful request starting
with "I cannot ...". Does that refusal live in the *token identity* of the opening ("I"), or in the
instruct-tuning machinery driving the forward pass? We test it directly by transplanting the instruct
model's own refusal opening onto the BASE model and asking whether the base model then carries the
refusal forward.

For each harmful prompt we:
  1. Render the user turn with the gemma-2-2b-it chat template (``--prompt_template chat``) -- or the
     neutral "User:/Assistant:" plaintext template (``--prompt_template plain``, better in-distribution
     for the base model; see analysis.attribution.run_attribution).
  2. Greedily generate the PREFILL-SOURCE model's continuation (default the instruct model = the refusal)
     and keep its first ``--max_prefill`` assistant token ids.
  3. For each prefix length k = 0..max_prefill and each ``--continue_with`` model, PREFILL the first k
     source tokens after the prompt and greedily continue that model for ``--max_new_tokens`` tokens.
  4. Judge the FULL assistant turn (prefilled opening + the continuation the model actually produced)
     with the same independent judge as find_strict_compliance_refusal, so labels are comparable.

Reading the result:
  - k=0 is each model's UN-prefilled behaviour (base should comply, instruct should refuse).
  - If the base curve stays COMPLIANCE as k grows -- even after prefilling "I", "I cannot", "I cannot
    provide instructions ..." -- then the opening tokens do NOT load the refusal; the instruct machinery
    does. If base flips to REFUSAL at some k, that token span is what carries it.
  - The instruct ``--continue_with`` arm is a sanity curve (prefilling its own opening should keep it
    refusing) and calibrates how much of any base flip is just language-modelling the injected words.

Prompt set: the verified base-complies/instruct-refuses flips from find_strict_compliance_refusal
(results.json ``records`` with ``selected: true``), sub-selected by ``--prompt_ids``.

Outputs (``--output_dir``): results.json, summary.md, prefill_flip.png, transcripts/<id>__<arm>__k<k>.md
(every generation saved -- the record of what each model actually said at each prefix length).

Example:
    PYTHONPATH=. uv run --extra viz python experiments/interesting_queries/scripts/refusal_prefill_flip.py \\
        --prompt_ids harm_125 harm_139 harm_116 --continue_with base instruct --max_prefill 10
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
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from helpers.log import logger, setup_logging  # noqa: E402

from experiments.interesting_queries.scripts.find_strict_compliance_refusal import (  # noqa: E402
    JUDGE_SYSTEM,
    generate_texts,
    judge_user_prompt,
    parse_judgment,
    release_model,
)

BASE_MODEL = "google/gemma-2-2b"
INSTRUCT_MODEL = "google/gemma-2-2b-it"
DEPLOYED_ADAPTER = "siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860"
STRICT_RESULTS = "experiments/interesting_queries/results/strict_compliance_refusal/results.json"

# Neutral plaintext dialogue template (kept in sync with run_attribution.PLAIN_* so the prefill test and
# the attribution graphs speak the same prompt language). Used by --prompt_template plain.
PLAIN_USER_PREFIX = "User: "
PLAIN_ASSISTANT_PREFIX = "Assistant:"

OUTCOMES = ("REFUSAL", "COMPLIANCE", "INCONCLUSIVE", "GIBBERISH")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt_ids", nargs="+", default=["harm_125", "harm_139", "harm_116"],
                   help="Which selected strict-flip records to test (ids in --strict_results). Default: meth (chem/bio), DDoS (cyber), pipe bomb (weapons) -- three distinct categories with crisp 'I cannot' refusals.")
    p.add_argument("--continue_with", nargs="+", default=["base", "instruct"],
                   help="Models that continue from the prefilled opening. 'base'=google/gemma-2-2b (the key arm), 'instruct'=gemma-2-2b-it (sanity), 'adapter'=the transcoder adapter. The base curve staying COMPLIANCE across k is the evidence the opening tokens don't load the refusal.")
    p.add_argument("--prefill_source", default="instruct", choices=["instruct", "adapter"],
                   help="Whose refusal opening is transplanted. 'instruct' = the real gemma-2-2b-it refusal (the intended test); 'adapter' = the transcoder adapter's refusal.")
    p.add_argument("--prompt_template", default="chat", choices=["chat", "plain"],
                   help="'chat' renders the gemma-2-2b-it chat template (matches how the instruct refusal was produced; base is somewhat off-distribution under it). 'plain' renders neutral 'User:/Assistant:' text (base is in-distribution -- use if the base arm comes out GIBBERISH under chat).")
    p.add_argument("--max_prefill", type=int, default=10, help="Longest source-refusal prefix (in tokens) to transplant; the sweep runs k=0..max_prefill.")
    p.add_argument("--base_model", default=BASE_MODEL, help="Base model = the un-tuned bottom of the ladder and the key --continue_with arm.")
    p.add_argument("--instruct_model", default=INSTRUCT_MODEL, help="Instruct model = the refusal source and the sanity --continue_with arm; also supplies the chat template for --prompt_template chat.")
    p.add_argument("--adapter_checkpoint", default=DEPLOYED_ADAPTER, help="Transcoder adapter checkpoint used when 'adapter' appears in --continue_with or --prefill_source.")
    p.add_argument("--strict_results", default=STRICT_RESULTS, help="results.json from find_strict_compliance_refusal; supplies the harmful prompts (records with selected=true).")
    p.add_argument("--max_new_tokens", type=int, default=120, help="Greedy tokens generated after the prefilled opening (per model, per k).")
    p.add_argument("--source_max_new_tokens", type=int, default=64, help="Greedy tokens generated from the prefill-source to harvest its refusal opening (only its first --max_prefill tokens are used).")
    p.add_argument("--batch_size", type=int, default=8, help="Generation batch size; lower if a large model runs out of GPU memory.")
    p.add_argument("--judge_model", default="Qwen/Qwen3-8B", help="Independent model labelling each full assistant turn; same judge as find_strict_compliance_refusal so labels are comparable.")
    p.add_argument("--judge_max_new_tokens", type=int, default=160, help="Token budget per judge verdict (LABEL/CONFIDENCE/RATIONALE).")
    p.add_argument("--judge_batch_size", type=int, default=8, help="Judge generation batch size.")
    p.add_argument("--device", default="cuda", help="Torch device.")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Model dtype; bfloat16 matches training/attribution.")
    p.add_argument("--output_dir", default=None, help="Where results.json / summary.md / plot / transcripts go (default: experiments/interesting_queries/results/refusal_prefill_flip).")
    p.add_argument("--skip_judge", action="store_true", help="Generate + save transcripts but skip judging (eyeball completions before spending judge compute).")
    return p.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def load_selected(args: argparse.Namespace) -> list[dict[str, Any]]:
    data = json.loads(Path(args.strict_results).read_text())
    by_id = {r["id"]: r for r in data["records"] if r.get("selected")}
    prompts = []
    for pid in args.prompt_ids:
        if pid not in by_id:
            raise SystemExit(f"prompt id {pid!r} not among selected records in {args.strict_results}")
        rec = by_id[pid]
        prompts.append({"id": pid, "category": rec.get("category", ""), "request": rec["prompt"]})
    logger.info("Prompts: %s", ", ".join(p["id"] for p in prompts))
    return prompts


def render_prompt_ids(tokenizer: Any, request: str, template: str) -> list[int]:
    """Prompt token ids up to (but not including) the first assistant token, per --prompt_template."""
    if template == "chat":
        return list(tokenizer.apply_chat_template(
            [{"role": "user", "content": request}], tokenize=True, add_generation_prompt=True))
    prefix = f"{PLAIN_USER_PREFIX}{request}\n{PLAIN_ASSISTANT_PREFIX}"
    return tokenizer.encode(prefix, add_special_tokens=True)


def _model_path(name: str, args: argparse.Namespace) -> tuple[str, str]:
    """(load kind, path) for an arm name. kind 'plain' -> HF causal LM, 'transcoder' -> adapter wrapper."""
    if name == "base":
        return "plain", args.base_model
    if name == "instruct":
        return "plain", args.instruct_model
    if name == "adapter":
        return "transcoder", args.adapter_checkpoint
    raise SystemExit(f"unknown model arm {name!r}")


def load_model(name: str, args: argparse.Namespace):
    kind, path = _model_path(name, args)
    dtype = _dtype(args.dtype)
    if kind == "transcoder":
        from models.auto import AutoModelForCausalLMWithTranscoder

        model = AutoModelForCausalLMWithTranscoder.from_pretrained(path, dtype=dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    model.to(args.device).eval()
    return model


def harvest_source_openings(args: argparse.Namespace, tokenizer: Any, prompts: list[dict[str, Any]]) -> None:
    """Greedy-generate the prefill-source continuation per prompt; store its first max_prefill token ids.

    Mutates each prompt dict in place with:
      prompt_ids   : rendered prompt token ids (no assistant tokens yet)
      source_ids   : the source model's assistant continuation token ids (first max_prefill kept)
      source_text  : decoded source continuation (the refusal we are transplanting)
    """
    logger.info("Harvesting %s refusal openings (%d tokens each)", args.prefill_source, args.max_prefill)
    model = load_model(args.prefill_source, args)
    try:
        for p in prompts:
            p["prompt_ids"] = render_prompt_ids(tokenizer, p["request"], args.prompt_template)
            enc = tokenizer.pad({"input_ids": [p["prompt_ids"]]}, padding=True, return_tensors="pt").to(args.device)
            gen = model.generate(**enc, max_new_tokens=args.source_max_new_tokens, do_sample=False,
                                 use_cache=True, pad_token_id=tokenizer.pad_token_id)
            cont_ids = gen[0][enc["input_ids"].shape[1]:].tolist()
            p["source_ids"] = cont_ids[: args.max_prefill]
            p["source_text"] = tokenizer.decode(p["source_ids"], skip_special_tokens=True)
            logger.info("  %s source opening: %r", p["id"], p["source_text"])
    finally:
        release_model(model)


def run_continuer(arm: str, args: argparse.Namespace, tokenizer: Any, prompts: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, str]]:
    """For one --continue_with model, sweep k=0..max_prefill over every prompt.

    Returns {(prompt_id, k): {"prefill": decoded prefilled opening, "continuation": model text,
    "assistant": prefill+continuation}}. One batched generate() over all (prompt, k) inputs.
    """
    keys: list[tuple[str, int]] = []
    encoded: list[list[int]] = []
    for p in prompts:
        for k in range(args.max_prefill + 1):
            keys.append((p["id"], k))
            encoded.append(p["prompt_ids"] + p["source_ids"][:k])

    model = load_model(arm, args)
    try:
        conts = generate_texts(model, tokenizer, encoded, device=args.device, batch_size=args.batch_size,
                               max_new_tokens=args.max_new_tokens, do_sample=False)
    finally:
        release_model(model)

    by_id = {p["id"]: p for p in prompts}
    out: dict[tuple[str, int], dict[str, str]] = {}
    for (pid, k), cont in zip(keys, conts):
        prefill = tokenizer.decode(by_id[pid]["source_ids"][:k], skip_special_tokens=True)
        assistant = (prefill + cont).strip()
        out[(pid, k)] = {"prefill": prefill, "continuation": cont, "assistant": assistant}
    return out


def judge_all(args: argparse.Namespace, prompts: list[dict[str, Any]],
              results: dict[str, dict[tuple[str, int], dict[str, str]]]) -> dict[str, dict[tuple[str, int], dict[str, Any]]]:
    """Label every (arm, prompt, k) full assistant turn with the independent judge."""
    logger.info("Loading judge %s", args.judge_model)
    jt = AutoTokenizer.from_pretrained(args.judge_model)
    if jt.pad_token_id is None:
        jt.pad_token = jt.eos_token
    jt.padding_side = "left"
    judge = AutoModelForCausalLM.from_pretrained(args.judge_model, dtype=_dtype(args.dtype))
    judge.to(args.device).eval()

    req = {p["id"]: p["request"] for p in prompts}
    verdicts: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    try:
        for arm, per_key in results.items():
            keys = list(per_key.keys())
            encoded = []
            for (pid, k) in keys:
                msgs = [{"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": judge_user_prompt(req[pid], per_key[(pid, k)]["assistant"] or "(empty)")}]
                encoded.append(jt.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False))
            raw = generate_texts(judge, jt, encoded, device=args.device, batch_size=args.judge_batch_size,
                                 max_new_tokens=args.judge_max_new_tokens, do_sample=False)
            verdicts[arm] = {key: parse_judgment(t) for key, t in zip(keys, raw)}
            labels = [v["label"] for v in verdicts[arm].values()]
            logger.info("judged %-10s %s", arm, {o: labels.count(o) for o in OUTCOMES})
    finally:
        release_model(judge)
    return verdicts


def write_outputs(args: argparse.Namespace, out: Path, prompts, arms, results, verdicts) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    tdir = out / "transcripts"
    tdir.mkdir(exist_ok=True)

    records = []
    for p in prompts:
        for arm in arms:
            for k in range(args.max_prefill + 1):
                e = results[arm][(p["id"], k)]
                rec = {"id": p["id"], "category": p["category"], "request": p["request"], "arm": arm, "k": k,
                       "prefill": e["prefill"], "continuation": e["continuation"], "assistant": e["assistant"]}
                if verdicts:
                    rec.update({key: verdicts[arm][(p["id"], k)][key] for key in ("label", "confidence", "rationale")})
                records.append(rec)
                # Transcript per cell.
                lines = [f"# {p['id']} — arm `{arm}` — k={k} prefilled tokens", "",
                         f"- Category: `{p['category']}`", f"- Prompt template: `{args.prompt_template}`",
                         f"- Prefill source: `{args.prefill_source}`  (opening: `{p['source_text']!r}`)", ""]
                if "label" in rec:
                    lines += [f"**Judge: {rec['label']}** (conf {rec['confidence']:.2f})", "", f"*{rec['rationale']}*", ""]
                lines += ["## Request", "", "```", p["request"], "```", "",
                          f"## Prefilled opening (k={k})", "", "```", e["prefill"] or "(none)", "```", "",
                          "## Model continuation", "", "```", e["continuation"] or "(empty)", "```", ""]
                (tdir / f"{p['id']}__{arm}__k{k:02d}.md").write_text("\n".join(lines) + "\n")

    # Refusal rate per (arm, k), averaged over prompts.
    stats: dict[str, dict[int, dict[str, Any]]] = {}
    if verdicts:
        for arm in arms:
            stats[arm] = {}
            for k in range(args.max_prefill + 1):
                labels = [verdicts[arm][(p["id"], k)]["label"] for p in prompts]
                counts = {o: labels.count(o) for o in OUTCOMES}
                n = len(prompts)
                stats[arm][k] = {"n": n, "counts": counts, "refusal_rate": counts["REFUSAL"] / n if n else 0.0}

    payload = {
        "configuration": {
            "prompt_ids": [p["id"] for p in prompts], "continue_with": arms, "prefill_source": args.prefill_source,
            "prompt_template": args.prompt_template, "max_prefill": args.max_prefill,
            "base_model": args.base_model, "instruct_model": args.instruct_model,
            "max_new_tokens": args.max_new_tokens, "decoding": "greedy",
            "judge_model": None if args.skip_judge else args.judge_model, "strict_results": args.strict_results,
        },
        "source_openings": {p["id"]: {"text": p["source_text"], "token_ids": p["source_ids"]} for p in prompts},
        "stats": stats,
        "records": records,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2) + "\n")

    # summary.md
    lines = ["# Refusal-prefill flip: do the opening tokens load the refusal?", "",
             f"Prompt template `{args.prompt_template}`; prefill source `{args.prefill_source}`; each `--continue_with` "
             f"model greedy-decodes {args.max_new_tokens} tokens after the transplanted opening; judged by "
             f"`{args.judge_model if not args.skip_judge else 'skipped'}`. Refusal rate = fraction of the "
             f"{len(prompts)} prompts judged REFUSAL (full assistant turn = prefill + continuation).", ""]
    for p in prompts:
        lines.append(f"- `{p['id']}` ({p['category']}) source opening: `{p['source_text']!r}`")
    lines.append("")
    if stats:
        lines += ["## Refusal rate vs. tokens prefilled", "",
                  "| k (prefilled) | " + " | ".join(f"`{a}`" for a in arms) + " |",
                  "|---:|" + "---:|" * len(arms)]
        for k in range(args.max_prefill + 1):
            row = " | ".join(f"{stats[a][k]['refusal_rate']*100:.0f}%" for a in arms)
            lines.append(f"| {k} | {row} |")
        lines.append("")
        if "base" in stats:
            b0, bmax = stats["base"][0]["refusal_rate"] * 100, stats["base"][args.max_prefill]["refusal_rate"] * 100
            lines += [f"**Base refusal:** {b0:.0f}% with no prefill → {bmax:.0f}% after prefilling {args.max_prefill} "
                      "instruct-refusal tokens. If this stays near 0%, the opening tokens do not load the refusal; the "
                      "instruct machinery does.", ""]
    lines += ["Transcripts: [`transcripts/`](transcripts) (every model's continuation at every k). Chart: `prefill_flip.png`.", ""]
    (out / "summary.md").write_text("\n".join(lines) + "\n")

    if stats:
        plot(out, arms, args, stats)
    return payload


def plot(out: Path, arms, args, stats) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = list(range(args.max_prefill + 1))
    colours = {"base": "#c53030", "instruct": "#2b6cb0", "adapter": "#2f855a"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for a in arms:
        ax.plot(ks, [stats[a][k]["refusal_rate"] * 100 for k in ks], marker="o",
                label=a, color=colours.get(a))
    ax.set_xlabel(f"# of {args.prefill_source} refusal tokens prefilled onto the model (k)")
    ax.set_ylabel("% of prompts judged REFUSAL")
    ax.set_ylim(-3, 103)
    ax.set_xticks(ks)
    ax.grid(alpha=0.3)
    ax.legend(title="continued with")
    ax.set_title(f"Does prefilling the {args.prefill_source} refusal opening make the base model refuse?\n"
                 f"template={args.prompt_template}, n={stats[arms[0]][0]['n']} prompts, greedy")
    fig.tight_layout()
    fig.savefig(out / "prefill_flip.png", dpi=160)
    logger.info("wrote %s", out / "prefill_flip.png")


def main() -> None:
    setup_logging()
    args = parse_args()
    logging.getLogger().setLevel(logging.INFO)
    out = Path(args.output_dir or "experiments/interesting_queries/results/refusal_prefill_flip")

    logger.info("Invocation: refusal_prefill_flip.py %s", " ".join(f"--{k} {v}" for k, v in vars(args).items()))

    tokenizer = AutoTokenizer.from_pretrained(args.instruct_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    prompts = load_selected(args)
    harvest_source_openings(args, tokenizer, prompts)

    results: dict[str, dict[tuple[str, int], dict[str, str]]] = {}
    for arm in args.continue_with:
        logger.info("=== continue_with %s ===", arm)
        results[arm] = run_continuer(arm, args, tokenizer, prompts)
        sample = results[arm][(prompts[0]["id"], 0)]["assistant"][:120]
        logger.info("arm %s k=0 sample: %r", arm, sample)

    verdicts = None if args.skip_judge else judge_all(args, prompts, results)
    payload = write_outputs(args, out, prompts, args.continue_with, results, verdicts)

    logger.info("Artifacts: %s", ", ".join(str(out / f) for f in ("results.json", "summary.md", "prefill_flip.png", "transcripts/")))
    for arm, per_k in payload["stats"].items():
        rates = " ".join(f"k{k}:{per_k[k]['refusal_rate']*100:.0f}%" for k in sorted(per_k))
        logger.info("  %-10s %s", arm, rates)


if __name__ == "__main__":
    main()
