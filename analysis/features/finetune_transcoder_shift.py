"""Re-fine-tune GemmaScope transcoders to the instruct input distribution.

<!-- Created by Claude Code session "implement: new 07/02 gemmascope transcoder experiments". 2026-07-09. -->

Follow-up to ``transcoder_input_shift.py`` (Experiment 1), which showed that the pretrained
GemmaScope transcoders reconstruct the base MLP worse when fed *instruct* hidden states
(FVU up +32-104% at the endpoint layers). The standard fix when transferring SAEs/transcoders
across a base->fine-tuned pair is a short re-fine-tune on the new input distribution. This
script does exactly that, per layer:

    x        = instruct model's blocks.L.ln2.hook_normalized   (the shifted input)
    target   = MLP_base(x)                                     (what the transcoder imitates,
                                                                via a patched base forward)
    loss     = MSE( transcoder(x), target ) + l1_coeff * L1(features)

The optional decoder-norm-weighted L1 sparsity penalty (``--l1_coeff``, matching
``models/gemma2_transcoder.py``) is off by default. Without it, reconstruction-only fine-tuning
lets L0 drift up (the frozen JumpReLU threshold no longer matches the shifted W_enc/b_enc, so more
features cross it): observed L25 60->81 vs base 65. With it, activations are pushed back under the
threshold so L0 stays near the base operating point while FVU still improves. Recommended for
longer runs (e.g. ~10M tokens); tune l1_coeff so after-FT L0 ~= base L0.

Only the selected transcoders' parameters (W_enc, b_enc, W_dec, b_dec; threshold optional and
frozen by default so the L0 operating point is preserved) are trained; both models stay frozen
and x/target are computed under no_grad. We eval FVU/L0 on a held-out instruct stream BEFORE
and AFTER, so the run reports exactly how much of the input-shift gap the fine-tune closes.

Reuses model/transcoder loading, hidden-state extraction, and the streaming metric accumulator
from ``transcoder_input_shift``.

Usage (fine-tune the worst-hit endpoint layers):
    uv run --extra viz python -m analysis.features.finetune_transcoder_shift \
        --gemmascope_width width_16k --gemmascope_l0 average_l0_76 \
        --layers 0 24 25 --train_tokens 2000000 --eval_tokens 200000 --lr 1e-4 --wandb
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path

import torch

from helpers.log import logger, setup_logging
from helpers.paths.output_path import generate_output_path

from analysis.attribution.run_base_adapter_comparison import (
    _load_gemmascope_transcoders,
    build_gemmascope_transcoder_config,
    resolve_gemmascope_l0_values,
)
from analysis.features.batching import compute_max_batch_tokens, form_length_packed_batches
from analysis.features.collection_wandb import CollectionWandbLogger
from analysis.features.transcoder_input_shift import (
    ShiftAccumulator,
    _torch_dtype,
    build_valid_mask,
    compute_base_target,
    compute_source_x,
    load_hooked_model,
    pad_batch,
    parse_layers,
    prepare_items,
)


def trainable_params(transcoder, train_threshold: bool) -> list[torch.nn.Parameter]:
    """Encoder/decoder + biases (and optionally the JumpReLU threshold) for one transcoder."""
    params = [transcoder.W_enc, transcoder.b_enc, transcoder.W_dec, transcoder.b_dec]
    params = [p for p in params if isinstance(p, torch.nn.Parameter)]
    if train_threshold and hasattr(transcoder.activation_function, "threshold"):
        params.append(transcoder.activation_function.threshold)
    for p in params:
        p.requires_grad_(True)
    return params


@torch.no_grad()
def evaluate(
    *, instruct_model, base_model, transcoders, layers, batches, args, d_model, device,
) -> dict[int, dict[str, float]]:
    """FVU/L0 per layer on a held-out instruct stream (target = MLP_base(x))."""
    n_features = transcoders[layers[0]].W_enc.shape[0]
    accs = {layer: ShiftAccumulator(d_model, n_features, device) for layer in layers}
    seen = 0
    for batch in batches:
        toks = pad_batch([it[0] for it in batch], args.pad_id, device)
        mask = build_valid_mask(toks, args.pad_id).reshape(-1)
        x_by_layer, _ = compute_source_x(instruct_model, toks, layers, args.feature_input_hook, args.feature_output_hook, capture_mlp_out=False)
        target_by_layer = compute_base_target(base_model, toks, x_by_layer, layers, args.feature_input_hook, args.feature_output_hook)
        for layer in layers:
            x = x_by_layer[layer].reshape(-1, d_model)[mask]
            target = target_by_layer[layer].reshape(-1, d_model)[mask]
            feats = transcoders[layer].encode(x)
            recon = transcoders[layer].decode(feats, x)
            accs[layer].update(feats, recon, target)
        seen += int(mask.sum().item())
        if seen >= args.eval_tokens:
            break
    return {layer: accs[layer].metrics() for layer in layers}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_model", default="google/gemma-2-2b")
    p.add_argument("--instruct_model", default="google/gemma-2-2b-it")
    p.add_argument("--layers", nargs="+", default=["0", "24", "25"], help="Layers to fine-tune (default: the worst-hit endpoints). 'all' = 0..n_layers-1.")
    p.add_argument("--gemmascope_repo", default="google/gemma-scope-2b-pt-transcoders")
    p.add_argument("--gemmascope_width", required=True)
    p.add_argument("--gemmascope_l0", required=True)
    p.add_argument("--gemmascope_l0_match", default="nearest", choices=["exact", "nearest"])
    p.add_argument("--gemmascope_n_layers", type=int, default=26)
    p.add_argument("--feature_input_hook", default="ln2.hook_normalized")
    p.add_argument("--feature_output_hook", default="hook_mlp_out")
    p.add_argument("--val_data", nargs="+", default=["chat:siddharthmb/2026.transcoder-adapters.lmsys-chat-1m-splits"])
    p.add_argument("--prompt_tokenizer_model", default="google/gemma-2-2b-it")
    p.add_argument("--train_tokens", type=int, default=2_000_000, help="Approx number of instruct tokens to train on (one pass).")
    p.add_argument("--eval_tokens", type=int, default=200_000, help="Held-out instruct tokens for before/after FVU/L0 eval.")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate for the transcoder params.")
    p.add_argument("--l1_coeff", type=float, default=0.0, help="Sparsity penalty coefficient on the decoder-norm-weighted L1 of the (post-JumpReLU) features. 0 = pure reconstruction (L0 drifts up as W_enc/b_enc shift under the frozen threshold). >0 holds L0 near the base operating point while reconstruction improves; tune so after-FT L0 ~= base L0 (try ~1e-3 and adjust). Recommended for longer runs (e.g. 10M tokens).")
    p.add_argument("--train_threshold", action="store_true", help="Also train the JumpReLU threshold (changes L0). Off by default to preserve the sparsity operating point.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_log_every", type=int, default=10)
    return p


def main() -> None:
    setup_logging()
    args = build_parser().parse_args()
    device = torch.device(args.device)
    dtype = _torch_dtype(args.dtype)
    layers = parse_layers(args.layers, args.gemmascope_n_layers)

    tag = f"ft_gemmascope_{args.gemmascope_width}_{args.gemmascope_l0}_L{'-'.join(map(str, layers))}"
    output_dir = args.output_dir or generate_output_path("transcoder_input_shift_finetune", tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Invocation: %s", " ".join(shlex.quote(a) for a in [sys.executable, *sys.argv]))
    logger.info("Output dir: %s | fine-tuning layers %s", output_dir, layers)

    # Transcoders (start from pretrained GemmaScope).
    l0_values = resolve_gemmascope_l0_values(repo=args.gemmascope_repo, width=args.gemmascope_width, l0=args.gemmascope_l0, n_layers=args.gemmascope_n_layers, l0_match=args.gemmascope_l0_match)
    ts_config = build_gemmascope_transcoder_config(repo=args.gemmascope_repo, width=args.gemmascope_width, l0=args.gemmascope_l0, n_layers=args.gemmascope_n_layers, model_name=args.base_model, feature_input_hook=args.feature_input_hook, feature_output_hook=args.feature_output_hook, l0_values=l0_values, l0_match=args.gemmascope_l0_match)
    (output_dir / "gemmascope_config.json").write_text(json.dumps(ts_config, indent=2) + "\n")
    transcoders = _load_gemmascope_transcoders(ts_config, device=device, dtype=dtype)

    base_model = load_hooked_model(args.base_model, device=device, dtype=dtype)
    instruct_model = load_hooked_model(args.instruct_model, device=device, dtype=dtype)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.prompt_tokenizer_model)
    args.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    d_model = base_model.cfg.d_model

    # Data: hold out the last chunk of (shuffled) batches for eval; train on the rest.
    items4 = [(toks, dom, {}, {}) for toks, dom in prepare_items(args, tokenizer)]
    max_batch_tokens = compute_max_batch_tokens(device=device, n_layers=len(layers), n_features=transcoders[layers[0]].W_enc.shape[0], hidden=d_model)
    batches = form_length_packed_batches(items4, batch_size=args.batch_size, max_batch_tokens=max_batch_tokens, shuffle=True, shuffle_seed=0)
    n_eval_batches = max(1, min(len(batches) // 5, int(args.eval_tokens / max(1, args.batch_size * 64)) + 1))
    eval_batches, train_batches = batches[:n_eval_batches], batches[n_eval_batches:]
    logger.info("Split: %d eval batches, %d train batches", len(eval_batches), len(train_batches))

    # Cast trainable transcoder params to fp32 for stable optimization.
    params = []
    for layer in layers:
        transcoders[layer].float()
        params += trainable_params(transcoders[layer], args.train_threshold)
    optimizer = torch.optim.Adam(params, lr=args.lr)
    logger.info("Training %d param tensors (%.1fM elems) across %d layers", len(params), sum(p.numel() for p in params) / 1e6, len(layers))

    wandb_logger = CollectionWandbLogger(enabled=args.wandb, project=args.wandb_project, entity=args.wandb_entity, run_name=args.wandb_run_name or output_dir.name, config=vars(args) | {"tag": tag, "layers": layers}, log_every=args.wandb_log_every)

    # --- BEFORE eval ---
    logger.info("Eval BEFORE fine-tuning...")
    before = evaluate(instruct_model=instruct_model, base_model=base_model, transcoders=transcoders, layers=layers, batches=eval_batches, args=args, d_model=d_model, device=device)
    for layer in layers:
        logger.info("  [before] L%d: FVU=%.4f L0=%.1f", layer, before[layer]["fvu"], before[layer]["l0_mean"])

    # --- Train (one pass, streaming) ---
    logger.info("Training...")
    wandb_logger.reset_timer()
    trained_tokens, step, t0 = 0, 0, time.time()
    for batch in train_batches:
        if trained_tokens >= args.train_tokens:
            break
        toks = pad_batch([it[0] for it in batch], args.pad_id, device)
        mask = build_valid_mask(toks, args.pad_id).reshape(-1)
        x_by_layer, _ = compute_source_x(instruct_model, toks, layers, args.feature_input_hook, args.feature_output_hook, capture_mlp_out=False)
        target_by_layer = compute_base_target(base_model, toks, x_by_layer, layers, args.feature_input_hook, args.feature_output_hook)
        optimizer.zero_grad()
        loss = torch.zeros((), device=device, dtype=torch.float32)
        mse_sum = l1_sum = l0_sum = 0.0
        for layer in layers:
            x = x_by_layer[layer].reshape(-1, d_model)[mask].float()
            target = target_by_layer[layer].reshape(-1, d_model)[mask].float()
            feats = transcoders[layer].encode(x)                       # post-JumpReLU, [N, n_features]
            recon = transcoders[layer].decode(feats, x)
            mse = torch.nn.functional.mse_loss(recon, target)
            loss = loss + mse
            mse_sum += mse.item()
            if args.l1_coeff > 0:
                # Decoder-norm-weighted L1 (matches models/gemma2_transcoder.py sparsity loss):
                # sum_i ||W_dec_i|| * feat_i, mean over tokens. Penalizes feature magnitude ->
                # pushes activations under the frozen JumpReLU threshold -> holds L0 down.
                dec_norms = transcoders[layer].W_dec.float().norm(dim=1)  # [n_features]
                l1 = (feats * dec_norms).sum(dim=-1).mean()
                loss = loss + args.l1_coeff * l1
                l1_sum += l1.item()
            l0_sum += (feats > 0).float().sum(dim=-1).mean().item()
        loss.backward()
        optimizer.step()
        n = int(mask.sum().item())
        trained_tokens += n
        step += 1
        wandb_logger.log_batch(n_seqs=len(batch), n_tokens=n)
        if args.wandb:
            wandb_logger._wandb.log({"train/loss": loss.item(), "train/mse": mse_sum, "train/l1": l1_sum, "train/l0": l0_sum / len(layers), "train/tokens": trained_tokens})
        if step % 25 == 0:
            logger.info("step %d | %d/%d tok | loss=%.5f mse=%.5f l1=%.4f l0=%.1f | %.0f tok/s", step, trained_tokens, args.train_tokens, loss.item(), mse_sum, l1_sum, l0_sum / len(layers), trained_tokens / max(1e-6, time.time() - t0))

    # --- AFTER eval ---
    logger.info("Eval AFTER fine-tuning...")
    after = evaluate(instruct_model=instruct_model, base_model=base_model, transcoders=transcoders, layers=layers, batches=eval_batches, args=args, d_model=d_model, device=device)

    # --- Save fine-tuned transcoder params + before/after report ---
    from safetensors.torch import save_file

    for layer in layers:
        t = transcoders[layer]
        sd = {"W_enc": t.W_enc.detach().cpu(), "b_enc": t.b_enc.detach().cpu(), "W_dec": t.W_dec.detach().cpu(), "b_dec": t.b_dec.detach().cpu()}
        if hasattr(t.activation_function, "threshold"):
            sd["threshold"] = t.activation_function.threshold.detach().cpu()
        save_file(sd, str(output_dir / f"finetuned_layer_{layer}.safetensors"))

    report = {"config": vars(args) | {"layers": layers, "scan_name": ts_config["scan_name"], "output_dir": str(output_dir)}, "before": {str(l): before[l] for l in layers}, "after": {str(l): after[l] for l in layers}}
    report["config"].pop("pad_id", None)
    (output_dir / "finetune_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    write_summary(output_dir, report, layers, args)

    for layer in layers:
        b, a = before[layer], after[layer]
        dfvu = 100 * (a["fvu"] - b["fvu"]) / b["fvu"]
        logger.info("L%d: FVU %.4f -> %.4f (%+.0f%%) | L0 %.1f -> %.1f", layer, b["fvu"], a["fvu"], dfvu, b["l0_mean"], a["l0_mean"])
    logger.info("Artifacts: %s (finetune_report.json, summary.md, finetuned_layer_*.safetensors)", output_dir)
    wandb_logger.finish(summary={f"final/L{l}/{k}/{ba}": v for l in layers for ba, d in (("before", before), ("after", after)) for k, v in d[l].items()})


def write_summary(output_dir: Path, report: dict, layers, args) -> None:
    lines = [
        f"# Transcoder re-fine-tune on instruct inputs — {report['config']['scan_name']}",
        "",
        f"Fine-tuned GemmaScope `{args.gemmascope_width}`/`{args.gemmascope_l0}` transcoders at layers "
        f"{layers} on ~{args.train_tokens:,} instruct tokens (loss = MSE(transcoder(x), MLP_base(x))"
        f"{f' + {args.l1_coeff:g}*L1(features)' if args.l1_coeff > 0 else ' (no sparsity penalty)'}, "
        f"x = instruct ln2.hook_normalized; threshold {'trained' if args.train_threshold else 'frozen'}). "
        f"FVU/L0 on {args.eval_tokens:,} held-out instruct tokens, before vs after.",
        "",
        "| layer | FVU before | FVU after | ΔFVU | L0 before | L0 after |",
        "|---|---|---|---|---|---|",
    ]
    for layer in layers:
        b, a = report["before"][str(layer)], report["after"][str(layer)]
        dfvu = 100 * (a["fvu"] - b["fvu"]) / b["fvu"]
        lines.append(f"| {layer} | {b['fvu']:.4f} | {a['fvu']:.4f} | {dfvu:+.0f}% | {b['l0_mean']:.1f} | {a['l0_mean']:.1f} |")
    lines += ["", "For reference, the pre-finetune base-input FVU (Exp 1) was ~0.082 (L0), ~0.21 (L24), ~0.15 (L25).", ""]
    (output_dir / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
