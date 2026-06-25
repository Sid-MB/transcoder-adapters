"""Real-weights integration check for build_combined_transcoder_set (T_base + T_adapter).

The synthetic-weights logic is already covered by tests/test_combined_transcoders.py. This
script does the REAL-weights / REAL-shapes integration check on the local GPU:

  1. Loads the REAL GemmaScope base transcoders (JumpReLU, 16384 feats/layer) via the same
     helpers run_base_adapter_comparison uses (_load_gemmascope_transcoders).
  2. Loads our REAL trained adapter checkpoint and wraps each layer's MLP transcoder branch
     (ReLU, 8192 feats/layer) into a circuit_tracer SingleLayerTranscoder. The crux is the
     weight orientation: this script self-checks that the wrapped SingleLayerTranscoder forward
     reproduces the actual adapter branch mlp.transcoder_dec(relu(mlp.transcoder_enc(x))) to
     bf16 tolerance before trusting it.
  3. Builds combined = build_combined_transcoder_set(base, adapter) and, for a handful of
     layers, verifies combined(x) == base(x) + adapter(x) (forward-equivalence) plus that the
     combined shapes are the expected 24576 features.

Run OUTSIDE the sandbox (needs CUDA):

  uv run --extra viz python -m misc_scripts.validate_combined_transcoder

All numeric thresholds are bf16-appropriate (atol/rtol ~1e-2). Uses helpers.log.logger; no print.
"""

from __future__ import annotations

import torch
from torch import nn

from circuit_tracer.transcoder.single_layer_transcoder import (
    SingleLayerTranscoder,
    TranscoderSet,
)

from analysis.attribution.combined_transcoders import (
    build_combined_transcoder_set,
    verify_combination,
)
from analysis.attribution.run_base_adapter_comparison import (
    _load_gemmascope_transcoders,
    build_gemmascope_transcoder_config,
    resolve_gemmascope_l0_values,
)
from helpers.log import logger, setup_logging
from models.auto import AutoModelForCausalLMWithTranscoder

# --- Fixed experiment configuration (the standard Gemma-2-2b + GemmaScope width_16k setup) ---
ADAPTER_REPO = "siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860"
BASE_MODEL = "google/gemma-2-2b"
GEMMASCOPE_REPO = "google/gemma-scope-2b-pt-transcoders"
GEMMASCOPE_WIDTH = "width_16k"
GEMMASCOPE_L0 = "average_l0_76"
GEMMASCOPE_L0_MATCH = "nearest"
N_LAYERS = 26
D_MODEL = 2304
N_BASE_FEATURES = 16384  # GemmaScope width_16k
N_ADAPTER_FEATURES = 8192  # adapter checkpoint transcoder_n_features
N_COMBINED_FEATURES = N_BASE_FEATURES + N_ADAPTER_FEATURES  # 24576
FEATURE_INPUT_HOOK = "ln2.hook_normalized"
FEATURE_OUTPUT_HOOK = "hook_mlp_out"

# Layers we spot-check for forward equivalence + shapes (early/mid/late + last).
TEST_LAYERS = [0, 6, 12, 18, 25]
# bf16 has ~3 decimal digits of mantissa; verify_combination is called with these for its raw
# (reported-only) bf16 allclose number.
ATOL = 1e-2
RTOL = 1e-2

# --- Why raw bf16 allclose is NOT the correctness signal here ---
# These real transcoder reconstructions reach magnitudes of a few hundred (GemmaScope decoder
# biases + reconstructions are large). Two artifacts follow:
#   (1) bf16 has only 8 mantissa bits (~2^-8 relative step), so a single elementwise rounding gap
#       on an O(100) value is O(1) by itself -- making torch.allclose(atol=1e-2,rtol=1e-2) fail
#       even when the construction is algebraically perfect.
#   (2) combined(x) sums all 24576 feature contributions in ONE reduction, while base(x)+adapter(x)
#       does two separate reductions (16384 then 8192) then adds. These are algebraically equal but
#       NOT bitwise equal: floating-point summation is non-associative, so even in fp32 they differ
#       by O(1e-1) on O(100) outputs (rel ~1e-4). This is reproduced exactly in fp64, confirming it
#       is reduction-order rounding, not a construction error.
# The robust correctness signal is therefore: compare combined(x) against a HIGH-PRECISION (fp64)
# reference of T_base(x) + T_adapter(x). If the construction is correct, combined(x) matches that
# fp64 reference to fp32 rounding, regardless of reduction order. We also report a bf16 RELATIVE
# error (max_abs_diff / max|expected|) for transparency.
FP64_REL_TOL = 1e-3   # combined(fp32) vs fp64 T_base+T_adapter reference: relative-error bound
BF16_REL_TOL = 1e-2   # combined(bf16) vs bf16 T_base+T_adapter: relative-error bound (bf16 ~2^-8)


def _build_adapter_single_layer_transcoder(
    mlp: nn.Module,
    *,
    layer_idx: int,
    device: torch.device,
    dtype: torch.dtype,
) -> SingleLayerTranscoder:
    """Wrap one adapter MLP's transcoder branch into a circuit_tracer SingleLayerTranscoder.

    Adapter branch (models/gemma2_transcoder.py):
        transcoder_enc: nn.Linear(d_model -> n_features, bias=True)
        transcoder_dec: nn.Linear(n_features -> d_model, bias=transcoder_dec_bias)
        branch(x) = transcoder_dec(relu(transcoder_enc(x)))

    circuit_tracer SingleLayerTranscoder (single_layer_transcoder.py):
        encode(x) = activation(F.linear(x, W_enc, b_enc)) = activation(x @ W_enc.T + b_enc)
        decode(a) = a @ W_dec + b_dec
    so W_enc is [d_transcoder, d_model] (== enc.weight) and W_dec is [d_transcoder, d_model]
    (== dec.weight.T, since dec.weight is [d_model, d_transcoder]).
    """
    enc: nn.Linear = mlp.transcoder_enc  # type: ignore[assignment]
    dec: nn.Linear = mlp.transcoder_dec  # type: ignore[assignment]
    d_transcoder, d_model = enc.weight.shape

    tc = SingleLayerTranscoder(
        d_model=d_model,
        d_transcoder=d_transcoder,
        activation_function=torch.nn.ReLU(),
        layer_idx=layer_idx,
        skip_connection=False,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        # W_enc <- enc.weight  ([n_features, d_model], matches encode's F.linear orientation)
        tc.W_enc.copy_(enc.weight.to(device, dtype))
        tc.b_enc.copy_(enc.bias.to(device, dtype))
        # W_dec <- dec.weight.T  (dec.weight is [d_model, n_features]; decode does acts @ W_dec)
        tc.W_dec.copy_(dec.weight.T.contiguous().to(device, dtype))
        if dec.bias is not None:
            tc.b_dec.copy_(dec.bias.to(device, dtype))
        else:
            tc.b_dec.zero_()
    return tc


@torch.no_grad()
def _selfcheck_adapter_orientation(
    tc: SingleLayerTranscoder,
    mlp: nn.Module,
    *,
    layer_idx: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    """Confirm the wrapped SingleLayerTranscoder == the real adapter branch on random x.

    This is the crux: if the orientation were wrong, the combined transcoder would silently
    encode the wrong adapter contribution.

    The correctness signal is the fp32-vs-fp64 diff: we recompute the wrapped transcoder in
    fp32 and compare it to the REAL adapter branch evaluated in fp64. A wrong orientation
    (transposed/swapped W) would NOT agree even in high precision. The bf16 diff is reported
    for transparency but judged via a relative bound, since bf16 rounding of O(100) activations
    yields O(1) absolute spikes by itself.
    """
    generator = torch.Generator(device="cpu").manual_seed(4321 + layer_idx)
    x = torch.randn(4, D_MODEL, generator=generator).to(device, dtype)

    wrapped = tc(x)
    # The real adapter branch contribution (NOT including the original Gemma MLP output).
    branch = mlp.transcoder_dec(torch.relu(mlp.transcoder_enc(x)))  # type: ignore[operator]
    branch = branch.to(device, dtype)
    diff = (wrapped - branch).abs()
    max_abs_bf16 = max(float(branch.abs().max()), 1e-12)
    bf16_rel = float(diff.max()) / max_abs_bf16

    # fp32 wrapped transcoder vs fp64 real adapter branch (orientation-correctness signal).
    xf = x.float()
    wrapped_f32 = (
        torch.relu(xf @ tc.W_enc.float().T + tc.b_enc.float()) @ tc.W_dec.float() + tc.b_dec.float()
    )
    enc, dec = mlp.transcoder_enc, mlp.transcoder_dec  # type: ignore[assignment]
    xd = x.double()
    branch_f64 = (
        torch.relu(xd @ enc.weight.double().T + enc.bias.double()) @ dec.weight.double().T
        + dec.bias.double()
    )
    diff_hp = (wrapped_f32.double() - branch_f64).abs()
    max_abs_hp = max(float(branch_f64.abs().max()), 1e-12)
    hp_rel = float(diff_hp.max()) / max_abs_hp

    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "bf16_rel": bf16_rel,
        "hp_rel": hp_rel,  # fp32-wrapped vs fp64-reference relative error
        "allclose": bool(torch.allclose(wrapped, branch, atol=ATOL, rtol=RTOL)),
        # PASS if fp32 wrapped matches fp64 reference (orientation exact) AND bf16 relative error
        # is within bf16 precision.
        "passed": bool(hp_rel < FP64_REL_TOL and bf16_rel < BF16_REL_TOL),
    }


def _node_feature_index(node: dict) -> int | None:
    """Local feature index for a 'cross layer transcoder' node.

    The graph JSON `feature` field is cantor_pairing(layer, feat_idx), NOT the raw feature
    index. The raw per-layer feature index is the middle component of node_id
    (f"{layer}_{feat_idx}_{pos}"), matching run_base_adapter_comparison._source_feature_index.
    """
    node_id = str(node.get("node_id", ""))
    parts = node_id.split("_")
    if len(parts) >= 2 and parts[1].lstrip("-").isdigit():
        return int(parts[1])
    return None


def run_attribution_integration_check(
    combined_set: TranscoderSet,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    """Validate the NEXT pivot step: combined set -> circuit-tracer attribution -> error nodes.

    Steps (each reported PASS/FAIL):
      1. Load combined_set into a circuit-tracer ReplacementModel (does it accept a TranscoderSet
         with BlockwiseActivation per-layer transcoders, d_transcoder=24576?).
      2. Run `attribute` on a tiny prompt (does standard linear attribution run over the combined
         transcoders end-to-end?).
      3. Export with create_graph_files and inspect the JSON: total nodes, count of
         'mlp reconstruction error' nodes (THE deliverable), and the min/max local feature index
         among 'cross layer transcoder' nodes to confirm BOTH base (<16384) and adapter (>=16384)
         features can appear.
    """
    import json
    import tempfile
    from pathlib import Path

    from circuit_tracer import ReplacementModel, attribute
    from circuit_tracer.utils.create_graph_files import create_graph_files

    summary: dict[str, object] = {
        "load_ok": False,
        "attribute_ok": False,
        "export_ok": False,
        "n_nodes": 0,
        "n_error_nodes": 0,
        "n_transcoder_nodes": 0,
        "feat_idx_min": None,
        "feat_idx_max": None,
        "has_base_feature": False,
        "has_adapter_feature": False,
        # Active-feature-level evidence that both blocks are reachable, independent of the
        # top-max_feature_nodes node cap (which can be dominated by base features on a given prompt).
        "n_active_features": 0,
        "n_active_base": 0,
        "n_active_adapter": 0,
        "active_feat_idx_max": None,
        "error": None,
    }

    # --- Step A: load into ReplacementModel ---
    logger.info(
        "ATTR STEP 1: loading combined TranscoderSet into circuit-tracer ReplacementModel "
        "(google/gemma-2-2b, transformerlens backend)"
    )
    try:
        model = ReplacementModel.from_pretrained_and_transcoders(
            model_name="google/gemma-2-2b",
            transcoders=combined_set,
            backend="transformerlens",
            device=device,
            dtype=dtype,
        )
        summary["load_ok"] = True
        logger.info("ATTR STEP 1 PASS: ReplacementModel accepted the combined set (d_transcoder=24576, BlockwiseActivation).")
    except Exception as exc:  # noqa: BLE001 - we want to report any failure mode
        summary["error"] = f"ReplacementModel load failed: {type(exc).__name__}: {exc}"
        logger.error(f"ATTR STEP 1 FAIL: {summary['error']}")
        return summary

    # --- Step B: attribution on a tiny prompt ---
    prompt = "The capital of France is"
    logger.info(f"ATTR STEP 2: running attribute() on tiny prompt {prompt!r}")
    try:
        graph = attribute(
            prompt=prompt,
            model=model,
            max_n_logits=5,
            batch_size=4,
            max_feature_nodes=128,
            offload=None,
            verbose=True,
        )
        graph.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        summary["attribute_ok"] = True
        # Active-feature evidence (independent of the top-N node cap): active_features is
        # (n_active, 3) = (layer, pos, feature_idx). Adapter-block features are feature_idx>=16384.
        active = graph.active_features
        active_feat_idx = active[:, 2]
        summary["n_active_features"] = int(active.shape[0])
        summary["n_active_base"] = int((active_feat_idx < N_BASE_FEATURES).sum())
        summary["n_active_adapter"] = int((active_feat_idx >= N_BASE_FEATURES).sum())
        summary["active_feat_idx_max"] = int(active_feat_idx.max())
        logger.info(
            f"ATTR STEP 2 PASS: attribute() completed. active_features={summary['n_active_features']} "
            f"(base<{N_BASE_FEATURES}: {summary['n_active_base']}, "
            f"adapter>={N_BASE_FEATURES}: {summary['n_active_adapter']}, max_feat_idx={summary['active_feat_idx_max']})"
        )
    except Exception as exc:  # noqa: BLE001
        summary["error"] = f"attribute() failed: {type(exc).__name__}: {exc}"
        logger.error(f"ATTR STEP 2 FAIL: {summary['error']}")
        return summary

    # --- Step C: export + inspect graph JSON ---
    logger.info("ATTR STEP 3: exporting graph with create_graph_files and inspecting nodes")
    try:
        with tempfile.TemporaryDirectory(prefix="combined_smoke_") as tmpdir:
            create_graph_files(
                graph_or_path=graph,
                slug="combined_smoke",
                output_path=tmpdir,
                scan_name="base-vs-adapter",
                node_threshold=0.8,
                edge_threshold=0.98,
            )
            json_paths = [
                p
                for p in Path(tmpdir).rglob("*.json")
                if p.name not in {"graph-metadata.json", "run_attribution_args.json"}
            ]
            payload = None
            for p in json_paths:
                candidate = json.loads(p.read_text())
                if {"metadata", "nodes", "links"}.issubset(candidate):
                    payload = candidate
                    break
            if payload is None:
                raise RuntimeError(f"No graph JSON with nodes/links found under {tmpdir} ({[p.name for p in json_paths]})")

            nodes = payload["nodes"]
            summary["n_nodes"] = len(nodes)
            error_nodes = [n for n in nodes if n.get("feature_type") == "mlp reconstruction error"]
            transcoder_nodes = [
                n for n in nodes if "transcoder" in str(n.get("feature_type") or "")
            ]
            summary["n_error_nodes"] = len(error_nodes)
            summary["n_transcoder_nodes"] = len(transcoder_nodes)

            feat_indices = [
                idx
                for n in transcoder_nodes
                if (idx := _node_feature_index(n)) is not None
            ]
            if feat_indices:
                summary["feat_idx_min"] = min(feat_indices)
                summary["feat_idx_max"] = max(feat_indices)
                summary["has_base_feature"] = any(i < N_BASE_FEATURES for i in feat_indices)
                summary["has_adapter_feature"] = any(i >= N_BASE_FEATURES for i in feat_indices)
            summary["export_ok"] = True
            logger.info(
                f"ATTR STEP 3 PASS: graph exported. total_nodes={summary['n_nodes']} "
                f"error_nodes={summary['n_error_nodes']} transcoder_nodes={summary['n_transcoder_nodes']} "
                f"feat_idx range=[{summary['feat_idx_min']}, {summary['feat_idx_max']}] "
                f"(base<{N_BASE_FEATURES}: {summary['has_base_feature']}, "
                f"adapter>={N_BASE_FEATURES}: {summary['has_adapter_feature']})"
            )
    except Exception as exc:  # noqa: BLE001
        summary["error"] = f"export/inspect failed: {type(exc).__name__}: {exc}"
        logger.error(f"ATTR STEP 3 FAIL: {summary['error']}")
        return summary

    return summary


def main() -> None:
    setup_logging()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run outside the sandbox on the GPU machine.")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    logger.info(f"Device: {device}, dtype: {dtype}")

    # --- Step 3: build the REAL GemmaScope base TranscoderSet ---
    logger.info("Resolving GemmaScope per-layer L0 folders (nearest to average_l0_76)")
    l0_values = resolve_gemmascope_l0_values(
        repo=GEMMASCOPE_REPO,
        width=GEMMASCOPE_WIDTH,
        l0=GEMMASCOPE_L0,
        n_layers=N_LAYERS,
        l0_match=GEMMASCOPE_L0_MATCH,
    )
    gemmascope_config = build_gemmascope_transcoder_config(
        repo=GEMMASCOPE_REPO,
        width=GEMMASCOPE_WIDTH,
        l0=GEMMASCOPE_L0,
        n_layers=N_LAYERS,
        model_name=BASE_MODEL,
        feature_input_hook=FEATURE_INPUT_HOOK,
        feature_output_hook=FEATURE_OUTPUT_HOOK,
        l0_values=l0_values,
        l0_match=GEMMASCOPE_L0_MATCH,
    )
    logger.info(f"Loading REAL GemmaScope base transcoders (scan {gemmascope_config['scan_name']})")
    base_set = _load_gemmascope_transcoders(gemmascope_config, device=device, dtype=dtype)
    logger.info(
        f"Base set: n_layers={base_set.n_layers}, d_transcoder={base_set.d_transcoder}, "
        f"input_hook={base_set.feature_input_hook}, output_hook={base_set.feature_output_hook}"
    )
    assert base_set.n_layers == N_LAYERS, base_set.n_layers
    assert base_set.d_transcoder == N_BASE_FEATURES, base_set.d_transcoder

    # --- Step 4: load the REAL adapter and wrap each layer's branch as a SingleLayerTranscoder ---
    logger.info(f"Loading REAL adapter checkpoint: {ADAPTER_REPO}")
    adapter_model = AutoModelForCausalLMWithTranscoder.from_pretrained(
        ADAPTER_REPO,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    adapter_model.eval()

    adapter_transcoders: dict[int, SingleLayerTranscoder] = {}
    selfcheck_results: dict[int, dict[str, float]] = {}
    selfcheck_all_pass = True
    for layer_idx, layer in enumerate(adapter_model.model.layers):
        mlp = layer.mlp
        tc = _build_adapter_single_layer_transcoder(
            mlp, layer_idx=layer_idx, device=device, dtype=dtype
        )
        adapter_transcoders[layer_idx] = tc
        # Self-check orientation on every layer (cheap), but log only the tested layers in detail.
        result = _selfcheck_adapter_orientation(
            tc, mlp, layer_idx=layer_idx, device=device, dtype=dtype
        )
        selfcheck_results[layer_idx] = result
        if not result["passed"]:
            selfcheck_all_pass = False
            logger.error(
                f"  [orientation SELF-CHECK FAIL] layer {layer_idx}: "
                f"hp_rel(fp32-vs-fp64)={result['hp_rel']:.3e} bf16_rel={result['bf16_rel']:.3e} "
                f"(bf16 max_abs_diff={result['max_abs_diff']:.3e} mean={result['mean_abs_diff']:.3e})"
            )
        if layer_idx in TEST_LAYERS:
            logger.info(
                f"  [orientation self-check] layer {layer_idx}: passed={result['passed']} "
                f"hp_rel(fp32-vs-fp64)={result['hp_rel']:.3e} bf16_rel={result['bf16_rel']:.3e} "
                f"| bf16 max_abs_diff={result['max_abs_diff']:.3e} mean_abs_diff={result['mean_abs_diff']:.3e} "
                f"bf16_allclose={result['allclose']}"
            )

    worst_hp = max(r["hp_rel"] for r in selfcheck_results.values())
    worst_bf16 = max(r["bf16_rel"] for r in selfcheck_results.values())
    if selfcheck_all_pass:
        logger.info(
            f"ADAPTER ORIENTATION SELF-CHECK PASSED for all 26 layers "
            f"(worst fp32-vs-fp64 rel={worst_hp:.3e} < {FP64_REL_TOL:.0e}, "
            f"worst bf16 rel={worst_bf16:.3e} < {BF16_REL_TOL:.0e}). Correct orientation: "
            "W_enc <- transcoder_enc.weight; W_dec <- transcoder_dec.weight.T; "
            "b_enc <- transcoder_enc.bias; b_dec <- transcoder_dec.bias."
        )
    else:
        logger.error("ADAPTER ORIENTATION SELF-CHECK FAILED on at least one layer (see above).")

    adapter_set = TranscoderSet(
        adapter_transcoders,
        feature_input_hook=FEATURE_INPUT_HOOK,
        feature_output_hook=FEATURE_OUTPUT_HOOK,
        scan_name="adapter",
    )
    logger.info(
        f"Adapter set: n_layers={adapter_set.n_layers}, d_transcoder={adapter_set.d_transcoder}"
    )
    assert adapter_set.n_layers == N_LAYERS, adapter_set.n_layers
    assert adapter_set.d_transcoder == N_ADAPTER_FEATURES, adapter_set.d_transcoder

    # --- Step 5: build the combined (T_base + T_adapter) set ---
    logger.info("Building combined transcoder set (base + adapter)")
    combined_set = build_combined_transcoder_set(
        base_set, adapter_set, scan_name="base-vs-adapter"
    )
    logger.info(
        f"Combined set: n_layers={combined_set.n_layers}, d_transcoder={combined_set.d_transcoder}"
    )

    # --- Steps 6 + 7: forward-equivalence and shape checks on the tested layers ---
    # We report the bf16 verify_combination numbers (max/mean_abs_diff, raw bf16 allclose) AND
    # the robust correctness criterion: combined(fp32) vs a high-precision (fp64) T_base+T_adapter
    # reference (immune to reduction-order rounding), plus a bf16 relative error. bf16 allclose
    # alone is a precision artifact here (outputs reach O(100), where one bf16 ulp is O(1)).
    forward_all_pass = True
    shapes_all_pass = True
    logger.info(
        f"Verifying combined == base + adapter on layers {TEST_LAYERS} "
        f"(report bf16 atol={ATOL}/rtol={RTOL}; correctness via fp32-vs-fp64 rel<{FP64_REL_TOL:.0e} "
        f"and bf16 rel<{BF16_REL_TOL:.0e})"
    )
    for layer_idx in TEST_LAYERS:
      with torch.no_grad():  # inference-only numeric check; attribution below needs grad and runs outside this
        base_tc = base_set[layer_idx]
        adapter_tc = adapter_set[layer_idx]
        combined_tc = combined_set[layer_idx]

        # Reported bf16 numbers (raw verify_combination from the module under test).
        result = verify_combination(
            base_tc, adapter_tc, combined_tc, n_samples=8, atol=ATOL, rtol=RTOL
        )

        gen = torch.Generator(device="cpu").manual_seed(1234)
        x = torch.randn(8, D_MODEL, generator=gen).to(device, dtype)

        # bf16 relative error (combined vs base+adapter, both bf16).
        combined_out = combined_tc(x)
        expected = base_tc(x.clone()).to(device, dtype) + adapter_tc(x.clone()).to(device, dtype)
        bf16_max_abs = max(float(expected.abs().max()), 1e-12)
        bf16_rel = float((combined_out - expected).abs().max()) / bf16_max_abs

        # combined(fp32) vs fp64 reference of T_base + T_adapter. The fp64 reference removes the
        # reduction-order confound; if the construction is correct, combined(fp32) tracks it to
        # fp32 rounding. Recompute manually (no in-place .float() on the JumpReLU modules).
        xf = x.float()
        xd = x.double()
        comb_pre = xf @ combined_tc.W_enc.float().T + combined_tc.b_enc.float()
        comb_acts = combined_tc.activation_function(comb_pre)  # BlockwiseActivation (JumpReLU|ReLU)
        comb_f32 = comb_acts @ combined_tc.W_dec.float() + combined_tc.b_dec.float()
        base_acts_d = base_tc.activation_function(
            (xd @ base_tc.W_enc.double().T + base_tc.b_enc.double())
        )
        base_f64 = base_acts_d @ base_tc.W_dec.double() + base_tc.b_dec.double()
        adapter_f64 = (
            torch.relu(xd @ adapter_tc.W_enc.double().T + adapter_tc.b_enc.double())
            @ adapter_tc.W_dec.double()
            + adapter_tc.b_dec.double()
        )
        ref_f64 = base_f64 + adapter_f64
        hp_max_abs = max(float(ref_f64.abs().max()), 1e-12)
        hp_rel = float((comb_f32.double() - ref_f64).abs().max()) / hp_max_abs

        forward_ok = hp_rel < FP64_REL_TOL and bf16_rel < BF16_REL_TOL
        if not forward_ok:
            forward_all_pass = False

        d_tc = combined_tc.d_transcoder
        w_enc_shape = tuple(combined_tc.W_enc.shape)
        w_dec_shape = tuple(combined_tc.W_dec.shape)
        b_enc_shape = tuple(combined_tc.b_enc.shape)
        b_dec_shape = tuple(combined_tc.b_dec.shape)
        shape_ok = (
            d_tc == N_COMBINED_FEATURES
            and w_enc_shape == (N_COMBINED_FEATURES, D_MODEL)
            and w_dec_shape == (N_COMBINED_FEATURES, D_MODEL)
            and b_enc_shape == (N_COMBINED_FEATURES,)
            and b_dec_shape == (D_MODEL,)
        )
        if not shape_ok:
            shapes_all_pass = False

        logger.info(
            f"  layer {layer_idx:2d}: forward_ok={forward_ok} "
            f"bf16 max_abs_diff={result['max_abs_diff']:.4e} mean_abs_diff={result['mean_abs_diff']:.4e} "
            f"| hp_rel(fp32-vs-fp64)={hp_rel:.3e} bf16_rel={bf16_rel:.3e} bf16_allclose={result['allclose']} "
            f"| d_transcoder={d_tc} W_enc={w_enc_shape} W_dec={w_dec_shape} "
            f"b_enc={b_enc_shape} b_dec={b_dec_shape} shapes_ok={shape_ok}"
        )

    # --- Next pivot step: combined set -> circuit-tracer attribution -> real error nodes ---
    attr = run_attribution_integration_check(combined_set, device=device, dtype=dtype)
    attr_load_pass = bool(attr["load_ok"])
    attr_run_pass = bool(attr["attribute_ok"])
    # The deliverable: real reconstruction-error nodes.
    attr_error_nodes_pass = bool(attr["export_ok"] and attr["n_error_nodes"] > 0)
    # Both base+adapter feature blocks are *reachable*: judged on the full active-feature set
    # (the top-max_feature_nodes graph nodes can be dominated by base features on a given prompt,
    # so use active_features as the authoritative evidence that adapter-block features fire too).
    attr_both_blocks_pass = bool(attr["n_active_base"] > 0 and attr["n_active_adapter"] > 0)

    # --- Verdict ---
    construction_pass = selfcheck_all_pass and forward_all_pass and shapes_all_pass
    attribution_pass = (
        attr_load_pass and attr_run_pass and attr_error_nodes_pass and attr_both_blocks_pass
    )
    overall_pass = construction_pass and attribution_pass
    logger.info("=" * 80)
    logger.info(f"Adapter-orientation self-check (all 26 layers): {'PASS' if selfcheck_all_pass else 'FAIL'}")
    logger.info(f"Forward-equivalence combined==base+adapter ({TEST_LAYERS}): {'PASS' if forward_all_pass else 'FAIL'}")
    logger.info(f"Combined shapes (24576 feats, W_enc/W_dec [24576,2304]): {'PASS' if shapes_all_pass else 'FAIL'}")
    logger.info(f"ATTR 1 ReplacementModel load (combined set): {'PASS' if attr_load_pass else 'FAIL'}")
    logger.info(f"ATTR 2 attribute() runs over combined transcoders: {'PASS' if attr_run_pass else 'FAIL'}")
    logger.info(
        f"ATTR 3 real error nodes produced: {'PASS' if attr_error_nodes_pass else 'FAIL'} "
        f"(n_error_nodes={attr['n_error_nodes']}, total_nodes={attr['n_nodes']}, "
        f"transcoder_nodes={attr['n_transcoder_nodes']})"
    )
    logger.info(
        f"ATTR 3b both base+adapter feature blocks reachable: {'PASS' if attr_both_blocks_pass else 'FAIL'} "
        f"| active_features: base<{N_BASE_FEATURES}={attr['n_active_base']}, "
        f"adapter>={N_BASE_FEATURES}={attr['n_active_adapter']} (max_feat_idx={attr['active_feat_idx_max']}) "
        f"| graph nodes (top-{128} cap): feat_idx range=[{attr['feat_idx_min']}, {attr['feat_idx_max']}], "
        f"base={attr['has_base_feature']}, adapter={attr['has_adapter_feature']}"
    )
    if attr["error"]:
        logger.error(f"Attribution-step error detail: {attr['error']}")
    logger.info(
        f"OVERALL VERDICT: {'PASS' if overall_pass else 'FAIL'} -- build_combined_transcoder_set "
        f"{'correctly yields' if construction_pass else 'does NOT yield'} T_base + T_adapter with real weights"
        f"; combined set {'runs through attribution and produces real error nodes' if attribution_pass else 'FAILED the attribution/error-node integration check'}."
    )
    logger.info("=" * 80)
    if not overall_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
