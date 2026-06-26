"""Parity test: the vectorized fast path in FeatureCollector.accumulate_batch must produce
byte-identical FeatureStats + histograms to the legacy per-event path, for the dense base
config (n_random=0, activation ranges disabled).

CPU-only; no model/GPU. Run directly:
    PYTHONPATH=. uv run --extra viz python tests/test_accumulate_vectorized_parity.py
or via pytest.
"""

import numpy as np
import torch

from analysis.features.collect_feature_activations import FeatureCollector

DOMAINS = ["chat", "fineweb"]


def _markers(seq_len: int):
    """Varied regions + a thinking span, so region/thinking vectorization is exercised."""
    return {
        "user_marker_positions": [1],
        "assistant_marker_positions": [min(8, seq_len - 1)],
        "think_start_positions": [min(10, seq_len - 1)],
        "think_end_positions": [min(15, seq_len - 1)],
    }


def _make_batches(seed: int, n_layers: int, n_features: int, n_batches: int):
    """Deterministic batches of sparse activation tensors + matching metadata."""
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    batches = []
    seq_idx_counter = 0
    for _ in range(n_batches):
        B = 3
        seq_lens = [18 + int(rng.integers(0, 6)) for _ in range(B)]
        max_len = max(seq_lens)
        layer_acts = {}
        for layer in range(n_layers):
            t = torch.zeros(B, max_len, n_features)
            for b in range(B):
                L = seq_lens[b]
                # ~40% density, strictly positive distinct-ish values in a binnable range.
                mask = torch.rand(L, n_features, generator=g) < 0.4
                vals = 0.05 + 9.0 * torch.rand(L, n_features, generator=g)
                t[b, :L][mask] = vals[mask]
            layer_acts[layer] = t
        batch_tokens = [
            [int(x) for x in rng.integers(5, 9, size=seq_lens[b])]  # tiny vocab -> token-count repeats
            for b in range(B)
        ]
        batch_domains = [DOMAINS[b % len(DOMAINS)] for b in range(B)]
        batch_markers = [_markers(seq_lens[b]) for b in range(B)]
        batch_seq_idxs = [seq_idx_counter + b for b in range(B)]
        batch_source_metadata = [
            {"prepared_item_idx": seq_idx_counter + b, "source_idx": 0} for b in range(B)
        ]
        seq_idx_counter += B
        batches.append(dict(
            batch_tokens=batch_tokens, batch_domains=batch_domains, batch_markers=batch_markers,
            batch_seq_idxs=batch_seq_idxs, batch_source_metadata=batch_source_metadata,
            layer_activations=layer_acts,
        ))
    return batches


def _collector(n_layers, n_features, top_k, domain_top_k, token_count_cap):
    return FeatureCollector(
        n_layers=n_layers, n_features=n_features, top_k=top_k, n_random=0,
        domain_top_k=domain_top_k, domain_names=DOMAINS,
        activation_example_ranges=[], activation_range_examples_per_domain=0,
        track_token_counts=True, token_count_cap=token_count_cap,
    )


def _ex_key(e):
    return (round(e.activation, 9), e.token_id, e.position, e.domain, e.region)


def _assert_parity(n_layers, n_features, top_k, domain_top_k, token_count_cap, seed=0,
                   n_batches=5, check_token_counts=True):
    batches = _make_batches(seed, n_layers, n_features, n_batches)
    A = _collector(n_layers, n_features, top_k, domain_top_k, token_count_cap)  # legacy
    A._force_legacy = True
    B = _collector(n_layers, n_features, top_k, domain_top_k, token_count_cap)  # fast
    for batch in batches:
        A.accumulate_batch(**batch)
        B.accumulate_batch(**batch)
    assert B._vec_activation is not None, "fast path did not create vec arrays"
    A.finalize()
    B.finalize()

    # Collector-level histograms
    assert np.array_equal(A.run_hist_total, B.run_hist_total)
    assert np.array_equal(A.run_hist_by_domain, B.run_hist_by_domain)
    for layer in range(n_layers):
        assert np.array_equal(A.feature_hist_total_by_layer[layer], B.feature_hist_total_by_layer[layer])
        assert np.array_equal(A.feature_hist_by_domain_by_layer[layer], B.feature_hist_by_domain_by_layer[layer])
    assert A.total_tokens == B.total_tokens
    assert dict(A.tokens_per_domain) == dict(B.tokens_per_domain)
    assert dict(A.tokens_per_region) == dict(B.tokens_per_region)
    assert list(A.tokens_per_thinking_bin) == list(B.tokens_per_thinking_bin)

    # Per-(layer, feature) FeatureStats
    n_checked = 0
    for layer in range(n_layers):
        for f in range(n_features):
            sa, sb = A.stats[layer][f], B.stats[layer][f]
            assert sa.activation_count == sb.activation_count, (layer, f, "activation_count")
            assert dict(sa.domain_counts) == dict(sb.domain_counts), (layer, f, "domain_counts")
            assert dict(sa.region_counts) == dict(sb.region_counts), (layer, f, "region_counts")
            assert list(sa.thinking_position_counts) == list(sb.thinking_position_counts), (layer, f, "thinking")
            if check_token_counts:
                # Exact only with no eviction (cap >= distinct tokens): legacy Space-Saving and
                # the fast path's exact-top-cap agree. Under eviction they differ by design
                # (see test_token_counts_exact_top_cap), so the caller disables this.
                assert dict(sa.token_counts) == dict(sb.token_counts), (layer, f, "token_counts")
            ka = sorted((_ex_key(e) for e in sa.top_k_examples), reverse=True)
            kb = sorted((_ex_key(e) for e in sb.top_k_examples), reverse=True)
            assert ka == kb, (layer, f, "top_k")
            doms = set(sa.domain_top_k_examples) | set(sb.domain_top_k_examples)
            for d in doms:
                da = sorted((_ex_key(e) for e in sa.domain_top_k_examples.get(d, [])), reverse=True)
                db = sorted((_ex_key(e) for e in sb.domain_top_k_examples.get(d, [])), reverse=True)
                assert da == db, (layer, f, d, "domain_top_k")
            if sa.activation_count:
                n_checked += 1
    return n_checked


def test_parity_no_eviction():
    # Large token_count_cap -> no Space-Saving eviction (token_counts unambiguous).
    n = _assert_parity(n_layers=3, n_features=40, top_k=4, domain_top_k=3, token_count_cap=1000)
    assert n > 0


def test_parity_with_eviction_deterministic():
    # Small cap forces token_counts eviction, where the fast path switched from approximate
    # Space-Saving to EXACT top-cap, so token_counts intentionally differs. Everything else
    # (counts, histograms, top-k) must still be byte-identical to legacy.
    n = _assert_parity(n_layers=3, n_features=40, top_k=4, domain_top_k=3, token_count_cap=2,
                       seed=7, check_token_counts=False)
    assert n > 0


def test_parity_smallish_topk():
    # cap=8 >= 4 distinct tokens -> no eviction -> token_counts also match.
    n = _assert_parity(n_layers=2, n_features=64, top_k=2, domain_top_k=2, token_count_cap=8, seed=3)
    assert n > 0


def _ground_truth_token_counts(batches, n_layers, cap):
    """Independent exact top-cap token counts, computed straight from the raw activations."""
    counts: dict = {}
    for batch in batches:
        for b, tokens in enumerate(batch["batch_tokens"]):
            L = len(tokens)
            for layer in range(n_layers):
                acts = batch["layer_activations"][layer][b, :L]
                nz = (acts > 0).nonzero()
                for pos, feat in nz.tolist():
                    key = (layer, int(feat), int(tokens[pos]))
                    counts[key] = counts.get(key, 0) + 1
    by_lf: dict = {}
    for (layer, feat, tok), c in counts.items():
        by_lf.setdefault((layer, feat), {})[tok] = c
    gt: dict = {}
    for lf, d in by_lf.items():
        ranked = sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[:cap]  # count desc, token asc
        gt[lf] = {tok: c for tok, c in ranked}
    return gt


def test_token_counts_exact_top_cap():
    """The fast path's token_counts must equal the independent exact-top-cap ground truth."""
    n_layers, n_features, cap = 3, 40, 2
    batches = _make_batches(seed=7, n_layers=n_layers, n_features=n_features, n_batches=5)
    B = _collector(n_layers, n_features, top_k=4, domain_top_k=3, token_count_cap=cap)
    for batch in batches:
        B.accumulate_batch(**batch)
    B.finalize()
    gt = _ground_truth_token_counts(batches, n_layers, cap)
    n_checked = 0
    for layer in range(n_layers):
        for f in range(n_features):
            got = dict(B.stats[layer][f].token_counts)
            expected = gt.get((layer, f), {})
            assert got == expected, (layer, f, "exact-top-cap", got, expected)
            if expected:
                n_checked += 1
    assert n_checked > 0


if __name__ == "__main__":
    test_parity_no_eviction()
    print("no-eviction parity: PASS")
    test_parity_with_eviction_deterministic()
    print("with-eviction deterministic parity: PASS")
    test_parity_smallish_topk()
    print("smallish-topk parity: PASS")
    test_token_counts_exact_top_cap()
    print("token_counts exact-top-cap: PASS")
    print("ALL PARITY TESTS PASSED")
