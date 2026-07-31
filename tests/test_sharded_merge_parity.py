"""Parity test for sharded feature collection: collecting a dataset as N disjoint shards and
merging (FeatureCollector.merge_from) must produce FeatureStats + histograms byte-identical to
a single unsharded run. This is what makes the multi-GPU sharded collectors
(collect_feature_activations / collect_base_feature_activations) correct.

Shards are partitioned exactly like the real collectors: by prepared_item_idx %% num_shards.
Merge requires the mergeable config (n_random=0, activation ranges disabled), which sharding
enforces. CPU-only; no model/GPU. Run directly or via pytest:
    LARGE_ARTIFACTS_DIR=/nlp/scr/siddharth uv run --no-sync python -m pytest tests/test_sharded_merge_parity.py -q
"""

import numpy as np
import torch

from analysis.features.collect_feature_activations import FeatureCollector

DOMAINS = ["chat", "fineweb"]


def _markers(seq_len: int):
    return {
        "user_marker_positions": [1],
        "assistant_marker_positions": [min(8, seq_len - 1)],
        "think_start_positions": [min(10, seq_len - 1)],
        "think_end_positions": [min(15, seq_len - 1)],
    }


def _make_items(seed: int, n_layers: int, n_features: int, n_items: int):
    """A flat list of single-sequence items, each with its own sparse activation tensors and a
    globally-unique prepared_item_idx (so the modulo shard split is well-defined)."""
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n_items):
        L = 18 + int(rng.integers(0, 6))
        layer_acts = {}
        for layer in range(n_layers):
            t = torch.zeros(1, L, n_features)
            mask = torch.rand(L, n_features, generator=g) < 0.4
            vals = 0.05 + 9.0 * torch.rand(L, n_features, generator=g)
            t[0][mask] = vals[mask]
            layer_acts[layer] = t
        items.append(dict(
            tokens=[int(x) for x in rng.integers(5, 9, size=L)],  # tiny vocab -> token-count repeats
            domain=DOMAINS[i % len(DOMAINS)],
            markers=_markers(L),
            prepared_item_idx=i,
            layer_acts=layer_acts,
        ))
    return items


def _collector(n_layers, n_features, top_k, domain_top_k, token_count_cap):
    return FeatureCollector(
        n_layers=n_layers, n_features=n_features, top_k=top_k, n_random=0,
        domain_top_k=domain_top_k, domain_names=DOMAINS,
        activation_example_ranges=[], activation_range_examples_per_domain=0,
        track_token_counts=True, token_count_cap=token_count_cap,
    )


def _feed(collector, item):
    """Feed one item as a 1-sequence batch (same call the real collector makes per batch)."""
    collector.accumulate_batch(
        batch_tokens=[item["tokens"]],
        batch_domains=[item["domain"]],
        batch_markers=[item["markers"]],
        batch_seq_idxs=[item["prepared_item_idx"]],
        batch_source_metadata=[{"prepared_item_idx": item["prepared_item_idx"], "source_idx": 0}],
        layer_activations=item["layer_acts"],
    )


def _ex_key(e):
    return (round(e.activation, 9), e.token_id, e.position, e.domain, e.region)


def _assert_collectors_equal(A, B, n_layers, n_features):
    assert A.total_tokens == B.total_tokens, "total_tokens"
    assert dict(A.tokens_per_domain) == dict(B.tokens_per_domain), "tokens_per_domain"
    assert dict(A.tokens_per_region) == dict(B.tokens_per_region), "tokens_per_region"
    assert list(A.tokens_per_thinking_bin) == list(B.tokens_per_thinking_bin), "thinking_bin"
    assert np.array_equal(A.run_hist_total, B.run_hist_total), "run_hist_total"
    assert np.array_equal(A.run_hist_by_domain, B.run_hist_by_domain), "run_hist_by_domain"
    n_checked = 0
    for layer in range(n_layers):
        assert np.array_equal(
            A.feature_hist_total_by_layer[layer], B.feature_hist_total_by_layer[layer]
        ), (layer, "feature_hist_total")
        assert np.array_equal(
            A.feature_hist_by_domain_by_layer[layer], B.feature_hist_by_domain_by_layer[layer]
        ), (layer, "feature_hist_by_domain")
        for f in range(n_features):
            sa, sb = A.stats[layer][f], B.stats[layer][f]
            assert sa.activation_count == sb.activation_count, (layer, f, "activation_count")
            assert dict(sa.domain_counts) == dict(sb.domain_counts), (layer, f, "domain_counts")
            assert dict(sa.region_counts) == dict(sb.region_counts), (layer, f, "region_counts")
            assert list(sa.thinking_position_counts) == list(sb.thinking_position_counts), (layer, f, "thinking")
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


def _assert_shard_parity(n_layers, n_features, top_k, domain_top_k, num_shards,
                         n_items=24, seed=0, token_count_cap=1000):
    items = _make_items(seed, n_layers, n_features, n_items)

    # Single unsharded run.
    single = _collector(n_layers, n_features, top_k, domain_top_k, token_count_cap)
    for it in items:
        _feed(single, it)
    single.finalize()

    # Sharded: partition by prepared_item_idx %% num_shards (exactly what the collectors do),
    # finalize each, then merge into shard 0.
    shards = [_collector(n_layers, n_features, top_k, domain_top_k, token_count_cap)
              for _ in range(num_shards)]
    for it in items:
        _feed(shards[it["prepared_item_idx"] % num_shards], it)
    for s in shards:
        s.finalize()
    merged = shards[0]
    for s in shards[1:]:
        merged.merge_from(s)

    n = _assert_collectors_equal(single, merged, n_layers, n_features)
    assert n > 0, "no active features were compared"
    return n


def test_shard_merge_parity_2shards():
    assert _assert_shard_parity(n_layers=3, n_features=40, top_k=4, domain_top_k=3, num_shards=2) > 0


def test_shard_merge_parity_3shards():
    assert _assert_shard_parity(n_layers=3, n_features=40, top_k=4, domain_top_k=3, num_shards=3, seed=7) > 0


def test_shard_merge_parity_many_shards_small_topk():
    # More shards than items per feature stress the union-then-keep-k heap merge.
    assert _assert_shard_parity(n_layers=2, n_features=64, top_k=2, domain_top_k=2,
                                num_shards=5, n_items=30, seed=3) > 0


if __name__ == "__main__":
    test_shard_merge_parity_2shards()
    print("2-shard merge parity: PASS")
    test_shard_merge_parity_3shards()
    print("3-shard merge parity: PASS")
    test_shard_merge_parity_many_shards_small_topk()
    print("many-shard merge parity: PASS")
    print("ALL SHARD-MERGE PARITY TESTS PASSED")
