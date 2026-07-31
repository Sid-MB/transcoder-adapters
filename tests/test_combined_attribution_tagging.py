"""Tests for tag_combined_graph: base/adapter feature tagging + error-node ranking.

The combined attribution graph carries base features (within-layer combined index < n_base),
adapter features (>= n_base), and reconstruction-error nodes. tag_combined_graph labels each,
rewrites adapter feature indices to their source-local cantor id (so /adapter_features loads
the right example), and keeps only the most influential error nodes.
"""

import unittest

from analysis.attribution.run_combined_attribution import (
    ERROR_NODE_SHAPE,
    cantor_pair,
    cantor_unpair,
    tag_combined_graph,
)
from analysis.attribution.run_base_adapter_comparison import (
    ADAPTER_NODE_SHAPE,
    BASE_NODE_SHAPE,
    SOURCE_ADAPTER,
    SOURCE_BASE,
)

N_BASE = 16384


def _feat_node(node_id, layer, within, *, influence=0.5):
    return {
        "node_id": node_id,
        "feature": cantor_pair(layer, within),
        "layer": layer,
        "ctx_idx": 1,
        "feature_type": "cross layer transcoder",
        "influence": influence,
        "jsNodeId": node_id,
        "clerp": "",
    }


def _error_node(node_id, layer, *, influence):
    return {
        "node_id": node_id,
        "feature": cantor_pair(layer, 0),
        "layer": layer,
        "ctx_idx": 1,
        "feature_type": "mlp reconstruction error",
        "influence": influence,
        "jsNodeId": node_id,
    }


def _payload(nodes, links):
    return {
        "metadata": {"slug": "combined__p__h0", "scan": "x", "prompt_tokens": ["a", "b"]},
        "qParams": {},
        "nodes": nodes,
        "links": links,
    }


class CantorTests(unittest.TestCase):
    def test_pair_unpair_roundtrip(self):
        for layer in (0, 5, 25):
            for feat in (0, 1, 16383, 16384, 24575):
                self.assertEqual(cantor_unpair(cantor_pair(layer, feat)), (layer, feat))


class TagCombinedGraphTests(unittest.TestCase):
    def test_tags_base_and_adapter_and_rewrites_adapter_index(self):
        base = _feat_node("0_5_1", 0, 5)
        adapter = _feat_node("0_16400_1", 0, N_BASE + 16)  # within = 16400 -> local 16
        logit = {"node_id": "27_2_1", "feature": 2, "feature_type": "logit", "influence": 1.0}
        payload = _payload(
            [base, adapter, logit],
            [{"source": "0_5_1", "target": "27_2_1", "weight": 1.0},
             {"source": "0_16400_1", "target": "27_2_1", "weight": 1.0}],
        )

        tagged = tag_combined_graph(
            payload, n_base=N_BASE,
            base_feature_scan="siddharthmb/base", adapter_feature_scan="/features",
        )
        by_id = {n["node_id"]: n for n in tagged["nodes"]}

        self.assertEqual(by_id["0_5_1"]["source_model"], SOURCE_BASE)
        self.assertEqual(by_id["0_5_1"]["node_shape"], BASE_NODE_SHAPE)
        self.assertEqual(by_id["0_5_1"]["feature"], cantor_pair(0, 5))  # unchanged
        self.assertEqual(by_id["0_5_1"]["source_feature_id"], "5")

        self.assertEqual(by_id["0_16400_1"]["source_model"], SOURCE_ADAPTER)
        self.assertEqual(by_id["0_16400_1"]["node_shape"], ADAPTER_NODE_SHAPE)
        # adapter feature index rewritten to source-local cantor(layer, within - n_base)
        self.assertEqual(by_id["0_16400_1"]["feature"], cantor_pair(0, 16))
        self.assertEqual(by_id["0_16400_1"]["source_feature_id"], "16")

        comp = tagged["metadata"]["comparison"]
        self.assertEqual(comp["n_base_features"], N_BASE)
        self.assertEqual(comp["base_feature_scan"], "siddharthmb/base")
        self.assertEqual(comp["adapter_feature_scan"], "/features")
        self.assertEqual(comp["node_counts"], {"base_features": 1, "adapter_features": 1, "error_nodes": 0})

    def test_keeps_top_influence_error_nodes_and_prunes_links(self):
        feat = _feat_node("0_5_1", 0, 5)
        errs = [_error_node(f"{l}_err_1", l, influence=inf) for l, inf in [(1, 0.1), (2, 0.9), (3, 0.5)]]
        links = [{"source": "2_err_1", "target": "0_5_1", "weight": 1.0},
                 {"source": "1_err_1", "target": "0_5_1", "weight": 1.0}]
        payload = _payload([feat, *errs], links)

        tagged = tag_combined_graph(
            payload, n_base=N_BASE, base_feature_scan=None, adapter_feature_scan=None,
            max_error_nodes=1,
        )
        kept_errors = [n for n in tagged["nodes"] if n.get("feature_type") == "mlp reconstruction error"]
        self.assertEqual(len(kept_errors), 1)
        self.assertEqual(kept_errors[0]["node_id"], "2_err_1")  # highest influence
        self.assertEqual(kept_errors[0]["node_shape"], ERROR_NODE_SHAPE)
        self.assertTrue(kept_errors[0]["isError"])
        # links to dropped error node are pruned; link to kept error survives
        kept_ids = {n["node_id"] for n in tagged["nodes"]}
        for link in tagged["links"]:
            self.assertIn(link["source"], kept_ids)
            self.assertIn(link["target"], kept_ids)
        self.assertEqual(len(tagged["links"]), 1)


if __name__ == "__main__":
    unittest.main()
