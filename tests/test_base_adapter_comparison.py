import json
import tempfile
import unittest
from urllib.request import Request, urlopen
from pathlib import Path

from analysis.attribution.comparison_frontend import (
    FEATURE_DETAIL_SCAN_NEW,
    FEATURE_EXAMPLES_LOAD_NEW,
    FEATURE_HISTOGRAM_GUARD_NEW,
    FEATURE_ID_PATCH_NEW,
    FEATURE_ROW_FONT_SIZE_NEW,
    FEATURE_TYPE_FUNCTION_OLD,
    FEATURE_TYPE_FUNCTION_NEW,
    FEATURE_URL_FUNCTION_OLD,
    FEATURE_URL_FUNCTION_NEW,
    LINK_GRAPH_FONT_SIZE_NEW,
    LINK_GRAPH_NODE_TEXT_NEW,
    NODE_CONNECTIONS_HEADER_ICON_NEW,
    patch_frontend_assets,
)
from analysis.attribution.prepare_comparison_overlay_view import (
    cap_from_arg,
    default_output_dir,
    infer_base_feature_scan,
)
from analysis.attribution.run_base_adapter_comparison import (
    ADAPTER_NODE_SHAPE,
    BASE_NODE_SHAPE,
    LOCAL_BASE_FEATURE_SCAN,
    LOCAL_FEATURE_SCAN,
    SOURCE_ADAPTER,
    SOURCE_BASE,
    STANDARD_GEMMA2_BASE_FEATURE_SCAN,
    build_gemmascope_transcoder_config,
    compact_overlay_payload,
    default_base_feature_scan_for_gemmascope,
    build_overlay_payload,
    gemmascope_layer_refs,
    normalize_overlay_payload,
    resolve_gemmascope_l0_values,
    write_compact_overlay_graphs,
    write_overlay_graphs,
)
from analysis.attribution.serve_comparison_graphs import start_comparison_server


def _node(node_id: str, feature_type: str, *, layer="0", feature=1, ctx_idx=0):
    return {
        "node_id": node_id,
        "feature": feature,
        "layer": layer,
        "ctx_idx": ctx_idx,
        "feature_type": feature_type,
        "token_prob": 0.0,
        "is_target_logit": False,
        "run_idx": 0,
        "reverse_ctx_idx": 0,
        "jsNodeId": node_id,
        "clerp": "",
        "influence": 0.5,
        "activation": 1.0,
    }


def _payload(slug: str, *, feature_node_id: str, logit_node_id: str):
    prompt_tokens = ["<bos>", "Hello"]
    return {
        "metadata": {
            "slug": slug,
            "scan": f"{slug}_scan",
            "transcoder_list": [],
            "prompt_tokens": prompt_tokens,
            "prompt": "<bos>Hello",
            "node_threshold": 0.8,
            "schema_version": 1,
        },
        "qParams": {
            "pinnedIds": [],
            "supernodes": [],
            "linkType": "both",
            "clickedId": logit_node_id,
            "sg_pos": "",
        },
        "nodes": [
            _node("E_1_0", "embedding", layer="E", feature=0, ctx_idx=0),
            _node(feature_node_id, "cross layer transcoder", layer="0", feature=1, ctx_idx=1),
            _node(logit_node_id, "logit", layer="27", feature=2, ctx_idx=1),
        ],
        "links": [
            {"source": "E_1_0", "target": feature_node_id, "weight": 1.0},
            {"source": feature_node_id, "target": logit_node_id, "weight": 2.0},
        ],
    }


class BaseAdapterComparisonTests(unittest.TestCase):
    def test_prepare_overlay_view_cap_from_arg(self):
        self.assertIsNone(cap_from_arg(None))
        self.assertIsNone(cap_from_arg(-1))
        self.assertEqual(cap_from_arg(0), 0)
        self.assertEqual(cap_from_arg(7), 7)

    def test_prepare_overlay_view_default_output_dir(self):
        output_dir = default_output_dir(
            Path("/tmp/run/overlay"),
            max_base_feature_nodes=64,
            max_base_error_nodes=0,
            max_adapter_feature_nodes=None,
        )
        self.assertEqual(output_dir, Path("/tmp/run/overlay_compact_base64_error0"))

    def test_prepare_overlay_view_default_output_dir_includes_adapter_cap_and_all(self):
        output_dir = default_output_dir(
            Path("/tmp/run/overlay"),
            max_base_feature_nodes=None,
            max_base_error_nodes=32,
            max_adapter_feature_nodes=10,
        )
        self.assertEqual(output_dir, Path("/tmp/run/overlay_compact_baseall_error32_adapter10"))

    def test_node_shape_metadata_uses_semantic_source_names(self):
        self.assertEqual(BASE_NODE_SHAPE, "base_model")
        self.assertEqual(ADAPTER_NODE_SHAPE, "adapter_model")

    def test_default_base_feature_scan_for_standard_gemma2(self):
        scan = default_base_feature_scan_for_gemmascope(
            base_model="google/gemma-2-2b",
            gemmascope_repo="google/gemma-scope-2b-pt-transcoders",
            gemmascope_width="width_16k",
        )

        self.assertEqual(scan, STANDARD_GEMMA2_BASE_FEATURE_SCAN)

    def test_default_base_feature_scan_rejects_nonstandard_model(self):
        scan = default_base_feature_scan_for_gemmascope(
            base_model="google/gemma-2-9b",
            gemmascope_repo="google/gemma-scope-2b-pt-transcoders",
            gemmascope_width="width_16k",
        )

        self.assertIsNone(scan)

    def test_gemmascope_layer_refs_are_explicit(self):
        refs = gemmascope_layer_refs(
            repo="google/gemma-scope-2b-pt-transcoders",
            width="width_16k",
            l0="average_l0_76",
            n_layers=3,
        )
        self.assertEqual(
            refs,
            [
                "hf://google/gemma-scope-2b-pt-transcoders/layer_0/width_16k/average_l0_76/params.npz",
                "hf://google/gemma-scope-2b-pt-transcoders/layer_1/width_16k/average_l0_76/params.npz",
                "hf://google/gemma-scope-2b-pt-transcoders/layer_2/width_16k/average_l0_76/params.npz",
            ],
        )

    def test_builds_config_without_width_l0_defaults(self):
        config = build_gemmascope_transcoder_config(
            repo="google/gemma-scope-2b-pt-transcoders",
            width="width_16k",
            l0="average_l0_76",
            n_layers=2,
            model_name="google/gemma-2-2b",
            feature_input_hook="ln2.hook_normalized",
            feature_output_hook="hook_mlp_out",
        )
        self.assertEqual(config["model_kind"], "transcoder_set")
        self.assertEqual(config["model_name"], "google/gemma-2-2b")
        self.assertEqual(config["scan_name"], "google/gemma-scope-2b-pt-transcoders/width_16k/average_l0_76")
        self.assertEqual(config["l0_by_layer"], ["average_l0_76", "average_l0_76"])
        self.assertEqual(len(config["transcoders"]), 2)

    def test_resolves_comma_separated_l0_values(self):
        values = resolve_gemmascope_l0_values(
            repo="google/gemma-scope-2b-pt-transcoders",
            width="width_16k",
            l0="average_l0_76,average_l0_104",
            n_layers=2,
            l0_match="nearest",
        )

        self.assertEqual(values, ["average_l0_76", "average_l0_104"])

    def test_overlay_prefixes_source_nodes_and_shares_embeddings(self):
        base_payload = _payload("run__capital__habc", feature_node_id="0_1_1", logit_node_id="27_2_1")
        adapter_payload = _payload("run__capital__habc", feature_node_id="0_1_1", logit_node_id="27_2_1")

        overlay = build_overlay_payload(
            base_payload=base_payload,
            adapter_payload=adapter_payload,
            base_feature_scan="siddharthmb/base-features",
        )
        node_ids = {node["node_id"] for node in overlay["nodes"]}
        self.assertIn("E_1_0", node_ids)
        self.assertIn("base__0_1_1", node_ids)
        self.assertIn("adapter__0_1_1", node_ids)
        self.assertIn("base__27_2_1", node_ids)
        self.assertIn("adapter__27_2_1", node_ids)

        by_id = {node["node_id"]: node for node in overlay["nodes"]}
        self.assertEqual(by_id["base__0_1_1"]["source_model"], SOURCE_BASE)
        self.assertEqual(by_id["base__0_1_1"]["original_node_id"], "0_1_1")
        self.assertEqual(by_id["base__0_1_1"]["node_shape"], BASE_NODE_SHAPE)
        self.assertEqual(by_id["base__0_1_1"]["source_feature_id"], "1")
        self.assertIn("Base GemmaScope feature L0/F1", by_id["base__0_1_1"]["clerp"])
        self.assertEqual(by_id["adapter__0_1_1"]["source_model"], SOURCE_ADAPTER)
        self.assertEqual(by_id["adapter__0_1_1"]["node_shape"], ADAPTER_NODE_SHAPE)

        links = {(link["source"], link["target"], link["source_model"]) for link in overlay["links"]}
        self.assertIn(("E_1_0", "base__0_1_1", SOURCE_BASE), links)
        self.assertIn(("base__0_1_1", "base__27_2_1", SOURCE_BASE), links)
        self.assertIn(("E_1_0", "adapter__0_1_1", SOURCE_ADAPTER), links)
        self.assertIn(("adapter__0_1_1", "adapter__27_2_1", SOURCE_ADAPTER), links)
        self.assertEqual(
            overlay["metadata"]["comparison"]["adapter_feature_scan"],
            LOCAL_FEATURE_SCAN,
        )
        self.assertEqual(
            overlay["metadata"]["comparison"]["base_feature_scan"],
            "siddharthmb/base-features",
        )

    def test_overlay_keeps_logits_as_logit_shape(self):
        base_payload = _payload("run__capital__habc", feature_node_id="0_1_1", logit_node_id="27_2_1")
        adapter_payload = _payload("run__capital__habc", feature_node_id="0_1_1", logit_node_id="27_2_1")

        overlay = build_overlay_payload(base_payload=base_payload, adapter_payload=adapter_payload)
        by_id = {node["node_id"]: node for node in overlay["nodes"]}

        self.assertEqual(by_id["base__27_2_1"]["node_shape"], "logit")
        self.assertEqual(by_id["adapter__27_2_1"]["node_shape"], "logit")

    def test_normalize_overlay_adds_labels_to_old_payloads(self):
        overlay = build_overlay_payload(
            base_payload=_payload("run__capital__habc", feature_node_id="0_1_1", logit_node_id="27_2_1"),
            adapter_payload=_payload("run__capital__habc", feature_node_id="0_1_1", logit_node_id="27_2_1"),
        )
        for node in overlay["nodes"]:
            node.pop("source_feature_label", None)
            if node.get("source_model") == SOURCE_BASE:
                node["node_shape"] = "base_model"
                node["clerp"] = ""

        normalized = normalize_overlay_payload(overlay)
        by_id = {node["node_id"]: node for node in normalized["nodes"]}

        self.assertEqual(by_id["base__27_2_1"]["node_shape"], "logit")
        self.assertEqual(by_id["base__0_1_1"]["source_feature_label"], "base L0/F1")
        self.assertIn("Base GemmaScope feature L0/F1", by_id["base__0_1_1"]["clerp"])

    def test_normalize_overlay_backfills_adapter_feature_scan(self):
        overlay = build_overlay_payload(
            base_payload=_payload("run__capital__habc", feature_node_id="0_1_1", logit_node_id="27_2_1"),
            adapter_payload=_payload("run__capital__habc", feature_node_id="0_1_1", logit_node_id="27_2_1"),
        )
        overlay["metadata"]["comparison"].pop("adapter_feature_scan")

        normalized = normalize_overlay_payload(overlay)

        self.assertEqual(
            normalized["metadata"]["comparison"]["adapter_feature_scan"],
            LOCAL_FEATURE_SCAN,
        )

    def test_compact_overlay_limits_base_feature_nodes(self):
        base_payload = _payload("run__capital__habc", feature_node_id="0_1_1", logit_node_id="27_2_1")
        adapter_payload = _payload("run__capital__habc", feature_node_id="0_1_1", logit_node_id="27_2_1")
        for index in range(2, 5):
            node_id = f"0_{index}_1"
            base_payload["nodes"].append(
                _node(node_id, "cross layer transcoder", layer="0", feature=index, ctx_idx=1)
            )
            base_payload["links"].append({"source": "E_1_0", "target": node_id, "weight": index})
            base_payload["links"].append({"source": node_id, "target": "27_2_1", "weight": index})

        overlay = build_overlay_payload(base_payload=base_payload, adapter_payload=adapter_payload)
        compact = compact_overlay_payload(overlay, max_base_feature_nodes=2)

        kept_base_features = [
            node
            for node in compact["nodes"]
            if node.get("source_model") == SOURCE_BASE
            and node.get("feature_type") == "cross layer transcoder"
        ]
        self.assertEqual(len(kept_base_features), 2)
        self.assertEqual(
            compact["metadata"]["comparison"]["compact_view"]["removed_feature_nodes"][SOURCE_BASE],
            2,
        )
        compact_node_ids = {node["node_id"] for node in compact["nodes"]}
        for link in compact["links"]:
            self.assertIn(link["source"], compact_node_ids)
            self.assertIn(link["target"], compact_node_ids)

    def test_compact_overlay_limits_base_error_nodes(self):
        base_payload = _payload("run__capital__habc", feature_node_id="0_1_1", logit_node_id="27_2_1")
        adapter_payload = _payload("run__capital__habc", feature_node_id="0_1_1", logit_node_id="27_2_1")
        for index in range(3):
            node_id = f"0_err{index}_1"
            base_payload["nodes"].append(
                _node(node_id, "mlp reconstruction error", layer="0", feature=index, ctx_idx=1)
            )
            base_payload["links"].append({"source": "E_1_0", "target": node_id, "weight": index + 1})
            base_payload["links"].append({"source": node_id, "target": "27_2_1", "weight": index + 1})

        overlay = build_overlay_payload(base_payload=base_payload, adapter_payload=adapter_payload)
        compact = compact_overlay_payload(overlay, max_base_error_nodes=1)

        kept_base_errors = [
            node
            for node in compact["nodes"]
            if node.get("source_model") == SOURCE_BASE
            and "error" in str(node.get("feature_type"))
        ]
        self.assertEqual(len(kept_base_errors), 1)
        self.assertEqual(
            compact["metadata"]["comparison"]["compact_view"]["removed_error_nodes"][SOURCE_BASE],
            2,
        )

    def test_overlay_hides_direct_embedding_logit_links_by_default(self):
        base_payload = _payload("run__capital__habc", feature_node_id="0_1_1", logit_node_id="27_2_1")
        adapter_payload = _payload("run__capital__habc", feature_node_id="0_1_1", logit_node_id="27_2_1")
        base_payload["links"].append({"source": "E_1_0", "target": "27_2_1", "weight": 3.0})
        adapter_payload["links"].append({"source": "E_1_0", "target": "27_2_1", "weight": 4.0})

        overlay = build_overlay_payload(base_payload=base_payload, adapter_payload=adapter_payload)

        links = {(link["source"], link["target"], link["source_model"]) for link in overlay["links"]}
        self.assertNotIn(("E_1_0", "base__27_2_1", SOURCE_BASE), links)
        self.assertNotIn(("E_1_0", "adapter__27_2_1", SOURCE_ADAPTER), links)
        self.assertEqual(
            overlay["metadata"]["comparison"]["link_filter"]["skipped_direct_embedding_logit_links"],
            {SOURCE_BASE: 1, SOURCE_ADAPTER: 1},
        )

    def test_overlay_can_preserve_direct_embedding_logit_links(self):
        base_payload = _payload("run__capital__habc", feature_node_id="0_1_1", logit_node_id="27_2_1")
        adapter_payload = _payload("run__capital__habc", feature_node_id="0_1_1", logit_node_id="27_2_1")
        base_payload["links"].append({"source": "E_1_0", "target": "27_2_1", "weight": 3.0})

        overlay = build_overlay_payload(
            base_payload=base_payload,
            adapter_payload=adapter_payload,
            hide_direct_embedding_logit_links=False,
        )

        links = {(link["source"], link["target"], link["source_model"]) for link in overlay["links"]}
        self.assertIn(("E_1_0", "base__27_2_1", SOURCE_BASE), links)
        self.assertFalse(overlay["metadata"]["comparison"]["link_filter"]["hide_direct_embedding_logit_links"])

    def test_overlay_rejects_mismatched_prompt_tokens(self):
        base_payload = _payload("base", feature_node_id="0_1_1", logit_node_id="27_2_1")
        adapter_payload = _payload("adapter", feature_node_id="0_1_1", logit_node_id="27_2_1")
        adapter_payload["metadata"]["prompt_tokens"] = ["<bos>", "Different"]

        with self.assertRaisesRegex(ValueError, "different prompt tokens"):
            build_overlay_payload(base_payload=base_payload, adapter_payload=adapter_payload)

    def test_write_overlay_graphs_writes_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_dir = root / "base"
            adapter_dir = root / "adapter"
            overlay_dir = root / "overlay"
            base_dir.mkdir()
            adapter_dir.mkdir()
            (base_dir / "run__capital__habc.json").write_text(
                json.dumps(_payload("run__capital__habc", feature_node_id="0_1_1", logit_node_id="27_2_1"))
            )
            (adapter_dir / "run__capital__habc.json").write_text(
                json.dumps(_payload("run__capital__habc", feature_node_id="0_1_1", logit_node_id="27_2_1"))
            )

            paths = write_overlay_graphs(
                base_graph_dir=base_dir,
                adapter_graph_dir=adapter_dir,
                overlay_graph_dir=overlay_dir,
            )

            self.assertEqual(len(paths), 1)
            self.assertTrue(paths[0].exists())
            metadata = json.loads((overlay_dir / "graph-metadata.json").read_text())
            self.assertEqual(metadata["graphs"][0]["slug"], "run__capital__habc__overlay")

    def test_write_compact_overlay_graphs_writes_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            overlay_dir = root / "overlay"
            compact_dir = root / "overlay_compact"
            overlay_dir.mkdir()
            overlay = build_overlay_payload(
                base_payload=_payload("run__capital__habc", feature_node_id="0_1_1", logit_node_id="27_2_1"),
                adapter_payload=_payload("run__capital__habc", feature_node_id="0_1_1", logit_node_id="27_2_1"),
            )
            overlay["metadata"]["comparison"][
                "base_scan"
            ] = "google/gemma-scope-2b-pt-transcoders/width_16k/average_l0_76_nearest"
            (overlay_dir / f"{overlay['metadata']['slug']}.json").write_text(json.dumps(overlay))

            self.assertEqual(infer_base_feature_scan(overlay_dir), STANDARD_GEMMA2_BASE_FEATURE_SCAN)

            paths = write_compact_overlay_graphs(
                overlay_graph_dir=overlay_dir,
                compact_overlay_graph_dir=compact_dir,
                max_base_feature_nodes=0,
                base_feature_scan=STANDARD_GEMMA2_BASE_FEATURE_SCAN,
                adapter_feature_scan=LOCAL_FEATURE_SCAN,
            )

            self.assertEqual(len(paths), 1)
            metadata = json.loads((compact_dir / "graph-metadata.json").read_text())
            self.assertEqual(
                metadata["graphs"][0]["comparison"]["compact_view"]["max_base_feature_nodes"],
                0,
            )
            self.assertEqual(
                metadata["graphs"][0]["comparison"]["base_feature_scan"],
                STANDARD_GEMMA2_BASE_FEATURE_SCAN,
            )
            self.assertEqual(
                metadata["graphs"][0]["comparison"]["adapter_feature_scan"],
                LOCAL_FEATURE_SCAN,
            )

    def test_frontend_patch_adds_source_feature_ids_and_node_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attribution_dir = root / "attribution_graph"
            feature_examples_dir = root / "feature_examples"
            attribution_dir.mkdir()
            feature_examples_dir.mkdir()
            util_path = attribution_dir / "util-cg.js"
            link_path = attribution_dir / "init-cg-link-graph.js"
            feature_detail_path = attribution_dir / "init-cg-feature-detail.js"
            feature_examples_path = feature_examples_dir / "init-feature-examples.js"
            node_connections_path = attribution_dir / "init-cg-node-connections.js"
            util_path.write_text(
                "\n".join(
                    [
                        "d.featureId = `${d.layer}_${d.feature}_${d.ctx_idx}`",
                        ".text(d => featureTypeToText(d.feature_type))",
                        "fontSize: 9,",
                        "        textAnchor:",
                        FEATURE_TYPE_FUNCTION_OLD.rstrip("\n"),
                        "    featureTypeToText,",
                    ]
                )
                + "\n"
            )
            link_path.write_text(
                "\n".join(
                    [
                        ".text(d => utilCg.featureTypeToText(d.feature_type))",
                        "fontSize: 9,",
                        "      fill:",
                        "dominantBaseline: 'central',",
                        "    })",
                    ]
                )
            )
            feature_detail_path.write_text(
                "\n".join(
                    [
                        "      const scan = data.metadata.scan?.startsWith('custom-') ? data.metadata.transcoder_list[d.layer] : data.metadata.scan;",
                        "      if (typeof currentActivation == 'number') {",
                        "      featureExamples.loadFeature(scan, d.featureIndex)",
                        "      renderFeatureExamples(scan, d.featureIndex)",
                        "      examplesSel.st({opacity: 1})",
                    ]
                )
                + "\n"
            )
            feature_examples_path.write_text(FEATURE_URL_FUNCTION_OLD)
            node_connections_path.write_text(
                "headerSel.append('span.feature-icon').text(utilCg.featureTypeToText(clickedNode.feature_type))\n"
            )

            patch_frontend_assets(root)

            self.assertIn(FEATURE_ID_PATCH_NEW, util_path.read_text())
            self.assertIn(FEATURE_ROW_FONT_SIZE_NEW, util_path.read_text())
            self.assertIn(FEATURE_TYPE_FUNCTION_NEW, util_path.read_text())
            self.assertIn(LINK_GRAPH_NODE_TEXT_NEW, link_path.read_text())
            self.assertIn(LINK_GRAPH_FONT_SIZE_NEW, link_path.read_text())
            self.assertIn(FEATURE_DETAIL_SCAN_NEW, feature_detail_path.read_text())
            self.assertIn(FEATURE_HISTOGRAM_GUARD_NEW, feature_detail_path.read_text())
            self.assertIn(FEATURE_URL_FUNCTION_NEW, feature_examples_path.read_text())
            self.assertIn(FEATURE_EXAMPLES_LOAD_NEW, feature_detail_path.read_text())
            self.assertIn(NODE_CONNECTIONS_HEADER_ICON_NEW, node_connections_path.read_text())

    def test_comparison_server_serves_adapter_and_base_feature_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph_dir = root / "graphs"
            adapter_dir = root / "adapter_features"
            base_dir = root / "base_features"
            graph_dir.mkdir()
            adapter_dir.mkdir()
            base_dir.mkdir()
            (graph_dir / "graph-metadata.json").write_text(json.dumps({"graphs": []}))
            (adapter_dir / "index.json.gz").write_bytes(b"adapter")
            (base_dir / "index.json.gz").write_bytes(b"base")

            server = start_comparison_server(
                graph_file_dir=graph_dir,
                adapter_features_dir=adapter_dir,
                base_features_dir=base_dir,
                port=0,
            )
            try:
                port = server.httpd.server_address[1]
                adapter_req = Request(f"http://localhost:{port}{LOCAL_FEATURE_SCAN}/index.json.gz")
                base_req = Request(f"http://localhost:{port}{LOCAL_BASE_FEATURE_SCAN}/index.json.gz")
                legacy_req = Request(f"http://localhost:{port}/features/index.json.gz")

                self.assertEqual(urlopen(adapter_req, timeout=2).read(), b"adapter")
                self.assertEqual(urlopen(base_req, timeout=2).read(), b"base")
                self.assertEqual(urlopen(legacy_req, timeout=2).read(), b"adapter")
            finally:
                server.stop()


if __name__ == "__main__":
    unittest.main()
