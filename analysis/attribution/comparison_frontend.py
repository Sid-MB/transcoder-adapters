"""Runtime frontend overrides for base-vs-adapter comparison graphs."""

from __future__ import annotations

import shutil
from importlib.resources import files
from pathlib import Path

from helpers.log import logger


FEATURE_ID_PATCH_OLD = "d.featureId = `${d.layer}_${d.feature}_${d.ctx_idx}`"
FEATURE_ID_PATCH_NEW = (
    "var sourceFeaturePrefix = d.source_model ? `${d.source_model}_` : ''\n"
    "      d.featureId = `${sourceFeaturePrefix}${d.layer}_${d.feature}_${d.ctx_idx}`"
)
LINK_GRAPH_NODE_TEXT_OLD = ".text(d => utilCg.featureTypeToText(d.feature_type))"
LINK_GRAPH_NODE_TEXT_NEW = (
    ".text(d => utilCg.nodeShapeToText ? utilCg.nodeShapeToText(d) : "
    "utilCg.featureTypeToText(d.feature_type))"
)
LINK_GRAPH_FONT_SIZE_OLD = "fontSize: 9,\n      fill:"
LINK_GRAPH_FONT_SIZE_NEW = (
    "fontSize: d => utilCg.nodeShapeFontSize ? utilCg.nodeShapeFontSize(d) : 9,\n"
    "      fill:"
)
LINK_GRAPH_NODE_OPACITY_OLD = "dominantBaseline: 'central',\n    })"
LINK_GRAPH_NODE_OPACITY_NEW = (
    "dominantBaseline: 'central',\n"
    "      opacity: d => utilCg.nodeShapeOpacity ? utilCg.nodeShapeOpacity(d) : 1,\n"
    "    })"
)
FEATURE_ROW_ICON_OLD = ".text(d => featureTypeToText(d.feature_type))"
FEATURE_ROW_ICON_NEW = ".text(d => nodeShapeToText(d))"
FEATURE_ROW_FONT_SIZE_OLD = "fontSize: 9,\n        textAnchor:"
FEATURE_ROW_FONT_SIZE_NEW = "fontSize: d => nodeShapeFontSize(d),\n        textAnchor:"
FEATURE_TYPE_RETURN_OLD = "    featureTypeToText,\n"
FEATURE_TYPE_RETURN_NEW = (
    "    featureTypeToText,\n    nodeShapeToText,\n    nodeShapeFontSize,\n    nodeShapeOpacity,\n"
    "    nodeFeatureScan,\n"
)
FEATURE_TYPE_FUNCTION_OLD = """  function featureTypeToText(type){
    if (type == 'logit') return '■'
    if (type == 'embedding') return '■'
    if (type === 'mlp reconstruction error') return '◆'
    return '●'
    
  }
"""
FEATURE_TYPE_FUNCTION_NEW = """  function nodeShapeToText(node){
    if (node?.feature_type == 'logit') return '■'
    if (node?.feature_type == 'embedding') return '■'
    if (node?.feature_type === 'mlp reconstruction error') return '◆'
    if (node?.node_shape == 'base_model') return '⬢'
    if (node?.node_shape == 'adapter_model') return '●'
    if (node?.node_shape == 'shared') return '■'
    if (node?.node_shape == 'triangle') return '▲'
    if (node?.node_shape == 'hexagon') return '⬢'
    if (node?.node_shape == 'circle') return '●'
    if (node?.node_shape == 'diamond') return '◆'
    if (node?.node_shape == 'square') return '■'
    return featureTypeToText(node?.feature_type)
  }

  function nodeShapeFontSize(node){
    if (node?.node_shape == 'base_model') return 11
    if (node?.node_shape == 'hexagon') return 11
    return 9
  }

  function nodeShapeOpacity(node){
    if (node?.node_shape == 'base_model') return 0.72
    return 1
  }

  function nodeFeatureScan(data, node){
    if (node?.source_model == 'base') {
      return data?.metadata?.comparison?.base_feature_scan || null
    }
    if (node?.source_model == 'adapter') {
      return data?.metadata?.comparison?.adapter_feature_scan || '/features'
    }
    if (data?.metadata?.scan?.startsWith('custom-')) return data.metadata.transcoder_list[node.layer]
    return data?.metadata?.scan
  }

  function featureTypeToText(type){
    if (type == 'logit') return '■'
    if (type == 'embedding') return '■'
    if (type === 'mlp reconstruction error') return '◆'
    return '●'
    
  }
"""
FEATURE_DETAIL_SCAN_OLD = (
    "      const scan = data.metadata.scan?.startsWith('custom-') ? "
    "data.metadata.transcoder_list[d.layer] : data.metadata.scan;"
)
FEATURE_DETAIL_SCAN_NEW = (
    "      const scan = utilCg.nodeFeatureScan ? utilCg.nodeFeatureScan(data, d) : "
    "(data.metadata.scan?.startsWith('custom-') ? data.metadata.transcoder_list[d.layer] : data.metadata.scan);"
)
FEATURE_HISTOGRAM_GUARD_OLD = "      if (typeof currentActivation == 'number') {"
FEATURE_HISTOGRAM_GUARD_NEW = "      if (typeof currentActivation == 'number' && scan) {"
FEATURE_EXAMPLES_LOAD_OLD = """      featureExamples.loadFeature(scan, d.featureIndex)
      renderFeatureExamples(scan, d.featureIndex)
      examplesSel.st({opacity: 1})
"""
FEATURE_EXAMPLES_LOAD_NEW = """      if (!scan) {
        examplesSel.st({opacity: 0})
      } else {
        featureExamples.loadFeature(scan, d.featureIndex)
        renderFeatureExamples(scan, d.featureIndex)
        examplesSel.st({opacity: 1})
      }
"""
NODE_CONNECTIONS_HEADER_ICON_OLD = (
    "headerSel.append('span.feature-icon').text(utilCg.featureTypeToText(clickedNode.feature_type))"
)
NODE_CONNECTIONS_HEADER_ICON_NEW = (
    "headerSel.append('span.feature-icon').text(utilCg.nodeShapeToText ? "
    "utilCg.nodeShapeToText(clickedNode) : utilCg.featureTypeToText(clickedNode.feature_type))"
)


def _replace_once(text: str, old: str, new: str, *, path: Path) -> str:
    if old not in text:
        raise RuntimeError(f"Could not patch {path}; expected snippet was not found.")
    return text.replace(old, new, 1)


def patch_frontend_assets(frontend_dir: Path) -> None:
    """Patch a copied circuit-tracer frontend directory in place."""
    util_path = frontend_dir / "attribution_graph" / "util-cg.js"
    link_graph_path = frontend_dir / "attribution_graph" / "init-cg-link-graph.js"
    feature_detail_path = frontend_dir / "attribution_graph" / "init-cg-feature-detail.js"
    node_connections_path = frontend_dir / "attribution_graph" / "init-cg-node-connections.js"

    util_text = util_path.read_text()
    util_text = _replace_once(util_text, FEATURE_ID_PATCH_OLD, FEATURE_ID_PATCH_NEW, path=util_path)
    util_text = _replace_once(util_text, FEATURE_ROW_ICON_OLD, FEATURE_ROW_ICON_NEW, path=util_path)
    util_text = _replace_once(util_text, FEATURE_ROW_FONT_SIZE_OLD, FEATURE_ROW_FONT_SIZE_NEW, path=util_path)
    util_text = _replace_once(util_text, FEATURE_TYPE_FUNCTION_OLD, FEATURE_TYPE_FUNCTION_NEW, path=util_path)
    util_text = _replace_once(util_text, FEATURE_TYPE_RETURN_OLD, FEATURE_TYPE_RETURN_NEW, path=util_path)
    util_path.write_text(util_text)

    link_graph_text = link_graph_path.read_text()
    link_graph_text = _replace_once(
        link_graph_text,
        LINK_GRAPH_NODE_TEXT_OLD,
        LINK_GRAPH_NODE_TEXT_NEW,
        path=link_graph_path,
    )
    link_graph_text = _replace_once(
        link_graph_text,
        LINK_GRAPH_FONT_SIZE_OLD,
        LINK_GRAPH_FONT_SIZE_NEW,
        path=link_graph_path,
    )
    link_graph_text = _replace_once(
        link_graph_text,
        LINK_GRAPH_NODE_OPACITY_OLD,
        LINK_GRAPH_NODE_OPACITY_NEW,
        path=link_graph_path,
    )
    link_graph_path.write_text(link_graph_text)

    feature_detail_text = feature_detail_path.read_text()
    feature_detail_text = _replace_once(
        feature_detail_text,
        FEATURE_DETAIL_SCAN_OLD,
        FEATURE_DETAIL_SCAN_NEW,
        path=feature_detail_path,
    )
    feature_detail_text = _replace_once(
        feature_detail_text,
        FEATURE_HISTOGRAM_GUARD_OLD,
        FEATURE_HISTOGRAM_GUARD_NEW,
        path=feature_detail_path,
    )
    feature_detail_text = _replace_once(
        feature_detail_text,
        FEATURE_EXAMPLES_LOAD_OLD,
        FEATURE_EXAMPLES_LOAD_NEW,
        path=feature_detail_path,
    )
    feature_detail_path.write_text(feature_detail_text)

    node_connections_text = node_connections_path.read_text()
    node_connections_text = _replace_once(
        node_connections_text,
        NODE_CONNECTIONS_HEADER_ICON_OLD,
        NODE_CONNECTIONS_HEADER_ICON_NEW,
        path=node_connections_path,
    )
    node_connections_path.write_text(node_connections_text)


def prepare_comparison_frontend(output_dir: Path) -> Path:
    """Copy upstream circuit-tracer assets and apply comparison graph overrides."""
    source_dir = Path(files("circuit_tracer") / "frontend" / "assets")
    target_dir = output_dir / "comparison_frontend_assets"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)
    patch_frontend_assets(target_dir)
    logger.info(f"Prepared comparison frontend assets: {target_dir}")
    return target_dir
