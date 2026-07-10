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
# clerpUUID keys saved feature annotations. Upstream takes the first two featureId parts
# (layer_feature), but our featureId patch above prefixes source_model, which would collapse
# the key to model_layer — one shared annotation per layer. Drop only the trailing ctx_idx
# instead so the key is layer_feature (plain) or model_layer_feature (overlay).
CLERP_UUID_PATCH_OLD = "return '🤖' + parts[0] + '_' + parts[1];"
CLERP_UUID_PATCH_NEW = "return '🤖' + parts.slice(0, -1).join('_');"
# hClerpUpdateFn extracts the feature index from the key as parts[1]; with a source_model
# prefix that would be the layer. The feature index is always the last part.
CLERP_URL_PARAM_PATCH_OLD = ".map(([key, value]) => [key.split('🤖')[1].split('_')[1], value])"
CLERP_URL_PARAM_PATCH_NEW = ".map(([key, value]) => [key.split('🤖')[1].split('_').at(-1), value])"
LINK_GRAPH_STATS_BANNER_OLD = (
    "  var c = d3.conventions({\n"
    "    sel: cgSel.select('.link-graph').html(''),\n"
    "    margin: {left: visState.isHideLayer ? 0 : 30, bottom: 85},\n"
    "    layers: 'sccccs',\n"
    "  })\n"
)
LINK_GRAPH_STATS_BANNER_NEW = (
    "  var c = d3.conventions({\n"
    "    sel: cgSel.select('.link-graph').html(''),\n"
    "    margin: {left: visState.isHideLayer ? 0 : 30, bottom: 85},\n"
    "    layers: 'sccccs',\n"
    "  })\n"
    "\n"
    "  // Comparison-graph addition: a small, graph-level stats banner summarizing the\n"
    "  // overall feature composition. The bold figure is the as-built composition\n"
    "  // (data.metadata.comparison.node_counts, written by tag_combined_graph at the build\n"
    "  // node_threshold). The dim '(N shown)' is recomputed live from the rendered node set\n"
    "  // so it tracks the pruning slider. Skipped entirely for plain circuit-tracer graphs.\n"
    "  if (data.metadata.comparison && data.metadata.comparison.node_counts) {\n"
    "    var comparisonNodeCounts = data.metadata.comparison.node_counts\n"
    "    var shownNodeCounts = {base: 0, adapter: 0, error: 0}\n"
    "    nodes.forEach(function(node){\n"
    "      if (node.isError || node.node_shape == 'error' || (node.feature_type && node.feature_type.indexOf('error') != -1)) shownNodeCounts.error++\n"
    "      else if (node.node_shape == 'base_model' || node.source_model == 'base') shownNodeCounts.base++\n"
    "      else if (node.node_shape == 'adapter_model' || node.source_model == 'adapter') shownNodeCounts.adapter++\n"
    "    })\n"
    "    var fmtComparisonStat = function(label, glyph, built, shown){\n"
    "      return `<span>${label} ${glyph} ${built || 0} <span style='opacity:.55'>(${shown} shown)</span></span>`\n"
    "    }\n"
    "    var linkGraphSel = cgSel.select('.link-graph')\n"
    "    // Anchor the absolute banner to .link-graph without clobbering any positioning\n"
    "    // gridsnap may already have applied (only promote a static container to relative).\n"
    "    if (linkGraphSel.node() && getComputedStyle(linkGraphSel.node()).position == 'static') linkGraphSel.st({position: 'relative'})\n"
    "    linkGraphSel.select('.comparison-stats-banner').remove()\n"
    "    linkGraphSel\n"
    "      .insert('div.comparison-stats-banner', ':first-child')\n"
    "      .st({\n"
    "        position: 'absolute',\n"
    "        top: 2,\n"
    "        left: 40,\n"
    "        zIndex: 10,\n"
    "        fontSize: 11,\n"
    "        color: '#555',\n"
    "        padding: '2px 6px',\n"
    "        background: 'rgba(245,244,238,0.85)',\n"
    "        borderRadius: '3px',\n"
    "        pointerEvents: 'none',\n"
    "      })\n"
    "      .html([\n"
    "        fmtComparisonStat('Base', '\\u2b22', comparisonNodeCounts.base_features, shownNodeCounts.base),\n"
    "        fmtComparisonStat('Adapter', '\\u25cf', comparisonNodeCounts.adapter_features, shownNodeCounts.adapter),\n"
    "        fmtComparisonStat('Error', '\\u25b2', comparisonNodeCounts.error_nodes, shownNodeCounts.error),\n"
    "      ].join('  \\u00b7  '))\n"
    "  }\n"
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
    if (node?.feature_type === 'mlp reconstruction error') return '▲'
    if (node?.node_shape == 'error') return '▲'
    if (node?.node_shape == 'triangle') return '▲'
    if (node?.feature_type == 'logit') return '■'
    if (node?.feature_type == 'embedding') return '■'
    if (node?.node_shape == 'base_model') return '⬢'
    if (node?.node_shape == 'adapter_model') return '●'
    if (node?.node_shape == 'shared') return '■'
    if (node?.node_shape == 'hexagon') return '⬢'
    if (node?.node_shape == 'circle') return '●'
    if (node?.node_shape == 'diamond') return '◆'
    if (node?.node_shape == 'square') return '■'
    return featureTypeToText(node?.feature_type)
  }

  function nodeShapeFontSize(node){
    if (node?.feature_type === 'mlp reconstruction error') return 11
    if (node?.node_shape == 'error') return 11
    if (node?.node_shape == 'triangle') return 11
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
# Base (GemmaScope) features have public Neuronpedia pages; adapter features are ours and
# don't. For base-source nodes, append a Neuronpedia link next to the feature title.
# Base nodes keep their GemmaScope within-layer index in d.feature (see tag_combined_graph),
# which is exactly Neuronpedia's feature id. The 16k scan name matches the width_16k
# GemmaScope transcoders every overlay uses.
FEATURE_DETAIL_NEURONPEDIA_OLD = (
    "      const featureTitleSel = headerTopRowSel.append('div.feature-title')\n"
    '        .html(`Feature&nbsp;<a style="color: inherit;" href="${d.url}" target="_blank">'
    '${label}</a> <span style="font-size: 0.9em; color: #777;">Act: ${actText}</span>`)\n'
)
FEATURE_DETAIL_NEURONPEDIA_NEW = (
    FEATURE_DETAIL_NEURONPEDIA_OLD
    + "      if (d.source_model == 'base'){\n"
    "        featureTitleSel.append('a')\n"
    "          .at({href: `https://www.neuronpedia.org/gemma-2-2b/${d.layer}-gemmascope-transcoder-16k/${d.feature}`, target: '_blank'})\n"
    "          .st({marginLeft: 6, fontSize: '0.85em', color: '#4a7bd0', textDecoration: 'none'})\n"
    "          .text('Neuronpedia \\u2197')\n"
    "      }\n"
)
FEATURE_HISTOGRAM_GUARD_OLD = "      if (typeof currentActivation == 'number') {"
FEATURE_HISTOGRAM_GUARD_NEW = "      if (typeof currentActivation == 'number' && scan) {"
FEATURE_URL_FUNCTION_OLD = """  function featureUrl(scan, path) {
    // If the scan is a local path, fetch features from the local directory
    if (scan.startsWith('/') || scan.startsWith('./')) {
      return `/features/${path}`
    }
    // else create the HuggingFace url
    const [repoId, rest] = scan.split('//')
    const [filePath, revision] = rest ? rest.split('@') : [null, scan.split('@')[1]]
    const prefix = filePath ? `${filePath}/` : ''
    return `https://huggingface.co/${repoId.split('@')[0]}/resolve/${revision || 'main'}/${prefix}features/${path}`
  }
"""
FEATURE_URL_FUNCTION_NEW = """  function featureUrl(scan, path) {
    // Local comparison scans are served under their own aliases so an overlay
    // can show base and adapter feature examples at the same time.
    if (scan.startsWith('/')) {
      return `${scan.replace(/\\/$/, '')}/${path}`
    }
    if (scan.startsWith('./')) {
      return `/features/${path}`
    }
    // else create the HuggingFace url
    const [repoId, rest] = scan.split('//')
    const [filePath, revision] = rest ? rest.split('@') : [null, scan.split('@')[1]]
    const prefix = filePath ? `${filePath}/` : ''
    return `https://huggingface.co/${repoId.split('@')[0]}/resolve/${revision || 'main'}/${prefix}features/${path}`
  }
"""
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
FEATURE_STATS_HELPER_OLD = (
    "  var renderFeatureExamples = util.throttleDebounce(featureExamples.renderFeature, 200)\n"
)
FEATURE_STATS_HELPER_NEW = """  var renderFeatureExamples = util.throttleDebounce(featureExamples.renderFeature, 200)

  // Comparison-graph addition: per-feature proportion stats injected above the
  // examples. activation_frequency = % of tokens this feature fires on. token_specificity
  // = of this feature's activations, what fraction land on each token.
  var featureStatsSel = examplesSel.insert('div.feature-proportion-stats', ':first-child')

  function ppFeatureStatsToken(token){
    if (token == null) return ''
    return String(token)
      .replace(/\\\\/g, '\\\\\\\\')
      .replace(/\\n/g, '\\\\n')
      .replace(/\\t/g, '\\\\t')
      .replace(/\\r/g, '\\\\r')
      .replace(/ /g, '·')
  }

  function renderFeatureStats(featureData){
    featureStatsSel.html('')
    if (!featureData) return
    var hasFreq = typeof featureData.activation_frequency == 'number'
    var specificity = Array.isArray(featureData.token_specificity) ? featureData.token_specificity : []
    if (!hasFreq && !specificity.length) return
    if (hasFreq){
      featureStatsSel.append('div.feature-stat-row')
        .html(`<span class="feature-stat-label">Fires on</span> <span class="feature-stat-value">${(featureData.activation_frequency * 100).toFixed(2)}%</span> <span class="feature-stat-note">of tokens</span>`)
    }
    if (specificity.length){
      var topTokens = specificity.slice(0, 5).map(t =>
        `<span class="feature-stat-token">${ppFeatureStatsToken(t.token)}</span> ${(t.fraction * 100).toFixed(0)}%`
      ).join('  ')
      featureStatsSel.append('div.feature-stat-row')
        .html(`<span class="feature-stat-label">Top tokens</span> <span class="feature-stat-value">${topTokens}</span>`)
    }
  }
"""
FEATURE_STATS_RENDER_OLD = """      if (!scan) {
        examplesSel.st({opacity: 0})
      } else {
        featureExamples.loadFeature(scan, d.featureIndex)
        renderFeatureExamples(scan, d.featureIndex)
        examplesSel.st({opacity: 1})
      }
"""
FEATURE_STATS_RENDER_NEW = """      if (!scan) {
        examplesSel.st({opacity: 0})
        renderFeatureStats(null)
      } else {
        Promise.resolve(featureExamples.loadFeature(scan, d.featureIndex))
          .then(renderFeatureStats)
          .catch(() => renderFeatureStats(null))
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

# Comparison-graph addition: append each graph's as-built composition (base/adapter/error
# node counts, baked into graph-metadata.json as d.node_counts) to its dropdown option text,
# with the same glyphs as the in-graph banner. Patched into BOTH option-builder sites in
# index.html (initial render + the refetch-on-click), hence _replace_all.
DROPDOWN_OPTION_COUNTS_OLD = "return prefix + scanName + ' — ' + d.prompt"
DROPDOWN_OPTION_COUNTS_NEW = (
    "return prefix + scanName + ' — ' + d.prompt + "
    "(d.node_counts ? `  [Base \\u2b22 ${d.node_counts.base_features || 0}  "
    "Adapter \\u25cf ${d.node_counts.adapter_features || 0}  "
    "Error \\u25b2 ${d.node_counts.error_nodes || 0}]` : '')"
)


# Fix the mouse hit-testing offset: upstream computes the pointer relative to the OUTER
# container (c.sel) and hand-subtracts the margin, then matches against node positions that
# live in the inner c.svg coordinate space. Any margin/scale/position mismatch between those
# two spaces offsets hover/click selection by a large constant (nodes appear far to the side
# of the cursor). Compute the pointer in c.svg's own space (d3.pointer(ev, c.svg.node())) so
# pointer and node coordinates pass through the identical SVG transform, then drop the manual
# margin subtraction. Applied to both the mousemove (hover) and click handlers.
LINK_GRAPH_HOVER_HITTEST_OLD = (
    "if (ev.shiftKey) return\n"
    "      var [mouseX, mouseY] = d3.pointer(ev)\n"
    "      var [closestNode, closestDistance] = findClosestPoint(mouseX - c.margin.left, mouseY - c.margin.top, nodes)"
)
LINK_GRAPH_HOVER_HITTEST_NEW = (
    "if (ev.shiftKey) return\n"
    "      var [mouseX, mouseY] = d3.pointer(ev, c.svg.node())\n"
    "      var [closestNode, closestDistance] = findClosestPoint(mouseX, mouseY, nodes)"
)
LINK_GRAPH_CLICK_HITTEST_OLD = (
    ".on('click', (ev) => {\n"
    "      var [mouseX, mouseY] = d3.pointer(ev)\n"
    "      var [closestNode, closestDistance] = findClosestPoint(mouseX - c.margin.left, mouseY - c.margin.top, nodes)"
)
LINK_GRAPH_CLICK_HITTEST_NEW = (
    ".on('click', (ev) => {\n"
    "      var [mouseX, mouseY] = d3.pointer(ev, c.svg.node())\n"
    "      var [closestNode, closestDistance] = findClosestPoint(mouseX, mouseY, nodes)"
)


def _replace_once(text: str, old: str, new: str, *, path: Path) -> str:
    if old not in text:
        raise RuntimeError(f"Could not patch {path}; expected snippet was not found.")
    return text.replace(old, new, 1)


def _replace_all(text: str, old: str, new: str, *, path: Path, min_count: int = 1) -> str:
    count = text.count(old)
    if count < min_count:
        raise RuntimeError(
            f"Could not patch {path}; expected snippet was not found "
            f"(needed >= {min_count}, found {count})."
        )
    return text.replace(old, new)


def patch_frontend_assets(frontend_dir: Path) -> None:
    """Patch a copied circuit-tracer frontend directory in place."""
    util_path = frontend_dir / "attribution_graph" / "util-cg.js"
    link_graph_path = frontend_dir / "attribution_graph" / "init-cg-link-graph.js"
    feature_detail_path = frontend_dir / "attribution_graph" / "init-cg-feature-detail.js"
    feature_examples_path = frontend_dir / "feature_examples" / "init-feature-examples.js"
    node_connections_path = frontend_dir / "attribution_graph" / "init-cg-node-connections.js"
    index_path = frontend_dir / "index.html"

    util_text = util_path.read_text()
    util_text = _replace_once(util_text, FEATURE_ID_PATCH_OLD, FEATURE_ID_PATCH_NEW, path=util_path)
    util_text = _replace_once(util_text, CLERP_UUID_PATCH_OLD, CLERP_UUID_PATCH_NEW, path=util_path)
    util_text = _replace_once(util_text, CLERP_URL_PARAM_PATCH_OLD, CLERP_URL_PARAM_PATCH_NEW, path=util_path)
    util_text = _replace_once(util_text, FEATURE_ROW_ICON_OLD, FEATURE_ROW_ICON_NEW, path=util_path)
    util_text = _replace_once(util_text, FEATURE_ROW_FONT_SIZE_OLD, FEATURE_ROW_FONT_SIZE_NEW, path=util_path)
    util_text = _replace_once(util_text, FEATURE_TYPE_FUNCTION_OLD, FEATURE_TYPE_FUNCTION_NEW, path=util_path)
    util_text = _replace_once(util_text, FEATURE_TYPE_RETURN_OLD, FEATURE_TYPE_RETURN_NEW, path=util_path)
    util_path.write_text(util_text)

    link_graph_text = link_graph_path.read_text()
    link_graph_text = _replace_once(
        link_graph_text,
        LINK_GRAPH_STATS_BANNER_OLD,
        LINK_GRAPH_STATS_BANNER_NEW,
        path=link_graph_path,
    )
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
    link_graph_text = _replace_once(
        link_graph_text,
        LINK_GRAPH_HOVER_HITTEST_OLD,
        LINK_GRAPH_HOVER_HITTEST_NEW,
        path=link_graph_path,
    )
    link_graph_text = _replace_once(
        link_graph_text,
        LINK_GRAPH_CLICK_HITTEST_OLD,
        LINK_GRAPH_CLICK_HITTEST_NEW,
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
        FEATURE_DETAIL_NEURONPEDIA_OLD,
        FEATURE_DETAIL_NEURONPEDIA_NEW,
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
    feature_detail_text = _replace_once(
        feature_detail_text,
        FEATURE_STATS_HELPER_OLD,
        FEATURE_STATS_HELPER_NEW,
        path=feature_detail_path,
    )
    # Must run after FEATURE_EXAMPLES_LOAD: this anchor is the snippet that patch emits.
    feature_detail_text = _replace_once(
        feature_detail_text,
        FEATURE_STATS_RENDER_OLD,
        FEATURE_STATS_RENDER_NEW,
        path=feature_detail_path,
    )
    feature_detail_path.write_text(feature_detail_text)

    feature_examples_text = feature_examples_path.read_text()
    feature_examples_text = _replace_once(
        feature_examples_text,
        FEATURE_URL_FUNCTION_OLD,
        FEATURE_URL_FUNCTION_NEW,
        path=feature_examples_path,
    )
    feature_examples_path.write_text(feature_examples_text)

    node_connections_text = node_connections_path.read_text()
    node_connections_text = _replace_once(
        node_connections_text,
        NODE_CONNECTIONS_HEADER_ICON_OLD,
        NODE_CONNECTIONS_HEADER_ICON_NEW,
        path=node_connections_path,
    )
    node_connections_path.write_text(node_connections_text)

    # index.html has two identical option-builder sites (initial render + refetch-on-click);
    # patch both so the dropdown shows each graph's as-built base/adapter/error counts.
    index_text = index_path.read_text()
    index_text = _replace_all(
        index_text,
        DROPDOWN_OPTION_COUNTS_OLD,
        DROPDOWN_OPTION_COUNTS_NEW,
        path=index_path,
        min_count=2,
    )
    index_path.write_text(index_text)


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
