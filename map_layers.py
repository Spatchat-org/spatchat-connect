"""
Map layers for the SpatChat: Connectivity room

- Nodes preview (points)
- LCP preview from **GeoJSONL** (full dataset)
- Raster overlays (cost surface, Connectivity (LCP Density)) with percentile stretch
- Graph model node overlay for centrality metrics

This is intentionally RAM-light: we DO NOT load full factorial outputs into
Folium. For LCPs, we now load the entire `.geojsonl` file so the UI reflects
every path. Full artifacts remain on disk.
"""
from __future__ import annotations

import os
import json
import math
import hashlib
import re
from typing import Iterable, List, Tuple, Optional, Dict, Any

import numpy as np
import folium
import rasterio
import matplotlib.pyplot as plt
from rasterio.transform import Affine, array_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import Transformer
from branca.element import MacroElement, Template
from lcp_flow import detect_nodes_lonlat
from crk_flow import KERNEL_PREVIEW_LABEL, kernel_preview_label

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas optional at runtime
    pd = None  # type: ignore

from map_utils import fit_map_to_bounds

# ---------------------------------------------------------------------------
# Base map
# ---------------------------------------------------------------------------

def _base_map(center_lat: float, center_lon: float, control_scale: bool = True, zoom: int = 9) -> folium.Map:
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, control_scale=control_scale)
    folium.TileLayer("OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB positron", attr="CartoDB").add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="OpenTopoMap", name="Topographic"
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite"
    ).add_to(m)
    return m

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_color(key: str) -> str:
    """Deterministic hex color from a string key (e.g., source id)."""
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return f"#{h[:6]}"


def _raster_bounds_wgs84(ds: rasterio.io.DatasetReader) -> List[List[float]]:
    b = ds.bounds
    if ds.crs is None:
        raise ValueError("Raster CRS is missing; cannot compute geographic bounds.")
    src_crs = ds.crs.to_string()
    if src_crs != "EPSG:4326":
        to_wgs = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
        pts = [(b.left, b.bottom), (b.left, b.top), (b.right, b.bottom), (b.right, b.top)]
        ll = [to_wgs.transform(x, y) for x, y in pts]  # (lon, lat)
        west = min(p[0] for p in ll); east = max(p[0] for p in ll)
        south = min(p[1] for p in ll); north = max(p[1] for p in ll)
        return [[south, west], [north, east]]
    return [[b.bottom, b.left], [b.top, b.right]]


def _read_raster_preview_wgs84(
    src: rasterio.io.DatasetReader,
    *,
    max_dim: int = 1400,
    resampling: Resampling = Resampling.bilinear,
) -> Tuple[np.ndarray, List[List[float]]]:
    arr = src.read(1, masked=True).astype(np.float32, copy=False).filled(np.nan)
    if src.crs is None:
        raise ValueError("Raster CRS is missing; assign the correct CRS before mapping.")
    if src.crs.to_epsg() == 4326:
        west, south, east, north = array_bounds(src.height, src.width, src.transform)
        return arr, [[south, west], [north, east]]

    dst_transform, dst_width, dst_height = calculate_default_transform(
        src.crs,
        "EPSG:4326",
        src.width,
        src.height,
        *src.bounds,
    )
    longest = max(int(dst_width), int(dst_height), 1)
    if longest > max_dim:
        scale = float(longest) / float(max_dim)
        dst_width = max(1, int(round(dst_width / scale)))
        dst_height = max(1, int(round(dst_height / scale)))
        dst_transform, _, _ = calculate_default_transform(
            src.crs,
            "EPSG:4326",
            src.width,
            src.height,
            *src.bounds,
            dst_width=dst_width,
            dst_height=dst_height,
        )

    dst_arr = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
    valid_src = np.isfinite(arr).astype(np.uint8)
    valid_dst = np.zeros((dst_height, dst_width), dtype=np.uint8)

    reproject(
        source=arr,
        destination=dst_arr,
        src_transform=src.transform,
        src_crs=src.crs,
        src_nodata=np.nan,
        dst_transform=dst_transform,
        dst_crs="EPSG:4326",
        dst_nodata=np.nan,
        resampling=resampling,
    )
    reproject(
        source=valid_src,
        destination=valid_dst,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dst_transform,
        dst_crs="EPSG:4326",
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    dst_arr = np.where(valid_dst > 0, dst_arr, np.nan)
    west, south, east, north = array_bounds(dst_height, dst_width, dst_transform)
    return dst_arr, [[south, west], [north, east]]


def _rgba_from_array(
    arr: np.ndarray,
    cmap_name: str = "magma",
    pmin: float = 2.0,
    pmax: float = 98.0,
    transform: Optional[str] = None,
    transparent_below: Optional[float] = None,
) -> Tuple[np.ndarray, float, float, float, float]:
    """Percentile stretch over valid cells only; NoData remains transparent."""
    arr = np.asarray(arr, dtype=np.float32)
    valid = np.isfinite(arr)
    if transparent_below is not None:
        valid &= arr > float(transparent_below)
    if not np.any(valid):
        return (np.zeros((*arr.shape, 4), dtype=np.uint8),
                float('nan'), float('nan'), float('nan'), float('nan'))
    display = np.where(np.isfinite(arr), arr, 0.0)
    if transform == "log1p":
        display = np.log1p(np.maximum(display, 0.0))
    values = display[valid]
    actual_min, actual_max = float(values.min()), float(values.max())
    lo, hi = float(np.percentile(values, pmin)), float(np.percentile(values, pmax))
    if hi <= lo:
        lo, hi = actual_min, actual_max
    if hi <= lo:
        # Constant valid surface: use a consistent middle colormap value.
        normalized = np.full(arr.shape, 0.5, dtype=np.float32)
    else:
        normalized = np.clip((display - lo) / (hi - lo), 0.0, 1.0)
    rgba = (plt.get_cmap(cmap_name)(normalized) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(valid, rgba[..., 3], 0).astype(np.uint8)
    return rgba, lo, hi, actual_min, actual_max


def _rgba_to_hex(rgba: Tuple[float, float, float, float]) -> str:
    r, g, b, _ = rgba
    return f"#{int(round(r * 255)):02x}{int(round(g * 255)):02x}{int(round(b * 255)):02x}"


def _cell_bounds_wgs84(transform: Affine, row: int, col: int, transformer: Optional[Transformer]) -> List[List[float]]:
    x0, y0 = transform * (col, row)
    x1, y1 = transform * (col + 1, row + 1)
    west = min(x0, x1)
    east = max(x0, x1)
    south = min(y0, y1)
    north = max(y0, y1)
    if transformer is not None:
        west_south = transformer.transform(west, south)
        west_north = transformer.transform(west, north)
        east_south = transformer.transform(east, south)
        east_north = transformer.transform(east, north)
        xs = [west_south[0], west_north[0], east_south[0], east_north[0]]
        ys = [west_south[1], west_north[1], east_south[1], east_north[1]]
        west, east = min(xs), max(xs)
        south, north = min(ys), max(ys)
    return [[south, west], [north, east]]


def _add_sparse_positive_raster(
    m: folium.Map,
    src: rasterio.io.DatasetReader,
    *,
    name: str,
    cmap: str,
    pmin: float,
    pmax: float,
    show: bool,
    transform_name: Optional[str] = None,
    max_cells: int = 20000,
) -> Optional[folium.map.FeatureGroup]:
    arr = src.read(1, masked=True).astype(np.float32, copy=False).filled(np.nan)
    mask = np.isfinite(arr) & (arr > 0)
    count = int(mask.sum())
    if count <= 0 or count > max_cells:
        return None

    display = np.where(mask, arr, np.nan)
    if transform_name == "log1p":
        display = np.where(mask, np.log1p(np.maximum(display, 0.0)), np.nan)
    valid = display[np.isfinite(display)]
    if valid.size == 0:
        return None
    actual_min = float(np.nanmin(valid))
    actual_max = float(np.nanmax(valid))

    vmin = float(np.nanpercentile(valid, pmin))
    vmax = float(np.nanpercentile(valid, pmax))
    if vmax <= vmin:
        vmin = float(np.nanmin(valid))
        vmax = float(np.nanmax(valid))
        if vmax <= vmin:
            vmax = vmin + 1e-12

    cmap_obj = plt.get_cmap(cmap)
    pane = _ensure_map_pane(m, name, z_index=360)
    layer = folium.FeatureGroup(name=name, show=show)
    transformer = None
    if src.crs is not None and str(src.crs).upper() != "EPSG:4326":
        transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)

    rect_features: List[dict] = []
    rows, cols = np.where(mask)
    for row, col in zip(rows.tolist(), cols.tolist()):
        value = float(display[row, col])
        norm = float(np.clip((value - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0))
        color = _rgba_to_hex(cmap_obj(norm))
        bounds = _cell_bounds_wgs84(src.transform, int(row), int(col), transformer)
        rect_features.append(
            {
                "bounds": bounds,
                "fillColor": color,
                "fillOpacity": 1.0,
            }
        )

    layer.add_to(m)
    _add_rectangle_feature_group_script(
        m,
        layer_var_name=layer.get_name(),
        pane=pane,
        features=rect_features,
    )
    _add_raster_legend(
        m,
        name=name,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        pmin=pmin,
        pmax=pmax,
        actual_min=actual_min,
        actual_max=actual_max,
    )
    return layer


def _format_value(value: float) -> str:
    if np.isnan(value):
        return "n/a"
    value = float(value)
    abs_val = abs(value)
    if abs_val >= 1000:
        return f"{value:,.0f}"
    if abs_val >= 100:
        return f"{value:,.1f}"
    if abs_val >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _pane_name(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", (name or "layer").lower()).strip("-")
    return f"pane-{cleaned or 'layer'}"


def _ensure_map_pane(m: folium.Map, name: str, z_index: int = 400) -> str:
    pane = _pane_name(name)
    registry = getattr(m, "_spatchat_panes", None)
    if not isinstance(registry, set):
        registry = set()
        setattr(m, "_spatchat_panes", registry)
    if pane not in registry:
        folium.map.CustomPane(pane, z_index=z_index).add_to(m)
        registry.add(pane)
    return pane


def _precreate_overlay_panes(
    m: folium.Map,
    *,
    base_rasters: Optional[List[Tuple[str, str]]] = None,
    cost_label: Optional[str] = None,
    extra_rasters: Optional[List[Tuple[str, str]]] = None,
    lcp_count_present: bool = False,
    has_nodes: bool = False,
    has_circuit: bool = False,
    has_lcp: bool = False,
    graph_metric_name: Optional[str] = None,
    graph_metric_names: Optional[List[str]] = None,
    has_graph_edges: bool = False,
) -> None:
    for _, label in base_rasters or ():
        _ensure_map_pane(m, label, z_index=330)
    if cost_label:
        _ensure_map_pane(m, cost_label, z_index=360)
    for _, label in extra_rasters or ():
        _ensure_map_pane(m, label, z_index=370)
    if lcp_count_present:
        _ensure_map_pane(m, "Connectivity (LCP Density)", z_index=380)
    if has_graph_edges:
        _ensure_map_pane(m, "Graph model edges", z_index=430)
    if has_circuit:
        _ensure_map_pane(m, "Circuit effective resistance", z_index=435)
    if has_lcp:
        _ensure_map_pane(m, "LCP sample", z_index=440)
    if has_nodes:
        _ensure_map_pane(m, "Nodes", z_index=460)
    metric_names = list(graph_metric_names or ())
    if graph_metric_name and graph_metric_name not in metric_names:
        metric_names.append(graph_metric_name)
    for idx, metric_name in enumerate(metric_names):
        _ensure_map_pane(m, f"Graph model ({metric_name})", z_index=470 + idx)


def _add_raster_legend(
    m: folium.Map,
    *,
    name: str,
    cmap: str,
    vmin: float,
    vmax: float,
    pmin: float,
    pmax: float,
    actual_min: Optional[float] = None,
    actual_max: Optional[float] = None,
) -> None:
    root = m.get_root()
    offset = getattr(root, "_spatchat_legend_offset", 0)
    bottom = 20 + offset
    setattr(root, "_spatchat_legend_offset", offset + 110)

    steps = 8
    cmap_obj = plt.get_cmap(cmap)
    colors = [_rgba_to_hex(cmap_obj(i / (steps - 1))) for i in range(steps)]
    stops = [f"{color} {int(round(100 * i / (steps - 1)))}%" for i, color in enumerate(colors)]
    gradient = ", ".join(stops)
    stretch_text = (
        "Constant value"
        if np.isfinite(vmin) and np.isfinite(vmax) and abs(vmax - vmin) <= 1e-12
        else f"Display stretch: p{pmin:.1f}-{pmax:.1f}"
    )
    actual_range_html = ""
    if (
        actual_min is not None and actual_max is not None
        and np.isfinite(actual_min) and np.isfinite(actual_max)
    ):
        actual_range_html = (
            f'<div style="font-size: 11px; margin-top: 4px;">'
            f'Data range: {_format_value(actual_min)} to {_format_value(actual_max)}'
            f'</div>'
        )

    legend_html = f"""
    {{% macro html(this, kwargs) %}}
    <div class="spatchat-map-legend" style="
        position: absolute;
        bottom: {bottom}px;
        left: 20px;
        padding: 10px 12px;
        background: rgba(30, 30, 30, 0.85);
        color: #f5f5f5;
        font-size: 13px;
        line-height: 1.3;
        border-radius: 6px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.35);
        min-width: 190px;
        z-index: 9999;
        pointer-events: none;
    ">
      <div style="font-weight: 600; margin-bottom: 6px;">{name}</div>
      <div style="
            height: 16px;
            border-radius: 4px;
            background: linear-gradient(to right, {gradient});
            margin-bottom: 6px;
        "></div>
      <div style="display: flex; justify-content: space-between; font-size: 12px;">
        <span>{_format_value(vmin)}</span>
        <span>{_format_value(vmax)}</span>
      </div>
      <div style="font-size: 11px; margin-top: 4px;">
        Stretch: p{pmin:.1f}–p{pmax:.1f}
      </div>
    </div>
    {{% endmacro %}}
    """
    legend_html = legend_html.replace(f"Stretch: p{pmin:.1f}â€“p{pmax:.1f}", stretch_text)

    legend_html = legend_html.replace("Stretch: p", "Display stretch: p", 1)
    if actual_range_html:
        legend_html = legend_html.replace(
            "\n    </div>\n    {% endmacro %}\n",
            f"\n      {actual_range_html}\n    </div>\n    {{% endmacro %}}\n",
        )

    legend = MacroElement()
    legend._template = Template(legend_html)
    root.add_child(legend)


def _add_graph_summary_legend(
    m: folium.Map,
    *,
    metric_ranges: Optional[Dict[str, Dict[str, float]]] = None,
    probability_decay: Optional[Dict[str, Any]] = None,
) -> None:
    metric_ranges = metric_ranges or {}
    probability_decay = probability_decay or {}
    rows = []
    metric_labels = {
        "betweenness": "Betweenness",
        "closeness": "Closeness",
        "pagerank": "PageRank",
    }
    for metric in ("betweenness", "closeness", "pagerank"):
        bounds = metric_ranges.get(metric)
        if not bounds:
            continue
        rows.append(
            f"<div style=\"display:flex;justify-content:space-between;gap:12px;\">"
            f"<span>{metric_labels.get(metric, metric.title())}</span>"
            f"<span>{_format_value(bounds.get('min', float('nan')))} to {_format_value(bounds.get('max', float('nan')))}</span>"
            f"</div>"
        )
    if not rows and not probability_decay:
        return

    root = m.get_root()
    offset = getattr(root, "_spatchat_legend_offset", 0)
    bottom = 20 + offset
    setattr(root, "_spatchat_legend_offset", offset + 150)

    decay_lines = []
    if probability_decay:
        if probability_decay.get("mode") == "exponential_decay":
            decay_lines.append(
                f"p = exp(-cost / {_format_value(probability_decay.get('decay_scale_cost', float('nan')))})"
            )
            decay_lines.append(
                f"Kept probability ≥ {_format_value(probability_decay.get('probability_threshold', float('nan')))}"
            )
        kept = probability_decay.get("edge_count_kept")
        total = probability_decay.get("edge_count_total")
        if kept is not None and total is not None:
            decay_lines.append(f"Links kept: {kept} / {total}")

    rows_html = "".join(rows) if rows else "<div>No graph metrics available.</div>"
    decay_html = "".join(f"<div>{line}</div>" for line in decay_lines)
    legend_html = f"""
    {{% macro html(this, kwargs) %}}
    <div class="spatchat-map-legend" style="
        position: absolute;
        bottom: {bottom}px;
        left: 20px;
        padding: 10px 12px;
        background: rgba(30, 30, 30, 0.85);
        color: #f5f5f5;
        font-size: 13px;
        line-height: 1.35;
        border-radius: 6px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.35);
        min-width: 260px;
        z-index: 9999;
        pointer-events: none;
    ">
      <div style="font-weight: 600; margin-bottom: 6px;">Graph Metrics Summary</div>
      <div style="display:flex;flex-direction:column;gap:3px;">{rows_html}</div>
      <div style="font-size: 11px; margin-top: 8px; color: #d9d9d9;">{decay_html}</div>
    </div>
    {{% endmacro %}}
    """
    legend = MacroElement()
    legend._template = Template(legend_html)
    root.add_child(legend)


def _add_graph_edge_legend(
    m: folium.Map,
    *,
    name: str = "Graph model edges",
    vmin: Optional[float],
    vmax: Optional[float],
) -> None:
    if vmin is None or vmax is None or not np.isfinite(vmin) or not np.isfinite(vmax):
        return

    root = m.get_root()
    offset = getattr(root, "_spatchat_legend_offset", 0)
    bottom = 20 + offset
    setattr(root, "_spatchat_legend_offset", offset + 110)

    steps = 8
    cmap_obj = plt.get_cmap("cividis")
    colors = [_rgba_to_hex(cmap_obj(i / (steps - 1))) for i in range(steps)]
    stops = [f"{color} {int(round(100 * i / (steps - 1)))}%" for i, color in enumerate(colors)]
    gradient = ", ".join(stops)

    legend_html = f"""
    {{% macro html(this, kwargs) %}}
    <div class="spatchat-map-legend" style="
        position: absolute;
        bottom: {bottom}px;
        left: 20px;
        padding: 10px 12px;
        background: rgba(30, 30, 30, 0.85);
        color: #f5f5f5;
        font-size: 13px;
        line-height: 1.3;
        border-radius: 6px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.35);
        min-width: 220px;
        z-index: 9999;
        pointer-events: none;
    ">
      <div style="font-weight: 600; margin-bottom: 6px;">{name}</div>
      <div style="
            height: 16px;
            border-radius: 4px;
            background: linear-gradient(to right, {gradient});
            margin-bottom: 6px;
        "></div>
      <div style="display: flex; justify-content: space-between; font-size: 12px;">
        <span>{_format_value(vmin)}</span>
        <span>{_format_value(vmax)}</span>
      </div>
      <div style="font-size: 11px; margin-top: 4px;">Edge probability</div>
    </div>
    {{% endmacro %}}
    """
    legend = MacroElement()
    legend._template = Template(legend_html)
    root.add_child(legend)


def _add_floating_ui_patch(m: folium.Map) -> None:
    root = m.get_root()
    if getattr(root, "_spatchat_floating_ui_patch_added", False):
        return
    setattr(root, "_spatchat_floating_ui_patch_added", True)
    patch_html = """
    {% macro html(this, kwargs) %}
    <style>
      .spatchat-legend-dock {
        position: absolute;
        top: 14px;
        left: 14px;
        width: 280px;
        min-width: 220px;
        max-width: calc(100% - 28px);
        min-height: 120px;
        max-height: calc(100% - 28px);
        display: flex;
        flex-direction: column;
        background: rgba(30, 30, 30, 0.9);
        color: #f5f5f5;
        border-radius: 8px;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.35);
        z-index: 9999;
        overflow: hidden;
      }
      .spatchat-legend-dock-header,
      .spatchat-layer-control-handle {
        padding: 8px 10px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.02em;
        cursor: move;
        user-select: none;
        background: rgba(255, 255, 255, 0.08);
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      }
      .spatchat-layer-control-tools {
        display: flex;
        padding: 8px 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.04);
      }
      .spatchat-layer-control-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        appearance: none;
        border: 1px solid rgba(255, 255, 255, 0.18);
        background: rgba(24, 24, 24, 0.92) !important;
        color: #ffffff !important;
        border-radius: 6px;
        padding: 4px 8px;
        font-size: 11px;
        font-weight: 600;
        line-height: 1.2;
        cursor: pointer;
        opacity: 1 !important;
        visibility: visible !important;
        box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.04);
      }
      .spatchat-layer-control-btn:hover {
        background: rgba(48, 48, 48, 0.98) !important;
      }
      .spatchat-legend-dock-body {
        overflow: auto;
        padding: 8px;
        display: flex;
        flex-direction: column;
        gap: 8px;
        flex: 1 1 auto;
        min-height: 0;
      }
      .spatchat-legend-dock .spatchat-map-legend {
        position: static !important;
        left: auto !important;
        right: auto !important;
        top: auto !important;
        bottom: auto !important;
        margin: 0 !important;
        min-width: 0 !important;
        width: auto !important;
        box-shadow: none !important;
        pointer-events: none !important;
      }
      .leaflet-control-layers,
      .leaflet-control-layers.spatchat-draggable-control {
        background: rgba(30, 30, 30, 0.9) !important;
        color: #f5f5f5 !important;
        border-radius: 8px !important;
        border: none !important;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.35) !important;
      }
      .leaflet-control-layers-toggle {
        background-color: transparent !important;
        border-radius: 8px !important;
        filter: invert(1) brightness(1.25);
      }
      .leaflet-control-layers-expanded {
        padding: 0 !important;
        background: transparent !important;
      }
      .leaflet-control-layers-list,
      .leaflet-control-layers form,
      .leaflet-control-layers .leaflet-control-layers-base,
      .leaflet-control-layers .leaflet-control-layers-overlays,
      .leaflet-control-layers .leaflet-control-layers-separator {
        background: transparent !important;
        color: inherit !important;
        border-color: rgba(255, 255, 255, 0.08) !important;
      }
      .leaflet-control-layers label {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 7px 10px;
        margin: 0;
        color: #f5f5f5 !important;
        font-size: 12px;
        line-height: 1.3;
      }
      .leaflet-control-layers label:hover {
        background: rgba(255, 255, 255, 0.05);
      }
      .leaflet-control-layers input[type="checkbox"],
      .leaflet-control-layers input[type="radio"] {
        accent-color: #8ec5ff;
      }
      .leaflet-control-layers .leaflet-control-layers-separator {
        margin: 0;
      }
      .leaflet-control-layers::-webkit-scrollbar,
      .leaflet-control-layers .leaflet-control-layers-list::-webkit-scrollbar,
      .spatchat-legend-dock-body::-webkit-scrollbar {
        width: 10px;
        height: 10px;
      }
      .leaflet-control-layers::-webkit-scrollbar-track,
      .leaflet-control-layers .leaflet-control-layers-list::-webkit-scrollbar-track,
      .spatchat-legend-dock-body::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.06);
      }
      .leaflet-control-layers::-webkit-scrollbar-thumb,
      .leaflet-control-layers .leaflet-control-layers-list::-webkit-scrollbar-thumb,
      .spatchat-legend-dock-body::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.18);
        border-radius: 999px;
      }
      .leaflet-control-layers::-webkit-scrollbar-thumb:hover,
      .leaflet-control-layers .leaflet-control-layers-list::-webkit-scrollbar-thumb:hover,
      .spatchat-legend-dock-body::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.26);
      }
      .leaflet-control-layers.spatchat-draggable-control {
        z-index: 9998 !important;
        width: 280px;
        min-width: 220px;
        max-width: calc(100% - 28px);
        min-height: 120px;
        max-height: calc(100% - 28px);
        display: flex;
        flex-direction: column;
        overflow: hidden;
      }
      .leaflet-control-layers.spatchat-draggable-control .leaflet-control-layers-list {
        flex: 1 1 auto;
        min-height: 0;
        overflow: auto;
      }
      .spatchat-dock-resize {
        position: absolute;
        right: 0;
        bottom: 0;
        width: 18px;
        height: 18px;
        cursor: nwse-resize;
        z-index: 10000;
        user-select: none;
      }
      .spatchat-dock-resize::before {
        content: "";
        position: absolute;
        right: 4px;
        bottom: 4px;
        width: 10px;
        height: 10px;
        border-right: 2px solid rgba(255, 255, 255, 0.65);
        border-bottom: 2px solid rgba(255, 255, 255, 0.65);
        border-bottom-right-radius: 2px;
      }
      .spatchat-dock-resize:hover::before {
        border-right-color: #8ec5ff;
        border-bottom-color: #8ec5ff;
      }
    </style>
    <script>
    (function() {
      function getLeafletMap() {
        if (!window.L) return null;
        for (var key in window) {
          try {
            if (window[key] instanceof L.Map) return window[key];
          } catch (err) {}
        }
        return null;
      }

      function enableDrag(container, handle, mapContainer) {
        if (!container || !handle || !mapContainer || handle.dataset.spatchatDragBound === '1') return;
        handle.dataset.spatchatDragBound = '1';
        handle.addEventListener('mousedown', function(event) {
          if (event.button !== 0) return;
          event.preventDefault();
          event.stopPropagation();

          var mapRect = mapContainer.getBoundingClientRect();
          var rect = container.getBoundingClientRect();
          if (container.parentElement !== mapContainer) {
            mapContainer.appendChild(container);
          }
          container.style.position = 'absolute';
          container.style.left = (rect.left - mapRect.left) + 'px';
          container.style.top = (rect.top - mapRect.top) + 'px';
          container.style.right = 'auto';
          container.style.bottom = 'auto';

          var startX = event.clientX;
          var startY = event.clientY;
          var startLeft = rect.left - mapRect.left;
          var startTop = rect.top - mapRect.top;

          function onMove(moveEvent) {
            var nextLeft = startLeft + (moveEvent.clientX - startX);
            var nextTop = startTop + (moveEvent.clientY - startY);
            var maxLeft = Math.max(0, mapContainer.clientWidth - container.offsetWidth);
            var maxTop = Math.max(0, mapContainer.clientHeight - container.offsetHeight);
            container.style.left = Math.max(0, Math.min(nextLeft, maxLeft)) + 'px';
            container.style.top = Math.max(0, Math.min(nextTop, maxTop)) + 'px';
          }

          function onUp() {
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
          }

          document.addEventListener('mousemove', onMove);
          document.addEventListener('mouseup', onUp);
        });
      }

      function clampPosition(container, mapContainer) {
        if (!container || !mapContainer) return;
        var left = parseFloat(container.style.left || '0');
        var top = parseFloat(container.style.top || '0');
        if (!isFinite(left)) left = 0;
        if (!isFinite(top)) top = 0;
        var maxLeft = Math.max(0, mapContainer.clientWidth - container.offsetWidth);
        var maxTop = Math.max(0, mapContainer.clientHeight - container.offsetHeight);
        container.style.left = Math.max(0, Math.min(left, maxLeft)) + 'px';
        container.style.top = Math.max(0, Math.min(top, maxTop)) + 'px';
      }

      function enableResize(container, handle, mapContainer, options) {
        if (!container || !handle || !mapContainer || handle.dataset.spatchatResizeBound === '1') return;
        handle.dataset.spatchatResizeBound = '1';
        var minWidth = (options && options.minWidth) || 220;
        var minHeight = (options && options.minHeight) || 120;
        handle.addEventListener('mousedown', function(event) {
          if (event.button !== 0) return;
          event.preventDefault();
          event.stopPropagation();

          var mapRect = mapContainer.getBoundingClientRect();
          var rect = container.getBoundingClientRect();
          if (container.parentElement !== mapContainer) {
            mapContainer.appendChild(container);
          }
          container.style.position = 'absolute';
          container.style.left = (rect.left - mapRect.left) + 'px';
          container.style.top = (rect.top - mapRect.top) + 'px';
          container.style.right = 'auto';
          container.style.bottom = 'auto';

          var startX = event.clientX;
          var startY = event.clientY;
          var startWidth = rect.width;
          var startHeight = rect.height;
          var startLeft = rect.left - mapRect.left;
          var startTop = rect.top - mapRect.top;

          function onMove(moveEvent) {
            var maxWidth = Math.max(minWidth, mapContainer.clientWidth - startLeft);
            var maxHeight = Math.max(minHeight, mapContainer.clientHeight - startTop);
            var nextWidth = startWidth + (moveEvent.clientX - startX);
            var nextHeight = startHeight + (moveEvent.clientY - startY);
            nextWidth = Math.max(minWidth, Math.min(nextWidth, maxWidth));
            nextHeight = Math.max(minHeight, Math.min(nextHeight, maxHeight));
            container.style.width = Math.round(nextWidth) + 'px';
            container.style.height = Math.round(nextHeight) + 'px';
            container.style.maxWidth = 'none';
            container.style.maxHeight = 'none';
          }

          function onUp() {
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
            clampPosition(container, mapContainer);
          }

          document.addEventListener('mousemove', onMove);
          document.addEventListener('mouseup', onUp);
        });
      }

      function initLegendDock(map) {
        var mapContainer = map && map.getContainer ? map.getContainer() : null;
        if (!mapContainer) return false;
        var legends = Array.from(document.querySelectorAll('.spatchat-map-legend'));
        if (!legends.length) return false;
        var existingDock = mapContainer.querySelector('.spatchat-legend-dock');
        if (existingDock) return true;

        var dock = document.createElement('div');
        dock.className = 'spatchat-legend-dock';

        var header = document.createElement('div');
        header.className = 'spatchat-legend-dock-header';
        header.textContent = 'Map Legends';

        var body = document.createElement('div');
        body.className = 'spatchat-legend-dock-body';

        var resize = document.createElement('div');
        resize.className = 'spatchat-dock-resize';
        resize.title = 'Resize legend dock';

        legends.forEach(function(legend) {
          body.appendChild(legend);
        });

        dock.appendChild(header);
        dock.appendChild(body);
        dock.appendChild(resize);
        mapContainer.appendChild(dock);
        enableDrag(dock, header, mapContainer);
        enableResize(dock, resize, mapContainer, { minWidth: 220, minHeight: 120 });
        return true;
      }

      function initLayerControlDrag(map) {
        var mapContainer = map && map.getContainer ? map.getContainer() : null;
        if (!mapContainer) return false;
        var layerControl = document.querySelector('.leaflet-control-layers');
        if (!layerControl) return false;
        layerControl.classList.add('spatchat-draggable-control');
        if (!layerControl.style.width) {
          layerControl.style.width = '280px';
        }

        var handle = layerControl.querySelector('.spatchat-layer-control-handle');
        if (!handle) {
          handle = document.createElement('div');
          handle.className = 'spatchat-layer-control-handle';
          handle.textContent = 'Layers';
          layerControl.insertBefore(handle, layerControl.firstChild);
        }

        var overlaysList = layerControl.querySelector('.leaflet-control-layers-overlays');
        var overlayInputs = Array.from(layerControl.querySelectorAll('.leaflet-control-layers-overlays input[type="checkbox"]'));
        var tools = layerControl.querySelector('.spatchat-layer-control-tools');
        if (!overlayInputs.length) {
          if (tools) tools.remove();
        } else if (!tools) {
          tools = document.createElement('div');
          tools.className = 'spatchat-layer-control-tools';

          var toggleBtn = document.createElement('button');
          toggleBtn.type = 'button';
          toggleBtn.className = 'spatchat-layer-control-btn';
          toggleBtn.textContent = 'All layers on/off';

          function setAllOverlays(checked) {
            var inputs = Array.from(layerControl.querySelectorAll('.leaflet-control-layers-overlays input[type="checkbox"]'));
            inputs.forEach(function(input) {
              if (!!input.checked !== checked) {
                input.click();
              }
            });
          }

          toggleBtn.addEventListener('mousedown', function(event) {
            event.stopPropagation();
          });
          toggleBtn.addEventListener('click', function(event) {
            event.preventDefault();
            event.stopPropagation();
            var inputs = Array.from(layerControl.querySelectorAll('.leaflet-control-layers-overlays input[type="checkbox"]'));
            var anyUnchecked = inputs.some(function(input) { return !input.checked; });
            setAllOverlays(anyUnchecked);
          });

          tools.appendChild(toggleBtn);
          if (overlaysList) {
            overlaysList.insertAdjacentElement('beforebegin', tools);
          } else {
            handle.insertAdjacentElement('afterend', tools);
          }
        }

        var resize = layerControl.querySelector('.spatchat-dock-resize');
        if (!resize) {
          resize = document.createElement('div');
          resize.className = 'spatchat-dock-resize';
          resize.title = 'Resize layers dock';
          layerControl.appendChild(resize);
        }

        enableDrag(layerControl, handle, mapContainer);
        enableResize(layerControl, resize, mapContainer, { minWidth: 220, minHeight: 120 });
        return true;
      }

      function initFloatingUi() {
        var map = getLeafletMap();
        if (!map) return false;
        var legendsOk = initLegendDock(map);
        var layersOk = initLayerControlDrag(map);
        return legendsOk || layersOk;
      }

      initFloatingUi();
      window.addEventListener('load', initFloatingUi);
      var attempts = 0;
      var timer = setInterval(function() {
        if (initFloatingUi() || attempts >= 20) {
          clearInterval(timer);
        }
        attempts += 1;
      }, 500);
    })();
    </script>
    {% endmacro %}
    """
    patch = MacroElement()
    patch._template = Template(patch_html)
    root.add_child(patch)


def _add_layer_control_sort_patch(m: folium.Map) -> None:
    root = m.get_root()
    if getattr(root, "_spatchat_layer_sort_patch_added", False):
        return
    setattr(root, "_spatchat_layer_sort_patch_added", True)
    patch_html = r"""
    {% macro html(this, kwargs) %}
    <style>
      .spatchat-layer-order-handle {
        display: inline-block;
        margin-right: 6px;
        color: #666;
        cursor: grab;
        user-select: none;
        font-size: 11px;
        line-height: 1;
      }
      .spatchat-layer-order-handle:active {
        cursor: grabbing;
      }
      .leaflet-control-layers-overlays label.spatchat-layer-drop-target {
        outline: 1px dashed rgba(0, 123, 255, 0.75);
        outline-offset: 2px;
      }
    </style>
    <script>
    (function() {
      var draggedOverlayLabel = null;
      var suppressOverlaySync = false;

      function getLeafletMap() {
        if (!window.L) return null;
        for (var key in window) {
          try {
            if (window[key] instanceof L.Map) return window[key];
          } catch (err) {}
        }
        return null;
      }

      function getLayerControlInstance() {
        if (!window.L) return null;
        for (var key in window) {
          try {
            if (window[key] instanceof L.Control.Layers) return window[key];
          } catch (err) {}
        }
        return null;
      }

      function extractLayerName(label) {
        if (!label) return '';
        var cloned = label.cloneNode(true);
        Array.from(cloned.querySelectorAll('.spatchat-layer-order-handle')).forEach(function(node) { node.remove(); });
        return (cloned.textContent || '').replace(/\\s+/g, ' ').trim();
      }

      function paneNameForLabel(name) {
        var normalized = String(name || 'layer')
          .replace(/\s*\(n=\d+\)\s*$/i, '')
          .trim();
        var cleaned = normalized
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, '-')
          .replace(/^-+|-+$/g, '');
        return 'pane-' + (cleaned || 'layer');
      }

      function panePriorityForLabel(name) {
        var normalized = String(name || '')
          .replace(/\s*\(n=\d+\)\s*$/i, '')
          .trim()
          .toLowerCase();
        if (!normalized) return 1;
        if (normalized === 'nodes' || normalized.indexOf('graph model (') === 0) {
          return 4;
        }
        if (normalized === 'connectivity (lcp density)') {
          return 2;
        }
        if (
          normalized === 'graph model edges' ||
          normalized === 'circuit effective resistance' ||
          normalized === 'lcp sample'
        ) {
          return 3;
        }
        return 1;
      }

      function findControlLayerByName(layerControlInstance, name) {
        if (!layerControlInstance || !layerControlInstance._layers || !name) return null;
        for (var i = 0; i < layerControlInstance._layers.length; i++) {
          var item = layerControlInstance._layers[i];
          if (!item || !item.overlay) continue;
          var itemName = String(item.name || '').replace(/\\s+/g, ' ').trim();
          if (itemName === name) return item.layer;
        }
        return null;
      }

      function retargetLayerToPane(layer, paneName, map) {
        if (!layer || !paneName || !map) return;

        if (layer.eachLayer && typeof layer.eachLayer === 'function') {
          layer.eachLayer(function(child) {
            retargetLayerToPane(child, paneName, map);
          });
        }

        if (layer.options) {
          layer.options.pane = paneName;
        }

        if (window.L && layer instanceof L.Path) {
          try {
            layer.options = layer.options || {};
            layer.options.pane = paneName;
            layer.options.renderer = L.svg({ pane: paneName });
            var wasOnMap = map.hasLayer(layer);
            if (wasOnMap) {
              map.removeLayer(layer);
              layer.addTo(map);
            } else if (layer.redraw) {
              layer.redraw();
            }
          } catch (err) {}
          return;
        }

        if (window.L && layer instanceof L.Marker) {
          try {
            var markerOnMap = map.hasLayer(layer);
            if (markerOnMap) {
              map.removeLayer(layer);
              layer.addTo(map);
            }
          } catch (err) {}
          return;
        }

        if (window.L && layer instanceof L.ImageOverlay) {
          try {
            var imageOnMap = map.hasLayer(layer);
            if (imageOnMap) {
              map.removeLayer(layer);
              layer.addTo(map);
            }
          } catch (err) {}
        }
      }

      function applyOverlayOrder(layerControl) {
        if (!layerControl) return;
        var map = getLeafletMap();
        var layerControlInstance = getLayerControlInstance();
        var labels = Array.from(layerControl.querySelectorAll('.leaflet-control-layers-overlays label'));
        var allEntries = labels.map(function(label) {
          var input = label.querySelector('input[type="checkbox"]');
          return {
            label: label,
            input: input,
            name: extractLayerName(label),
          };
        });
        allEntries.forEach(function(entry) {
          var layer = findControlLayerByName(layerControlInstance, entry.name);
          if (!layer) return;
          retargetLayerToPane(layer, paneNameForLabel(entry.name), map);
        });
        var paneBase = 2000;
        var paneEntries = [];
        allEntries.forEach(function(entry, index) {
          var paneName = paneNameForLabel(entry.name);
          var paneEl = map && map.getPane ? map.getPane(paneName) : null;
          if (!paneEl) return;
          var zIndex = paneBase + (panePriorityForLabel(entry.name) * 100) + (allEntries.length - index);
          paneEl.style.zIndex = String(zIndex);
          paneEntries.push({ paneEl: paneEl, name: entry.name, zIndex: zIndex });
        });
        if (map && map.getPanes && map.getPanes().mapPane) {
          var mapPane = map.getPanes().mapPane;
          paneEntries
            .slice()
            .sort(function(a, b) { return a.zIndex - b.zIndex; })
            .forEach(function(item) {
              try {
                mapPane.appendChild(item.paneEl);
              } catch (err) {}
            });
        }

        var checkedEntries = allEntries
          .filter(function(entry) {
            return !!(entry.input && entry.input.checked);
          });
        if (!checkedEntries.length) return;

        suppressOverlaySync = true;
        checkedEntries.forEach(function(entry) {
          try {
            entry.input.click();
          } catch (err) {}
        });
        for (var i = checkedEntries.length - 1; i >= 0; i--) {
          try {
            checkedEntries[i].input.click();
          } catch (err) {}
        }
        setTimeout(function() {
          suppressOverlaySync = false;
        }, 0);
      }

      function initOverlayOrdering(layerControl) {
        var overlaysList = layerControl ? layerControl.querySelector('.leaflet-control-layers-overlays') : null;
        if (!overlaysList) return;

        if (overlaysList.dataset.spatchatInitialOrder !== '1') {
          var initialLabels = Array.from(overlaysList.querySelectorAll('label'));
          initialLabels.reverse().forEach(function(label) { overlaysList.appendChild(label); });
          overlaysList.dataset.spatchatInitialOrder = '1';
        }

        Array.from(overlaysList.querySelectorAll('label')).forEach(function(label) {
          if (!label.querySelector('.spatchat-layer-order-handle')) {
            var handleSpan = document.createElement('span');
            handleSpan.className = 'spatchat-layer-order-handle';
            handleSpan.textContent = '↕';
            handleSpan.draggable = true;
            handleSpan.addEventListener('dragstart', function(event) {
              draggedOverlayLabel = label;
              if (event.dataTransfer) {
                event.dataTransfer.effectAllowed = 'move';
                try { event.dataTransfer.setData('text/plain', 'spatchat-layer'); } catch (e) {}
              }
            });
            handleSpan.addEventListener('dragend', function() {
              draggedOverlayLabel = null;
              Array.from(overlaysList.querySelectorAll('label')).forEach(function(item) {
                item.classList.remove('spatchat-layer-drop-target');
              });
            });
            label.insertBefore(handleSpan, label.firstChild);
          }

          if (label.dataset.spatchatDropBound === '1') return;
          label.dataset.spatchatDropBound = '1';
          label.addEventListener('dragover', function(event) {
            if (!draggedOverlayLabel || draggedOverlayLabel === label) return;
            event.preventDefault();
            label.classList.add('spatchat-layer-drop-target');
          });
          label.addEventListener('dragleave', function() {
            label.classList.remove('spatchat-layer-drop-target');
          });
          label.addEventListener('drop', function(event) {
            if (!draggedOverlayLabel || draggedOverlayLabel === label) return;
            event.preventDefault();
            label.classList.remove('spatchat-layer-drop-target');
            var rect = label.getBoundingClientRect();
            var insertAfter = event.clientY > rect.top + rect.height / 2;
            if (insertAfter) {
              overlaysList.insertBefore(draggedOverlayLabel, label.nextSibling);
            } else {
              overlaysList.insertBefore(draggedOverlayLabel, label);
            }
            applyOverlayOrder(layerControl);
          });
        });

        if (layerControl.dataset.spatchatOrderBound !== '1') {
          layerControl.dataset.spatchatOrderBound = '1';
          layerControl.addEventListener('change', function() {
            if (suppressOverlaySync) return;
            setTimeout(function() { applyOverlayOrder(layerControl); }, 0);
          });
        }

        applyOverlayOrder(layerControl);
      }

      function tryInitLayerControl() {
        var layerControl = document.querySelector('.leaflet-control-layers');
        if (!layerControl) return false;
        initOverlayOrdering(layerControl);
        return true;
      }

      tryInitLayerControl();
      window.addEventListener('load', tryInitLayerControl);
      var attempts = 0;
      var timer = setInterval(function() {
        tryInitLayerControl();
        attempts += 1;
        if (attempts >= 20) {
          clearInterval(timer);
        }
      }, 500);
    })();
    </script>
    {% endmacro %}
    """
    patch = MacroElement()
    patch._template = Template(patch_html)
    root.add_child(patch)


def _add_polyline_feature_group_script(
    m: folium.Map,
    *,
    layer_var_name: str,
    pane: str,
    features: List[dict],
) -> None:
    if not features:
        return
    root = m.get_root()
    patch_html = """
    {% macro html(this, kwargs) %}
    <script>
    (function() {
      var features = {{ this.features | tojson }};
      var paneName = {{ this.pane | tojson }};
      var layerVarName = {{ this.layer_var_name | tojson }};

      function getLeafletMap() {
        if (!window.L) return null;
        for (var key in window) {
          try {
            if (window[key] instanceof L.Map) return window[key];
          } catch (err) {}
        }
        return null;
      }

      function bindPolylines() {
        if (!window.L) return false;
        var map = getLeafletMap();
        var layer = window[layerVarName];
        if (!map || !layer) return false;
        if (layer._spatchatPolylineBound) return true;
        try {
          var renderer = L.svg({ pane: paneName });
          features.forEach(function(feature) {
            try {
              var poly = L.polyline(feature.locations, {
                color: feature.color,
                weight: feature.weight,
                opacity: feature.opacity,
                pane: paneName,
                renderer: renderer,
              });
              if (feature.tooltip) {
                poly.bindTooltip(feature.tooltip, { sticky: true });
              }
              poly.addTo(layer);
            } catch (err) {}
          });
          layer._spatchatPolylineBound = true;
          return true;
        } catch (err) {
          return false;
        }
      }

      bindPolylines();
      window.addEventListener('load', bindPolylines);
      var attempts = 0;
      var timer = setInterval(function() {
        if (bindPolylines() || attempts >= 20) {
          clearInterval(timer);
        }
        attempts += 1;
      }, 500);
    })();
    </script>
    {% endmacro %}
    """
    patch = MacroElement()
    patch._template = Template(patch_html)
    patch.features = features
    patch.pane = pane
    patch.layer_var_name = layer_var_name
    root.add_child(patch)


def _add_circle_marker_feature_group_script(
    m: folium.Map,
    *,
    layer_var_name: str,
    pane: str,
    features: List[dict],
) -> None:
    if not features:
        return
    root = m.get_root()
    patch_html = """
    {% macro html(this, kwargs) %}
    <script>
    (function() {
      var features = {{ this.features | tojson }};
      var paneName = {{ this.pane | tojson }};
      var layerVarName = {{ this.layer_var_name | tojson }};

      function getLeafletMap() {
        if (!window.L) return null;
        for (var key in window) {
          try {
            if (window[key] instanceof L.Map) return window[key];
          } catch (err) {}
        }
        return null;
      }

      function bindCircleMarkers() {
        if (!window.L) return false;
        var map = getLeafletMap();
        var layer = window[layerVarName];
        if (!map || !layer) return false;
        if (layer._spatchatCircleMarkersBound) return true;
        try {
          var renderer = L.svg({ pane: paneName });
          features.forEach(function(feature) {
            try {
              var circle = L.circleMarker(feature.location, {
                radius: feature.radius,
                color: feature.color,
                weight: feature.weight,
                fill: true,
                fillColor: feature.fillColor,
                fillOpacity: feature.fillOpacity,
                pane: paneName,
                renderer: renderer,
                bubblingMouseEvents: false,
              });
              if (feature.tooltip) {
                circle.bindTooltip(feature.tooltip, { sticky: true });
              }
              circle.addTo(layer);
            } catch (err) {}
          });
          layer._spatchatCircleMarkersBound = true;
          return true;
        } catch (err) {
          return false;
        }
      }

      bindCircleMarkers();
      window.addEventListener('load', bindCircleMarkers);
      var attempts = 0;
      var timer = setInterval(function() {
        if (bindCircleMarkers() || attempts >= 20) {
          clearInterval(timer);
        }
        attempts += 1;
      }, 500);
    })();
    </script>
    {% endmacro %}
    """
    patch = MacroElement()
    patch._template = Template(patch_html)
    patch.features = features
    patch.pane = pane
    patch.layer_var_name = layer_var_name
    root.add_child(patch)


def _add_rectangle_feature_group_script(
    m: folium.Map,
    *,
    layer_var_name: str,
    pane: str,
    features: List[dict],
) -> None:
    if not features:
        return
    root = m.get_root()
    patch_html = """
    {% macro html(this, kwargs) %}
    <script>
    (function() {
      var features = {{ this.features | tojson }};
      var paneName = {{ this.pane | tojson }};
      var layerVarName = {{ this.layer_var_name | tojson }};

      function getLeafletMap() {
        if (!window.L) return null;
        for (var key in window) {
          try {
            if (window[key] instanceof L.Map) return window[key];
          } catch (err) {}
        }
        return null;
      }

      function bindRectangles() {
        if (!window.L) return false;
        var map = getLeafletMap();
        var layer = window[layerVarName];
        if (!map || !layer) return false;
        if (layer._spatchatRectanglesBound) return true;
        try {
          var renderer = L.svg({ pane: paneName });
          features.forEach(function(feature) {
            try {
              var rect = L.rectangle(feature.bounds, {
                stroke: false,
                fill: true,
                fillColor: feature.fillColor,
                fillOpacity: feature.fillOpacity,
                pane: paneName,
                renderer: renderer,
                interactive: false,
              });
              rect.addTo(layer);
            } catch (err) {}
          });
          layer._spatchatRectanglesBound = true;
          return true;
        } catch (err) {
          return false;
        }
      }

      bindRectangles();
      window.addEventListener('load', bindRectangles);
      var attempts = 0;
      var timer = setInterval(function() {
        if (bindRectangles() || attempts >= 20) {
          clearInterval(timer);
        }
        attempts += 1;
      }, 500);
    })();
    </script>
    {% endmacro %}
    """
    patch = MacroElement()
    patch._template = Template(patch_html)
    patch.features = features
    patch.pane = pane
    patch.layer_var_name = layer_var_name
    root.add_child(patch)


def _add_geotiff_overlay(
    m: folium.Map,
    tif_path: str,
    *,
    name: str,
    opacity: float = 1.0,
    cmap: str = "magma",
    pmin: float = 2.0,
    pmax: float = 98.0,
    show: bool = True,
    transform: Optional[str] = None,
    transparent_below: Optional[float] = None,
    preview_max_dim: int = 1400,
    preview_resampling: str = "bilinear",
    sparse_positive_only: bool = False,
) -> Optional[folium.map.FeatureGroup]:
    if not (tif_path and os.path.exists(tif_path)):
        return None
    try:
        with rasterio.open(tif_path) as src:
            if sparse_positive_only:
                sparse_layer = _add_sparse_positive_raster(
                    m,
                    src,
                    name=name,
                    cmap=cmap,
                    pmin=pmin,
                    pmax=pmax,
                    show=show,
                    transform_name=transform,
                )
                if sparse_layer is not None:
                    return sparse_layer
            resampling_enum = Resampling.nearest if str(preview_resampling).lower() == "nearest" else Resampling.bilinear
            arr, bounds_ll = _read_raster_preview_wgs84(
                src,
                max_dim=preview_max_dim,
                resampling=resampling_enum,
            )
            rgba, vmin, vmax, actual_min, actual_max = _rgba_from_array(
                arr,
                cmap_name=cmap,
                pmin=pmin,
                pmax=pmax,
                transform=transform,
                transparent_below=transparent_below,
            )
            pane = _ensure_map_pane(m, name, z_index=360)
            layer = folium.FeatureGroup(name=name, show=show)
            layer.add_child(
                folium.raster_layers.ImageOverlay(
                    image=np.dstack([rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]]),
                    bounds=bounds_ll,
                    opacity=opacity,
                    interactive=False,
                    mercator_project=True,
                    origin="upper",
                    pane=pane,
                )
            )
            layer.add_to(m)
            _add_raster_legend(
                m,
                name=name,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                pmin=pmin,
                pmax=pmax,
                actual_min=actual_min,
                actual_max=actual_max,
            )
            return layer
    except Exception as exc:
        print(f"[map] Raster overlay failed for {name!r}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Nodes & LCP layers
# ---------------------------------------------------------------------------

def _build_node_marker_features(
    nodes_df,
    *,
    id_col: str = "node_id",
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    raster_path: Optional[str] = None,
) -> List[dict]:
    features: List[dict] = []
    snapped_points = None
    if raster_path and os.path.exists(raster_path):
        try:
            with rasterio.open(raster_path) as ds:
                if ds.crs is not None:
                    to_raster = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
                    to_wgs = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True)
                    snapped_points = {}
                    for _, row in nodes_df.iterrows():
                        nid = str(row[id_col])
                        lon = float(row[lon_col]); lat = float(row[lat_col])
                        x, y = to_raster.transform(lon, lat)
                        rr, cc = ds.index(x, y)
                        cx, cy = ds.transform * (cc + 0.5, rr + 0.5)
                        snapped_lon, snapped_lat = to_wgs.transform(cx, cy)
                        snapped_points[nid] = [float(snapped_lat), float(snapped_lon)]
        except Exception:
            snapped_points = None

    for _, row in nodes_df.iterrows():
        nid = str(row[id_col])
        lat = float(row[lat_col]); lon = float(row[lon_col])
        if isinstance(snapped_points, dict) and nid in snapped_points:
            lat, lon = snapped_points[nid]
        features.append(
            {
                "location": [lat, lon],
                "radius": 7,
                "color": "#ffffff",
                "weight": 2,
                "fillColor": "#3d8cff",
                "fillOpacity": 1.0,
                "tooltip": nid,
            }
        )
    return features

def make_nodes_layer(nodes_df, *, id_col: str = "node_id", lon_col: str = "longitude", lat_col: str = "latitude",
                     name: str = "Nodes", show: bool = True, pane: Optional[str] = None,
                     raster_path: Optional[str] = None) -> folium.map.FeatureGroup:
    _ = (id_col, lon_col, lat_col, pane, raster_path)
    return folium.FeatureGroup(name=name, show=show)


def _snap_nodes_to_raster_centers(
    nodes_df,
    *,
    raster_path: Optional[str],
    id_col: str = "node_id",
    lon_col: str = "longitude",
    lat_col: str = "latitude",
):
    if nodes_df is None or nodes_df.empty or not raster_path or not os.path.exists(raster_path):
        return nodes_df
    if not {id_col, lon_col, lat_col}.issubset(nodes_df.columns):
        return nodes_df
    try:
        with rasterio.open(raster_path) as ds:
            transform = ds.transform
            raster_crs = ds.crs
            if raster_crs is None:
                return nodes_df
            to_raster = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
            to_wgs = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)
            snapped = nodes_df.copy()
            out_lon = []
            out_lat = []
            for row in snapped.itertuples(index=False):
                lon = float(getattr(row, lon_col))
                lat = float(getattr(row, lat_col))
                x, y = to_raster.transform(lon, lat)
                rr, cc = ds.index(x, y)
                cx, cy = transform * (cc + 0.5, rr + 0.5)
                slon, slat = to_wgs.transform(cx, cy)
                out_lon.append(float(slon))
                out_lat.append(float(slat))
            snapped[lon_col] = out_lon
            snapped[lat_col] = out_lat
            return snapped
    except Exception:
        return nodes_df


def make_graph_metric_layer(
    nodes_df,
    metrics_df,
    *,
    metric_col: str,
    id_col: str = "node_id",
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    name: Optional[str] = None,
    show: bool = True,
    pane: Optional[str] = None,
) -> Optional[Tuple[folium.map.FeatureGroup, float, float]]:
    """Use original node WGS84 locations and centered, fixed-size CircleMarkers."""
    if nodes_df is None or metrics_df is None or metric_col not in metrics_df.columns:
        return None
    metrics_id_col = "node_id" if "node_id" in metrics_df.columns else "node"
    if metrics_id_col not in metrics_df.columns:
        return None
    if not {id_col, lon_col, lat_col}.issubset(nodes_df.columns):
        return None

    # Do not merge coordinates from metrics: source-of-truth is uploaded nodes.
    locations = nodes_df[[id_col, lon_col, lat_col]].copy()
    locations["_join_id"] = locations[id_col].astype(str).str.strip()
    locations = locations.drop_duplicates("_join_id", keep="first")
    metrics = metrics_df[[metrics_id_col, metric_col]].copy()
    metrics["_join_id"] = metrics[metrics_id_col].astype(str).str.strip()
    metrics = metrics.drop_duplicates("_join_id", keep="first")
    joined = locations.merge(metrics[["_join_id", metric_col]], on="_join_id", how="left", validate="one_to_one")
    vals = pd.to_numeric(joined[metric_col], errors="coerce")
    valid = vals[np.isfinite(vals)]
    if valid.empty:
        return None
    vmin, vmax = float(valid.min()), float(valid.max())
    span = vmax - vmin
    cmap = plt.get_cmap("plasma")
    layer = folium.FeatureGroup(name=name or f"Graph model ({metric_col})", show=show)
    for _, row in joined.iterrows():
        try:
            val = float(row[metric_col]); lat = float(row[lat_col]); lon = float(row[lon_col])
        except (TypeError, ValueError):
            continue
        if not all(np.isfinite(v) for v in (val, lat, lon)):
            continue
        norm = (val - vmin) / span if span > 1e-12 else 0.5
        color = _rgba_to_hex(cmap(float(np.clip(norm, 0.0, 1.0))))
        folium.CircleMarker(
            location=[lat, lon], radius=7,
            color="#ffffff", weight=2,
            fill=True, fill_color=color, fill_opacity=1.0,
            tooltip=f"{row[id_col]} | {metric_col}: {_format_value(val)}",
            pane=pane,
        ).add_to(layer)
    return layer, vmin, vmax


def make_graph_edge_layer(
    nodes_df,
    edges_df,
    *,
    id_col: str = "node_id",
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    name: str = "Graph model edges",
    show: bool = True,
    pane: Optional[str] = None,
) -> Optional[folium.map.FeatureGroup]:
    if nodes_df is None or edges_df is None:
        return None
    required_cols = {id_col, lon_col, lat_col}
    if not required_cols.issubset(nodes_df.columns) or not {"source", "target"}.issubset(edges_df.columns):
        return None

    lookup = nodes_df[[id_col, lon_col, lat_col]].copy()
    lookup[id_col] = lookup[id_col].astype(str)
    coords = {
        row[id_col]: (float(row[lat_col]), float(row[lon_col]))
        for _, row in lookup.iterrows()
    }
    if not coords:
        return None

    prob_series = pd.to_numeric(edges_df["probability"], errors="coerce") if "probability" in edges_df.columns else None
    valid_probs = prob_series[np.isfinite(prob_series)] if prob_series is not None else None
    min_prob = float(valid_probs.min()) if valid_probs is not None and not valid_probs.empty else None
    max_prob = float(valid_probs.max()) if valid_probs is not None and not valid_probs.empty else None
    layer = folium.FeatureGroup(name=name, show=show)
    added = 0
    for idx, row in edges_df.iterrows():
        source = str(row.get("source"))
        target = str(row.get("target"))
        if source not in coords or target not in coords:
            continue
        edge_cost = None
        edge_prob = None
        try:
            raw_cost = row.get("cost")
            if raw_cost is not None:
                edge_cost = float(raw_cost)
        except Exception:
            edge_cost = None
        try:
            raw_prob = row.get("probability")
            if raw_prob is not None:
                edge_prob = float(raw_prob)
        except Exception:
            edge_prob = None
        if edge_prob is not None and min_prob is not None and max_prob is not None:
            norm = (edge_prob - min_prob) / max(max_prob - min_prob, 1e-12)
            color = _rgba_to_hex(plt.get_cmap("cividis")(norm))
        else:
            color = "#3b7dd8"
        tooltip = f"{source} → {target}"
        if edge_cost is not None and np.isfinite(edge_cost):
            tooltip = f"{tooltip}<br>cost: {_format_value(edge_cost)}"
        if edge_prob is not None and np.isfinite(edge_prob):
            tooltip = f"{tooltip}<br>probability: {_format_value(edge_prob)}"
        folium.PolyLine(
            locations=[coords[source], coords[target]],
            color=color,
            weight=2.5,
            opacity=0.75,
            tooltip=tooltip,
            pane=pane,
        ).add_to(layer)
        added += 1

    return layer if added else None


def _build_graph_edge_features(
    nodes_df,
    edges_df,
    *,
    id_col: str = "node_id",
    lon_col: str = "longitude",
    lat_col: str = "latitude",
 ) -> tuple[List[dict], Optional[float], Optional[float]]:
    if nodes_df is None or edges_df is None:
        return [], None, None
    required_cols = {id_col, lon_col, lat_col}
    if not required_cols.issubset(nodes_df.columns) or not {"source", "target"}.issubset(edges_df.columns):
        return [], None, None

    lookup = nodes_df[[id_col, lon_col, lat_col]].copy()
    lookup[id_col] = lookup[id_col].astype(str)
    coords = {
        row[id_col]: (float(row[lat_col]), float(row[lon_col]))
        for _, row in lookup.iterrows()
    }
    if not coords:
        return [], None, None

    prob_series = pd.to_numeric(edges_df["probability"], errors="coerce") if "probability" in edges_df.columns else None
    valid_probs = prob_series[np.isfinite(prob_series)] if prob_series is not None else None
    min_prob = float(valid_probs.min()) if valid_probs is not None and not valid_probs.empty else None
    max_prob = float(valid_probs.max()) if valid_probs is not None and not valid_probs.empty else None

    features: List[dict] = []
    for _, row in edges_df.iterrows():
        source = str(row.get("source"))
        target = str(row.get("target"))
        if source not in coords or target not in coords:
            continue
        edge_cost = None
        edge_prob = None
        try:
            raw_cost = row.get("cost")
            if raw_cost is not None:
                edge_cost = float(raw_cost)
        except Exception:
            edge_cost = None
        try:
            raw_prob = row.get("probability")
            if raw_prob is not None:
                edge_prob = float(raw_prob)
        except Exception:
            edge_prob = None
        if edge_prob is not None and min_prob is not None and max_prob is not None:
            norm = (edge_prob - min_prob) / max(max_prob - min_prob, 1e-12)
            color = _rgba_to_hex(plt.get_cmap("cividis")(norm))
        else:
            color = "#3b7dd8"
        tooltip = f"{source} → {target}"
        if edge_cost is not None and np.isfinite(edge_cost):
            tooltip = f"{tooltip}<br>cost: {_format_value(edge_cost)}"
        if edge_prob is not None and np.isfinite(edge_prob):
            tooltip = f"{tooltip}<br>probability: {_format_value(edge_prob)}"
        features.append(
            {
                "locations": [list(coords[source]), list(coords[target])],
                "color": color,
                "weight": 2.5,
                "opacity": 0.75,
                "tooltip": tooltip,
            }
        )
    return features, min_prob, max_prob


def _build_circuit_edge_features(
    nodes_df,
    pairs_df,
    *,
    id_col: str = "node_id",
    lon_col: str = "longitude",
    lat_col: str = "latitude",
) -> tuple[List[dict], Optional[float], Optional[float]]:
    if nodes_df is None or pairs_df is None:
        return [], None, None
    required_cols = {id_col, lon_col, lat_col}
    if not required_cols.issubset(nodes_df.columns) or not {"source", "target"}.issubset(pairs_df.columns):
        return [], None, None

    lookup = nodes_df[[id_col, lon_col, lat_col]].copy()
    lookup[id_col] = lookup[id_col].astype(str)
    coords = {
        row[id_col]: (float(row[lat_col]), float(row[lon_col]))
        for _, row in lookup.iterrows()
    }
    if not coords:
        return [], None, None

    resistance_series = (
        pd.to_numeric(pairs_df["effective_resistance"], errors="coerce")
        if "effective_resistance" in pairs_df.columns
        else None
    )
    valid_resistance = (
        resistance_series[np.isfinite(resistance_series)]
        if resistance_series is not None
        else None
    )
    min_resistance = float(valid_resistance.min()) if valid_resistance is not None and not valid_resistance.empty else None
    max_resistance = float(valid_resistance.max()) if valid_resistance is not None and not valid_resistance.empty else None

    features: List[dict] = []
    for _, row in pairs_df.iterrows():
        source = str(row.get("source"))
        target = str(row.get("target"))
        if source not in coords or target not in coords:
            continue
        resistance = None
        try:
            raw_resistance = row.get("effective_resistance")
            if raw_resistance is not None:
                resistance = float(raw_resistance)
        except Exception:
            resistance = None
        if resistance is not None and min_resistance is not None and max_resistance is not None:
            norm = (resistance - min_resistance) / max(max_resistance - min_resistance, 1e-12)
            color = _rgba_to_hex(plt.get_cmap("plasma")(norm))
        else:
            color = "#b12a90"
        tooltip = f"{source} → {target}"
        if resistance is not None and np.isfinite(resistance):
            tooltip = f"{tooltip}<br>effective resistance: {_format_value(resistance)}"
        features.append(
            {
                "locations": [list(coords[source]), list(coords[target])],
                "color": color,
                "weight": 2.25,
                "opacity": 0.65,
                "tooltip": tooltip,
            }
        )
    return features, min_resistance, max_resistance


def _read_geojsonl_features(geojsonl_path: str) -> List[dict]:
    feats: List[dict] = []
    if not (geojsonl_path and os.path.exists(geojsonl_path)):
        return feats
    with open(geojsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                feats.append(json.loads(line))
            except Exception:
                continue
    return feats


def _build_lcp_sample_features(
    geojsonl_path: str,
    *,
    color_by: str = "source",
    weight: int = 2,
    opacity: float = 0.8,
) -> List[dict]:
    feats = _read_geojsonl_features(geojsonl_path)
    features: List[dict] = []
    for feat in feats:
        geometry = feat.get("geometry") or {}
        if geometry.get("type") != "LineString":
            continue
        coords = geometry.get("coordinates") or []
        if not coords:
            continue
        locations = []
        for xy in coords:
            if not isinstance(xy, (list, tuple)) or len(xy) < 2:
                continue
            locations.append([float(xy[1]), float(xy[0])])
        if len(locations) < 2:
            continue
        props = feat.get("properties") or {}
        key = str(props.get(color_by, "lcp"))
        tooltip = f"{props.get('source', '')} → {props.get('target', '')}<br>cost: {_format_value(float(props.get('cost', float('nan'))))}"
        features.append(
            {
                "locations": locations,
                "color": _hash_color(key),
                "weight": weight,
                "opacity": opacity,
                "tooltip": tooltip,
            }
        )
    return features


def make_lcp_preview_layer(geojsonl_path: str, *, name: str = "LCP sample", limit: Optional[int] = None, show: bool = True,
                           color_by: str = "source", weight: int = 2, opacity: float = 0.8,
                           pane: Optional[str] = None) -> Optional[folium.map.FeatureGroup]:
    # `limit` is kept for API compatibility but intentionally ignored so we load every feature.
    _ = limit
    feats = _read_geojsonl_features(geojsonl_path)
    if not feats:
        return None

    def _style_func(feat):
        props = feat.get("properties") or {}
        key = str(props.get(color_by, "lcp"))
        return {"color": _hash_color(key), "weight": weight, "opacity": opacity, "pane": pane}

    fc = {"type": "FeatureCollection", "features": feats}
    layer = folium.FeatureGroup(name=f"{name} (n={len(feats)})", show=show)
    tooltip = folium.GeoJsonTooltip(fields=["source", "target", "cost"])
    folium.GeoJson(data=fc, style_function=_style_func, tooltip=tooltip).add_to(layer)
    return layer


# ---------------------------------------------------------------------------
# Composition entrypoints
# ---------------------------------------------------------------------------

def build_nodes_preview_map(nodes_df, *, id_col: str = "node_id", lon_col: str = "longitude", lat_col: str = "latitude") -> str:
    if nodes_df is None or nodes_df.empty:
        # generic world map
        m = _base_map(0.0, 0.0, control_scale=True, zoom=2)
        _add_floating_ui_patch(m)
        return m._repr_html_()

    m = _base_map(nodes_df[lat_col].mean(), nodes_df[lon_col].mean(), control_scale=True, zoom=7)
    nodes_layer = make_nodes_layer(nodes_df, id_col=id_col, lon_col=lon_col, lat_col=lat_col)
    nodes_layer.add_to(m)
    _add_floating_ui_patch(m)
    folium.LayerControl(collapsed=False).add_to(m)
    # Fit to node extent
    try:
        m = fit_map_to_bounds(m, nodes_df.rename(columns={lon_col: "longitude", lat_col: "latitude"}))
    except Exception:
        pass
    return m._repr_html_()


def build_preview_from_cache(cache: Dict[str, Any]) -> str:
    nodes_df = cache.get("nodes_df")
    df = None
    if pd is not None and isinstance(nodes_df, pd.DataFrame) and not nodes_df.empty:
        df = nodes_df

    lon_col = cache.get("nodes_lon_col")
    lat_col = cache.get("nodes_lat_col")
    if df is not None and (not lon_col or not lat_col):
        guess = detect_nodes_lonlat(df)
        if guess:
            lon_col, lat_col = guess

    if df is not None and not (lon_col and lat_col):
        df = None

    surface_type = cache.get("surface_type")
    input_raster = cache.get("raster_path")
    analysis_cost_raster = cache.get("analysis_cost_tif")

    if surface_type == "suitability":
        input_label = "Suitability surface"
    elif surface_type == "conductance":
        input_label = "Conductance surface"
    elif surface_type == "resistance":
        input_label = "Cost surface"
    else:
        input_label = "Input surface"

    cost_tif = input_raster
    cost_label = input_label
    base_rasters: List[Tuple[str, str]] = []
    extra_rasters: List[Tuple[str, str]] = []

    if analysis_cost_raster and os.path.exists(analysis_cost_raster):
        cost_tif = analysis_cost_raster
        label_override = None
        conversion_path = cache.get("conversion_path")
        if conversion_path and os.path.exists(conversion_path):
            try:
                if os.path.abspath(conversion_path) == os.path.abspath(analysis_cost_raster):
                    label_override = cache.get("conversion_label")
            except Exception:
                label_override = cache.get("conversion_label")
        if input_raster and os.path.exists(input_raster) and os.path.abspath(input_raster) != os.path.abspath(analysis_cost_raster):
            # Keep the original raster available for exports, but hide it from map overlays
            # once a dedicated analysis cost surface exists.
            cost_label = label_override or "Converted cost surface"
        else:
            cost_label = label_override or input_label

    preview_layers = cache.get("preview_rasters")
    if isinstance(preview_layers, dict):
        for label, path in preview_layers.items():
            if not path or not os.path.exists(path):
                continue
            extra_rasters.append((path, label or "Converted surface"))
    # Ensure CRK appears in Layer Control even if preview_rasters state is stale.
    kernel_surface_tif = cache.get("kernel_surface_tif")
    if kernel_surface_tif and os.path.exists(kernel_surface_tif):
        kernel_distance = cache.get("kernel_dispersal_distance_m")
        if kernel_distance is not None:
            try:
                kernel_label = kernel_preview_label(float(kernel_distance))
            except Exception:
                kernel_label = KERNEL_PREVIEW_LABEL
        else:
            kernel_label = KERNEL_PREVIEW_LABEL
        already_listed = any(
            os.path.abspath(path) == os.path.abspath(kernel_surface_tif)
            for path, _ in extra_rasters
            if path
        )
        if not already_listed:
            extra_rasters.append((kernel_surface_tif, kernel_label))

    graph_metrics_df = cache.get("graph_metrics_df")
    graph_metric_names: List[str] = []
    if pd is not None and isinstance(graph_metrics_df, pd.DataFrame):
        for metric_name in ("betweenness", "closeness", "pagerank"):
            if metric_name in graph_metrics_df.columns:
                graph_metric_names.append(metric_name)

    # All map layers use the same original WGS84 node coordinates.
    # LCP computation may snap to raster cells internally, but markers do not.
    preview_nodes_df = df

    return build_connectivity_map(
        preview_nodes_df,
        lcp_geojsonl=cache.get("lcp_geojsonl"),
        circuit_pairs_df=cache.get("circuit_pairs_df"),
        cost_tif=cost_tif,
        surface_label=cost_label,
        extra_rasters=extra_rasters,
        lcp_count_tif=cache.get("lcp_count_tif"),
        graph_metrics_df=graph_metrics_df,
        graph_metric_name=cache.get("graph_metric_name"),
        graph_metric_names=graph_metric_names,
        graph_edges_df=cache.get("graph_edges_df"),
        graph_metric_ranges=cache.get("graph_metric_ranges"),
        graph_probability_decay=cache.get("graph_probability_decay"),
        base_rasters=base_rasters,
        preview_limit=0,
        id_col="node_id",
        lon_col=lon_col or "longitude",
        lat_col=lat_col or "latitude",
    )


def build_connectivity_map(
    nodes_df,
    *,
    lcp_geojsonl: Optional[str] = None,
    circuit_pairs_df = None,
    cost_tif: Optional[str] = None,
    surface_label: str = "Cost surface",
    extra_rasters: Optional[List[Tuple[str, str]]] = None,
    lcp_count_tif: Optional[str] = None,
    graph_metrics_df = None,
    graph_metric_name: Optional[str] = None,
    graph_metric_names: Optional[List[str]] = None,
    graph_edges_df = None,
    graph_metric_ranges: Optional[Dict[str, Dict[str, float]]] = None,
    graph_probability_decay: Optional[Dict[str, Any]] = None,
    base_rasters: Optional[List[Tuple[str, str]]] = None,
    preview_limit: Optional[int] = None,
    id_col: str = "node_id",
    lon_col: str = "longitude",
    lat_col: str = "latitude",
) -> str:
    """Compose a UI-safe map of nodes + (optional) LCP preview + rasters.

    - If `lcp_geojsonl` exists, show every path from the `.geojsonl` file.
    - `preview_limit` is kept for compatibility but is not applied.
    - Any `base_rasters` are added first so they sit underneath other surfaces.
    - If `cost_tif` exists, overlay a semi-transparent raster labelled with `surface_label`.
    - If `extra_rasters` is provided, overlay each `(path, label)` pair in order (duplicates skipped).
    - If `lcp_count_tif` exists, overlay the path-count raster (on top by default).
    """
    if nodes_df is None or nodes_df.empty:
        center = (0.0, 0.0)
        zoom = 2
    else:
        center = (float(nodes_df[lat_col].mean()), float(nodes_df[lon_col].mean()))
        zoom = 6

    m = _base_map(center[0], center[1], control_scale=True, zoom=zoom)
    has_overlay_layers = False
    _precreate_overlay_panes(
        m,
        base_rasters=base_rasters,
        cost_label=surface_label if cost_tif else None,
        extra_rasters=extra_rasters,
        lcp_count_present=bool(lcp_count_tif),
        has_nodes=bool(nodes_df is not None and not nodes_df.empty),
        has_circuit=bool(nodes_df is not None and not nodes_df.empty and circuit_pairs_df is not None),
        has_lcp=bool(lcp_geojsonl and os.path.exists(lcp_geojsonl)),
        graph_metric_name=graph_metric_name if graph_metrics_df is not None else None,
        graph_metric_names=graph_metric_names if graph_metrics_df is not None else None,
        has_graph_edges=bool(nodes_df is not None and not nodes_df.empty and graph_edges_df is not None),
    )

    # Add base rasters (e.g., suitability) before any analysis layers so they remain underneath.
    if base_rasters:
        for tif_path, label in base_rasters:
            if not tif_path or not os.path.exists(tif_path):
                continue
            if cost_tif and os.path.abspath(tif_path) == os.path.abspath(cost_tif):
                continue
            label_lower = (label or "").lower()
            is_kernel = "cumulative resistant kernel" in label_lower
            is_current_density = "current density" in label_lower
            show_layer = False
            _add_geotiff_overlay(
                m,
                tif_path,
                name=label,
                opacity=1.0,
                cmap="magma" if (is_current_density or is_kernel) else "viridis",
                pmin=0.0 if is_kernel else 2.0,
                pmax=99.5 if (is_current_density or is_kernel) else 98.0,
                show=show_layer,
                transform="log1p" if (is_current_density or is_kernel) else None,
                transparent_below=0.0 if is_current_density else None,
                preview_max_dim=4000 if (is_current_density or is_kernel) else 1400,
                preview_resampling="nearest" if (is_current_density or is_kernel) else "bilinear",
                sparse_positive_only=False,
            )
            has_overlay_layers = True

    # Ras ters first (so vectors sit above)
    if cost_tif:
        _add_geotiff_overlay(
            m,
            cost_tif,
            name=surface_label,
            cmap="viridis",
            pmin=2.0,
            pmax=98.0,
            show=not bool(lcp_count_tif),
        )
        has_overlay_layers = True

    if extra_rasters:
        for tif_path, label in extra_rasters:
            if not tif_path or not os.path.exists(tif_path):
                continue
            if cost_tif and os.path.abspath(tif_path) == os.path.abspath(cost_tif):
                continue
            label_lower = (label or "").lower()
            is_kernel = "cumulative resistant kernel" in label_lower
            is_current_density = "current density" in label_lower
            show_layer = bool(label_lower and (is_kernel or is_current_density) and not lcp_count_tif)
            _add_geotiff_overlay(
                m,
                tif_path,
                name=label,
                opacity=1.0,
                cmap="magma" if (is_current_density or is_kernel) else "viridis",
                pmin=0.0 if is_kernel else 2.0,
                pmax=99.5 if (is_current_density or is_kernel) else 98.0,
                show=show_layer,
                transform="log1p" if (is_current_density or is_kernel) else None,
                transparent_below=0.0 if is_current_density else None,
                preview_max_dim=4000 if (is_current_density or is_kernel) else 1400,
                preview_resampling="nearest" if (is_current_density or is_kernel) else "bilinear",
                sparse_positive_only=False,
            )
            has_overlay_layers = True

    if lcp_count_tif:
        _add_geotiff_overlay(
            m,
            lcp_count_tif,
            name="Connectivity (LCP Density)",
            cmap="magma",
            pmin=0.0,
            pmax=99.5,
            show=True,
            transparent_below=0.0,
            preview_max_dim=4000,
            preview_resampling="nearest",
            sparse_positive_only=True,
        )
        has_overlay_layers = True

    # Nodes
    if nodes_df is not None and not nodes_df.empty:
        nodes_pane = _ensure_map_pane(m, "Nodes", z_index=460)
        nodes_layer = make_nodes_layer(
            nodes_df,
            id_col=id_col,
            lon_col=lon_col,
            lat_col=lat_col,
            name="Nodes",
            show=True,
            pane=nodes_pane,
            raster_path=cost_tif,
        )
        nodes_layer.add_to(m)
        node_features = _build_node_marker_features(
            nodes_df,
            id_col=id_col,
            lon_col=lon_col,
            lat_col=lat_col,
            raster_path=None,
        )
        _add_circle_marker_feature_group_script(
            m,
            layer_var_name=nodes_layer.get_name(),
            pane=nodes_pane,
            features=node_features,
        )
        has_overlay_layers = True

    if nodes_df is not None and not nodes_df.empty and graph_edges_df is not None:
        graph_edges_pane = _ensure_map_pane(m, "Graph model edges", z_index=430)
        edge_features, edge_vmin, edge_vmax = _build_graph_edge_features(
            nodes_df,
            graph_edges_df,
            id_col=id_col,
            lon_col=lon_col,
            lat_col=lat_col,
        )
        if edge_features:
            edge_layer = folium.FeatureGroup(name="Graph model edges", show=True)
            edge_layer.add_to(m)
            _add_polyline_feature_group_script(
                m,
                layer_var_name=edge_layer.get_name(),
                pane=graph_edges_pane,
                features=edge_features,
            )
            has_overlay_layers = True
            _add_graph_edge_legend(
                m,
                name="Graph model edges",
                vmin=edge_vmin,
                vmax=edge_vmax,
            )

    if nodes_df is not None and not nodes_df.empty and circuit_pairs_df is not None:
        circuit_pane = _ensure_map_pane(m, "Circuit effective resistance", z_index=435)
        circuit_features, circuit_vmin, circuit_vmax = _build_circuit_edge_features(
            nodes_df,
            circuit_pairs_df,
            id_col=id_col,
            lon_col=lon_col,
            lat_col=lat_col,
        )
        if circuit_features:
            circuit_layer = folium.FeatureGroup(name="Circuit effective resistance", show=True)
            circuit_layer.add_to(m)
            _add_polyline_feature_group_script(
                m,
                layer_var_name=circuit_layer.get_name(),
                pane=circuit_pane,
                features=circuit_features,
            )
            has_overlay_layers = True
            if (
                circuit_vmin is not None
                and circuit_vmax is not None
                and np.isfinite(circuit_vmin)
                and np.isfinite(circuit_vmax)
            ):
                _add_raster_legend(
                    m,
                    name="Circuit effective resistance",
                    cmap="plasma",
                    vmin=float(circuit_vmin),
                    vmax=float(circuit_vmax),
                    pmin=0.0,
                    pmax=100.0,
                )

    if nodes_df is not None and not nodes_df.empty and graph_metrics_df is not None:
        metric_names = list(graph_metric_names or ())
        if graph_metric_name and graph_metric_name not in metric_names:
            metric_names.insert(0, graph_metric_name)
        if not metric_names:
            metric_names = [
                metric_name
                for metric_name in ("betweenness", "closeness", "pagerank")
                if metric_name in getattr(graph_metrics_df, "columns", [])
            ]
        legend_added = False
        for idx, metric_name in enumerate(metric_names):
            graph_nodes_pane = _ensure_map_pane(m, f"Graph model ({metric_name})", z_index=470 + idx)
            graph_layer = make_graph_metric_layer(
                nodes_df,
                graph_metrics_df,
                metric_col=metric_name,
                id_col=id_col,
                lon_col=lon_col,
                lat_col=lat_col,
                name=f"Graph model ({metric_name})",
                show=(metric_name == graph_metric_name),
                pane=graph_nodes_pane,
            )
            if graph_layer is None:
                continue
            lyr, vmin, vmax = graph_layer
            lyr.add_to(m)
            has_overlay_layers = True
            _add_raster_legend(
                m,
                name=f"Graph model ({metric_name})",
                cmap="plasma",
                vmin=vmin,
                vmax=vmax,
                pmin=0.0,
                pmax=100.0,
            )
            legend_added = True
    # LCP preview layer
    if lcp_geojsonl and os.path.exists(lcp_geojsonl):
        lcp_pane = _ensure_map_pane(m, "LCP sample", z_index=440)
        lcp_features = _build_lcp_sample_features(
            lcp_geojsonl,
            color_by="source",
            weight=2,
            opacity=0.8,
        )
        if lcp_features:
            lcp_layer = folium.FeatureGroup(name="LCP sample", show=True)
            lcp_layer.add_to(m)
            _add_polyline_feature_group_script(
                m,
                layer_var_name=lcp_layer.get_name(),
                pane=lcp_pane,
                features=lcp_features,
            )
            has_overlay_layers = True

    if has_overlay_layers:
        _add_layer_control_sort_patch(m)
    _add_floating_ui_patch(m)

    if has_overlay_layers:
        folium.LayerControl(collapsed=False).add_to(m)

    # Fit to data bounds: prefer nodes; else try rasters
    try:
        if nodes_df is not None and not nodes_df.empty:
            m = fit_map_to_bounds(m, nodes_df.rename(columns={lon_col: "longitude", lat_col: "latitude"}))
        elif lcp_count_tif and os.path.exists(lcp_count_tif):
            with rasterio.open(lcp_count_tif) as ds:
                b = _raster_bounds_wgs84(ds)
            m.fit_bounds(b)
        elif cost_tif and os.path.exists(cost_tif):
            with rasterio.open(cost_tif) as ds:
                b = _raster_bounds_wgs84(ds)
            m.fit_bounds(b)
    except Exception:
        pass

    return m._repr_html_()


__all__ = [
    "build_nodes_preview_map",
    "build_preview_from_cache",
    "build_connectivity_map",
    "make_nodes_layer",
    "make_lcp_preview_layer",
]


