"""
Map layers for the SpatChat: Connectivity room

- Nodes preview (points)
- LCP preview from **GeoJSONL** (sampled — UI-safe)
- Raster overlays (cost surface, LCP count raster) with percentile stretch

This is intentionally RAM-light: we DO NOT load full factorial outputs into
Folium. For LCPs, we read only the first `preview_limit` features from the
`.geojsonl` file and render those. Full artifacts remain on disk.
"""
from __future__ import annotations

import os
import json
import math
import random
import hashlib
from typing import Iterable, List, Tuple, Optional

import numpy as np
import folium
import rasterio
import matplotlib.pyplot as plt
from rasterio.transform import Affine
from pyproj import Transformer

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
    src_crs = ds.crs.to_string() if ds.crs else "EPSG:4326"
    if src_crs != "EPSG:4326":
        to_wgs = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
        pts = [(b.left, b.bottom), (b.left, b.top), (b.right, b.bottom), (b.right, b.top)]
        ll = [to_wgs.transform(x, y) for x, y in pts]  # (lon, lat)
        west = min(p[0] for p in ll); east = max(p[0] for p in ll)
        south = min(p[1] for p in ll); north = max(p[1] for p in ll)
        return [[south, west], [north, east]]
    return [[b.bottom, b.left], [b.top, b.right]]


def _rgba_from_array(arr: np.ndarray, cmap_name: str = "magma", pmin: float = 2.0, pmax: float = 98.0) -> np.ndarray:
    """Percentile stretch to 0..1 then map to RGBA uint8 (A preserved)."""
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0)
    if not np.isfinite(arr).any():
        return np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    lo = float(np.nanpercentile(arr, pmin))
    hi = float(np.nanpercentile(arr, pmax))
    if hi <= lo:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        if hi <= lo:
            return np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    arr = np.clip((arr - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    cmap = plt.get_cmap(cmap_name)
    rgba = (cmap(arr) * 255).astype(np.uint8)
    return rgba


def _add_geotiff_overlay(m: folium.Map, tif_path: str, *, name: str, opacity: float = 0.7, cmap: str = "magma",
                          pmin: float = 2.0, pmax: float = 98.0, show: bool = True) -> Optional[folium.map.FeatureGroup]:
    if not (tif_path and os.path.exists(tif_path)):
        return None
    try:
        with rasterio.open(tif_path) as src:
            arr = src.read(1)
            rgba = _rgba_from_array(arr, cmap_name=cmap, pmin=pmin, pmax=pmax)
            bounds_ll = _raster_bounds_wgs84(src)
            layer = folium.FeatureGroup(name=name, show=show)
            layer.add_child(
                folium.raster_layers.ImageOverlay(
                    image=np.dstack([rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]]),
                    bounds=bounds_ll,
                    opacity=opacity,
                    interactive=False,
                )
            )
            layer.add_to(m)
            return layer
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Nodes & LCP layers
# ---------------------------------------------------------------------------

def make_nodes_layer(nodes_df, *, id_col: str = "node_id", lon_col: str = "longitude", lat_col: str = "latitude",
                     name: str = "Nodes", show: bool = True) -> folium.map.FeatureGroup:
    layer = folium.FeatureGroup(name=name, show=show)
    for _, row in nodes_df.iterrows():
        nid = str(row[id_col])
        lat = float(row[lat_col]); lon = float(row[lon_col])
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color="#2c7fb8",
            fill=True,
            fill_opacity=0.9,
            popup=f"{nid}",
        ).add_to(layer)
    return layer


def _read_geojsonl_sample(geojsonl_path: str, limit: int) -> List[dict]:
    feats: List[dict] = []
    if not (geojsonl_path and os.path.exists(geojsonl_path)):
        return feats
    with open(geojsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= int(limit):
                break
            try:
                feats.append(json.loads(line))
            except Exception:
                continue
    return feats


def make_lcp_preview_layer(geojsonl_path: str, *, name: str = "LCP sample", limit: int = 2000, show: bool = True,
                           color_by: str = "source", weight: int = 2, opacity: float = 0.8) -> Optional[folium.map.FeatureGroup]:
    feats = _read_geojsonl_sample(geojsonl_path, limit)
    if not feats:
        return None

    def _style_func(feat):
        props = feat.get("properties") or {}
        key = str(props.get(color_by, "lcp"))
        return {"color": _hash_color(key), "weight": weight, "opacity": opacity}

    fc = {"type": "FeatureCollection", "features": feats}
    layer = folium.FeatureGroup(name=f"{name} (n={len(feats)})", show=show)
    folium.GeoJson(data=fc, style_function=_style_func, tooltip=folium.GeoJsonTooltip(fields=["source", "target", "cost"]).add_to(layer))
    return layer


# ---------------------------------------------------------------------------
# Composition entrypoints
# ---------------------------------------------------------------------------

def build_nodes_preview_map(nodes_df, *, id_col: str = "node_id", lon_col: str = "longitude", lat_col: str = "latitude") -> str:
    if nodes_df is None or nodes_df.empty:
        # generic world map
        m = _base_map(0.0, 0.0, control_scale=True, zoom=2)
        folium.LayerControl(collapsed=False).add_to(m)
        return m._repr_html_()

    m = _base_map(nodes_df[lat_col].mean(), nodes_df[lon_col].mean(), control_scale=True, zoom=7)
    make_nodes_layer(nodes_df, id_col=id_col, lon_col=lon_col, lat_col=lat_col).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    # Fit to node extent
    try:
        m = fit_map_to_bounds(m, nodes_df.rename(columns={lon_col: "longitude", lat_col: "latitude"}))
    except Exception:
        pass
    return m._repr_html_()


def build_connectivity_map(
    nodes_df,
    *,
    lcp_geojsonl: Optional[str] = None,
    cost_tif: Optional[str] = None,
    lcp_count_tif: Optional[str] = None,
    preview_limit: int = 2000,
    id_col: str = "node_id",
    lon_col: str = "longitude",
    lat_col: str = "latitude",
) -> str:
    """Compose a UI-safe map of nodes + (optional) LCP preview + rasters.

    - If `lcp_geojsonl` exists, show the first `preview_limit` paths as a layer.
    - If `cost_tif` exists, overlay a semi-transparent cost surface.
    - If `lcp_count_tif` exists, overlay the path-count raster (on top by default).
    """
    if nodes_df is None or nodes_df.empty:
        center = (0.0, 0.0)
        zoom = 2
    else:
        center = (float(nodes_df[lat_col].mean()), float(nodes_df[lon_col].mean()))
        zoom = 6

    m = _base_map(center[0], center[1], control_scale=False, zoom=zoom)

    # Ras ters first (so vectors sit above)
    if cost_tif:
        _add_geotiff_overlay(m, cost_tif, name="Cost surface", opacity=0.55, cmap="viridis", pmin=2.0, pmax=98.0, show=False)

    if lcp_count_tif:
        _add_geotiff_overlay(m, lcp_count_tif, name="LCP count raster", opacity=0.75, cmap="magma", pmin=0.0, pmax=99.5, show=True)

    # Nodes
    if nodes_df is not None and not nodes_df.empty:
        make_nodes_layer(nodes_df, id_col=id_col, lon_col=lon_col, lat_col=lat_col, name="Nodes", show=True).add_to(m)

    # LCP preview layer
    if lcp_geojsonl and os.path.exists(lcp_geojsonl):
        lyr = make_lcp_preview_layer(lcp_geojsonl, name="LCP sample", limit=int(preview_limit), show=True)
        if lyr is not None:
            lyr.add_to(m)

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
    "build_connectivity_map",
    "make_nodes_layer",
    "make_lcp_preview_layer",
]
