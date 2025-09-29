# factorial_lcp.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Sequence, Iterable
import os
import math
import json

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from pyproj import Transformer, CRS, Geod
from skimage.graph import MCP, MCP_Geometric
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from shapely.geometry import LineString
except Exception:  # optional; only needed if you enable simplification
    LineString = None  # type: ignore


# ----------------------------- Data classes ---------------------------------

@dataclass
class FactorialLCPParams:
    id_col: str = "node_id"
    lon_col: str = "longitude"
    lat_col: str = "latitude"
    nodes_crs: Optional[str] = "EPSG:4326"
    fully_connected: bool = True        # 8-neighborhood
    geometric: bool = True              # diagonal steps use Euclidean weight
    # Adaptive crop margin in pixels: margin = max(crop_min, crop_base + crop_alpha * d_max)
    crop_base: int = 128
    crop_alpha: float = 0.2
    crop_min: int = 256
    barrier_cost: float = 1e12
    n_jobs: int = 1
    use_float32: bool = True            # read windows as float32
    simplify_tol_m: Optional[float] = None  # visual-only simplification (meters)
    max_vertices: Optional[int] = None      # visual-only vertex cap


# ----------------------------- Helpers --------------------------------------

def _xy_to_rc(transform: Affine, x: float, y: float) -> tuple[int, int]:
    col, row = (~transform) * (x, y)
    return int(math.floor(row + 0.5)), int(math.floor(col + 0.5))


def _rc_to_xy(transform: Affine, r: int, c: int) -> tuple[float, float]:
    x, y = transform * (c + 0.5, r + 0.5)
    return float(x), float(y)


def _union_crop_window(points_rc: Sequence[tuple[int, int]], h: int, w: int, margin: int) -> Window:
    r_vals = [p[0] for p in points_rc]
    c_vals = [p[1] for p in points_rc]
    rmin = max(0, min(r_vals) - margin)
    rmax = min(h, max(r_vals) + margin)
    cmin = max(0, min(c_vals) - margin)
    cmax = min(w, max(c_vals) + margin)
    return Window.from_slices((int(rmin), int(rmax)), (int(cmin), int(cmax)))


def _geodesic_length_m(coords_ll: list[tuple[float, float]]) -> float:
    geod = Geod(ellps="WGS84")
    length = 0.0
    for (x0, y0), (x1, y1) in zip(coords_ll[:-1], coords_ll[1:]):
        _, _, d = geod.inv(x0, y0, x1, y1)
        length += d
    return float(length)


def _simplify_coords(coords_ll: list[tuple[float, float]], tol_m: Optional[float], max_vertices: Optional[int]) -> list[tuple[float, float]]:
    if tol_m and LineString is not None and len(coords_ll) > 2:
        # Rough conversion: 1 deg latitude ≈ 111_000 m
        deg_tol = float(tol_m) / 111_000.0
        line = LineString(coords_ll)
        line = line.simplify(deg_tol, preserve_topology=True)
        coords_ll = [(float(x), float(y)) for x, y in line.coords]
    if max_vertices and len(coords_ll) > max_vertices:
        stride = max(1, len(coords_ll) // int(max_vertices))
        coords_ll = coords_ll[::stride] + coords_ll[-1:]
    return coords_ll


# ----------------------------- Worker ---------------------------------------

def _solve_for_source(
    src_id: str,
    src_rc: tuple[int, int],
    targets: list[tuple[str, tuple[int, int]]],
    raster_path: str,
    crop_base: int,
    crop_alpha: float,
    crop_min: int,
    fully_connected: bool,
    geometric: bool,
    barrier_cost: float,
    use_float32: bool,
    simplify_tol_m: Optional[float],
    max_vertices: Optional[int],
) -> dict:
    features = []
    skipped: list[tuple[str, str, str]] = []

    with rasterio.open(raster_path) as ds:
        H, W = ds.height, ds.width
        nodata = ds.nodata
        rcrs = ds.crs

        # Adaptive margin based on farthest target (in pixels)
        d_max = 0.0
        for _, rc in targets:
            dr = abs(rc[0] - src_rc[0]); dc = abs(rc[1] - src_rc[1])
            d_max = max(d_max, (dr*dr + dc*dc) ** 0.5)
        margin_eff = max(int(crop_min), int(crop_base + crop_alpha * d_max))

        # Build one union window covering source + all targets
        pts = [src_rc] + [rc for _, rc in targets]
        win = _union_crop_window(pts, H, W, margin_eff)

        # Read cost window
        arr = ds.read(1, window=win, masked=True)
        if use_float32:
            arr = arr.astype(np.float32, copy=False)
        else:
            arr = arr.astype(np.float64, copy=False)
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        arr = np.where(~np.isfinite(arr), np.nan, arr)
        if np.isnan(arr).all():
            for dst_id, _ in targets:
                skipped.append((src_id, dst_id, "no valid cost pixels in window"))
            return {"features": features, "skipped": skipped}
        arr = np.where(np.isnan(arr), np.float32(barrier_cost) if use_float32 else float(barrier_cost), arr)

        # Local coords in window
        r_off, c_off = int(win.row_off), int(win.col_off)
        start = (src_rc[0] - r_off, src_rc[1] - c_off)
        ends = [(rc[0] - r_off, rc[1] - c_off) for _, rc in targets]

        # Build MCP solver and compute SPT
        mcp = MCP_Geometric(arr, fully_connected=fully_connected) if geometric else MCP(arr, fully_connected=fully_connected)
        costs, _ = mcp.find_costs(starts=[start], ends=ends)

        # Prepare transform/projection to WGS84
        win_transform = ds.window_transform(win)
        if rcrs and rcrs.to_string() != "EPSG:4326":
            to_wgs = Transformer.from_crs(rcrs, "EPSG:4326", always_xy=True)
            def _to_ll(xy):
                return to_wgs.transform(xy[0], xy[1])
        else:
            to_wgs = None
            def _to_ll(xy):
                return (float(xy[0]), float(xy[1]))

        # Reconstruct each target path
        for (dst_id, rc_dst), end in zip(targets, ends):
            r_end, c_end = end
            cval = float(costs[r_end, c_end])
            if not np.isfinite(cval):
                skipped.append((src_id, dst_id, "unreachable"))
                continue
            try:
                path_rc = mcp.traceback(end)
            except Exception:
                skipped.append((src_id, dst_id, "traceback failure"))
                continue

            # RC → raster CRS xy → WGS84
            coords_xy = [_rc_to_xy(win_transform, r, c) for r, c in path_rc]
            coords_ll = [_to_ll(xy) for xy in coords_xy]

            # Metrics on full coordinates
            length_m = _geodesic_length_m(coords_ll)

            # Optional simplification for output geometry only
            coords_out = _simplify_coords(coords_ll, simplify_tol_m, max_vertices)

            feat = {
                "type": "Feature",
                "properties": {
                    "source": src_id,
                    "target": dst_id,
                    "cost": cval,
                    "length_m": float(length_m),
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [(float(x), float(y)) for x, y in coords_out],
                },
            }
            features.append(feat)

    return {"features": features, "skipped": skipped}


# ----------------------------- Public API -----------------------------------

def compute_all_pairs_lcps(
    cost_tif: str,
    nodes_df: pd.DataFrame,
    *,
    id_col: str = "node_id",
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    nodes_crs: Optional[str] = "EPSG:4326",
    fully_connected: bool = True,
    geometric: bool = True,
    crop_base: int = 128,
    crop_alpha: float = 0.2,
    crop_min: int = 256,
    barrier_cost: float = 1e12,
    n_jobs: int = 1,
    use_float32: bool = True,
    simplify_tol_m: Optional[float] = None,
    max_vertices: Optional[int] = None,
    outdir: Optional[str] = None,
) -> dict:
    """Compute exact least-cost paths for **all unordered pairs (i<j)**.

    Returns dict with keys: features, count, skipped_pairs, params, geojson_path (optional).
    """
    if not os.path.exists(cost_tif):
        raise FileNotFoundError(cost_tif)
    if nodes_df is None or nodes_df.empty:
        raise ValueError("nodes_df is empty")
    for col in (id_col, lon_col, lat_col):
        if col not in nodes_df.columns:
            raise ValueError(f"nodes_df must contain column '{col}'")

    # Raster metadata (open once here to get transform/crs; reopen in workers)
    with rasterio.open(cost_tif) as ds:
        H, W = ds.height, ds.width
        transform = ds.transform
        raster_crs = ds.crs

    # Prepare transformer for node coords → raster CRS
    if nodes_crs:
        if raster_crs is None:
            raise ValueError("Cost raster lacks CRS; cannot reproject nodes.")
        if CRS.from_user_input(nodes_crs) != raster_crs:
            to_raster = Transformer.from_crs(nodes_crs, raster_crs, always_xy=True)
        else:
            to_raster = None
    else:
        to_raster = None

    # Project nodes, snap to pixels, and keep those on-grid
    ids: list[str] = []
    rc_lookup: dict[str, tuple[int, int]] = {}
    for _, row in nodes_df.iterrows():
        nid = str(row[id_col])
        x, y = float(row[lon_col]), float(row[lat_col])
        if to_raster is not None:
            x, y = to_raster.transform(x, y)
        r, c = _xy_to_rc(transform, x, y)
        if 0 <= r < H and 0 <= c < W:
            ids.append(nid)
            rc_lookup[nid] = (r, c)

    if len(ids) < 2:
        raise ValueError("Need at least two valid nodes within the raster extent.")

    # Build unordered pair list i<j
    pair_list: list[tuple[str, str]] = []
    n = len(ids)
    for i in range(n):
        a = ids[i]
        for j in range(i + 1, n):
            b = ids[j]
            pair_list.append((a, b))

    # Group targets by source index i (a fixed, b varies)
    by_src: dict[str, list[tuple[str, tuple[int, int]]]] = {}
    for a, b in pair_list:
        # (ids already filtered to rc_lookup keys)
        by_src.setdefault(a, []).append((b, rc_lookup[b]))

    # Solve per source (optionally in parallel)
    features_all = []
    skipped_all: list[tuple[str, str, str]] = []

    # Modest cap for Spaces
    if n_jobs is None or n_jobs < 1:
        n_jobs = 1
    n_jobs = min(int(n_jobs), max(1, (os.cpu_count() or 2) // 2), 8)

    if n_jobs > 1 and len(by_src) > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = []
            for src_id, targets in by_src.items():
                fut = ex.submit(
                    _solve_for_source,
                    src_id,
                    rc_lookup[src_id],
                    targets,
                    cost_tif,
                    int(crop_base),
                    float(crop_alpha),
                    int(crop_min),
                    bool(fully_connected),
                    bool(geometric),
                    float(barrier_cost),
                    bool(use_float32),
                    simplify_tol_m,
                    max_vertices,
                )
                futs.append(fut)
            for fut in as_completed(futs):
                res = fut.result()
                features_all.extend(res.get("features", []))
                skipped_all.extend(res.get("skipped", []))
    else:
        for src_id, targets in by_src.items():
            res = _solve_for_source(
                src_id,
                rc_lookup[src_id],
                targets,
                cost_tif,
                int(crop_base),
                float(crop_alpha),
                int(crop_min),
                bool(fully_connected),
                bool(geometric),
                float(barrier_cost),
                bool(use_float32),
                simplify_tol_m,
                max_vertices,
            )
            features_all.extend(res.get("features", []))
            skipped_all.extend(res.get("skipped", []))

    result = {
        "features": features_all,
        "count": len(features_all),
        "skipped_pairs": skipped_all,
        "params": {
            "id_col": id_col,
            "lon_col": lon_col,
            "lat_col": lat_col,
            "nodes_crs": nodes_crs,
            "fully_connected": fully_connected,
            "geometric": geometric,
            "crop_base": crop_base,
            "crop_alpha": crop_alpha,
            "crop_min": crop_min,
            "barrier_cost": barrier_cost,
            "n_jobs": n_jobs,
            "use_float32": use_float32,
            "simplify_tol_m": simplify_tol_m,
            "max_vertices": max_vertices,
            "cost_tif": os.path.basename(cost_tif),
        },
    }

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        out_geojson = os.path.join(outdir, "lcps_all.geojson")
        with open(out_geojson, "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": features_all}, f)
        result["geojson_path"] = out_geojson

    return result


__all__ = ["FactorialLCPParams", "compute_all_pairs_lcps"]
