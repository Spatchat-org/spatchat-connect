# flcp.py

"""
Factorial (all unordered pairs) Least-Cost Paths — STREAMING & RAM-SAFE

This module computes **exact** least-cost paths (LCPs) for **all unordered
pairs** of nodes over a cost raster, while keeping memory stable:
- Builds **one SPT per source** (Dijkstra via scikit-image MCP/MCP_Geometric).
- Writes **per-source** GEOJSONL and CSV temp files (no giant in-RAM lists).
- Merges temps into final artifacts at the end (ordered by node order).

Outputs
-------
If `outdir` is provided, the function writes:
- `lcps_all.geojsonl`       – all paths, one feature per line (WGS84)
- `lcp_index.csv`           – rows: source,target,cost,length_m,n_vertices
- `unreachable_pairs.csv`   – rows: source,target,reason
- `nodes_order.csv`         – the exact node order used (one id per line)
- `lcp_path_counts.tif`     – per-pixel path counts
- `lcp_cost_matrix.csv`     – symmetric cost matrix aligned with node order

By default, **no features are kept in memory**. Set `return_features=True` to
also collect features in RAM (not recommended for large N).

Example
-------
from methods.flcp import compute_all_pairs_lcps
res = compute_all_pairs_lcps(
    cost_tif="uploads/cost.tif",
    nodes_df=nodes_df,  # columns: node_id, longitude, latitude (WGS84 default)
    n_jobs=4,
    outdir="outputs",
)
print(res["count"], "paths →", res.get("geojsonl_path"))
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Iterable
import os
import math
import json
import csv
import shutil

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine, rowcol
from pyproj import Transformer, CRS, Geod
from skimage.graph import MCP, MCP_Geometric
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from shapely.geometry import LineString
except Exception:  # optional; only needed if you enable simplification
    LineString = None  # type: ignore


# ----------------------------- Data classes ---------------------------------

@dataclass
class FLCPParams:
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
    return_features: bool = False


# ----------------------------- Helpers --------------------------------------

def _xy_to_rc(transform: Affine, x: float, y: float) -> tuple[int, int]:
    row, col = rowcol(transform, x, y, op=math.floor)
    return int(row), int(col)


def _rc_to_xy(transform: Affine, r: int, c: int) -> tuple[float, float]:
    x, y = transform * (c + 0.5, r + 0.5)
    return float(x), float(y)


def _estimate_pixel_size(transform: Affine) -> float:
    sx = math.hypot(transform.a, transform.b)
    sy = math.hypot(transform.d, transform.e)
    if sx == 0 and sy == 0:
        raise ValueError("Invalid transform; pixel size cannot be zero.")
    if sx == 0:
        return float(sy)
    if sy == 0:
        return float(sx)
    return float((sx + sy) / 2.0)


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


def _safe_name(s: str) -> str:
    # Safe-ish filename from node id
    keep = [c if c.isalnum() or c in ("-", "_") else "_" for c in str(s)]
    out = "".join(keep)
    return out[:64] if len(out) > 64 else out


def write_cost_matrix_from_index(index_csv_path: str, nodes_order_path: str, out_csv_path: str) -> str:
    nodes: list[str] = []
    with open(nodes_order_path, "r", encoding="utf-8") as f:
        for line in f:
            nid = line.strip()
            if nid:
                nodes.append(nid)
    if not nodes:
        raise ValueError("nodes_order.csv is empty")

    n = len(nodes)
    idx_lookup = {nid: i for i, nid in enumerate(nodes)}
    matrix = np.full((n, n), np.nan, dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)

    with open(index_csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row.get("source")
            dst = row.get("target")
            if src not in idx_lookup or dst not in idx_lookup:
                continue
            cost_val = float(row.get("cost", "nan"))
            i = idx_lookup[src]
            j = idx_lookup[dst]
            matrix[i, j] = cost_val
            matrix[j, i] = cost_val

    df = pd.DataFrame(matrix, index=nodes, columns=nodes)
    df.to_csv(out_csv_path)
    return out_csv_path


def _append_node_coordinates_to_pair_csv(
    csv_path: Optional[str],
    nodes_df: pd.DataFrame,
    *,
    id_col: str,
    lon_col: str,
    lat_col: str,
) -> None:
    """Add source/target coordinate columns to pairwise result CSVs."""
    if not csv_path or not os.path.exists(csv_path):
        return
    if nodes_df is None or nodes_df.empty:
        return
    required = {id_col, lon_col, lat_col}
    if not required.issubset(nodes_df.columns):
        return
    try:
        pairs_df = pd.read_csv(csv_path, dtype={"source": str, "target": str})
    except Exception:
        return
    if pairs_df.empty or "source" not in pairs_df.columns or "target" not in pairs_df.columns:
        return

    coords_df = (
        nodes_df[[id_col, lon_col, lat_col]]
        .copy()
        .rename(
            columns={
                id_col: "node_id",
                lon_col: "x",
                lat_col: "y",
            }
        )
    )
    coords_df["node_id"] = coords_df["node_id"].astype(str)
    pairs_df["source"] = pairs_df["source"].astype(str)
    pairs_df["target"] = pairs_df["target"].astype(str)

    source_coords = coords_df.rename(
        columns={
            "node_id": "source",
            "x": "source_x",
            "y": "source_y",
        }
    )
    target_coords = coords_df.rename(
        columns={
            "node_id": "target",
            "x": "target_x",
            "y": "target_y",
        }
    )

    merged = pairs_df.merge(source_coords, on="source", how="left")
    merged = merged.merge(target_coords, on="target", how="left")
    merged.to_csv(csv_path, index=False)


# ----------------------------- Worker ---------------------------------------

def _solve_for_source(
    src_index: int,
    src_id: str,
    src_rc: tuple[int, int],
    targets: list[tuple[str, tuple[int, int]]],
    raster_path: str,
    crop_base: int,
    crop_alpha: float,
    crop_min: int,
    fully_connected: bool,
    geometric: bool,
    pixel_size: float,
    barrier_cost: float,
    use_float32: bool,
    simplify_tol_m: Optional[float],
    max_vertices: Optional[int],
    tmp_geojsonl_dir: Optional[str],
    tmp_index_dir: Optional[str],
    tmp_unreach_dir: Optional[str],
    tmp_counts_dir: Optional[str],
    return_features: bool,
) -> dict:
    features_mem = [] if return_features else None
    count = 0
    skipped_count = 0

    # Prepare per-source temp files
    gjl_path = None
    idx_path = None
    unr_path = None
    counts_meta: Optional[dict] = None
    gjl_f = idx_f = unr_f = None

    try:
        if tmp_geojsonl_dir:
            os.makedirs(tmp_geojsonl_dir, exist_ok=True)
            gjl_path = os.path.join(tmp_geojsonl_dir, f"lcp_src_{src_index:06d}_{_safe_name(src_id)}.geojsonl")
            gjl_f = open(gjl_path, "w", buffering=8*1024*1024, encoding="utf-8")
        if tmp_index_dir:
            os.makedirs(tmp_index_dir, exist_ok=True)
            idx_path = os.path.join(tmp_index_dir, f"lcp_idx_src_{src_index:06d}.csv")
            idx_f = open(idx_path, "w", buffering=4*1024*1024, newline="", encoding="utf-8")
            idx_writer = csv.writer(idx_f)
            # no header here; parent will write the header in the merged file
        if tmp_unreach_dir:
            os.makedirs(tmp_unreach_dir, exist_ok=True)
            unr_path = os.path.join(tmp_unreach_dir, f"unreach_src_{src_index:06d}.csv")
            unr_f = open(unr_path, "w", buffering=1*1024*1024, newline="", encoding="utf-8")
            unr_writer = csv.writer(unr_f)

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
            arr = arr.astype(np.float32 if use_float32 else np.float64, copy=False)
            if nodata is not None:
                arr = np.where(arr == nodata, np.nan, arr)
            arr = np.where(~np.isfinite(arr), np.nan, arr)
            if np.isnan(arr).all():
                if unr_f:
                    for dst_id, _ in targets:
                        unr_writer.writerow([src_id, dst_id, "no valid cost pixels in window"])  # type: ignore
                skipped_count += len(targets)
                return {
                    "count": 0,
                    "skipped": skipped_count,
                    "gjl": gjl_path,
                    "idx": idx_path,
                    "unr": unr_path,
                    "counts": None,
                    "features": features_mem or [],
                }
            arr = np.where(np.isnan(arr), np.float32(barrier_cost) if use_float32 else float(barrier_cost), arr)
            arr = arr * (np.float32(pixel_size) if use_float32 else float(pixel_size))
            counts_arr = np.zeros(arr.shape, dtype=np.uint32)

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
                def _to_ll(xy):
                    return (float(xy[0]), float(xy[1]))

            # Reconstruct each target path
            for (dst_id, rc_dst), end in zip(targets, ends):
                r_end, c_end = end
                cval = float(costs[r_end, c_end])
                if not np.isfinite(cval):
                    if unr_f:
                        unr_writer.writerow([src_id, dst_id, "unreachable"])  # type: ignore
                    skipped_count += 1
                    continue
                try:
                    path_rc = mcp.traceback(end)
                except Exception:
                    if unr_f:
                        unr_writer.writerow([src_id, dst_id, "traceback failure"])  # type: ignore
                    skipped_count += 1
                    continue

                # RC → raster CRS xy → WGS84
                coords_xy = [_rc_to_xy(win_transform, r, c) for r, c in path_rc]
                coords_ll = [_to_ll(xy) for xy in coords_xy]

                n_vertices_full = len(coords_ll)
                length_m = _geodesic_length_m(coords_ll)

                # Optional simplification (visual-only)
                coords_out = _simplify_coords(coords_ll, simplify_tol_m, max_vertices)

                # Stream GeoJSONL
                if gjl_f:
                    feat = {
                        "type": "Feature",
                        "properties": {
                            "source": src_id,
                            "target": dst_id,
                            "cost": cval,
                            "length_m": float(length_m),
                            "n_vertices": int(n_vertices_full),
                        },
                        "geometry": {"type": "LineString", "coordinates": [(float(x), float(y)) for x, y in coords_out]},
                    }
                    gjl_f.write(json.dumps(feat) + "\n")

                # Stream CSV index
                if idx_f:
                    idx_writer.writerow([src_id, dst_id, cval, float(length_m), int(n_vertices_full)])  # type: ignore

                # Optional in-RAM feature collect (not recommended at scale)
                if features_mem is not None:
                    features_mem.append(feat)

                count += 1

                for r, c in path_rc:
                    counts_arr[r, c] += 1

        if tmp_counts_dir and counts_arr.any():
            os.makedirs(tmp_counts_dir, exist_ok=True)
            counts_path = os.path.join(tmp_counts_dir, f"counts_src_{src_index:06d}.npy")
            np.save(counts_path, counts_arr)
            counts_meta = {
                "path": counts_path,
                "row_off": r_off,
                "col_off": c_off,
                "height": counts_arr.shape[0],
                "width": counts_arr.shape[1],
            }
    finally:
        for fh in (gjl_f, idx_f, unr_f):
            try:
                fh and fh.close()
            except Exception:
                pass

    return {
        "count": count,
        "skipped": skipped_count,
        "gjl": gjl_path,
        "idx": idx_path,
        "unr": unr_path,
        "counts": counts_meta,
        "features": features_mem or [],
    }


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
    return_features: bool = False,
    cleanup_tmp: bool = True,
) -> dict:
    """Compute exact least-cost paths for **all unordered pairs (i<j)**, streaming to disk.

    Returns dict with keys: count, skipped_pairs_count, params, geojsonl_path, index_csv_path,
    unreachable_csv_path, nodes_order_path, path_counts_tif, cost_matrix_path, and optionally
    an in-memory `features` list when `return_features=True`.
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
        if raster_crs is None:
            raise ValueError("Cost raster has no CRS; prepare a metric analysis raster first.")
        crs_obj = CRS.from_user_input(raster_crs)
        units = [a.unit_name.lower() for a in crs_obj.axis_info if a.unit_name]
        if not crs_obj.is_projected or len(units) < 2 or not all(
            "metre" in u or "meter" in u for u in units[:2]
        ):
            raise ValueError("LCP requires a projected raster with metre units; run metric raster preparation.")
        if (transform.a <= 0 or transform.e >= 0 or
                abs(transform.b) > 1e-8 or abs(transform.d) > 1e-8 or
                abs(abs(transform.a) - abs(transform.e)) >
                1e-6 * max(abs(transform.a), abs(transform.e), 1)):
            raise ValueError("LCP requires a north-up, square-cell metric raster.")
        pixel_size = _estimate_pixel_size(transform)

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

    # Build unordered pair list i<j and group by source id
    by_src: dict[str, list[tuple[str, tuple[int, int]]]] = {}
    n = len(ids)
    for i in range(n):
        a = ids[i]
        for j in range(i + 1, n):
            b = ids[j]
            by_src.setdefault(a, []).append((b, rc_lookup[b]))

    # Prepare output dirs/paths
    features_mem_all = [] if return_features else None
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        tmp_geojsonl_dir = os.path.join(outdir, "lcps_tmp")
        tmp_index_dir = os.path.join(outdir, "lcp_index_tmp")
        tmp_unreach_dir = os.path.join(outdir, "unreach_tmp")
        tmp_counts_dir = os.path.join(outdir, "lcp_counts_tmp")
        os.makedirs(tmp_counts_dir, exist_ok=True)
        mmap_path = os.path.join(tmp_counts_dir, "counts_memmap.dat")
        counts_mem = np.memmap(mmap_path, dtype=np.uint32, mode="w+", shape=(H, W))
        counts_mem[:] = 0
        final_geojsonl = os.path.join(outdir, "lcps_all.geojsonl")
        final_index_csv = os.path.join(outdir, "lcp_index.csv")
        final_unreach_csv = os.path.join(outdir, "unreachable_pairs.csv")
        final_count_tif = os.path.join(outdir, "lcp_path_counts.tif")
        nodes_order_csv = os.path.join(outdir, "nodes_order.csv")
        matrix_csv = None
    else:
        tmp_geojsonl_dir = tmp_index_dir = tmp_unreach_dir = tmp_counts_dir = None
        mmap_path = None
        counts_mem = None
        final_geojsonl = final_index_csv = final_unreach_csv = final_count_tif = nodes_order_csv = None
        matrix_csv = None

    # Modest cap for Spaces
    if n_jobs is None or n_jobs < 1:
        n_jobs = 1
    n_jobs = min(int(n_jobs), max(1, (os.cpu_count() or 2) // 2), 8)

    # Solve per source (optionally in parallel)
    total_paths = 0
    total_skipped = 0
    worker_files: list[dict] = []

    if n_jobs > 1 and len(by_src) > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = []
            for src_index, src_id in enumerate(ids):
                targets = by_src.get(src_id)
                if not targets:
                    continue
                fut = ex.submit(
                    _solve_for_source,
                    src_index,
                    src_id,
                    rc_lookup[src_id],
                    targets,
                    cost_tif,
                    int(crop_base),
                    float(crop_alpha),
                    int(crop_min),
                    bool(fully_connected),
                    bool(geometric),
                    float(pixel_size),
                    float(barrier_cost),
                    bool(use_float32),
                    simplify_tol_m,
                    max_vertices,
                    tmp_geojsonl_dir,
                    tmp_index_dir,
                    tmp_unreach_dir,
                    tmp_counts_dir,
                    bool(return_features),
                )
                futs.append((src_index, fut))
            for src_index, fut in futs:
                res = fut.result()
                total_paths += int(res.get("count", 0))
                total_skipped += int(res.get("skipped", 0))
                worker_files.append(
                    {
                        "src_index": src_index,
                        "gjl": res.get("gjl"),
                        "idx": res.get("idx"),
                        "unr": res.get("unr"),
                        "counts": res.get("counts"),
                    }
                )
                if features_mem_all is not None:
                    features_mem_all.extend(res.get("features", []))
    else:
        for src_index, src_id in enumerate(ids):
            targets = by_src.get(src_id)
            if not targets:
                continue
            res = _solve_for_source(
                src_index,
                src_id,
                rc_lookup[src_id],
                targets,
                cost_tif,
                int(crop_base),
                float(crop_alpha),
                int(crop_min),
                bool(fully_connected),
                bool(geometric),
                float(pixel_size),
                float(barrier_cost),
                bool(use_float32),
                simplify_tol_m,
                max_vertices,
                tmp_geojsonl_dir,
                tmp_index_dir,
                tmp_unreach_dir,
                tmp_counts_dir,
                bool(return_features),
            )
            total_paths += int(res.get("count", 0))
            total_skipped += int(res.get("skipped", 0))
            worker_files.append(
                {
                    "src_index": src_index,
                    "gjl": res.get("gjl"),
                    "idx": res.get("idx"),
                    "unr": res.get("unr"),
                    "counts": res.get("counts"),
                }
            )
            if features_mem_all is not None:
                features_mem_all.extend(res.get("features", []))

    # Merge temp files → finals (ordered by src_index)
    if outdir:
        # nodes order
        with open(nodes_order_csv, "w", encoding="utf-8") as fno:
            for nid in ids:
                fno.write(str(nid) + "\n")

        # GEOJSONL
        if final_geojsonl:
            with open(final_geojsonl, "w", buffering=8*1024*1024, encoding="utf-8") as out_gjl:
                for wf in sorted(worker_files, key=lambda x: x["src_index"]):
                    gjl_path = wf.get("gjl")
                    if gjl_path and os.path.exists(gjl_path):
                        with open(gjl_path, "r", encoding="utf-8") as f:
                            shutil.copyfileobj(f, out_gjl)

        # CSV index (write header once)
        if final_index_csv:
            with open(final_index_csv, "w", newline="", encoding="utf-8") as out_csv:
                writer = csv.writer(out_csv)
                writer.writerow(["source", "target", "cost", "length_m", "n_vertices"])
                for wf in sorted(worker_files, key=lambda x: x["src_index"]):
                    idx_path = wf.get("idx")
                    if idx_path and os.path.exists(idx_path):
                        with open(idx_path, "r", newline="", encoding="utf-8") as f:
                            shutil.copyfileobj(f, out_csv)

        # Unreachable report (header once)
        if final_unreach_csv:
            with open(final_unreach_csv, "w", newline="", encoding="utf-8") as out_unr:
                writer = csv.writer(out_unr)
                writer.writerow(["source", "target", "reason"])
                for wf in sorted(worker_files, key=lambda x: x["src_index"]):
                    unr_path = wf.get("unr")
                    if unr_path and os.path.exists(unr_path):
                        with open(unr_path, "r", newline="", encoding="utf-8") as f:
                            shutil.copyfileobj(f, out_unr)

        _append_node_coordinates_to_pair_csv(
            final_index_csv,
            nodes_df,
            id_col=id_col,
            lon_col=lon_col,
            lat_col=lat_col,
        )
        _append_node_coordinates_to_pair_csv(
            final_unreach_csv,
            nodes_df,
            id_col=id_col,
            lon_col=lon_col,
            lat_col=lat_col,
        )

        if counts_mem is not None and final_count_tif:
            for wf in sorted(worker_files, key=lambda x: x["src_index"]):
                counts_info = wf.get("counts")
                if not counts_info:
                    continue
                c_path = counts_info.get("path")
                if not c_path or not os.path.exists(c_path):
                    continue
                row_off = int(counts_info.get("row_off", 0))
                col_off = int(counts_info.get("col_off", 0))
                height = int(counts_info.get("height", 0))
                width = int(counts_info.get("width", 0))
                arr = np.load(c_path)
                counts_mem[row_off:row_off + height, col_off:col_off + width] += arr.astype(np.uint32, copy=False)
            counts_mem.flush()
            with rasterio.open(
                final_count_tif,
                "w",
                driver="GTiff",
                height=H,
                width=W,
                count=1,
                dtype="uint32",
                transform=transform,
                crs=raster_crs,
                nodata=0,
                compress="LZW",
            ) as dst:
                dst.write(np.asarray(counts_mem, dtype=np.uint32), 1)
            del counts_mem
            if mmap_path and os.path.exists(mmap_path):
                os.remove(mmap_path)

        if outdir and final_index_csv and nodes_order_csv:
            try:
                matrix_csv = write_cost_matrix_from_index(
                    final_index_csv,
                    nodes_order_csv,
                    os.path.join(outdir, "lcp_cost_matrix.csv"),
                )
            except Exception:
                matrix_csv = None

        if cleanup_tmp:
            for p in (tmp_geojsonl_dir, tmp_index_dir, tmp_unreach_dir, tmp_counts_dir):
                try:
                    p and shutil.rmtree(p)
                except Exception:
                    pass
    else:
        nodes_order_csv = final_geojsonl = final_index_csv = final_unreach_csv = final_count_tif = matrix_csv = None

    result = {
        "count": int(total_paths),
        "skipped_pairs_count": int(total_skipped),
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
        "geojsonl_path": final_geojsonl,
        "index_csv_path": final_index_csv,
        "unreachable_csv_path": final_unreach_csv,
        "nodes_order_path": nodes_order_csv,
        "path_counts_tif": final_count_tif,
        "cost_matrix_path": matrix_csv,
    }

    if features_mem_all is not None:
        result["features"] = features_mem_all

    return result


__all__ = ["FLCPParams", "compute_all_pairs_lcps", "write_cost_matrix_from_index"]
