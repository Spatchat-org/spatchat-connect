"""Least-cost path workflow helpers."""
from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from analysis_tracker import record_step
from methods.flcp import compute_all_pairs_lcps

Cache = Dict[str, Any]


def _auto_node_ids(count: int) -> list[str]:
    width = max(2, len(str(max(1, count))))
    return [f"node_{i:0{width}d}" for i in range(1, count + 1)]


def detect_nodes_lonlat(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    cols = {c.lower(): c for c in df.columns}
    for lon_key, lat_key in [("longitude", "latitude"), ("lon", "lat"), ("x", "y")]:
        if lon_key in cols and lat_key in cols:
            lon_col, lat_col = cols[lon_key], cols[lat_key]
            try:
                lon_ok = pd.to_numeric(df[lon_col], errors="coerce").between(-180, 180).all()
                lat_ok = pd.to_numeric(df[lat_col], errors="coerce").between(-90, 90).all()
                if lon_ok and lat_ok:
                    return lon_col, lat_col
            except Exception:
                pass
    return None


def prepare_nodes_for_lcp(cache: Cache) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str], Optional[str], Optional[str]]:
    nodes_df = cache.get("nodes_df")
    if not isinstance(nodes_df, pd.DataFrame) or nodes_df.empty:
        return None, None, None, None, "Upload a nodes CSV before running least-cost paths."

    guess = detect_nodes_lonlat(nodes_df)
    if not guess:
        return None, None, None, None, "Could not detect longitude/latitude columns in the nodes CSV."

    lon_col, lat_col = guess
    df = nodes_df.copy()
    if len(df) < 2:
        return None, None, None, None, "Need at least two nodes to compute least-cost paths."

    if "node_id" not in df.columns:
        df = df.copy()
        df["node_id"] = _auto_node_ids(len(df))

    cache["nodes_lon_col"] = lon_col
    cache["nodes_lat_col"] = lat_col

    return df, "node_id", lon_col, lat_col, None


def run_lcp(
    cache: Cache,
    output_dir: str,
    job_id: str,
    write_session_snapshot,
) -> Tuple[bool, str, Optional[str]]:
    if not cache.get("raster_path"):
        return False, "Upload a raster surface before running least-cost paths.", None

    cache["lcp_geojsonl"] = None
    cache["lcp_index_csv"] = None
    cache["lcp_unreachable_csv"] = None
    cache["lcp_count_tif"] = None
    cache["lcp_cost_matrix"] = None
    cache["lcp_zip_path"] = None

    nodes_df, id_col, lon_col, lat_col, error = prepare_nodes_for_lcp(cache)
    if error:
        return False, error, None

    surface_type = cache.get("surface_type") or "resistance"
    source_raster = cache.get("raster_path")
    job_dir = os.path.join(output_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)

    analysis_cost_path = cache.get("analysis_cost_tif")
    if analysis_cost_path and not os.path.exists(analysis_cost_path):
        analysis_cost_path = None

    if surface_type in ("suitability", "conductance"):
        if not analysis_cost_path:
            return (
                False,
                "Please convert the raster to a cost surface first (e.g., 'convert to cost 1-100') before running least-cost paths.",
                None,
            )
    else:
        analysis_cost_path = analysis_cost_path or source_raster

    if not analysis_cost_path or not os.path.exists(analysis_cost_path):
        return False, "Cost surface not found. Try converting again before running least-cost paths.", None

    cache["analysis_cost_tif"] = analysis_cost_path

    try:
        result = compute_all_pairs_lcps(
            cost_tif=analysis_cost_path,
            nodes_df=nodes_df,
            id_col=id_col,
            lon_col=lon_col,
            lat_col=lat_col,
            outdir=job_dir,
        )
    except Exception as exc:
        return False, f"LCP computation failed: {exc}", None

    geojson_path = result.get("geojsonl_path")
    index_path = result.get("index_csv_path")
    unreachable_path = result.get("unreachable_csv_path")
    count_tif = result.get("path_counts_tif")
    matrix_path = result.get("cost_matrix_path")
    cache["lcp_geojsonl"] = geojson_path
    cache["lcp_index_csv"] = index_path
    cache["lcp_unreachable_csv"] = unreachable_path
    cache["lcp_count_tif"] = count_tif
    cache["lcp_cost_matrix"] = matrix_path
    cache["download_bundle_dirty"] = True
    cache["results_zip_path"] = None

    inputs = {
        "cost_tif": source_raster,
        "analysis_cost_tif": analysis_cost_path,
        "surface_type": cache.get("surface_type"),
        "nodes_count": int(len(nodes_df)),
        "id_col": id_col,
        "lon_col": lon_col,
        "lat_col": lat_col,
        "outdir": job_dir,
    }
    try:
        node_ids_full = nodes_df[id_col].astype(str).tolist()
        inputs["node_ids_sample"] = node_ids_full[:20]
        if len(node_ids_full) > 20:
            inputs["node_ids_truncated"] = len(node_ids_full) - 20
    except Exception:
        pass
    try:
        inputs["nodes_preview"] = nodes_df[[id_col, lon_col, lat_col]].head(10).to_dict(orient="records")
    except Exception:
        pass

    outputs = {
        "geojsonl_path": geojson_path,
        "index_csv_path": index_path,
        "unreachable_csv_path": unreachable_path,
        "path_counts_tif": count_tif,
        "cost_matrix_path": matrix_path,
    }

    record_step(
        kind="least_cost_paths",
        function="methods.flcp.compute_all_pairs_lcps",
        inputs=inputs,
        outputs=outputs,
        code_obj=compute_all_pairs_lcps,
        session_id=cache.get("session_id"),
    )

    write_session_snapshot(cache, job_dir, result)

    zip_path = shutil.make_archive(job_dir, "zip", job_dir)
    cache["lcp_zip_path"] = zip_path

    count = result.get("count", 0)

    message = (
        f"LCP computation complete. Generated {count} paths. "
        "Click \"Download Results\" to grab the outputs."
    )
    return True, message, zip_path


__all__ = [
    "detect_nodes_lonlat",
    "prepare_nodes_for_lcp",
    "run_lcp",
]
