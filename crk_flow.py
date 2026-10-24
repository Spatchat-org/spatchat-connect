"""Cumulative resistant kernel workflow with shared metric raster preparation."""
from __future__ import annotations
import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
import pandas as pd
from rasterio.warp import Resampling
from crs_utils import prepare_metric_analysis_raster
from lcp_flow import detect_nodes_lonlat
from methods.crk import CRKParams, compute_cumulative_resistant_kernel

Cache = Dict[str, Any]
KERNEL_PREVIEW_LABEL = "Cumulative resistant kernel"


def kernel_preview_label(dispersal_distance_m: float) -> str:
    distance = float(dispersal_distance_m)
    if distance >= 1000 and math.isclose(distance % 1000.0, 0.0, abs_tol=1e-9):
        distance_text = f"{int(round(distance / 1000.0))} km"
    elif distance >= 1000:
        distance_text = f"{distance / 1000.0:g} km"
    else:
        distance_text = f"{distance:g} m"
    return f"{KERNEL_PREVIEW_LABEL} ({distance_text})"


def _auto_node_ids(count: int) -> list[str]:
    width = max(2, len(str(max(1, count))))
    return [f"node_{i:0{width}d}" for i in range(1, count + 1)]


def prepare_nodes_for_crk(cache: Cache) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str], Optional[str], Optional[str]]:
    nodes_df = cache.get("nodes_df")
    if not isinstance(nodes_df, pd.DataFrame) or nodes_df.empty:
        return None, None, None, None, "Upload a nodes CSV before running a cumulative resistant kernel."
    guess = detect_nodes_lonlat(nodes_df)
    if not guess:
        return None, None, None, None, "Could not detect longitude/latitude columns in the nodes CSV."
    lon_col, lat_col = guess
    df = nodes_df.copy()
    if "node_id" not in df.columns:
        df["node_id"] = _auto_node_ids(len(df))
    cache["nodes_lon_col"] = lon_col
    cache["nodes_lat_col"] = lat_col
    return df, "node_id", lon_col, lat_col, None


def run_crk(cache: Cache, output_dir: str, job_id: str, dispersal_distance_m: float) -> Tuple[bool, str, Optional[str]]:
    if not cache.get("raster_path"):
        return False, "Upload a raster surface before running a cumulative resistant kernel.", None
    if not math.isfinite(float(dispersal_distance_m)) or float(dispersal_distance_m) <= 0:
        return False, "Dispersal distance must be a positive finite value.", None
    nodes_df, id_col, lon_col, lat_col, error = prepare_nodes_for_crk(cache)
    if error:
        return False, error, None
    surface_type = cache.get("surface_type") or "resistance"
    source_raster = cache.get("raster_path")
    analysis_cost_path = cache.get("analysis_cost_tif")
    if analysis_cost_path and not os.path.exists(analysis_cost_path):
        analysis_cost_path = None
    if surface_type in ("suitability", "conductance"):
        if not analysis_cost_path:
            return False, "Please convert the raster to a cost surface first before running a cumulative resistant kernel.", None
    else:
        analysis_cost_path = analysis_cost_path or source_raster
    if not analysis_cost_path or not os.path.exists(analysis_cost_path):
        return False, "Cost surface not found. Try converting again before running the kernel.", None
    cache["analysis_cost_tif"] = analysis_cost_path
    job_dir = os.path.join(output_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)
    kernel_tif = os.path.join(job_dir, "cumulative_resistant_kernel.tif")
    summary_json = os.path.join(job_dir, "crk_summary.json")
    params = CRKParams(id_col=id_col, lon_col=lon_col, lat_col=lat_col,
                       cutoff=float(dispersal_distance_m))
    kernel_sigma_m = max(float(dispersal_distance_m) / 3.0, 1.0)
    try:
        metric_path, metadata = prepare_metric_analysis_raster(
            analysis_cost_path, os.path.join(output_dir, "_metric_grid_cache"),
            resampling=Resampling.nearest)
        result = compute_cumulative_resistant_kernel(
            cost_tif=metric_path, nodes_df=nodes_df, bandwidth_meters=kernel_sigma_m,
            params=params, out_tif=kernel_tif, return_surface=False)
    except Exception as exc:
        return False, f"Cumulative resistant kernel failed: {exc}", None
    cache["crk_analysis_cost_tif"] = metric_path
    cache["crk_crs_metadata"] = metadata
    summary = {
        "kernel_surface_tif": kernel_tif,
        "analysis_cost_tif": metric_path,
        "crs_preparation": metadata,
        "dispersal_distance_m": float(dispersal_distance_m),
        "kernel_sigma_m": float(kernel_sigma_m),
        "pixel_size_m": result.get("pixel_size"),
        "kernel": result.get("kernel"),
        "cutoff": result.get("cutoff"),
        "sources_used": result.get("sources_used"),
        "sources_skipped": result.get("sources_skipped"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    except Exception:
        summary_json = None
    preview_rasters = cache.get("preview_rasters")
    if not isinstance(preview_rasters, dict):
        preview_rasters = {}
        cache["preview_rasters"] = preview_rasters
    preview_rasters[kernel_preview_label(dispersal_distance_m)] = kernel_tif
    cache["kernel_surface_tif"] = kernel_tif
    cache["kernel_summary_json"] = summary_json
    cache["download_bundle_dirty"] = True
    cache["results_zip_path"] = None
    pixel_size_m = result.get("pixel_size")
    try:
        pixel_size_m = float(pixel_size_m) if pixel_size_m is not None else None
    except Exception:
        pixel_size_m = None
    compact_note = ""
    if pixel_size_m and dispersal_distance_m <= 3 * pixel_size_m:
        compact_note = f" The raster cell size is about {pixel_size_m:.0f} m, so kernels at this distance will appear compact on the map."
    message = (
        f"Cumulative resistant kernel complete using a dispersal distance of {dispersal_distance_m:g} meters "
        "(treated as the resistance-weighted cost-distance cutoff). "
        "The map now shows the kernel layer by default. Click \"Download Results\" to grab the outputs."
        f"{compact_note}"
    )
    if metadata.get("reprojected"):
        message = f"Prepared metric analysis grid ({metadata['analysis_crs']}). {message}"
    return True, message, kernel_tif


__all__ = ["KERNEL_PREVIEW_LABEL", "kernel_preview_label", "prepare_nodes_for_crk", "run_crk"]
