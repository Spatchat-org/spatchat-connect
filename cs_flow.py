"""Circuit-theory workflow helpers."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from pyproj import CRS

from crs_utils import choose_meter_crs_for_raster, reproject_raster_to_crs
from lcp_flow import detect_nodes_lonlat
from methods.cs import CircuitTheoryParams, compute_effective_resistances

Cache = Dict[str, Any]
CIRCUIT_PREVIEW_LABEL = "Circuit current density"
CIRCUIT_EDGE_PREVIEW_MAX_PAIRS = 500


def _auto_node_ids(count: int) -> list[str]:
    width = max(2, len(str(max(1, count))))
    return [f"node_{i:0{width}d}" for i in range(1, count + 1)]


def _ensure_projected_cost_for_distance(
    cache: Cache,
    cost_path: str,
) -> Tuple[str, Optional[str]]:
    centroid = cache.get("raster_centroid_wgs84")
    lon = lat = None
    if isinstance(centroid, dict):
        lon = centroid.get("longitude")
        lat = centroid.get("latitude")
    centroid_hint = (float(lon), float(lat)) if lon is not None and lat is not None else None
    target_epsg = choose_meter_crs_for_raster(cost_path, centroid_hint=centroid_hint)
    if not target_epsg:
        return cost_path, None

    converted_dir = cache.get("converted_dir") or os.path.dirname(cost_path)
    base = os.path.splitext(os.path.basename(cost_path))[0]
    projected_path = os.path.join(converted_dir, f"{base}_projected_epsg{target_epsg}.tif")
    needs_refresh = (not os.path.exists(projected_path)) or (
        os.path.getmtime(projected_path) < os.path.getmtime(cost_path)
    )
    if needs_refresh:
        reproject_raster_to_crs(cost_path, projected_path, CRS.from_epsg(int(target_epsg)))
    return projected_path, f"Auto-projected cost surface to EPSG:{target_epsg} for meter-consistent circuit analysis."


def _append_node_coordinates_to_pair_csv(
    csv_path: Optional[str],
    nodes_df: pd.DataFrame,
    *,
    id_col: str,
    lon_col: str,
    lat_col: str,
) -> None:
    if not csv_path or not os.path.exists(csv_path):
        return
    if nodes_df is None or nodes_df.empty:
        return
    required = {id_col, lon_col, lat_col}
    if not required.issubset(nodes_df.columns):
        return
    try:
        pairs_df = pd.read_csv(csv_path)
    except Exception:
        return
    if pairs_df.empty:
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

    if "source" in pairs_df.columns and "target" in pairs_df.columns:
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
        return

    if "node_id" in pairs_df.columns:
        merged = pairs_df.merge(coords_df, on="node_id", how="left")
        merged.to_csv(csv_path, index=False)


def prepare_nodes_for_circuit(
    cache: Cache,
) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str], Optional[str], Optional[str]]:
    nodes_df = cache.get("nodes_df")
    if not isinstance(nodes_df, pd.DataFrame) or nodes_df.empty:
        return None, None, None, None, "Upload a nodes CSV before running circuit analysis."

    guess = detect_nodes_lonlat(nodes_df)
    if not guess:
        return None, None, None, None, "Could not detect longitude/latitude columns in the nodes CSV."

    lon_col, lat_col = guess
    df = nodes_df.copy()
    if len(df) < 2:
        return None, None, None, None, "Need at least two nodes to compute circuit effective resistance."

    if "node_id" not in df.columns:
        df = df.copy()
        df["node_id"] = _auto_node_ids(len(df))

    cache["nodes_lon_col"] = lon_col
    cache["nodes_lat_col"] = lat_col
    return df, "node_id", lon_col, lat_col, None


def run_circuit(
    cache: Cache,
    output_dir: str,
    job_id: str,
    write_session_snapshot,
) -> Tuple[bool, str, Optional[str]]:
    if not cache.get("raster_path"):
        return False, "Upload a raster surface before running circuit analysis.", None

    cache["circuit_pairs_csv"] = None
    cache["circuit_pairs_df"] = None
    cache["circuit_unplaced_csv"] = None
    cache["circuit_summary_json"] = None
    cache["circuit_current_density_tif"] = None

    nodes_df, id_col, lon_col, lat_col, error = prepare_nodes_for_circuit(cache)
    if error:
        return False, error, None

    surface_type = cache.get("surface_type") or "resistance"
    source_raster = cache.get("raster_path")
    analysis_cost_path = cache.get("analysis_cost_tif")
    if analysis_cost_path and not os.path.exists(analysis_cost_path):
        analysis_cost_path = None

    if surface_type in ("suitability", "conductance"):
        if not analysis_cost_path:
            return (
                False,
                "Please convert the raster to a cost surface first before running circuit analysis.",
                None,
            )
    else:
        analysis_cost_path = analysis_cost_path or source_raster

    if not analysis_cost_path or not os.path.exists(analysis_cost_path):
        return False, "Cost surface not found. Try converting again before running circuit analysis.", None

    cache["analysis_cost_tif"] = analysis_cost_path

    job_dir = os.path.join(output_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)
    summary_json = os.path.join(job_dir, "circuit_summary.json")

    params = CircuitTheoryParams(
        id_col=id_col,
        lon_col=lon_col,
        lat_col=lat_col,
    )

    circuit_cost_path = analysis_cost_path
    projection_note = None
    try:
        circuit_cost_path, projection_note = _ensure_projected_cost_for_distance(cache, analysis_cost_path)
    except Exception:
        circuit_cost_path = analysis_cost_path
        projection_note = None

    try:
        result = compute_effective_resistances(
            cost_tif=circuit_cost_path,
            nodes_df=nodes_df,
            params=params,
            outdir=job_dir,
        )
    except Exception as exc:
        return False, f"Circuit analysis failed: {exc}", None

    pairs_csv = result.get("pairs_csv")
    unplaced_csv = result.get("unplaced_csv")
    current_density_tif = result.get("current_density_tif")
    _append_node_coordinates_to_pair_csv(
        pairs_csv,
        nodes_df,
        id_col=id_col,
        lon_col=lon_col,
        lat_col=lat_col,
    )
    _append_node_coordinates_to_pair_csv(
        unplaced_csv,
        nodes_df,
        id_col=id_col,
        lon_col=lon_col,
        lat_col=lat_col,
    )

    pairs_df = result.get("pairs")
    if not isinstance(pairs_df, pd.DataFrame):
        pairs_df = pd.DataFrame()
    unplaced_df = result.get("unplaced_nodes")
    if not isinstance(unplaced_df, pd.DataFrame):
        unplaced_df = pd.DataFrame()
    disconnected_df = result.get("disconnected_pairs")
    if not isinstance(disconnected_df, pd.DataFrame):
        disconnected_df = pd.DataFrame()

    summary = {
        "circuit_pairs_csv": pairs_csv,
        "circuit_unplaced_csv": unplaced_csv,
        "circuit_disconnected_csv": result.get("disconnected_csv"),
        "circuit_current_density_tif": current_density_tif,
        "pair_count": int(len(pairs_df)),
        "unplaced_count": int(len(unplaced_df)),
        "disconnected_pairs_count": int(len(disconnected_df)),
        "laplacian_size": int(result.get("laplacian_size") or 0),
        "effective_resistance_min": float(pairs_df["effective_resistance"].min()) if "effective_resistance" in pairs_df.columns and not pairs_df.empty else None,
        "effective_resistance_max": float(pairs_df["effective_resistance"].max()) if "effective_resistance" in pairs_df.columns and not pairs_df.empty else None,
        "effective_resistance_mean": float(pairs_df["effective_resistance"].mean()) if "effective_resistance" in pairs_df.columns and not pairs_df.empty else None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    except Exception:
        summary_json = None

    cache["circuit_pairs_csv"] = pairs_csv
    cache["circuit_pairs_df"] = pairs_df.copy() if len(pairs_df) <= CIRCUIT_EDGE_PREVIEW_MAX_PAIRS else None
    cache["circuit_unplaced_csv"] = unplaced_csv
    cache["circuit_summary_json"] = summary_json
    cache["circuit_current_density_tif"] = current_density_tif
    preview_rasters = cache.get("preview_rasters")
    if not isinstance(preview_rasters, dict):
        preview_rasters = {}
        cache["preview_rasters"] = preview_rasters
    if current_density_tif and os.path.exists(current_density_tif):
        preview_rasters[CIRCUIT_PREVIEW_LABEL] = current_density_tif
    cache["download_bundle_dirty"] = True
    cache["results_zip_path"] = None

    snapshot_result = {
        "pairs_csv": pairs_csv,
        "unplaced_csv": unplaced_csv,
        "disconnected_csv": result.get("disconnected_csv"),
        "current_density_tif": current_density_tif,
        "pair_count": int(len(pairs_df)),
        "unplaced_count": int(len(unplaced_df)),
        "disconnected_pairs_count": int(len(disconnected_df)),
        "laplacian_size": int(result.get("laplacian_size") or 0),
    }
    write_session_snapshot(cache, job_dir, snapshot_result)

    if len(disconnected_df):
        message = (
            f"Circuit analysis complete. Computed effective resistance for {len(pairs_df)} connected node pairs, "
            f"and {len(disconnected_df)} cross-component pairs were reported as disconnected. "
            "A current-density raster was added to the map. Click \"Download Results\" to grab the outputs."
        )
    else:
        message = (
            f"Circuit analysis complete. Computed effective resistance for {len(pairs_df)} node pairs and added a current-density raster to the map. "
            "Click \"Download Results\" to grab the outputs."
        )
    if projection_note:
        message = f"{projection_note} {message}"
    if len(pairs_df) > CIRCUIT_EDGE_PREVIEW_MAX_PAIRS:
        message = (
            f"{message} The straight-line pair overlay was skipped in the preview to keep the app responsive."
        )
    return True, message, pairs_csv


__all__ = [
    "CIRCUIT_PREVIEW_LABEL",
    "prepare_nodes_for_circuit",
    "run_circuit",
]
