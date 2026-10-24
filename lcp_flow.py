"""Least-cost path workflow helpers with metric raster preparation."""
from __future__ import annotations
import os
import shutil
from typing import Any, Dict, Optional, Tuple
import pandas as pd
from rasterio.warp import Resampling
from crs_utils import prepare_metric_analysis_raster
from analysis_tracker import record_step
from methods.flcp import compute_all_pairs_lcps

Cache = Dict[str, Any]


def _auto_node_ids(count: int) -> list[str]:
    width = max(2, len(str(max(1, count))))
    return [f"node_{i:0{width}d}" for i in range(1, count + 1)]


def detect_nodes_lonlat(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    cols = {str(c).lower(): c for c in df.columns}
    for lon_key, lat_key in [('longitude', 'latitude'), ('lon', 'lat'), ('x', 'y')]:
        if lon_key in cols and lat_key in cols:
            lon_col, lat_col = cols[lon_key], cols[lat_key]
            lon = pd.to_numeric(df[lon_col], errors='coerce')
            lat = pd.to_numeric(df[lat_col], errors='coerce')
            if lon.notna().all() and lat.notna().all() and lon.between(-180, 180).all() and lat.between(-90, 90).all():
                return lon_col, lat_col
    return None


def prepare_nodes_for_lcp(cache: Cache):
    nodes_df = cache.get('nodes_df')
    if not isinstance(nodes_df, pd.DataFrame) or nodes_df.empty:
        return None, None, None, None, 'Upload a nodes CSV before running least-cost paths.'
    guess = detect_nodes_lonlat(nodes_df)
    if not guess:
        return None, None, None, None, 'Could not detect WGS84 longitude/latitude columns in the nodes CSV.'
    lon_col, lat_col = guess
    df = nodes_df.copy()
    if len(df) < 2:
        return None, None, None, None, 'Need at least two nodes to compute least-cost paths.'
    if 'node_id' not in df.columns:
        df['node_id'] = _auto_node_ids(len(df))
    cache['nodes_lon_col'], cache['nodes_lat_col'] = lon_col, lat_col
    return df, 'node_id', lon_col, lat_col, None


def run_lcp(cache: Cache, output_dir: str, job_id: str, write_session_snapshot):
    if not cache.get('raster_path'):
        return False, 'Upload a raster surface before running least-cost paths.', None
    for key in ('lcp_geojsonl', 'lcp_index_csv', 'lcp_unreachable_csv', 'lcp_count_tif',
                'lcp_cost_matrix', 'lcp_zip_path'):
        cache[key] = None
    nodes_df, id_col, lon_col, lat_col, error = prepare_nodes_for_lcp(cache)
    if error:
        return False, error, None
    surface_type = cache.get('surface_type') or 'resistance'
    source_raster = cache.get('raster_path')
    job_dir = os.path.join(output_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)
    analysis_cost_path = cache.get('analysis_cost_tif')
    if analysis_cost_path and not os.path.exists(analysis_cost_path):
        analysis_cost_path = None
    if surface_type in ('suitability', 'conductance'):
        if not analysis_cost_path:
            return False, "Please convert the raster to a cost surface first (e.g., 'convert to cost 1-100').", None
    else:
        analysis_cost_path = analysis_cost_path or source_raster
    if not analysis_cost_path or not os.path.exists(analysis_cost_path):
        return False, 'Cost surface not found. Try converting again before running least-cost paths.', None
    # Preserve the original converted cost surface in cache. The metric grid is
    # an LCP-specific analysis copy, not a replacement for the uploaded surface.
    try:
        metric_path, crs_metadata = prepare_metric_analysis_raster(
            analysis_cost_path, os.path.join(output_dir, '_metric_grid_cache'),
            resampling=Resampling.nearest)
        result = compute_all_pairs_lcps(
            cost_tif=metric_path, nodes_df=nodes_df, id_col=id_col,
            lon_col=lon_col, lat_col=lat_col, outdir=job_dir)
    except Exception as exc:
        return False, f'LCP computation failed: {exc}', None
    cache['lcp_analysis_metric_tif'] = metric_path
    cache['lcp_crs_metadata'] = crs_metadata
    geojson_path = result.get('geojsonl_path')
    index_path = result.get('index_csv_path')
    unreachable_path = result.get('unreachable_csv_path')
    count_tif = result.get('path_counts_tif')
    matrix_path = result.get('cost_matrix_path')
    cache.update(lcp_geojsonl=geojson_path, lcp_index_csv=index_path,
                 lcp_unreachable_csv=unreachable_path, lcp_count_tif=count_tif,
                 lcp_cost_matrix=matrix_path, download_bundle_dirty=True,
                 results_zip_path=None)
    inputs = dict(cost_tif=source_raster, analysis_cost_tif=metric_path,
                  source_cost_tif=analysis_cost_path, crs_preparation=crs_metadata,
                  surface_type=surface_type, nodes_count=int(len(nodes_df)),
                  id_col=id_col, lon_col=lon_col, lat_col=lat_col, outdir=job_dir)
    try:
        ids = nodes_df[id_col].astype(str).tolist()
        inputs['node_ids_sample'] = ids[:20]
        if len(ids) > 20:
            inputs['node_ids_truncated'] = len(ids) - 20
        inputs['nodes_preview'] = nodes_df[[id_col, lon_col, lat_col]].head(10).to_dict(orient='records')
    except Exception:
        pass
    outputs = dict(geojsonl_path=geojson_path, index_csv_path=index_path,
                   unreachable_csv_path=unreachable_path, path_counts_tif=count_tif,
                   cost_matrix_path=matrix_path)
    record_step(kind='least_cost_paths', function='methods.flcp.compute_all_pairs_lcps',
                inputs=inputs, outputs=outputs, code_obj=compute_all_pairs_lcps,
                session_id=cache.get('session_id'))
    write_session_snapshot(cache, job_dir, result)
    zip_path = shutil.make_archive(job_dir, 'zip', job_dir)
    cache['lcp_zip_path'] = zip_path
    return True, (f"LCP computation complete. Generated {result.get('count', 0)} paths. "
                  'Click "Download Results" to grab the outputs.'), zip_path


__all__ = ['detect_nodes_lonlat', 'prepare_nodes_for_lcp', 'run_lcp']
