# app.py

import os
import shutil
import json
import time
import threading
import uuid
import math
from datetime import datetime, timezone
from typing import Optional, Tuple, Union

import inspect, gradio as gr
import pandas as pd
import re
from difflib import get_close_matches
from gradio.context import LocalContext

from analysis_tracker import reset as reset_analysis_steps, set_current_session as set_analysis_session
try:
    from llm_utils import ask_llm, ask_llm_stream
except Exception:  # pragma: no cover
    ask_llm = None
    ask_llm_stream = None

try:
    from gradio_client import utils as grc_utils
except Exception:  # pragma: no cover
    grc_utils = None
    _get_type_original = None
else:
    _get_type_original = getattr(grc_utils, "get_type", None)

# Optional deps for raster preview (bounds only). If rasterio is missing, preview will show empty map with a note.
try:
    import rasterio
except Exception:  # pragma: no cover
    rasterio = None

try:
    from pyproj import CRS, Transformer
except Exception:  # pragma: no cover
    CRS = None
    Transformer = None

import folium
from branca.element import MacroElement, Template
from map_layers import build_preview_from_cache
from conversion_flow import conversion_prompt, process_conversion_chat, execute_surface_conversion, interpret_conversion_message
from lcp_flow import detect_nodes_lonlat, run_lcp
from graph_flow import run_graph_model
from crk_flow import KERNEL_PREVIEW_LABEL, run_crk
from cs_flow import CIRCUIT_PREVIEW_LABEL, run_circuit
from session_snapshot import write_session_snapshot

print("Starting SpatChat: Connectivity Analysis")
reset_analysis_steps()

# Disable Gradio API schema/docs early to avoid known serialization crash
os.environ["GRADIO_SHOW_API"] = "0"
REPRO_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "reproducible_scripts")

# --------------------------------------------------------------------------------------
# In-memory cache template (per Gradio session via gr.State)
# --------------------------------------------------------------------------------------
def _fresh_cache() -> dict:
    return {
        "raster_path": None,
        "raster_bounds": None,   # (minx, miny, maxx, maxy) in raster CRS
        "raster_crs": None,      # proj string or EPSG code
        "raster_width": None,
        "raster_height": None,
        "raster_count": None,
        "raster_resolution": None,
        "raster_bounds_wgs84": None,
        "raster_centroid_wgs84": None,
        "surface_type": None,    # suitability / resistance / conductance
        "awaiting_surface_type": False,
        "last_upload_kind": None,  # "raster" or "nodes"
        "nodes_df": None,        # pandas DataFrame with detected lon/lat columns
        "nodes_lon_col": None,
        "nodes_lat_col": None,
        "lcp_geojsonl": None,
        "lcp_index_csv": None,
        "lcp_unreachable_csv": None,
        "lcp_count_tif": None,
        "lcp_cost_matrix": None,
        "lcp_zip_path": None,
        "circuit_pairs_csv": None,
        "circuit_pairs_df": None,
        "circuit_unplaced_csv": None,
        "circuit_summary_json": None,
        "circuit_current_density_tif": None,
        "graph_metrics_csv": None,
        "graph_metrics_df": None,
        "graph_metric_name": None,
        "graph_edges_df": None,
        "graph_metric_ranges": None,
        "graph_probability_decay": None,
        "graph_summary_json": None,
        "kernel_surface_tif": None,
        "kernel_summary_json": None,
        "kernel_dispersal_distance_m": None,
        "awaiting_kernel_distance": False,
        "pending_post_kernel_analysis": None,
        "results_zip_path": None,
        "analysis_cost_tif": None,
        "session_info": {},
        "awaiting_conversion_choice": False,
        "conversion_target": None,
        "conversion_min_resistance": 1.0,
        "conversion_max_resistance": 100.0,
        "conversion_scale_mode": "linear",
        "conversion_path": None,
        "conversion_label": None,
        "preview_rasters": {},
        "session_id": None,
        "session_root": None,
        "upload_dir": None,
        "output_dir": None,
        "converted_dir": None,
        "download_bundle_dirty": True,
        "out_of_bounds_node_count": 0,
        "out_of_bounds_node_ids": [],
    }

SESSIONS_ROOT = "sessions"
os.makedirs(SESSIONS_ROOT, exist_ok=True)
GLOBAL_OUTPUT_DIR = "outputs"
os.makedirs(GLOBAL_OUTPUT_DIR, exist_ok=True)
_GRADIO_SESSION_MAP: dict[str, str] = {}


def _current_gradio_session_hash() -> Optional[str]:
    try:
        request = LocalContext.request.get()
    except LookupError:
        request = None
    if request is None:
        return None
    session_hash = getattr(request, "session_hash", None)
    if not session_hash:
        return None
    return str(session_hash).strip() or None


def _register_gradio_session(session_id: Optional[str]) -> None:
    sid = str(session_id or "").strip()
    if not sid:
        return
    request_hash = _current_gradio_session_hash()
    if request_hash:
        previous_sid = _GRADIO_SESSION_MAP.get(request_hash)
        _GRADIO_SESSION_MAP[request_hash] = sid
        # Browser refresh can rebind the same session hash to a new session id.
        # Clean the old workspace immediately so disk usage does not accumulate.
        if previous_sid and previous_sid != sid:
            _cleanup_session_workspace(previous_sid)


def _cleanup_session_workspace(session_id: Optional[str]) -> None:
    sid = str(session_id or "").strip()
    if not sid:
        return
    reset_analysis_steps(sid)
    session_root = os.path.join(SESSIONS_ROOT, sid)
    if os.path.exists(session_root):
        shutil.rmtree(session_root, ignore_errors=True)
    for key, value in list(_GRADIO_SESSION_MAP.items()):
        if value == sid:
            _GRADIO_SESSION_MAP.pop(key, None)


def _cleanup_current_browser_session() -> None:
    request_hash = _current_gradio_session_hash()
    if not request_hash:
        return
    sid = _GRADIO_SESSION_MAP.pop(request_hash, None)
    if sid:
        _cleanup_session_workspace(sid)


def _prune_stale_sessions(max_age_seconds: int = 60 * 60 * 12) -> None:
    cutoff = time.time() - max_age_seconds
    try:
        for entry in os.scandir(SESSIONS_ROOT):
            if not entry.is_dir():
                continue
            try:
                if entry.stat().st_mtime < cutoff:
                    _cleanup_session_workspace(entry.name)
            except OSError:
                continue
    except OSError:
        pass


def _start_session_pruner(interval_seconds: int = 10 * 60, max_age_seconds: int = 60 * 60 * 24) -> None:
    """Run stale-session cleanup periodically in a daemon thread."""
    interval_seconds = max(30, int(interval_seconds))
    max_age_seconds = max(60, int(max_age_seconds))

    def _loop() -> None:
        while True:
            try:
                _prune_stale_sessions(max_age_seconds=max_age_seconds)
            except Exception:
                pass
            time.sleep(interval_seconds)

    threading.Thread(target=_loop, name="spatchat-session-pruner", daemon=True).start()


def _ensure_placeholder_download(cache: Optional[dict] = None) -> str:
    output_dir = None
    if isinstance(cache, dict):
        output_dir = cache.get("output_dir")
    base_dir = output_dir or GLOBAL_OUTPUT_DIR
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, "download_results_placeholder.txt")
    if not os.path.exists(path):
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("No downloadable results yet. Run an analysis first.\n")
        except Exception:
            pass
    return path

# --------------------------------------------------------------------------------------
# Monkeypatches for older Gradio builds
# --------------------------------------------------------------------------------------

if grc_utils:
    def _safe_get_type(schema):
        """Handle boolean schemas produced by some Gradio builds."""
        if isinstance(schema, bool):
            return "Any" if schema else "Never"
        if _get_type_original:
            return _get_type_original(schema)
        raise TypeError("Unexpected schema without fallback")

    if _get_type_original:
        grc_utils.get_type = _safe_get_type

    _json_schema_original = getattr(grc_utils, "_json_schema_to_python_type", None)

    def _safe_json_schema_to_python_type(schema, defs=None):
        if isinstance(schema, bool):
            return "Any" if schema else "Never"
        if _json_schema_original:
            return _json_schema_original(schema, defs)
        raise grc_utils.APIInfoParseError(f"Cannot parse schema {schema}")

    if _json_schema_original:
        grc_utils._json_schema_to_python_type = _safe_json_schema_to_python_type

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _file_to_path(file: Union[str, os.PathLike, dict, object]) -> Optional[str]:
    """Return an on-disk path for Gradio uploads across return types."""
    if file is None:
        return None
    if isinstance(file, (str, os.PathLike)):
        candidate = os.fspath(file)
        return candidate if os.path.exists(candidate) else None
    if isinstance(file, dict):
        for key in ("path", "name"):
            candidate = file.get(key)
            if candidate and os.path.exists(candidate):
                return candidate
        return None
    for attr in ("path", "name"):
        candidate = getattr(file, attr, None)
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def _render_empty_map(msg: Optional[str] = None) -> str:
    m = folium.Map(location=[20, 0], zoom_start=2, control_scale=True, tiles=None)
    folium.TileLayer("OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB positron", attr="CartoDB").add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="OpenTopoMap",
        name="Topographic",
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
    ).add_to(m)
    if msg:
        folium.map.Marker([0, 0], popup=msg).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    _add_empty_map_style_patch(m)
    return m._repr_html_()


def _add_empty_map_style_patch(m: folium.Map) -> None:
    root = m.get_root()
    if getattr(root, "_spatchat_empty_map_style_patch_added", False):
        return
    setattr(root, "_spatchat_empty_map_style_patch_added", True)
    patch = MacroElement() # type: ignore
    patch._template = Template(
        """
        {% macro html(this, kwargs) %}
        <style>
          .leaflet-control-layers {
            background: rgba(30, 30, 30, 0.9) !important;
            color: #f5f5f5 !important;
            border-radius: 8px !important;
            border: none !important;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.35) !important;
          }
          .leaflet-control-layers-toggle {
            background-color: rgba(30, 30, 30, 0.9) !important;
            border-radius: 8px !important;
            filter: invert(1) brightness(1.25);
          }
          .leaflet-control-layers-expanded {
            padding: 0 !important;
            background: rgba(30, 30, 30, 0.9) !important;
          }
          .leaflet-control-layers-list,
          .leaflet-control-layers form,
          .leaflet-control-layers .leaflet-control-layers-base,
          .leaflet-control-layers .leaflet-control-layers-overlays,
          .leaflet-control-layers .leaflet-control-layers-separator {
            background: rgba(30, 30, 30, 0.9) !important;
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
        </style>
        {% endmacro %}
        """
    )
    root.add_child(patch)


def _auto_node_ids(count: int) -> list[str]:
    width = max(2, len(str(max(1, count))))
    return [f"node_{i:0{width}d}" for i in range(1, count + 1)]


def _normalize_cached_nodes_df(cache: dict) -> None:
    nodes_df = cache.get("nodes_df")
    if not isinstance(nodes_df, pd.DataFrame) or nodes_df.empty:
        return
    if "node_id" in nodes_df.columns:
        return
    df = nodes_df.copy()
    df["node_id"] = _auto_node_ids(len(df))
    cache["nodes_df"] = df


def _ensure_session_workspace(cache: dict) -> dict:
    session_id = str(cache.get("session_id") or uuid.uuid4().hex)
    session_root = os.path.join(SESSIONS_ROOT, session_id)
    upload_dir = os.path.join(session_root, "uploads")
    output_dir = os.path.join(session_root, "outputs")
    converted_dir = os.path.join(output_dir, "converted")
    for path in (session_root, upload_dir, output_dir, converted_dir):
        os.makedirs(path, exist_ok=True)

    cache["session_id"] = session_id
    cache["session_root"] = session_root
    cache["upload_dir"] = upload_dir
    cache["output_dir"] = output_dir
    cache["converted_dir"] = converted_dir
    _register_gradio_session(session_id)

    session_info = cache.get("session_info")
    if not isinstance(session_info, dict):
        session_info = {}
        cache["session_info"] = session_info
    session_info.setdefault("session_id", session_id)
    return cache


def _mark_results_dirty(cache: dict, *, clear_bundle_path: bool = False) -> None:
    cache["download_bundle_dirty"] = True
    if clear_bundle_path:
        cache["results_zip_path"] = None


def _reset_raster_cache(cache: dict) -> None:
    """Drop cached raster metadata so stale surfaces do not linger between runs."""
    cache["raster_path"] = None
    cache["raster_bounds"] = None
    cache["raster_crs"] = None
    cache["raster_width"] = None
    cache["raster_height"] = None
    cache["raster_count"] = None
    cache["raster_resolution"] = None
    cache["raster_bounds_wgs84"] = None
    cache["raster_centroid_wgs84"] = None
    cache["surface_type"] = None
    cache["awaiting_surface_type"] = False
    cache["awaiting_conversion_choice"] = False
    cache["analysis_cost_tif"] = None
    cache["conversion_target"] = None
    cache["conversion_path"] = None
    cache["conversion_label"] = None
    cache["preview_rasters"] = {}
    cache["kernel_surface_tif"] = None
    cache["kernel_summary_json"] = None
    cache["kernel_dispersal_distance_m"] = None
    cache["awaiting_kernel_distance"] = False
    cache["circuit_pairs_csv"] = None
    cache["circuit_pairs_df"] = None
    cache["circuit_unplaced_csv"] = None
    cache["circuit_summary_json"] = None
    cache["circuit_current_density_tif"] = None
    cache["results_zip_path"] = None
    cache["download_bundle_dirty"] = True
    cache["conversion_min_resistance"] = 1.0
    cache["conversion_max_resistance"] = 100.0
    cache["conversion_scale_mode"] = "linear"
    cache["out_of_bounds_node_count"] = 0
    cache["out_of_bounds_node_ids"] = []
    session_id = cache.get("session_id")
    cache["session_info"] = {"session_id": session_id} if session_id else {}
    reset_analysis_steps(session_id)


def _xy_to_rc(transform, x: float, y: float) -> tuple[int, int]:
    row, col = rasterio.transform.rowcol(transform, x, y, op=math.floor)
    return int(row), int(col)


def _refresh_out_of_bounds_node_cache(cache: dict) -> Optional[int]:
    cache["out_of_bounds_node_count"] = 0
    cache["out_of_bounds_node_ids"] = []

    if rasterio is None or CRS is None or Transformer is None:
        return None

    raster_path = cache.get("raster_path")
    nodes_df = cache.get("nodes_df")
    if not raster_path or not os.path.exists(raster_path):
        return None
    if not isinstance(nodes_df, pd.DataFrame) or nodes_df.empty:
        return None

    lon_col = cache.get("nodes_lon_col")
    lat_col = cache.get("nodes_lat_col")
    if not lon_col or not lat_col or lon_col not in nodes_df.columns or lat_col not in nodes_df.columns:
        return None

    try:
        with rasterio.open(raster_path) as ds:
            raster_crs = ds.crs
            transform = ds.transform
            height = ds.height
            width = ds.width
    except Exception:
        return None

    if raster_crs is None:
        return None

    transformer = None
    try:
        if CRS.from_user_input("EPSG:4326") != raster_crs:
            transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    except Exception:
        return None

    out_of_bounds_ids: list[str] = []
    for row in nodes_df.itertuples(index=False):
        node_id = str(getattr(row, "node_id", ""))
        try:
            x = float(getattr(row, lon_col))
            y = float(getattr(row, lat_col))
        except Exception:
            continue

        try:
            if transformer is not None:
                x, y = transformer.transform(x, y)
            r, c = _xy_to_rc(transform, x, y)
        except Exception:
            continue

        if not (0 <= r < height and 0 <= c < width):
            out_of_bounds_ids.append(node_id or f"row_{len(out_of_bounds_ids) + 1}")

    cache["out_of_bounds_node_count"] = len(out_of_bounds_ids)
    cache["out_of_bounds_node_ids"] = out_of_bounds_ids
    if isinstance(cache.get("session_info"), dict):
        cache["session_info"]["out_of_bounds_node_count"] = len(out_of_bounds_ids)
        cache["session_info"]["out_of_bounds_node_ids"] = out_of_bounds_ids[:50]
    return len(out_of_bounds_ids)


def _out_of_bounds_nodes_message(cache: dict, *, reminder: bool = False) -> Optional[str]:
    count = cache.get("out_of_bounds_node_count") or 0
    if count <= 0:
        return None
    noun = "point was" if count == 1 else "points were"
    if reminder:
        return (
            f"Reminder: {count} uploaded {noun} detected outside the raster extent and "
            "were excluded from the analysis."
        )
    return (
        f"Warning: {count} uploaded {noun} detected outside the raster extent. "
        "If you proceed, those points will be excluded from the analysis."
    )


def _append_out_of_bounds_analysis_reminder(chat_history: list[dict], cache: dict, *, success: bool) -> None:
    if not success:
        return
    reminder = _out_of_bounds_nodes_message(cache, reminder=True)
    if reminder:
        chat_history.append({"role": "assistant", "content": reminder})


def _clear_outputs_dir(output_dir: str, *, preserve: Optional[Tuple[str, ...]] = None) -> None:
    """Remove files/directories inside an output dir, optionally preserving specific paths."""
    try:
        entries = list(os.listdir(output_dir))
    except Exception as exc:
        print(f"Warning: could not list outputs directory '{output_dir}': {exc}")
        return

    preserve_roots: set[str] = set()
    if preserve:
        for item in preserve:
            if not item:
                continue
            preserve_roots.add(os.path.abspath(item))

    for name in entries:
        target = os.path.join(output_dir, name)
        abs_target = os.path.abspath(target)
        if any(abs_target == root or abs_target.startswith(root + os.sep) for root in preserve_roots):
            continue
        try:
            if os.path.isdir(target):
                shutil.rmtree(target)
            else:
                os.remove(target)
        except Exception as exc:
            print(f"Warning: failed to remove output artifact '{target}': {exc}")


def _clear_uploads_dir(upload_dir: str, *, keep: Optional[str] = None) -> None:
    """Remove previously uploaded artifacts from an uploads directory."""
    try:
        entries = list(os.listdir(upload_dir))
    except Exception as exc:
        print(f"Warning: could not list uploads directory '{upload_dir}': {exc}")
        return

    for name in entries:
        if keep and name == keep:
            continue
        target = os.path.join(upload_dir, name)
        try:
            if os.path.isdir(target):
                shutil.rmtree(target)
            else:
                os.remove(target)
        except Exception as exc:
            print(f"Warning: failed to remove upload artifact '{target}': {exc}")


def _repro_manifest_from_cache(cache: dict) -> dict:
    analysis_cost_path = cache.get("analysis_cost_tif")
    raster_path = cache.get("raster_path")
    same_cost_as_input = False
    try:
        if analysis_cost_path and raster_path:
            same_cost_as_input = os.path.abspath(analysis_cost_path) == os.path.abspath(raster_path)
    except Exception:
        same_cost_as_input = analysis_cost_path == raster_path

    return {
        "surface_type": cache.get("surface_type"),
        "nodes_lon_col": cache.get("nodes_lon_col"),
        "nodes_lat_col": cache.get("nodes_lat_col"),
        "input_raster": "input_raster.tif" if raster_path and os.path.exists(raster_path) else None,
        "analysis_cost_tif": (
            "input_raster.tif" if analysis_cost_path and same_cost_as_input
            else "analysis_cost.tif" if analysis_cost_path and os.path.exists(analysis_cost_path)
            else None
        ),
        "nodes_csv": "nodes.csv" if isinstance(cache.get("nodes_df"), pd.DataFrame) and not cache["nodes_df"].empty else None,
        "conversion": {
            "enabled": bool(cache.get("conversion_path")),
            "target": cache.get("conversion_target"),
            "scale_mode": cache.get("conversion_scale_mode"),
            "range": [
                cache.get("conversion_min_resistance"),
                cache.get("conversion_max_resistance"),
            ] if cache.get("conversion_target") == "resistance" else None,
        },
        "lcp": {
            "enabled": bool(cache.get("lcp_index_csv") or cache.get("lcp_count_tif") or cache.get("lcp_geojsonl")),
        },
        "crk": {
            "enabled": bool(cache.get("kernel_surface_tif")),
            "dispersal_distance_m": cache.get("kernel_dispersal_distance_m"),
            "cutoff_m": cache.get("kernel_dispersal_distance_m"),
            "kernel_sigma_m": (
                float(cache.get("kernel_dispersal_distance_m")) / 3.0
                if cache.get("kernel_dispersal_distance_m") not in (None, "")
                else None
            ),
        },
        "circuit": {
            "enabled": bool(cache.get("circuit_pairs_csv") or cache.get("circuit_current_density_tif")),
        },
        "graph": {
            "enabled": bool(cache.get("graph_metrics_csv") or cache.get("graph_summary_json")),
            "selected_metric": cache.get("graph_metric_name"),
        },
    }


def _write_repro_scripts(bundle_dir: str, cache: dict) -> list[str]:
    script_dir = os.path.join(bundle_dir, "reproducible_scripts")
    os.makedirs(script_dir, exist_ok=True)
    created: list[str] = []

    manifest = _repro_manifest_from_cache(cache)
    config_path = os.path.join(script_dir, "repro_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    created.append(config_path)

    raster_path = cache.get("raster_path")
    if raster_path and os.path.exists(raster_path):
        dst = os.path.join(script_dir, "input_raster.tif")
        shutil.copy2(raster_path, dst)
        created.append(dst)

    analysis_cost_path = cache.get("analysis_cost_tif")
    try:
        same_cost_as_input = bool(raster_path and analysis_cost_path and os.path.abspath(raster_path) == os.path.abspath(analysis_cost_path))
    except Exception:
        same_cost_as_input = analysis_cost_path == raster_path
    if analysis_cost_path and os.path.exists(analysis_cost_path) and not same_cost_as_input:
        dst = os.path.join(script_dir, "analysis_cost.tif")
        shutil.copy2(analysis_cost_path, dst)
        created.append(dst)

    nodes_df = cache.get("nodes_df")
    if isinstance(nodes_df, pd.DataFrame) and not nodes_df.empty:
        nodes_path = os.path.join(script_dir, "nodes.csv")
        nodes_df.to_csv(nodes_path, index=False)
        created.append(nodes_path)

    if os.path.exists(os.path.join(REPRO_TEMPLATE_DIR, "README.txt")):
        target = os.path.join(script_dir, "README.txt")
        shutil.copyfile(os.path.join(REPRO_TEMPLATE_DIR, "README.txt"), target)
        created.append(target)

    template_map = {
        "convert": ("convert_reproduce.txt", bool(manifest.get("conversion", {}).get("enabled"))),
        "lcp": ("lcp_reproduce.txt", bool(manifest.get("lcp", {}).get("enabled"))),
        "crk": ("crk_reproduce.txt", bool(manifest.get("crk", {}).get("enabled"))),
        "circuit": ("circuit_reproduce.txt", bool(manifest.get("circuit", {}).get("enabled"))),
        "graph": ("graph_reproduce.txt", bool(manifest.get("graph", {}).get("enabled"))),
    }
    for _, (filename, needed) in template_map.items():
        if not needed:
            continue
        src = os.path.join(REPRO_TEMPLATE_DIR, filename)
        if not os.path.exists(src):
            continue
        dst = os.path.join(script_dir, filename)
        shutil.copyfile(src, dst)
        created.append(dst)

    return created


def _refresh_results_bundle(cache: dict) -> Optional[str]:
    cache = _ensure_session_workspace(cache)
    if not cache.get("download_bundle_dirty") and cache.get("results_zip_path") and os.path.exists(cache["results_zip_path"]):
        return cache.get("results_zip_path")

    output_dir = cache["output_dir"]
    bundle_dir = os.path.join(output_dir, "download_bundle")
    try:
        if os.path.isdir(bundle_dir):
            shutil.rmtree(bundle_dir)
        os.makedirs(bundle_dir, exist_ok=True)
    except Exception:
        cache["results_zip_path"] = None
        return None

    copied_any = False

    def _copy_file(src: Optional[str], dest_name: Optional[str] = None) -> None:
        nonlocal copied_any
        if not src or not os.path.exists(src):
            return
        name = dest_name or os.path.basename(src)
        dst = os.path.join(bundle_dir, name)
        try:
            shutil.copy2(src, dst)
            copied_any = True
        except Exception:
            pass

    _copy_file(cache.get("conversion_path"))
    _copy_file(cache.get("analysis_cost_tif"))
    _copy_file(cache.get("lcp_geojsonl"))
    _copy_file(cache.get("lcp_index_csv"))
    _copy_file(cache.get("lcp_unreachable_csv"))
    _copy_file(cache.get("lcp_count_tif"))
    _copy_file(cache.get("lcp_cost_matrix"))
    _copy_file(cache.get("circuit_pairs_csv"))
    _copy_file(cache.get("circuit_unplaced_csv"))
    _copy_file(cache.get("circuit_summary_json"))
    _copy_file(cache.get("circuit_current_density_tif"))
    _copy_file(cache.get("graph_metrics_csv"))
    _copy_file(cache.get("graph_summary_json"))
    _copy_file(cache.get("kernel_surface_tif"))
    _copy_file(cache.get("kernel_summary_json"))

    preview_rasters = cache.get("preview_rasters")
    if isinstance(preview_rasters, dict):
        for label, path in preview_rasters.items():
            safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", str(label or "preview"))
            ext = os.path.splitext(path or "")[1] or ".tif"
            _copy_file(path, f"{safe_label}{ext}")

    graph_edges_df = cache.get("graph_edges_df")
    if isinstance(graph_edges_df, pd.DataFrame) and not graph_edges_df.empty:
        edges_csv = os.path.join(bundle_dir, "graph_edges_filtered.csv")
        try:
            graph_edges_df.to_csv(edges_csv, index=False)
            copied_any = True
        except Exception:
            pass

    session_info = cache.get("session_info")
    if isinstance(session_info, dict) and session_info:
        session_json = os.path.join(bundle_dir, "session_info.json")
        try:
            with open(session_json, "w", encoding="utf-8") as f:
                json.dump(session_info, f, indent=2, default=str)
            copied_any = True
        except Exception:
            pass

    try:
        repro_paths = _write_repro_scripts(bundle_dir, cache)
        if repro_paths:
            copied_any = True
    except Exception as exc:
        print(f"Warning: failed to write reproducible scripts: {exc}")

    if not copied_any:
        cache["results_zip_path"] = None
        cache["download_bundle_dirty"] = False
        return None

    zip_base = os.path.join(output_dir, "download_bundle")
    zip_path = f"{zip_base}.zip"
    try:
        if os.path.exists(zip_path):
            os.remove(zip_path)
    except Exception:
        pass

    try:
        cache["results_zip_path"] = shutil.make_archive(zip_base, "zip", bundle_dir)
    except Exception:
        cache["results_zip_path"] = None
    cache["download_bundle_dirty"] = False
    return cache.get("results_zip_path")


_SURFACE_TYPE_PATTERNS = [
    (re.compile(r"\bsuitability\b"), "suitability"),
    (re.compile(r"\bhabitat\b"), "suitability"),
    (re.compile(r"\bresistance\b"), "resistance"),
    (re.compile(r"\bcost\b"), "resistance"),
    (re.compile(r"\bcost\s+resistance\b"), "resistance"),
    (re.compile(r"\bcost\s+surface\b"), "resistance"),
    (re.compile(r"\bresistance\s+surface\b"), "resistance"),
    (re.compile(r"\bcost\s+raster\b"), "resistance"),
    (re.compile(r"\bconductance\b"), "conductance"),
    (re.compile(r"\bconductivity\b"), "conductance"),
]

_SURFACE_LABELS = {
    "suitability": "suitability",
    "resistance": "resistance (cost)",
    "conductance": "conductance",
}

_SURFACE_ACKS = {
    "suitability": "I'll convert it to resistance (cost) for path analyses and to conductance for circuit analyses.",
    "resistance": "I'll use it directly as the cost surface.",
    "conductance": "I'll convert it to resistance when a workflow expects costs.",
}


def _surface_prompt() -> str:
    return (
        "Is this raster a suitability, resistance (cost), or conductance surface?\n"
        "Reply with one of those words so I can convert it appropriately for analyses."
    )


def _normalize_surface_type_answer(message: str) -> Optional[str]:
    msg = message.lower()
    analysis_request_patterns = [
        r"\bleast[-\s]*cost\b",
        r"\blcp\b",
        r"\bpath(s)?\b",
        r"\bcorridor(s)?\b",
        r"\bcost distance\b",
        r"\bkernel\b",
        r"\bcircuit\b",
        r"\bcurrent density\b",
        r"\brun\b",
        r"\bcompute\b",
        r"\bi want\b",
    ]
    if any(re.search(pattern, msg) for pattern in analysis_request_patterns):
        return None

    classification_prefixes = (
        "it's ",
        "its ",
        "it is ",
        "this is ",
        "this raster is ",
        "the raster is ",
        "the surface is ",
        "surface is ",
        "raster is ",
    )
    normalized = msg.strip()
    for prefix in classification_prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):].strip()
            break
    normalized = re.sub(r"^(a|an|the)\s+", "", normalized)
    normalized = re.sub(r"\b(surface|raster|map|layer)\b", "", normalized)
    normalized = re.sub(r"[^a-z\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    matches = []
    for pattern, canonical in _SURFACE_TYPE_PATTERNS:
        if pattern.fullmatch(normalized):
            matches.append(canonical)
    # Deduplicate while preserving order
    ordered = []
    for item in matches:
        if item not in ordered:
            ordered.append(item)
    if len(ordered) == 1:
        return ordered[0]
    # Fuzzy match common typos (e.g., "resitance")
    tokens = re.findall(r"[a-z]+", normalized)
    for token in tokens:
        closest = get_close_matches(token, list(_SURFACE_LABELS.keys()), n=1, cutoff=0.8)
        if closest:
            return closest[0]
    return None


def _auto_prepare_surface(cache: dict, surface_type: Optional[str]) -> Tuple[bool, Tuple[str, ...]]:
    """Perform default conversions after the surface type is confirmed.

    Returns (needs_manual_choice, messages). If `needs_manual_choice` is True, the caller
    should prompt the user for manual conversion instructions.
    """
    if not surface_type:
        return False, ()
    cache = _ensure_session_workspace(cache)
    converted_dir = cache["converted_dir"]

    messages: list[str] = []
    needs_manual = False

    if surface_type == "suitability":
        cost_success, cost_msg = execute_surface_conversion(
            cache,
            converted_dir,
            "resistance",
            announce_path=False,
        )
        cost_path = cache.get("analysis_cost_tif") if cost_success else None
        conduct_success, conduct_msg = (False, "Cost surface unavailable for conductance conversion.")
        if cost_path:
            conduct_success, conduct_msg = execute_surface_conversion(
                cache,
                converted_dir,
                "conductance",
                announce_path=False,
                source_path_override=cost_path,
                source_type_override="resistance",
            )

        if cost_success and conduct_success:
            messages.append(
                "Default cost (linear 1-100) and conductance surfaces are ready. "
                "Ask me to 'convert to cost 5-150' or 'use exponential decay' if you want a different resistance range."
            )
        else:
            if cost_success:
                messages.append(
                    "Default cost (linear 1-100) surface is ready. "
                    "You can rescale it anytime with commands like 'convert to cost 5-150'."
                )
            else:
                needs_manual = True
                messages.append(f"Automatic cost conversion failed ({cost_msg}). Try 'convert to cost 5-150'.")

            if conduct_success:
                messages.append("Conductance surface is ready for circuit-style analyses.")
            else:
                messages.append(f"Automatic conductance conversion failed ({conduct_msg}).")

    elif surface_type == "conductance":
        cost_success, cost_msg = execute_surface_conversion(
            cache,
            converted_dir,
            "resistance",
            announce_path=False,
        )
        if cost_success:
            messages.append(
                "Default cost (linear 1-100) surface is ready. Adjust it later with 'convert to cost 5-150' if needed."
            )
        else:
            needs_manual = True
            messages.append(f"Automatic cost conversion failed ({cost_msg}). Try 'convert to cost 5-150'.")

    elif surface_type == "resistance":
        conduct_success, conduct_msg = execute_surface_conversion(
            cache,
            converted_dir,
            "conductance",
            announce_path=False,
        )
        if conduct_success:
            messages.append(
                "A conductance surface is also ready for circuit-style analyses and is shown on the map."
            )
        else:
            messages.append(f"Automatic conductance conversion failed ({conduct_msg}).")

    return needs_manual, tuple(messages)


def _build_preview_map(cache: dict) -> str:
    _normalize_cached_nodes_df(cache)
    return build_preview_from_cache(cache)


def _download_update(cache: dict):
    cache = _ensure_session_workspace(cache)
    zip_path = _refresh_results_bundle(cache)
    return gr.update(value=zip_path or _ensure_placeholder_download(cache), visible=True)


def _analysis_error_message(label: str, exc: Exception) -> str:
    detail = str(exc).strip() or exc.__class__.__name__
    return f"{label} failed: {detail}"


def _detect_intent(message: str) -> Optional[str]:
    intents = _detect_intents(message)
    return intents[0] if intents else None


def _detect_intents(message: str) -> list[str]:
    msg = message.lower()
    question_like = bool(
        msg.strip().endswith("?")
        or re.match(r"\s*(what|which|where|when|why|how|is|are|do|does|did|can|could|should)\b", msg)
    )
    explanation_like = bool(
        re.search(r"\b(what\s+does|what\s+do|mean|means|explain|interpret|interpretation|describe)\b", msg)
    )
    instruction_question = bool(re.match(r"\s*how\s+(can|do|should)\s+i\b", msg))
    action_terms = (
        "run", "compute", "calculate", "create", "generate", "build", "make",
        "analyze", "analyse", "map", "use",
    )
    has_action_term = any(re.search(rf"\b{term}\b", msg) for term in action_terms)
    if (question_like or explanation_like) and (not has_action_term or instruction_question):
        return []

    if any(phrase in msg for phrase in ("run all models", "all models", "run all analyses", "all analyses", "all workflows")):
        return ["run_kernel", "run_lcp", "run_circuit", "run_graph"]
    lcp_keywords = [
        "least cost", "lcp", "run lcp", "run flcp", "factorial lcp", "compute lcp", "paths", "least-cost",
    ]
    kernel_keywords = [
        "resistant kernel", "cumulative resistant kernel", "cumulative kernel", "kernel map", "kernel model",
        "run kernel", "run crk", "compute kernel", "compute crk", " crk ", "crk",
    ]
    circuit_keywords = [
        "circuitscape", "circuit analysis", "effective resistance", "run circuit", "compute circuit", "circuit-style",
    ]
    graph_keywords = [
        "graph model", "graph theory", "graph analysis", "network model", "network analysis",
    ]
    intents: list[str] = []
    if any(keyword in msg for keyword in lcp_keywords):
        intents.append("run_lcp")
    if any(keyword in msg for keyword in kernel_keywords):
        intents.append("run_kernel")
    if any(keyword in msg for keyword in circuit_keywords):
        intents.append("run_circuit")
    if any(keyword in msg for keyword in graph_keywords):
        intents.append("run_graph")
    return intents


def _extract_distance_text(message: str) -> Optional[str]:
    matches = re.findall(r"([0-9]*\.?[0-9]+\s*(?:km|kilometers?|kilometres?|m|meters?|metres?)?)", message.lower())
    for match in matches:
        text = (match or "").strip()
        if text:
            return text
    return None


def _parse_kernel_distance_m(message: str) -> Optional[float]:
    try:
        from methods.crk import parse_bandwidth_text
    except Exception:
        return None
    candidate = _extract_distance_text(message)
    if not candidate:
        return None
    try:
        return float(parse_bandwidth_text(candidate))
    except Exception:
        return None


def _clear_kernel_outputs(cache: dict) -> None:
    preview_rasters = cache.get("preview_rasters")
    if isinstance(preview_rasters, dict):
        for label in list(preview_rasters.keys()):
            if str(label).startswith(KERNEL_PREVIEW_LABEL):
                preview_rasters.pop(label, None)
    cache["kernel_surface_tif"] = None
    cache["kernel_summary_json"] = None
    cache["kernel_dispersal_distance_m"] = None
    cache["awaiting_kernel_distance"] = False
    cache["pending_post_kernel_analysis"] = None


def _clear_circuit_outputs(cache: dict) -> None:
    preview_rasters = cache.get("preview_rasters")
    if isinstance(preview_rasters, dict):
        preview_rasters.pop(CIRCUIT_PREVIEW_LABEL, None)
    cache["circuit_pairs_csv"] = None
    cache["circuit_pairs_df"] = None
    cache["circuit_unplaced_csv"] = None
    cache["circuit_summary_json"] = None
    cache["circuit_current_density_tif"] = None


def _kernel_distance_prompt() -> str:
    return (
        "What dispersal distance should I use for the cumulative resistant kernel? "
        "This will be treated as the cutoff distance, meaning the limit of how far the species can travel under regular permeability. "
        "For example: `500 m` or `2 km`."
    )


def _bounds_tuple(bounds) -> Optional[Tuple[float, float, float, float]]:
    if bounds is None:
        return None
    try:
        return (float(bounds.left), float(bounds.bottom), float(bounds.right), float(bounds.top))
    except Exception:
        pass
    try:
        values = tuple(float(v) for v in bounds)
    except Exception:
        return None
    if len(values) != 4:
        return None
    return values


def _projected_bounds_context(bounds) -> Optional[dict]:
    values = _bounds_tuple(bounds)
    if values is None:
        return None
    left, bottom, right, top = values
    return {
        "left": left,
        "bottom": bottom,
        "right": right,
        "top": top,
    }


def _set_raster_wgs84_location(cache: dict, bounds, crs) -> bool:
    cache["raster_bounds_wgs84"] = None
    cache["raster_centroid_wgs84"] = None
    values = _bounds_tuple(bounds)
    if values is None or CRS is None or Transformer is None or crs is None:
        return False

    left, bottom, right, top = values
    try:
        source_crs = CRS.from_user_input(crs)
    except Exception:
        return False

    west = south = east = north = None
    try:
        from rasterio.warp import transform_bounds

        west, south, east, north = transform_bounds(
            source_crs,
            "EPSG:4326",
            left,
            bottom,
            right,
            top,
            densify_pts=21,
        )
    except Exception:
        try:
            transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
            corners = [
                transformer.transform(left, bottom),
                transformer.transform(left, top),
                transformer.transform(right, bottom),
                transformer.transform(right, top),
            ]
            lon_values = [lon for lon, _lat in corners]
            lat_values = [lat for _lon, lat in corners]
            west, east = min(lon_values), max(lon_values)
            south, north = min(lat_values), max(lat_values)
        except Exception:
            return False

    try:
        transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
        centroid_lon, centroid_lat = transformer.transform((left + right) / 2.0, (bottom + top) / 2.0)
    except Exception:
        centroid_lon = (float(west) + float(east)) / 2.0
        centroid_lat = (float(south) + float(north)) / 2.0

    cache["raster_bounds_wgs84"] = {
        "west_longitude": float(west),
        "south_latitude": float(south),
        "east_longitude": float(east),
        "north_latitude": float(north),
    }
    cache["raster_centroid_wgs84"] = {
        "longitude": float(centroid_lon),
        "latitude": float(centroid_lat),
    }
    return True


def _refresh_raster_metadata(cache: dict) -> bool:
    raster_path = cache.get("raster_path")
    if rasterio is None or not raster_path or not os.path.exists(raster_path):
        return False
    try:
        with rasterio.open(raster_path) as ds:
            cache["raster_bounds"] = ds.bounds
            cache["raster_crs"] = str(ds.crs) if ds.crs else None
            cache["raster_width"] = int(ds.width)
            cache["raster_height"] = int(ds.height)
            cache["raster_count"] = int(ds.count)
            cache["raster_resolution"] = tuple(float(v) for v in ds.res)
            _set_raster_wgs84_location(cache, ds.bounds, ds.crs)
    except Exception:
        return False

    session = cache.get("session_info")
    if isinstance(session, dict):
        session["raster_bounds"] = cache["raster_bounds"]
        session["raster_crs"] = cache["raster_crs"]
        session["raster_width"] = cache["raster_width"]
        session["raster_height"] = cache["raster_height"]
        session["raster_count"] = cache["raster_count"]
        session["raster_resolution"] = cache["raster_resolution"]
        session["raster_bounds_wgs84"] = cache["raster_bounds_wgs84"]
        session["raster_centroid_wgs84"] = cache["raster_centroid_wgs84"]
    return True


def _to_jsonable(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return _to_jsonable(value.item())
        except Exception:
            pass
    return str(value)


def _read_json_context(path: Optional[str]) -> Optional[dict]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    return _to_jsonable(data) if isinstance(data, dict) else {"value": _to_jsonable(data)}


def _dataframe_context(df: Optional[pd.DataFrame], max_rows: int = 10) -> Optional[dict]:
    if not isinstance(df, pd.DataFrame):
        return None
    sample = df.head(max_rows).copy()
    sample = sample.astype(object).where(pd.notna(sample), None)
    return {
        "row_count": int(len(df)),
        "columns": [str(col) for col in df.columns],
        "preview_rows": _to_jsonable(sample.to_dict(orient="records")),
    }


def _csv_context(path: Optional[str], max_rows: int = 10) -> Optional[dict]:
    if not path or not os.path.exists(path):
        return None
    try:
        preview = pd.read_csv(path, nrows=max_rows)
    except Exception:
        return None
    row_count = None
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            row_count = max(sum(1 for _ in f) - 1, 0)
    except Exception:
        pass
    ctx = _dataframe_context(preview, max_rows=max_rows) or {}
    if row_count is not None:
        ctx["row_count"] = int(row_count)
    ctx["path_name"] = os.path.basename(path)
    return ctx


def _build_llm_context(cache: dict) -> dict:
    if cache.get("raster_path") and (cache.get("raster_width") is None or cache.get("raster_height") is None):
        _refresh_raster_metadata(cache)
    session_info = cache.get("session_info") if isinstance(cache.get("session_info"), dict) else {}
    nodes_df = cache.get("nodes_df")
    preview_rasters = cache.get("preview_rasters") if isinstance(cache.get("preview_rasters"), dict) else {}
    raster_width = cache.get("raster_width")
    raster_height = cache.get("raster_height")
    raster_count = cache.get("raster_count")

    context = {
        "has_raster": bool(cache.get("raster_path")),
        "raster_name": session_info.get("raster_name"),
        "raster_crs": cache.get("raster_crs"),
        "raster_bounds": cache.get("raster_bounds"),
        "raster_bounds_projected": _projected_bounds_context(cache.get("raster_bounds")),
        "raster_dimensions": {
            "width_columns": int(raster_width) if raster_width is not None else None,
            "height_rows": int(raster_height) if raster_height is not None else None,
            "band_count": int(raster_count) if raster_count is not None else None,
        },
        "raster_resolution": cache.get("raster_resolution"),
        "raster_location": {
            "centroid_wgs84": cache.get("raster_centroid_wgs84"),
            "bounds_wgs84": cache.get("raster_bounds_wgs84"),
            "note": (
                "Coordinates are derived from raster bounds. A precise place name requires reverse geocoding "
                "or an administrative boundary lookup."
            ),
        },
        "surface_type": cache.get("surface_type"),
        "has_cost_surface": bool(cache.get("analysis_cost_tif")),
        "has_converted_surface": bool(cache.get("conversion_path")),
        "conversion_target": cache.get("conversion_target"),
        "conversion_scale_mode": cache.get("conversion_scale_mode"),
        "conversion_range": {
            "min": cache.get("conversion_min_resistance"),
            "max": cache.get("conversion_max_resistance"),
        },
        "has_nodes": isinstance(nodes_df, pd.DataFrame) and not nodes_df.empty,
        "nodes_count": int(len(nodes_df)) if isinstance(nodes_df, pd.DataFrame) else 0,
        "node_columns": list(nodes_df.columns) if isinstance(nodes_df, pd.DataFrame) else [],
        "nodes_lon_col": cache.get("nodes_lon_col"),
        "nodes_lat_col": cache.get("nodes_lat_col"),
        "nodes": _dataframe_context(nodes_df, max_rows=10),
        "lcp_ready": bool(cache.get("lcp_geojsonl") or cache.get("lcp_count_tif") or cache.get("lcp_cost_matrix")),
        "circuit_ready": bool(cache.get("circuit_pairs_csv") or cache.get("circuit_summary_json") or cache.get("circuit_current_density_tif")),
        "graph_ready": bool(cache.get("graph_metrics_csv") or cache.get("graph_summary_json")),
        "result_layer_meanings": {
            "least_cost_paths": (
                "Least-cost path edges are pairwise connections between nodes that minimize accumulated cost "
                "through the resistance/cost raster. The edge cost is not straight-line distance; it reflects "
                "the modeled movement cost along the cheapest route."
            ),
            "graph_edges": (
                "Graph model edges are network links derived from the least-cost path results. In this app, "
                "least-cost links are converted to a graph and filtered by probability decay: higher-cost links "
                "receive lower movement probability, and links below the probability threshold are dropped. "
                "Use analysis_results.graph.probability_decay for how many links were retained."
            ),
            "graph_node_metric": (
                "The graph node layer is colored by the selected graph metric. Betweenness highlights nodes "
                "that sit on many shortest routes through the network; high values can indicate potential "
                "stepping-stones, bottlenecks, or connector sites."
            ),
            "circuit_current_density": (
                "Circuit current density shows where current concentrates across the conductance surface when "
                "nodes are connected as circuit terminals. Higher current suggests areas important for multiple "
                "potential movement routes."
            ),
            "kernel_surface": (
                "The cumulative resistant kernel surface sums cost-limited dispersal influence from all source "
                "nodes. Higher values indicate places reachable by more or stronger source contributions under "
                "the selected dispersal distance."
            ),
        },
        "analysis_results": {
            "least_cost_paths": {
                "ready": bool(cache.get("lcp_geojsonl") or cache.get("lcp_count_tif") or cache.get("lcp_cost_matrix")),
                "index": _csv_context(cache.get("lcp_index_csv"), max_rows=10),
                "unreachable": _csv_context(cache.get("lcp_unreachable_csv"), max_rows=10),
                "cost_matrix": _csv_context(cache.get("lcp_cost_matrix"), max_rows=8),
                "outputs": {
                    "geojsonl": os.path.basename(cache["lcp_geojsonl"]) if cache.get("lcp_geojsonl") else None,
                    "path_counts_tif": os.path.basename(cache["lcp_count_tif"]) if cache.get("lcp_count_tif") else None,
                    "zip": os.path.basename(cache["lcp_zip_path"]) if cache.get("lcp_zip_path") else None,
                },
            },
            "circuit": {
                "ready": bool(cache.get("circuit_pairs_csv") or cache.get("circuit_summary_json") or cache.get("circuit_current_density_tif")),
                "summary": _read_json_context(cache.get("circuit_summary_json")),
                "pairs": _dataframe_context(cache.get("circuit_pairs_df"), max_rows=10) or _csv_context(cache.get("circuit_pairs_csv"), max_rows=10),
                "unplaced": _csv_context(cache.get("circuit_unplaced_csv"), max_rows=10),
                "current_density_layer": bool(cache.get("circuit_current_density_tif")),
            },
            "kernel": {
                "ready": bool(cache.get("kernel_surface_tif") or cache.get("kernel_summary_json")),
                "summary": _read_json_context(cache.get("kernel_summary_json")),
                "surface_layer": bool(cache.get("kernel_surface_tif")),
            },
            "graph": {
                "ready": bool(cache.get("graph_metrics_csv") or cache.get("graph_summary_json")),
                "summary": _read_json_context(cache.get("graph_summary_json")),
                "selected_metric": cache.get("graph_metric_name"),
                "metric_ranges": cache.get("graph_metric_ranges"),
                "probability_decay": cache.get("graph_probability_decay"),
                "node_metrics": _dataframe_context(cache.get("graph_metrics_df"), max_rows=10) or _csv_context(cache.get("graph_metrics_csv"), max_rows=10),
                "edges": _dataframe_context(cache.get("graph_edges_df"), max_rows=10),
            },
        },
        "preview_layers": list(preview_rasters.keys()),
    }
    return _to_jsonable(context)


def _local_fallback_reply(cache: dict) -> str:
    if cache.get("raster_path"):
        return (
            "The language model is unavailable right now. "
            "I can still help with the loaded data through the built-in workflows: surface setup, least-cost paths, circuit analysis, and the graph workflow."
        )
    return (
        "The language model is unavailable right now. "
        "You can still use the built-in workflows by uploading a raster and optionally a node CSV."
    )


def _status_message(label: str, started_at: Optional[float] = None) -> str:
    _ = started_at
    suffix = "..." if not str(label).endswith("...") else ""
    return f"<span class='spatchat-status'><span class='spatchat-status-dot'></span>{label}{suffix}</span>"


def _status_clear_update():
    return gr.update(value="")


def _run_with_status(chat_history, cache, status_label: str, fn):
    started_at = time.time()
    status_html = _status_message(status_label, started_at)
    yield gr.skip(), gr.skip(), gr.skip(), gr.update(value=status_html), gr.skip()

    result_box = {}
    session_id = cache.get("session_id")

    def _target():
        previous_session = set_analysis_session(session_id)
        try:
            result_box["result"] = fn()
        except Exception as exc:  # pragma: no cover
            result_box["error"] = exc
        finally:
            set_analysis_session(previous_session)

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()

    last_second = -1
    while worker.is_alive():
        current_second = int(time.time() - started_at)
        if current_second != last_second:
            status_html = _status_message(status_label, started_at)
            yield gr.skip(), gr.skip(), gr.skip(), gr.update(value=status_html), gr.skip()
            last_second = current_second
        time.sleep(0.15)

    worker.join()

    if "error" in result_box:
        raise result_box["error"]
    return result_box.get("result")


def _run_post_kernel_analyses(chat_history, cache, output_dir: str, intents: list[str]):
    messages: list[str] = []
    success = True
    ordered = [item for item in ("run_lcp", "run_circuit", "run_graph") if item in intents]
    for intent in ordered:
        if intent == "run_lcp":
            job_id = datetime.now(timezone.utc).strftime("lcp_%Y%m%d_%H%M%S")
            step_success, step_message, _ = yield from _run_with_status(
                chat_history,
                cache,
                "Thinking",
                lambda: run_lcp(cache, output_dir, job_id, write_session_snapshot),
            )
        elif intent == "run_circuit":
            job_id = datetime.now(timezone.utc).strftime("circuit_%Y%m%d_%H%M%S")
            step_success, step_message, _ = yield from _run_with_status(
                chat_history,
                cache,
                "Thinking",
                lambda: run_circuit(cache, output_dir, job_id, write_session_snapshot),
            )
        else:
            job_id = datetime.now(timezone.utc).strftime("graph_%Y%m%d_%H%M%S")
            def _ensure_lcp(cache, base_job_id):
                lcp_job_id = f"{base_job_id}_lcp"
                return run_lcp(cache, output_dir, lcp_job_id, write_session_snapshot)
            step_success, step_message, _ = yield from _run_with_status(
                chat_history,
                cache,
                "Thinking",
                lambda: run_graph_model(
                    cache,
                    output_dir,
                    job_id,
                    ensure_lcp=_ensure_lcp,
                ),
            )
        messages.append(step_message)
        success = bool(success and step_success)
    return success, " ".join(m.strip() for m in messages if m).strip()


def _should_stream_llm_reply(message: str) -> bool:
    msg = (message or "").strip().lower()
    question_like = bool(
        msg.endswith("?")
        or re.match(r"(what|which|where|when|why|how|is|are|do|does|did|can|could|should)\b", msg)
    )
    explanation_like = bool(
        re.search(r"\b(what\s+does|what\s+do|mean|means|explain|interpret|interpretation|describe)\b", msg)
    )
    if question_like or explanation_like:
        return True

    toolish_terms = (
        "least cost", "least-cost", "lcp", "path", "corridor", "graph model", "network analysis",
        "kernel", "crk", "current density", "circuit", "run ", "compute ", "convert ",
    )
    return not any(term in msg for term in toolish_terms)


def _extract_llm_tool_call(text: str) -> Optional[dict]:
    raw = (text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _tool_call_followup_message(tool_call: Optional[dict], cache: dict) -> str:
    if not isinstance(tool_call, dict):
        if cache.get("raster_path"):
            return (
                "I interpreted that as an analysis request, but I couldn't turn it into a runnable action. "
                "Try asking for a cumulative resistant kernel, least-cost paths, circuit analysis, or graph model."
            )
        return (
            "I interpreted that as an analysis request, but I need a raster first. "
            "Upload a raster and optionally a nodes CSV, then ask for a kernel, least-cost paths, circuit analysis, or graph model."
        )

    tool_name = str(tool_call.get("tool") or "").strip().lower()
    method = str(tool_call.get("method") or "").strip().lower()
    if tool_name and tool_name != "connectivity":
        return (
            f"I received a tool request for `{tool_name}`, but this room only supports connectivity workflows. "
            "Try asking for a cumulative resistant kernel, least-cost paths, circuit analysis, or graph model."
        )
    if method and method not in {"lcp", "kernel", "circuit", "graph"}:
        return (
            f"I received a connectivity request for `{method}`, but that method is not available here. "
            "Try `kernel`, `lcp`, `circuit`, or `graph`."
        )
    return (
        "I interpreted that as a connectivity tool request, but it was incomplete. "
        "Please ask in plain language, for example: `run a cumulative resistant kernel with dispersal distance 2 km`."
    )


def _execute_connectivity_tool_call(tool_call: dict, chat_history, cache):
    method = str(tool_call.get("method") or "").strip().lower()
    cache = _ensure_session_workspace(cache)
    output_dir = cache["output_dir"]

    if method == "lcp":
        job_id = datetime.now(timezone.utc).strftime("lcp_%Y%m%d_%H%M%S")
        try:
            success, message, _ = yield from _run_with_status(
                chat_history,
                cache,
                "Thinking",
                lambda: run_lcp(cache, output_dir, job_id, write_session_snapshot),
            )
            chat_history.append({"role": "assistant", "content": message})
            map_html = _build_preview_map(cache)
            download_update = _download_update(cache) if success else gr.update(value=_ensure_placeholder_download(cache), visible=True)
            yield chat_history, gr.update(value=map_html), download_update, _status_clear_update(), cache
        except Exception as exc:
            chat_history.append({"role": "assistant", "content": _analysis_error_message("Least-cost path analysis", exc)})
            yield chat_history, gr.skip(), _download_update(cache), _status_clear_update(), cache
        return

    if method == "kernel":
        dispersal_distance_m = tool_call.get("cutoff") or tool_call.get("sigma")
        try:
            dispersal_distance_m = float(dispersal_distance_m) if dispersal_distance_m is not None else None
        except Exception:
            dispersal_distance_m = None

        if dispersal_distance_m is None or dispersal_distance_m <= 0:
            cache["awaiting_kernel_distance"] = True
            chat_history.append({"role": "assistant", "content": _kernel_distance_prompt()})
            yield chat_history, gr.update(value=_build_preview_map(cache)), _download_update(cache), _status_clear_update(), cache
            return

        cache["kernel_dispersal_distance_m"] = dispersal_distance_m
        job_id = datetime.now(timezone.utc).strftime("crk_%Y%m%d_%H%M%S")
        try:
            success, message, _ = yield from _run_with_status(
                chat_history,
                cache,
                "Thinking",
                lambda: run_crk(cache, output_dir, job_id, dispersal_distance_m),
            )
            chat_history.append({"role": "assistant", "content": message})
            _append_out_of_bounds_analysis_reminder(chat_history, cache, success=success)
            map_html = _build_preview_map(cache)
            yield chat_history, gr.update(value=map_html), _download_update(cache), _status_clear_update(), cache
        except Exception as exc:
            chat_history.append({"role": "assistant", "content": _analysis_error_message("Kernel analysis", exc)})
            yield chat_history, gr.skip(), _download_update(cache), _status_clear_update(), cache
        return

    if method == "circuit":
        job_id = datetime.now(timezone.utc).strftime("circuit_%Y%m%d_%H%M%S")
        try:
            success, message, _ = yield from _run_with_status(
                chat_history,
                cache,
                "Thinking",
                lambda: run_circuit(cache, output_dir, job_id, write_session_snapshot),
            )
            chat_history.append({"role": "assistant", "content": message})
            yield chat_history, gr.update(value=_build_preview_map(cache)), _download_update(cache), _status_clear_update(), cache
        except Exception as exc:
            chat_history.append({"role": "assistant", "content": _analysis_error_message("Circuit analysis", exc)})
            yield chat_history, gr.skip(), _download_update(cache), _status_clear_update(), cache
        return

    chat_history.append(
        {
            "role": "assistant",
            "content": f"I couldn't execute the requested connectivity method `{method or 'unknown'}`.",
        }
    )
    yield chat_history, gr.update(value=_build_preview_map(cache)), _download_update(cache), _status_clear_update(), cache


def _handle_llm_fallback(chat_history, prior_history, user_message, cache):
    if ask_llm is None:
        chat_history.append({"role": "assistant", "content": _local_fallback_reply(cache)})
        yield chat_history, gr.update(value=_build_preview_map(cache)), _download_update(cache), _status_clear_update(), cache
        return

    if ask_llm_stream is not None and _should_stream_llm_reply(user_message):
        started_at = time.time()
        assistant_message = None
        yield gr.skip(), gr.skip(), gr.skip(), gr.update(value=_status_message("Thinking", started_at)), gr.skip()
        try:
            streamed_any = False
            last_second = -1
            for chunk in ask_llm_stream(prior_history, user_message, context=_build_llm_context(cache)):
                if not chunk:
                    current_second = int(time.time() - started_at)
                    if current_second != last_second:
                        yield gr.skip(), gr.skip(), gr.skip(), gr.update(value=_status_message("Thinking", started_at)), gr.skip()
                        last_second = current_second
                    continue
                streamed_any = True
                if assistant_message is None:
                    assistant_message = {"role": "assistant", "content": ""}
                    chat_history.append(assistant_message)
                else:
                    assistant_message["content"] = assistant_message["content"].rstrip("▌")
                assistant_message["content"] += str(chunk) + "▌"
                yield chat_history, gr.skip(), gr.skip(), gr.update(value=_status_message("Thinking", started_at)), cache
            if not streamed_any:
                chat_history.append({"role": "assistant", "content": _local_fallback_reply(cache)})
                yield chat_history, gr.skip(), gr.skip(), _status_clear_update(), cache
                return
            assistant_message["content"] = assistant_message["content"].rstrip("▌").strip() or _local_fallback_reply(cache)
            yield chat_history, gr.skip(), gr.skip(), _status_clear_update(), cache
            tool_call = _extract_llm_tool_call(assistant_message["content"])
            if isinstance(tool_call, dict):
                chat_history.pop()
                if tool_call.get("tool") == "connectivity":
                    yield from _execute_connectivity_tool_call(tool_call, chat_history, cache)
                    return
                chat_history.append({"role": "assistant", "content": _tool_call_followup_message(tool_call, cache)})
                yield chat_history, gr.skip(), gr.skip(), _status_clear_update(), cache
                return
            return
        except Exception:
            if assistant_message is not None and chat_history and chat_history[-1] is assistant_message:
                chat_history.pop()
            chat_history.append({"role": "assistant", "content": _local_fallback_reply(cache)})
            yield chat_history, gr.update(value=_build_preview_map(cache)), _download_update(cache), _status_clear_update(), cache
            return

    started_at = time.time()
    yield gr.skip(), gr.skip(), gr.skip(), gr.update(value=_status_message("Thinking", started_at)), gr.skip()
    try:
        tool_call, llm_text = ask_llm(prior_history, user_message, context=_build_llm_context(cache))
    except Exception:
        chat_history.append({"role": "assistant", "content": _local_fallback_reply(cache)})
        yield chat_history, gr.update(value=_build_preview_map(cache)), _download_update(cache), _status_clear_update(), cache
        return

    if isinstance(tool_call, dict) and tool_call.get("tool") == "connectivity":
        yield from _execute_connectivity_tool_call(tool_call, chat_history, cache)
        return

    parsed_tool_call = _extract_llm_tool_call(llm_text)
    if isinstance(parsed_tool_call, dict):
        if parsed_tool_call.get("tool") == "connectivity":
            yield from _execute_connectivity_tool_call(parsed_tool_call, chat_history, cache)
            return
        chat_history.append({"role": "assistant", "content": _tool_call_followup_message(parsed_tool_call, cache)})
        yield chat_history, gr.update(value=_build_preview_map(cache)), _download_update(cache), _status_clear_update(), cache
        return

    reply = (llm_text or "").strip() or _help_text()
    chat_history.append({"role": "assistant", "content": reply})
    yield chat_history, gr.update(value=_build_preview_map(cache)), _download_update(cache), _status_clear_update(), cache
    return


def _help_text() -> str:
    return (
        "Hi, I'm SpatChat! This room helps you prepare landscape connectivity inputs and run connectivity analyses from a raster surface and optional node locations.\n\n"
        "Upload a raster (required): a GeoTIFF representing a landscape surface such as habitat suitability, resistance/cost, or conductance.\n"
        "Upload nodes CSV (optional): point locations to connect or evaluate, such as habitat patches, source populations, detections, sampling sites, or restoration targets, with longitude/latitude columns.\n\n"
        "You can then ask for: \n"
        "• Cumulative resistant kernel (specify a dispersal distance)\n"
        "• Least‑cost paths / cost distance / corridors\n"
        "• Circuit-style effective resistance between nodes\n"
        "• Graph model from the LCP network\n"
        "\n"
        "Examples:\n"
        "• Run a cumulative resistant kernel with dispersal distance 2 km\n"
        "• Compute least-cost paths between nodes\n"
        "• Run Circuitscape effective resistance between nodes\n"
        "• Run a graph model\n"
    )

# --------------------------------------------------------------------------------------
# Upload handlers
# --------------------------------------------------------------------------------------

def handle_raster_upload(file, cache):
    """Cache raster on disk, read bounds, and refresh preview map."""
    cache = _ensure_session_workspace(dict(cache or _fresh_cache()))
    if file is None:
        _reset_raster_cache(cache)
        cache["last_upload_kind"] = None
        return (
            [{"role": "assistant", "content": "No raster selected."}],
            _render_empty_map("No raster selected"),
            _download_update(cache),
            _status_clear_update(),
            cache,
        )

    src = _file_to_path(file)
    if not src:
        cache["awaiting_surface_type"] = False
        return (
            [{"role": "assistant", "content": "Could not access uploaded raster path."}],
            _render_empty_map("Raster read error"),
            _download_update(cache),
            _status_clear_update(),
            cache,
        )

    # Preserve any already-uploaded nodes so users can upload raster and nodes in either order.
    existing_nodes_df = cache.get("nodes_df")
    existing_nodes_lon_col = cache.get("nodes_lon_col")
    existing_nodes_lat_col = cache.get("nodes_lat_col")
    existing_session = cache.get("session_info") if isinstance(cache.get("session_info"), dict) else {}
    existing_nodes_csv = existing_session.get("nodes_csv")
    existing_nodes_uploaded_at = existing_session.get("nodes_uploaded_at")

    # Save
    upload_dir = cache["upload_dir"]
    output_dir = cache["output_dir"]
    converted_dir = cache["converted_dir"]
    dst_name = os.path.basename(src)
    dst = os.path.join(upload_dir, dst_name)
    try:
        _clear_outputs_dir(output_dir)
        _clear_uploads_dir(upload_dir, keep=dst_name)
        need_copy = True
        try:
            if os.path.exists(dst) and os.path.samefile(src, dst):
                need_copy = False
        except OSError:
            # If samefile check fails (e.g., different drive), fall back to copy
            need_copy = True
        if need_copy:
            shutil.copy(src, dst)
        cache["raster_path"] = dst
        cache["surface_type"] = None
        cache["awaiting_surface_type"] = True
        cache["awaiting_conversion_choice"] = False
        cache["last_upload_kind"] = "raster"
        cache["analysis_cost_tif"] = None
        cache["conversion_target"] = None
        cache["conversion_path"] = None
        cache["conversion_label"] = None
        cache["preview_rasters"] = {}
        _clear_kernel_outputs(cache)
        _clear_circuit_outputs(cache)
        _mark_results_dirty(cache, clear_bundle_path=True)
        cache["nodes_df"] = existing_nodes_df
        cache["nodes_lon_col"] = existing_nodes_lon_col
        cache["nodes_lat_col"] = existing_nodes_lat_col
        cache["conversion_min_resistance"] = 1.0
        cache["conversion_max_resistance"] = 100.0
        cache["conversion_scale_mode"] = "linear"
        cache["session_info"] = {
            "session_id": cache.get("session_id"),
            "raster_path": dst,
            "raster_name": os.path.basename(dst),
            "raster_uploaded_at": datetime.now(timezone.utc).isoformat(),
            "raster_bounds": None,
            "raster_crs": None,
            "raster_width": None,
            "raster_height": None,
            "raster_count": None,
            "raster_resolution": None,
            "raster_bounds_wgs84": None,
            "raster_centroid_wgs84": None,
            "surface_type": None,
            "nodes_csv": existing_nodes_csv,
            "nodes_uploaded_at": existing_nodes_uploaded_at,
            "nodes_lon_col": existing_nodes_lon_col,
            "nodes_lat_col": existing_nodes_lat_col,
        }
        reset_analysis_steps(cache.get("session_id"))
        cache["lcp_geojsonl"] = None
        cache["lcp_index_csv"] = None
        cache["lcp_unreachable_csv"] = None
        cache["lcp_count_tif"] = None
        cache["lcp_cost_matrix"] = None
        cache["lcp_zip_path"] = None
        _clear_circuit_outputs(cache)
        cache["graph_metrics_csv"] = None
        cache["graph_metrics_df"] = None
        cache["graph_metric_name"] = None
        cache["graph_edges_df"] = None
        cache["graph_metric_ranges"] = None
        cache["graph_probability_decay"] = None
        cache["graph_summary_json"] = None
        _mark_results_dirty(cache, clear_bundle_path=True)
    except Exception as e:
        cache["awaiting_surface_type"] = False
        return (
            [{"role": "assistant", "content": f"Failed to save raster: {e}"}],
            _render_empty_map("Raster save error"),
            _download_update(cache),
            _status_clear_update(),
            cache,
        )

    # Read basic metadata
    if rasterio is None:
        cache["raster_bounds"] = None
        cache["raster_crs"] = None
        cache["raster_width"] = None
        cache["raster_height"] = None
        cache["raster_count"] = None
        cache["raster_resolution"] = None
        cache["raster_bounds_wgs84"] = None
        cache["raster_centroid_wgs84"] = None
        msg = (
            "Raster uploaded. (Install rasterio to see extent preview.)\n\n"
            + _surface_prompt()
            + "\n\n"
            + _help_text()
        )
        return ([{"role": "assistant", "content": msg}], _build_preview_map(cache), _download_update(cache), _status_clear_update(), cache)

    try:
        with rasterio.open(dst) as ds:
            cache["raster_bounds"] = ds.bounds  # (minx, miny, maxx, maxy)
            cache["raster_crs"] = str(ds.crs) if ds.crs else None
            cache["raster_width"] = int(ds.width)
            cache["raster_height"] = int(ds.height)
            cache["raster_count"] = int(ds.count)
            cache["raster_resolution"] = tuple(float(v) for v in ds.res)
            _set_raster_wgs84_location(cache, ds.bounds, ds.crs)
    except Exception as e:
        cache["raster_bounds"] = None
        cache["raster_crs"] = None
        cache["raster_width"] = None
        cache["raster_height"] = None
        cache["raster_count"] = None
        cache["raster_resolution"] = None
        cache["raster_bounds_wgs84"] = None
        cache["raster_centroid_wgs84"] = None
        msg = (
            f"Raster uploaded, but metadata read failed: {e}\n\n"
            + _surface_prompt()
        )
        return ([{"role": "assistant", "content": msg}], _build_preview_map(cache), _download_update(cache), _status_clear_update(), cache)

    centroid = cache.get("raster_centroid_wgs84") or {}
    location_line = ""
    if centroid.get("latitude") is not None and centroid.get("longitude") is not None:
        location_line = (
            f"- Approx. WGS84 centroid: {centroid['latitude']:.6f}, {centroid['longitude']:.6f}"
        )

    info_lines = [
        "Raster uploaded.",
        f"- Bounds: {cache['raster_bounds']}",
        f"- CRS: {cache['raster_crs']}",
        f"- Dimensions: {cache['raster_width']} columns x {cache['raster_height']} rows ({cache['raster_count']} band{'s' if cache['raster_count'] != 1 else ''})",
    ]
    if location_line:
        info_lines.append(location_line)
    info_lines.extend([
        "",
        _help_text(),
        "",
        _surface_prompt(),
    ])
    info = "\n".join(info_lines)
    _refresh_out_of_bounds_node_cache(cache)
    out_of_bounds_note = _out_of_bounds_nodes_message(cache)
    if out_of_bounds_note:
        info = f"{info}\n\n{out_of_bounds_note}"
    if isinstance(cache.get("session_info"), dict):
        session = cache["session_info"]
        session["raster_bounds"] = cache["raster_bounds"]
        session["raster_crs"] = cache["raster_crs"]
        session["raster_width"] = cache["raster_width"]
        session["raster_height"] = cache["raster_height"]
        session["raster_count"] = cache["raster_count"]
        session["raster_resolution"] = cache["raster_resolution"]
        session["raster_bounds_wgs84"] = cache["raster_bounds_wgs84"]
        session["raster_centroid_wgs84"] = cache["raster_centroid_wgs84"]
    return ([{"role": "assistant", "content": info}], _build_preview_map(cache), _download_update(cache), _status_clear_update(), cache)


def handle_nodes_upload(file, chat_history, cache):
    """Cache nodes CSV in memory, detect lon/lat, and refresh preview map."""
    chat_history = list(chat_history)
    cache = _ensure_session_workspace(dict(cache or _fresh_cache()))
    output_dir = cache["output_dir"]
    converted_dir = cache["converted_dir"]

    if file is None:
        # Do not reset raster when clearing nodes; preserve current surface and preview
        _clear_outputs_dir(output_dir, preserve=(converted_dir,))
        cache["nodes_df"] = None
        cache["nodes_lon_col"] = None
        cache["nodes_lat_col"] = None
        cache["out_of_bounds_node_count"] = 0
        cache["out_of_bounds_node_ids"] = []
        cache["last_upload_kind"] = None
        # Clear LCP-related artifacts only
        cache["lcp_geojsonl"] = None
        cache["lcp_index_csv"] = None
        cache["lcp_unreachable_csv"] = None
        cache["lcp_count_tif"] = None
        cache["lcp_cost_matrix"] = None
        cache["lcp_zip_path"] = None
        _clear_circuit_outputs(cache)
        cache["graph_metrics_csv"] = None
        cache["graph_metrics_df"] = None
        cache["graph_metric_name"] = None
        cache["graph_edges_df"] = None
        cache["graph_metric_ranges"] = None
        cache["graph_probability_decay"] = None
        cache["graph_summary_json"] = None
        _mark_results_dirty(cache, clear_bundle_path=True)
        chat_history.append({"role": "assistant", "content": "Nodes cleared."})
        return chat_history, _build_preview_map(cache), _download_update(cache), _status_clear_update(), cache

    try:
        path = _file_to_path(file)
        df = pd.read_csv(path if path else file)
    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"Failed to read nodes CSV: {e}"})
        return chat_history, _build_preview_map(cache), _download_update(cache), _status_clear_update(), cache

    upload_dir = cache["upload_dir"]
    persisted_nodes_path = None
    if path:
        dst_name = os.path.basename(path)
        persisted_nodes_path = os.path.join(upload_dir, dst_name)
        try:
            if not (os.path.exists(persisted_nodes_path) and os.path.samefile(path, persisted_nodes_path)):
                shutil.copy(path, persisted_nodes_path)
        except OSError:
            try:
                shutil.copy(path, persisted_nodes_path)
            except Exception as e:
                chat_history.append({"role": "assistant", "content": f"Failed to save nodes CSV: {e}"})
                return chat_history, _build_preview_map(cache), _download_update(cache), _status_clear_update(), cache

    # Never reset the raster when nodes are uploaded; just clear outputs
    _clear_outputs_dir(output_dir, preserve=(converted_dir,))

    note = "Nodes CSV uploaded. "
    guess = detect_nodes_lonlat(df) or (None, None)
    lon_col, lat_col = guess

    if lon_col and lat_col:
        cache["nodes_lon_col"] = lon_col
        cache["nodes_lat_col"] = lat_col
        note += f"Detected lon/lat columns: {lon_col} / {lat_col}."
    else:
        cache["nodes_lon_col"] = None
        cache["nodes_lat_col"] = None
        note += (
            "Lat/Lon columns not detected. Include 'longitude'/'latitude' (or 'lon'/'lat')."
        )

    if "node_id" not in df.columns:
        df = df.copy()
        df["node_id"] = _auto_node_ids(len(df))

    _clear_kernel_outputs(cache)
    _clear_circuit_outputs(cache)
    cache["nodes_df"] = df
    _normalize_cached_nodes_df(cache)
    cache["lcp_geojsonl"] = None
    cache["lcp_index_csv"] = None
    cache["lcp_unreachable_csv"] = None
    cache["lcp_count_tif"] = None
    cache["lcp_cost_matrix"] = None
    cache["lcp_zip_path"] = None
    cache["graph_metrics_csv"] = None
    cache["graph_metrics_df"] = None
    cache["graph_metric_name"] = None
    cache["graph_edges_df"] = None
    cache["graph_metric_ranges"] = None
    cache["graph_probability_decay"] = None
    cache["graph_summary_json"] = None
    _mark_results_dirty(cache, clear_bundle_path=True)
    _refresh_out_of_bounds_node_cache(cache)
    out_of_bounds_note = _out_of_bounds_nodes_message(cache)
    if out_of_bounds_note:
        note = f"{note} {out_of_bounds_note}"

    chat_history.append({"role": "assistant", "content": note})
    cache["last_upload_kind"] = "nodes"
    if isinstance(cache.get("session_info"), dict):
        session = cache["session_info"]
        session["nodes_csv"] = persisted_nodes_path or path or getattr(file, "name", None)
        session["nodes_uploaded_at"] = datetime.now(timezone.utc).isoformat()
        session["nodes_lon_col"] = cache["nodes_lon_col"]
        session["nodes_lat_col"] = cache["nodes_lat_col"]
    return chat_history, _build_preview_map(cache), _download_update(cache), _status_clear_update(), cache


# --------------------------------------------------------------------------------------
# Chat handler (placeholder)
# --------------------------------------------------------------------------------------

def handle_chat(chat_history, user_message, cache):
    chat_history = list(chat_history)
    cache = _ensure_session_workspace(dict(cache or _fresh_cache()))
    msg = user_message.strip()
    prior_history = list(chat_history)
    output_dir = cache["output_dir"]
    converted_dir = cache["converted_dir"]

    if not msg:
        chat_history.append({"role": "assistant", "content": "How can I help?"})
        yield chat_history, gr.skip(), _download_update(cache), _status_clear_update(), cache
        return

    chat_history.append({"role": "user", "content": user_message})
    yield chat_history, gr.skip(), gr.skip(), _status_clear_update(), cache

    intent = _detect_intent(msg)
    intents = _detect_intents(msg)
    if cache.get("awaiting_kernel_distance"):
        pass
    elif "run_kernel" in intents and len(intents) > 1:
        cache["pending_post_kernel_analysis"] = [item for item in intents if item != "run_kernel"]
    else:
        cache["pending_post_kernel_analysis"] = None

    if cache.get("awaiting_surface_type"):
        surface_choice = _normalize_surface_type_answer(msg)
        if surface_choice:
            cache["surface_type"] = surface_choice
            cache["awaiting_surface_type"] = False
            cache["analysis_cost_tif"] = None
            cache["conversion_target"] = None
            cache["conversion_path"] = None
            cache["conversion_label"] = None
            cache["preview_rasters"] = {}
            cache["awaiting_conversion_choice"] = False
            label = _SURFACE_LABELS.get(surface_choice, surface_choice)
            ack = f"Got it. I'll treat the raster as {label}."
            extra = _SURFACE_ACKS.get(surface_choice)
            if extra:
                ack = f"{ack} {extra}"
            chat_history.append({"role": "assistant", "content": ack})
            try:
                needs_manual, auto_messages = yield from _run_with_status(
                    chat_history,
                    cache,
                    "Converting surface",
                    lambda: _auto_prepare_surface(cache, surface_choice),
                )
            except Exception as exc:
                chat_history.append(
                    {
                        "role": "assistant",
                        "content": f"Automatic surface preparation failed: {exc}",
                    }
                )
                yield (
                    chat_history,
                    gr.update(value=_build_preview_map(cache)),
                    _download_update(cache),
                    _status_clear_update(),
                    cache,
                )
                return
            for text in auto_messages:
                chat_history.append({"role": "assistant", "content": text})
            if needs_manual:
                cache["awaiting_conversion_choice"] = True
                chat_history.append({"role": "assistant", "content": conversion_prompt(cache, _SURFACE_LABELS)})
            if isinstance(cache.get("session_info"), dict):
                session = cache["session_info"]
                session["surface_type"] = surface_choice
                session["surface_type_set_at"] = datetime.now(timezone.utc).isoformat()
                session.pop("conversion_target", None)
                session.pop("conversion_path", None)
                session.pop("conversion_resistance_range", None)
                session.pop("conversion_completed_at", None)
        else:
            if intent is None:
                yield from _handle_llm_fallback(chat_history, prior_history, user_message, cache)
                return
            reminder = (
                "Before we continue, I need to know what kind of surface the raster represents.\n"
                + _surface_prompt()
            )
            chat_history.append({"role": "assistant", "content": reminder})
            yield (
                chat_history,
                gr.update(value=_build_preview_map(cache)),
                _download_update(cache),
                _status_clear_update(),
                cache,
            )
            return

        yield (
            chat_history,
            gr.update(value=_build_preview_map(cache)),
            _download_update(cache),
            _status_clear_update(),
            cache,
        )
        return

    if cache.get("awaiting_kernel_distance"):
        dispersal_distance_m = _parse_kernel_distance_m(msg)
        if dispersal_distance_m is None:
            chat_history.append({"role": "assistant", "content": _kernel_distance_prompt()})
            yield (
                chat_history,
                gr.update(value=_build_preview_map(cache)),
                _download_update(cache),
                _status_clear_update(),
                cache,
            )
            return

        cache["awaiting_kernel_distance"] = False
        cache["kernel_dispersal_distance_m"] = dispersal_distance_m
        job_id = datetime.now(timezone.utc).strftime("crk_%Y%m%d_%H%M%S")
        try:
            success, message, _ = yield from _run_with_status(
                chat_history,
                cache,
                "Thinking",
                lambda: run_crk(cache, output_dir, job_id, dispersal_distance_m),
            )
            post_kernel_intents = cache.get("pending_post_kernel_analysis") or []
            if success and post_kernel_intents:
                post_success, post_message = yield from _run_post_kernel_analyses(
                    chat_history,
                    cache,
                    output_dir,
                    list(post_kernel_intents),
                )
                message = f"{message} {post_message}".strip()
                success = bool(success and post_success)
            cache["pending_post_kernel_analysis"] = None
            chat_history.append({"role": "assistant", "content": message})
            _append_out_of_bounds_analysis_reminder(chat_history, cache, success=success)
            map_html = _build_preview_map(cache)
            yield (
                chat_history,
                gr.update(value=map_html),
                _download_update(cache),
                _status_clear_update(),
                cache,
            )
        except Exception as exc:
            cache["pending_post_kernel_analysis"] = None
            chat_history.append({"role": "assistant", "content": _analysis_error_message("Kernel analysis", exc)})
            yield chat_history, gr.skip(), _download_update(cache), _status_clear_update(), cache
        return

    if intents == ["run_lcp", "run_circuit", "run_graph"]:
        try:
            success, message = yield from _run_post_kernel_analyses(
                chat_history,
                cache,
                output_dir,
                ["run_lcp", "run_circuit", "run_graph"],
            )
            chat_history.append({"role": "assistant", "content": message})
            _append_out_of_bounds_analysis_reminder(chat_history, cache, success=success)
            map_html = _build_preview_map(cache)
            yield chat_history, gr.update(value=map_html), _download_update(cache), _status_clear_update(), cache
        except Exception as exc:
            chat_history.append({"role": "assistant", "content": _analysis_error_message("Combined analysis", exc)})
            yield chat_history, gr.skip(), _download_update(cache), _status_clear_update(), cache
        return

    if intent == "run_lcp":
        job_id = datetime.now(timezone.utc).strftime("lcp_%Y%m%d_%H%M%S")
        try:
            success, message, _ = yield from _run_with_status(
                chat_history,
                cache,
                "Thinking",
                lambda: run_lcp(cache, output_dir, job_id, write_session_snapshot),
            )
            chat_history.append({"role": "assistant", "content": message})
            _append_out_of_bounds_analysis_reminder(chat_history, cache, success=success)
            map_html = _build_preview_map(cache)
            download_update = _download_update(cache) if success else gr.update(value=_ensure_placeholder_download(cache), visible=True)
            yield chat_history, gr.update(value=map_html), download_update, _status_clear_update(), cache
        except Exception as exc:
            chat_history.append({"role": "assistant", "content": _analysis_error_message("Least-cost path analysis", exc)})
            yield chat_history, gr.skip(), _download_update(cache), _status_clear_update(), cache
        return

    if intent == "run_kernel":
        dispersal_distance_m = _parse_kernel_distance_m(msg)
        if dispersal_distance_m is None:
            cache["awaiting_kernel_distance"] = True
            chat_history.append({"role": "assistant", "content": _kernel_distance_prompt()})
            yield (
                chat_history,
                gr.update(value=_build_preview_map(cache)),
                _download_update(cache),
                _status_clear_update(),
                cache,
            )
            return

        cache["kernel_dispersal_distance_m"] = dispersal_distance_m
        job_id = datetime.now(timezone.utc).strftime("crk_%Y%m%d_%H%M%S")
        try:
            success, message, _ = yield from _run_with_status(
                chat_history,
                cache,
                "Thinking",
                lambda: run_crk(cache, output_dir, job_id, dispersal_distance_m),
            )
            chat_history.append({"role": "assistant", "content": message})
            _append_out_of_bounds_analysis_reminder(chat_history, cache, success=success)
            map_html = _build_preview_map(cache)
            yield (
                chat_history,
                gr.update(value=map_html),
                _download_update(cache),
                _status_clear_update(),
                cache,
            )
        except Exception as exc:
            chat_history.append({"role": "assistant", "content": _analysis_error_message("Kernel analysis", exc)})
            yield chat_history, gr.skip(), _download_update(cache), _status_clear_update(), cache
        return

    if intent == "run_circuit":
        job_id = datetime.now(timezone.utc).strftime("circuit_%Y%m%d_%H%M%S")
        try:
            success, message, _ = yield from _run_with_status(
                chat_history,
                cache,
                "Thinking",
                lambda: run_circuit(cache, output_dir, job_id, write_session_snapshot),
            )
            chat_history.append({"role": "assistant", "content": message})
            _append_out_of_bounds_analysis_reminder(chat_history, cache, success=success)
            map_html = _build_preview_map(cache)
            yield (
                chat_history,
                gr.update(value=map_html),
                _download_update(cache),
                _status_clear_update(),
                cache,
            )
        except Exception as exc:
            chat_history.append({"role": "assistant", "content": _analysis_error_message("Circuit analysis", exc)})
            yield chat_history, gr.skip(), _download_update(cache), _status_clear_update(), cache
        return

    if intent == "run_graph":
        job_id = datetime.now(timezone.utc).strftime("graph_%Y%m%d_%H%M%S")
        def _ensure_lcp(cache, base_job_id):
            lcp_job_id = f"{base_job_id}_lcp"
            return run_lcp(cache, output_dir, lcp_job_id, write_session_snapshot)

        try:
            success, message, _ = yield from _run_with_status(
                chat_history,
                cache,
                "Thinking",
                lambda: run_graph_model(
                    cache,
                    output_dir,
                    job_id,
                    ensure_lcp=_ensure_lcp,
                ),
            )
            chat_history.append({"role": "assistant", "content": message})
            _append_out_of_bounds_analysis_reminder(chat_history, cache, success=success)
            map_html = _build_preview_map(cache)
            yield (
                chat_history,
                gr.update(value=map_html),
                _download_update(cache),
                _status_clear_update(),
                cache,
            )
        except Exception as exc:
            chat_history.append({"role": "assistant", "content": _analysis_error_message("Graph analysis", exc)})
            yield chat_history, gr.skip(), _download_update(cache), _status_clear_update(), cache
        return

    conversion_status_actions = {"convert", "range", "scale_mode"}
    conversion_action = interpret_conversion_message(msg).get("action")
    if conversion_action in conversion_status_actions:
        conversion_handled, refresh_map = yield from _run_with_status(
            chat_history,
            cache,
            "Thinking",
            lambda: process_conversion_chat(
                cache,
                converted_dir,
                chat_history,
                msg,
                _SURFACE_LABELS,
            ),
        )
    else:
        conversion_handled, refresh_map = process_conversion_chat(
            cache,
            converted_dir,
            chat_history,
            msg,
            _SURFACE_LABELS,
        )
    if conversion_handled:
        map_update = gr.update(value=_build_preview_map(cache)) if refresh_map else gr.update()
        yield (
            chat_history,
            map_update,
            _download_update(cache),
            _status_clear_update(),
            cache,
        )
        return

    yield from _handle_llm_fallback(chat_history, prior_history, user_message, cache)


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
_SPLITTER_HEAD = """
<script>
window.spatchatBeginResize = function(event) {
  event.preventDefault();
  const workarea = document.getElementById("spatchat-workarea");
  const sidebar = document.getElementById("spatchat-sidebar");
  const mapcol = document.getElementById("spatchat-mapcol");
  if (!workarea || !sidebar || !mapcol) return false;

  const rect = workarea.getBoundingClientRect();
  const minWidth = 80;
  const maxWidth = Math.max(minWidth, rect.width - 80);
  let latestClientX = event.clientX;
  let rafId = 0;

  function apply() {
    rafId = 0;
    const next = Math.min(Math.max(latestClientX - rect.left, minWidth), maxWidth);
    const value = `${Math.round(next)}px`;
    sidebar.style.width = value;
    sidebar.style.flexBasis = value;
    sidebar.style.maxWidth = value;
    mapcol.style.flex = "1 1 auto";
  }

  function scheduleApply(clientX) {
    latestClientX = clientX;
    if (!rafId) {
      rafId = window.requestAnimationFrame(apply);
    }
  }

  function onMove(moveEvent) {
    scheduleApply(moveEvent.clientX);
  }

  function onUp() {
    document.body.classList.remove("spatchat-resizing");
    if (rafId) {
      window.cancelAnimationFrame(rafId);
      rafId = 0;
    }
    window.removeEventListener("mousemove", onMove);
    window.removeEventListener("mouseup", onUp);
  }

  document.body.classList.add("spatchat-resizing");
  scheduleApply(event.clientX);
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
  return false;
};
</script>
"""

with gr.Blocks(title="SpatChat: Connectivity Analysis") as demo:
    session_cache = gr.State(_fresh_cache()) # type: ignore
    gr.Image(
        value="logo_long1.png",
        show_label=False,
        type="filepath",
        buttons=[],  # no download/share/fullscreen buttons
        elem_id="logo-img"
    )
    gr.HTML(
    """
    <style>
    .gradio-container {
        --spatchat-connect-bg: #f6f8fb;
        --spatchat-chat-text: #131722;
    }

    .dark .gradio-container {
        --spatchat-connect-bg: var(--body-background-fill, var(--background-fill-primary, #0f141d));
        --spatchat-chat-text: var(--body-text-color, #edf2ff);
    }

    .gradio-container {
        max-width: 100% !important;
        padding-left: 4px !important;
        padding-right: 4px !important;
    }

    #spatchat-workarea {
        overflow-x: auto;
        overflow-y: visible;
        padding-bottom: 8px;
    }

    #spatchat-workarea > .gradio-row {
        min-width: 900px;
        align-items: stretch;
        flex-wrap: nowrap;
        gap: 0 !important;
    }

    #spatchat-sidebar {
        min-width: 0 !important;
        max-width: 1200px;
        flex: 0 0 420px !important;
        width: 420px;
        overflow: visible;
        padding-right: 4px !important;
        font-size: 15px;
        scrollbar-color: color-mix(in srgb, var(--background-fill-secondary) 78%, white 22%) var(--background-fill-secondary);
        scrollbar-width: thin;
    }

    #spatchat-sidebar::-webkit-scrollbar {
        width: 12px;
    }

    #spatchat-sidebar::-webkit-scrollbar-track {
        background: var(--background-fill-secondary);
        border-radius: 999px;
    }

    #spatchat-sidebar::-webkit-scrollbar-thumb {
        background: color-mix(in srgb, var(--background-fill-secondary) 78%, white 22%);
        border-radius: 999px;
        border: 2px solid var(--background-fill-secondary);
    }

    #spatchat-sidebar::-webkit-scrollbar-thumb:hover {
        background: color-mix(in srgb, var(--background-fill-secondary) 70%, white 30%);
    }

    #spatchat-sidebar::-webkit-scrollbar-button:single-button {
        display: block;
        height: 12px;
        background-color: color-mix(in srgb, var(--background-fill-secondary) 78%, white 22%);
        border-radius: 999px;
        border: 2px solid var(--background-fill-secondary);
        background-repeat: no-repeat;
        background-position: center;
        background-size: 7px 7px;
    }

    #spatchat-sidebar::-webkit-scrollbar-button:single-button:vertical:decrement {
        background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 10'><path d='M2 6.5 5 3.5 8 6.5' fill='none' stroke='%23606b7a' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/></svg>");
    }

    #spatchat-sidebar::-webkit-scrollbar-button:single-button:vertical:increment {
        background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 10'><path d='M2 3.5 5 6.5 8 3.5' fill='none' stroke='%23606b7a' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/></svg>");
    }

    .dark #spatchat-sidebar {
        scrollbar-color: color-mix(in srgb, var(--background-fill-secondary) 86%, white 14%) var(--background-fill-secondary);
    }

    .dark #spatchat-sidebar::-webkit-scrollbar-thumb,
    .dark #spatchat-sidebar::-webkit-scrollbar-button:single-button {
        background-color: color-mix(in srgb, var(--background-fill-secondary) 86%, white 14%);
        background: color-mix(in srgb, var(--background-fill-secondary) 86%, white 14%);
    }

    #spatchat-sidebar > div {
        margin-left: 0 !important;
        margin-right: 0 !important;
    }

    #spatchat-chatbot,
    #spatchat-user-input,
    #spatchat-raster-input,
    #spatchat-nodes-input,
    #spatchat-download {
        margin-left: 0 !important;
        margin-right: 0 !important;
    }

    /* Chat container */
    #spatchat-chatbot {
        padding-left: 0 !important;
        padding-right: 0 !important;
        --chatbot-body-text-size: 16px;
        font-size: 16px !important;
        height: 56vh !important;
        min-height: 56vh !important;
        background: var(--spatchat-connect-bg) !important;
        color: var(--spatchat-chat-text) !important;
        border: none !important;
        box-shadow: none !important;
        border-radius: var(--radius-lg) var(--radius-lg) 0 0 !important;
        overflow: hidden !important;
    }

    #spatchat-chatbot > div {
        background: var(--spatchat-connect-bg) !important;
        color: var(--spatchat-chat-text) !important;
        border: none !important;
        box-shadow: none !important;
        border-radius: var(--radius-lg) var(--radius-lg) 0 0 !important;
    }

    #spatchat-chatbot,
    #spatchat-chatbot > div {
        scrollbar-color: rgba(133, 146, 171, 0.75) var(--spatchat-connect-bg);
        scrollbar-width: thin;
    }

    #spatchat-chatbot::-webkit-scrollbar,
    #spatchat-chatbot > div::-webkit-scrollbar {
        width: 12px;
    }

    #spatchat-chatbot::-webkit-scrollbar-track,
    #spatchat-chatbot > div::-webkit-scrollbar-track {
        background: var(--spatchat-connect-bg);
        border-radius: 999px;
    }

    #spatchat-chatbot::-webkit-scrollbar-thumb,
    #spatchat-chatbot > div::-webkit-scrollbar-thumb {
        background: #323845;
        border-radius: 999px;
        border: 2px solid var(--spatchat-connect-bg);
    }

    #spatchat-chatbot::-webkit-scrollbar-thumb:hover,
    #spatchat-chatbot > div::-webkit-scrollbar-thumb:hover {
        background: #323845;
    }

    #spatchat-chatbot::-webkit-scrollbar-button:single-button,
    #spatchat-chatbot > div::-webkit-scrollbar-button:single-button {
        display: block;
        height: 12px;
        background-color: #323845;
        border-radius: 999px;
        border: 2px solid var(--spatchat-connect-bg);
        background-repeat: no-repeat;
        background-position: center;
        background-size: 7px 7px;
    }

    #spatchat-chatbot::-webkit-scrollbar-button:single-button:vertical:decrement,
    #spatchat-chatbot > div::-webkit-scrollbar-button:single-button:vertical:decrement {
        background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 10'><path d='M2 6.5 5 3.5 8 6.5' fill='none' stroke='%23c7ced9' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/></svg>");
    }

    #spatchat-chatbot::-webkit-scrollbar-button:single-button:vertical:increment,
    #spatchat-chatbot > div::-webkit-scrollbar-button:single-button:vertical:increment {
        background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 10'><path d='M2 3.5 5 6.5 8 3.5' fill='none' stroke='%23c7ced9' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/></svg>");
    }

    /* Remove default action buttons/icons */
    #spatchat-chatbot button[aria-label*="Copy"],
    #spatchat-chatbot button[aria-label*="copy"],
    #spatchat-chatbot button[aria-label*="Share"],
    #spatchat-chatbot button[aria-label*="share"] {
        display: none !important;
    }

    /* Input area */
    #spatchat-user-input {
        background: color-mix(in srgb, var(--background-fill-secondary) 78%, white 22%) !important;
        border-radius: 0 0 var(--radius-lg) var(--radius-lg) !important;
        padding: 6px 10px !important;
        margin-top: -8px !important;
        border-top: none !important;
    }

    .dark #spatchat-user-input {
        background: color-mix(in srgb, var(--background-fill-secondary) 86%, white 14%) !important;
    }

    #spatchat-user-input > div,
    #spatchat-user-input .wrap,
    #spatchat-user-input label {
        background: transparent !important;
    }

    #spatchat-user-input .block-label,
    #spatchat-user-input .label-wrap,
    #spatchat-user-input [data-testid="block-label"] {
        display: none !important;
    }

    #spatchat-user-input .input-container {
        align-items: center !important;
    }

    #spatchat-user-input textarea,
    #spatchat-user-input input {
        background: color-mix(in srgb, var(--background-fill-secondary) 78%, white 22%) !important;
        border: none !important;
        box-shadow: none !important;
        border-radius: var(--radius-md) !important;
        height: 48px !important;
        min-height: 48px !important;
        box-sizing: border-box !important;
        padding: 13px 12px !important;
        line-height: 22px !important;
    }

    #spatchat-user-input textarea {
        max-height: 240px !important;
        overflow-y: auto !important;
        resize: vertical !important;
    }

    #spatchat-user-input input {
        overflow: hidden !important;
    }

    .dark #spatchat-user-input textarea,
    .dark #spatchat-user-input input {
        background: color-mix(in srgb, var(--background-fill-secondary) 86%, white 14%) !important;
    }

    #spatchat-user-input textarea::placeholder,
    #spatchat-user-input input::placeholder {
        color: var(--body-text-color-subdued) !important;
        text-align: left !important;
    }

    /* Chat rows */
    #spatchat-chatbot .message-row,
    #spatchat-chatbot .message-wrap,
    #spatchat-chatbot .bubble-wrap,
    #spatchat-chatbot .wrap {
        background: transparent !important;
    }

    /* Assistant: no visible bubble */
    #spatchat-chatbot .message-row.bot,
    #spatchat-chatbot .message-row.assistant,
    #spatchat-chatbot .message-row.bot-row,
    #spatchat-chatbot .bubble.bot-row,
    #spatchat-chatbot [data-testid="chatbot-message-assistant"] {
        background: var(--spatchat-connect-bg) !important;
        background-color: var(--spatchat-connect-bg) !important;
        justify-content: flex-start !important;
    }

    #spatchat-chatbot .message-row.bot-row,
    #spatchat-chatbot .message-row.bot-row .flex-wrap,
    #spatchat-chatbot .message-row.bot-row .message,
    #spatchat-chatbot .message-row.bot-row .message-bubble-border,
    #spatchat-chatbot .message-row.bot-row .bot,
    #spatchat-chatbot .bubble.bot-row,
    #spatchat-chatbot .bubble.bot-row .message,
    #spatchat-chatbot .bubble.bot-row .message-bubble-border,
    #spatchat-chatbot .bubble.bot-row .bot,
    #spatchat-chatbot .bot,
    #spatchat-chatbot .bot-row .message,
    #spatchat-chatbot .bot-row .message-bubble-border,
    #spatchat-chatbot .bot-row .bot,
    #spatchat-chatbot .bubble.pending,
    #spatchat-chatbot .bubble.pending .message-content {
        background: var(--spatchat-connect-bg) !important;
        background-color: var(--spatchat-connect-bg) !important;
        border: none !important;
        border-color: transparent !important;
        box-shadow: none !important;
        outline: none !important;
    }

    #spatchat-chatbot .message-row.bot-row .bot,
    #spatchat-chatbot .message-row.bot-row,
    #spatchat-chatbot .bubble.bot-row,
    #spatchat-chatbot .bubble.bot-row .bot,
    #spatchat-chatbot .bot,
    #spatchat-chatbot .bot-row .bot,
    #spatchat-chatbot .bubble.pending {
        padding: 0 !important;
        border-radius: 0 !important;
    }

    #spatchat-chatbot .message-row.bot .bubble-wrap,
    #spatchat-chatbot .message-row.assistant .bubble-wrap,
    #spatchat-chatbot [data-testid="chatbot-message-assistant"] .bubble-wrap,
    #spatchat-chatbot .message-row.bot .bubble,
    #spatchat-chatbot .message-row.assistant .bubble,
    #spatchat-chatbot [data-testid="chatbot-message-assistant"] .bubble,
    #spatchat-chatbot .message-row.bot .avatar-container,
    #spatchat-chatbot .message-row.assistant .avatar-container,
    #spatchat-chatbot [data-testid="chatbot-message-assistant"] .avatar-container,
    #spatchat-chatbot .message-row.bot .message,
    #spatchat-chatbot .message-row.assistant .message,
    #spatchat-chatbot [data-testid="chatbot-message-assistant"] .message {
        background: var(--spatchat-connect-bg) !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }

    #spatchat-chatbot .message-row.bot .message,
    #spatchat-chatbot .message-row.assistant .message,
    #spatchat-chatbot [data-testid="chatbot-message-assistant"] .message {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
        width: auto !important;
        color: var(--spatchat-chat-text) !important;
    }

    #spatchat-chatbot .message-row.bot .message *,
    #spatchat-chatbot .message-row.assistant .message *,
    #spatchat-chatbot [data-testid="chatbot-message-assistant"] .message * {
        background: transparent !important;
        color: inherit !important;
    }

    /* User row aligned right */
    #spatchat-chatbot .message-row.user,
    #spatchat-chatbot [data-testid="chatbot-message-user"] {
        background: transparent !important;
        display: flex !important;
        justify-content: flex-end !important;
    }

    /* User bubble */
    #spatchat-chatbot .message-row.user .message,
    #spatchat-chatbot [data-testid="chatbot-message-user"] .message {
        margin-left: auto !important;
        margin-right: 0 !important;
        width: auto !important;
        max-width: 90% !important;
        padding: 10px 14px !important;
        border: none !important;
        box-shadow: none !important;
        border-radius: 16px 16px 4px 16px !important;
        background: #5b86c5 !important;
        color: #ffffff !important;
    }

    .dark #spatchat-chatbot .message-row.user .message,
    .dark #spatchat-chatbot [data-testid="chatbot-message-user"] .message {
        background: #4d79bd !important;
        color: #ffffff !important;
    }

    #spatchat-chatbot .message-row.user .message *,
    #spatchat-chatbot [data-testid="chatbot-message-user"] .message * {
        color: inherit !important;
        background: transparent !important;
    }

    /* Status */
    #spatchat-chatbot .spatchat-status,
    #spatchat-status .spatchat-status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        color: var(--body-text-color-subdued);
        font-weight: 500;
    }

    #spatchat-chatbot .spatchat-status-dot,
    #spatchat-status .spatchat-status-dot {
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: #5b86c5;
        animation: spatchat-pulse 1.2s ease-in-out infinite;
    }

    #spatchat-chatbot .spatchat-status-time,
    #spatchat-status .spatchat-status-time {
        color: var(--body-text-color-subdued);
        font-size: 0.92em;
    }

    #spatchat-status {
        min-height: 24px;
        margin: 4px 0 8px 0 !important;
        color: var(--body-text-color-subdued);
        font-size: 14px;
    }

    @keyframes spatchat-pulse {
        0%, 100% { opacity: 0.35; }
        50% { opacity: 1; }
    }

    /* Splitter */
    #spatchat-splitter {
        min-width: 10px;
        max-width: 10px;
        width: 10px;
        flex: 0 0 10px !important;
        position: relative;
        padding: 0 !important;
        margin: 0 !important;
        min-height: 78vh;
        align-self: stretch !important;
        background: transparent !important;
        overflow: visible !important;
    }

    #spatchat-splitter > div,
    #spatchat-splitter .gradio-html,
    #spatchat-splitter .gradio-html > div {
        height: 100%;
        min-height: 78vh;
        padding: 0 !important;
        margin: 0 !important;
        background: transparent !important;
    }

    #spatchat-splitter .spatchat-splitter-handle {
        position: absolute;
        inset: 0;
        display: block;
        width: 100%;
        height: 100%;
        cursor: col-resize;
        user-select: none;
        background: transparent;
    }

    #spatchat-splitter .spatchat-splitter-handle::before {
        content: "";
        position: absolute;
        top: 8px;
        bottom: 8px;
        left: 50%;
        width: 1px;
        transform: translateX(-50%);
        background: rgba(120, 130, 145, 0.55);
        border-radius: 999px;
        transition: background 120ms ease, width 120ms ease;
    }

    #spatchat-splitter:hover .spatchat-splitter-handle::before,
    body.spatchat-resizing #spatchat-splitter .spatchat-splitter-handle::before {
        width: 2px;
        background: #5b86c5;
    }

    body.spatchat-resizing,
    body.spatchat-resizing * {
        cursor: col-resize !important;
        user-select: none !important;
    }

    /* Map */
    #spatchat-mapcol {
        min-width: 0 !important;
        flex: 1 1 auto !important;
        width: auto !important;
    }

    #spatchat-map {
        min-height: 78vh;
        height: 78vh;
        overflow: visible;
        width: 100%;
    }

    #spatchat-map > div {
        height: 100%;
    }

    #spatchat-map iframe {
        width: 100% !important;
        height: 100% !important;
        min-height: 78vh;
        border: none;
    }

    body.spatchat-resizing #spatchat-map iframe {
        pointer-events: none !important;
    }

    /* Download */
    #spatchat-download {
        display: flex;
        justify-content: center;
        width: 100%;
        margin-top: 10px;
    }

    #spatchat-download button {
        min-width: 220px;
        width: 100%;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 8px;
        text-align: center;
    }

    /* Logo */
    #logo-img img {
        height: 90px;
        margin: 10px 50px 10px 10px;
        border-radius: 6px;
    }
    </style>
    """
    )
    gr.Markdown("## 🌐 SpatChat: Connectivity Analysis {connect} 🦌🦅🦝")
    gr.HTML(
        """
    <div style="margin-top: -10px; margin-bottom: 15px;">
      <input type="text" value="https://spatchat.org/browse/?room=connect" id="shareLink" readonly style="width: 50%; padding: 5px; background-color: #f8f8f8; color: #222; font-weight: 500; border: 1px solid #ccc; border-radius: 4px;">
      <button onclick="navigator.clipboard.writeText(document.getElementById('shareLink').value)" style="padding: 5px 10px; background-color: #007BFF; color: white; border: none; border-radius: 4px; cursor: pointer;">
        📋 Copy Share Link
      </button>
      <div style="margin-top: 10px; font-size: 14px;"> 
        <b>Share:</b> 
        <a href="https://twitter.com/intent/tweet?text=Checkout+Spatchat!&url=https://spatchat.org/browse/?room=connect" target="_blank">🐦 Twitter</a> | 
        <a href="https://www.facebook.com/sharer/sharer.php?u=https://spatchat.org/browse/?room=connect" target="_blank">📘 Facebook</a> 
      </div>
    </div>
    """
    )
    gr.Markdown(
        """
        <div style="font-size: 14px;">
        © 2025 Ho Yi Wan & Logan Hysen. All rights reserved.<br>
        If you use Spatchat in research, please cite:<br>
        <b>Wan, H.Y.</b> & <b>Hysen, L.</b> (2025). <i>SpatChat: Connectivity Analysis.</i>
        </div>
    """
    )

    with gr.Row(elem_id="spatchat-workarea"):
        with gr.Column(scale=3, min_width=0, elem_id="spatchat-sidebar"):
            chatbot = gr.Chatbot(
                label="SpatChat",
                show_label=False,
                layout="bubble",
                buttons=[],
                value=[
                    {
                        "role": "assistant",
                        "content": "Hi, I'm Spatchat! This room helps you analyze landscape connectivity using a raster surface and optional node locations.\n\nUpload a raster (GeoTIFF) to begin: it should represent habitat suitability, resistance/cost, or conductance across the landscape.\nOptionally add a nodes CSV: these should be the locations you want to connect or evaluate, such as habitat patches, populations, detections, or sites, with longitude/latitude columns."
                    }
                ],
                elem_id="spatchat-chatbot",
            )
            status_output = gr.HTML(value="", visible=True, elem_id="spatchat-status")
            user_input = gr.Textbox(
                label=None,
                show_label=False,
                placeholder="Ask Spatchat...",
                lines=1,
                elem_id="spatchat-user-input",
            )

            # NEW: Separate uploaders for raster and nodes
            raster_input = gr.File(
                label="Upload Raster (GeoTIFF .tif/.tiff; required)",
                file_types=[".tif", ".tiff"],
                elem_id="spatchat-raster-input",
            )
            nodes_input = gr.File(
                label="Upload Nodes CSV (optional)",
                file_types=[".csv"],
                interactive=True,
                elem_id="spatchat-nodes-input",
            )
            download_btn = gr.DownloadButton(
                "⭳ Download Results",
                value=_ensure_placeholder_download(),
                visible=True,
                elem_id="spatchat-download"
            )
        with gr.Column(scale=0, min_width=14, elem_id="spatchat-splitter"):
            gr.HTML("<div class='spatchat-splitter-handle' onmousedown='return window.spatchatBeginResize ? window.spatchatBeginResize(event) : false;' title='Drag to resize panels'></div>")
        with gr.Column(scale=5, min_width=0, elem_id="spatchat-mapcol"):
            map_output = gr.HTML(label="Map Preview", value=_render_empty_map(), show_label=False, elem_id="spatchat-map")

    # Queue
    _prune_stale_sessions()
    _start_session_pruner()
    demo.queue(max_size=16)
    demo.unload(_cleanup_current_browser_session)

    # Wire events
    raster_input.change(
        fn=handle_raster_upload,
        inputs=[raster_input, session_cache],
        outputs=[chatbot, map_output, download_btn, status_output, session_cache]
    )
    nodes_input.change(
        fn=handle_nodes_upload,
        inputs=[nodes_input, chatbot, session_cache],
        outputs=[chatbot, map_output, download_btn, status_output, session_cache]
    )
    chat_submit = user_input.submit(
        fn=handle_chat,
        inputs=[chatbot, user_input, session_cache],
        outputs=[chatbot, map_output, download_btn, status_output, session_cache],
        show_progress="hidden",
        js="""
        (chatHistory, message, cache) => {
            const text = message ?? "";
            setTimeout(() => {
                const input = document.querySelector('#spatchat-user-input textarea, #spatchat-user-input input');
                if (input) {
                    input.value = "";
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                }
            }, 0);
            return [chatHistory, text, cache];
        }
        """,
    )

def _launch_demo(blocks: gr.Blocks) -> None:
    launch_kwargs = {"share": True, "head": _SPLITTER_HEAD}
    try:
        sig = inspect.signature(gr.Blocks.launch)
        params = getattr(sig, "parameters", {}) or {}
    except (ValueError, TypeError):
        params = {}

    if hasattr(params, "keys"):
        available = params.keys()
    else:
        try:
            available = list(params)
        except TypeError:
            available = []

    if "ssr_mode" in available:
        launch_kwargs["ssr_mode"] = False

    blocks.launch(**launch_kwargs)


# HF Spaces-friendly launch
if os.environ.get("SPATCHAT_SKIP_LAUNCH") != "1":
    _launch_demo(demo)
