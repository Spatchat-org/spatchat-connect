"""Workflow helpers for interactive raster conversion."""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from convert_surfaces import convert_surface


Cache = Dict[str, Any]


def format_range(min_val: float, max_val: float) -> str:
    return f"{min_val:g}-{max_val:g}"


def conversion_prompt(cache: Cache, surface_labels: Dict[str, str]) -> str:
    surface_type = cache.get("surface_type")
    label = surface_labels.get(surface_type, surface_type) if surface_type else "surface"
    if isinstance(label, str):
        label_text = label.replace("(cost)", "").strip()
    else:
        label_text = "surface"
    min_r = float(cache.get("conversion_min_resistance", 1.0))
    max_r = float(cache.get("conversion_max_resistance", 100.0))
    mode = (cache.get("conversion_scale_mode") or "linear").lower()
    mode_text = "linearly" if mode == "linear" else "with exponential decay"
    return (
        f"The {label_text} is ready. Convert it to a cost surface (rescaled {mode_text} to {format_range(min_r, max_r)}) "
        "or to a conductance surface? Reply with 'convert to cost', 'convert to conductance', optionally add a range "
        "like 'convert to cost 5-150'. Say 'use exponential decay' or 'use linear scaling' to switch modes, or say "
        "'skip conversion' to leave the raster as-is."
    )


def _parse_range_tokens(message: str) -> Tuple[Optional[float], Optional[float], bool]:
    # Treat hyphen/emdash between digits as a range separator, not a negative sign.
    cleaned = re.sub(r"(?<=\d)\s*[-\u2013\u2014]\s*(?=\d)", " ", message)
    nums = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if not nums:
        return None, None, False
    try:
        values = [float(n) for n in nums]
    except ValueError:
        return None, None, False
    if len(values) >= 2:
        first, second = values[0], values[1]
        if first > second:
            first, second = second, first
        return first, second, True
    single = values[0]
    if "max" in message:
        return None, single, True
    if "min" in message:
        return single, None, True
    return None, None, False


def _detect_scale_mode(message: str) -> Optional[str]:
    if re.search(r"\b(exp(?:onential)?(?:\s+decay)?)\b", message):
        return "exponential"
    if "linear" in message:
        return "linear"
    if "decay" in message and "exp" in message:
        return "exponential"
    return None


def interpret_conversion_message(message: str) -> Dict[str, Any]:
    msg = message.strip().lower()
    if not msg:
        return {"action": "none"}
    if msg in {"no", "no thanks", "nah", "not now", "nope"}:
        return {"action": "skip"}
    if "skip" in msg or "keep as is" in msg or "no conversion" in msg or "leave it" in msg:
        return {"action": "skip"}
    if (
        "don't need to adjust" in msg
        or "do not need to adjust" in msg
        or "no need to adjust" in msg
        or "no adjustment" in msg
        or "leave the range" in msg
        or "keep the range" in msg
    ):
        return {"action": "keep_current"}

    min_val, max_val, range_explicit = _parse_range_tokens(msg)

    target = None
    if re.search(r"\bconduct(?:ance|ivity|ance surface)?\b", msg):
        target = "conductance"
    elif re.search(r"\b(cost|resistance|resistant|friction)\b", msg):
        target = "resistance"

    scale_mode = _detect_scale_mode(msg)
    has_convert = "convert" in msg or "conversion" in msg
    mentions_range = any(word in msg for word in ("range", "scale", "min", "max"))

    if scale_mode and not target and not range_explicit:
        return {"action": "scale_mode", "scale_mode": scale_mode}

    if target:
        return {
            "action": "convert",
            "target": target,
            "min": min_val,
            "max": max_val,
            "range_explicit": range_explicit,
            "scale_mode": scale_mode,
        }

    if range_explicit and mentions_range:
        return {"action": "range", "min": min_val, "max": max_val, "scale_mode": scale_mode}

    if mentions_range and not range_explicit:
        return {"action": "range_missing"}

    if has_convert and not target:
        return {"action": "ask_target"}

    if scale_mode:
        return {"action": "scale_mode", "scale_mode": scale_mode}

    return {"action": "none"}


def resolve_conversion_range(
    cache: Cache,
    min_override: Optional[float],
    max_override: Optional[float],
) -> Tuple[bool, str, float, float, bool]:
    current_min = float(cache.get("conversion_min_resistance", 1.0))
    current_max = float(cache.get("conversion_max_resistance", 100.0))
    changed = False

    if min_override is not None:
        current_min = float(min_override)
        changed = True
    if max_override is not None:
        current_max = float(max_override)
        changed = True

    if current_min <= 0 or current_max <= 0:
        return False, "Resistance range values must be positive.", current_min, current_max, changed
    if current_min >= current_max:
        return (
            False,
            "Please provide a minimum resistance value that is lower than the maximum.",
            current_min,
            current_max,
            changed,
        )

    cache["conversion_min_resistance"] = current_min
    cache["conversion_max_resistance"] = current_max
    return True, "", current_min, current_max, changed


def conversion_output_path(cache: Cache, converted_dir: str, target: str) -> str:
    raster_path = cache.get("raster_path") or "surface.tif"
    base = os.path.splitext(os.path.basename(raster_path))[0] or "surface"
    suffix = "cost" if target == "resistance" else target
    os.makedirs(converted_dir, exist_ok=True)
    return os.path.join(converted_dir, f"{base}_{suffix}.tif")


def execute_surface_conversion(
    cache: Cache,
    converted_dir: str,
    target: str,
    *,
    min_override: Optional[float] = None,
    max_override: Optional[float] = None,
    scale_mode: Optional[str] = None,
    announce_path: bool = True,
    source_path_override: Optional[str] = None,
    source_type_override: Optional[str] = None,
) -> Tuple[bool, str]:
    source_path = source_path_override or cache.get("raster_path")
    if not source_path or not os.path.exists(source_path):
        return False, "Upload a raster before asking for conversions."

    surface_type = source_type_override or cache.get("surface_type")
    if not surface_type:
        return False, "Let me know whether the raster is suitability, resistance, or conductance first."

    mode = (scale_mode or cache.get("conversion_scale_mode") or "linear").strip().lower()
    if mode not in {"linear", "exponential"}:
        mode = "linear"
    if target != "resistance":
        mode = "linear"

    if target == "resistance":
        ok, msg, min_val, max_val, _ = resolve_conversion_range(cache, min_override, max_override)
        if not ok:
            return False, msg
        cache["conversion_min_resistance"] = min_val
        cache["conversion_max_resistance"] = max_val
    else:
        min_val = float(cache.get("conversion_min_resistance", 1.0))
        max_val = float(cache.get("conversion_max_resistance", 100.0))

    dest_path = conversion_output_path(cache, converted_dir, target)

    try:
        convert_surface(
            input_path=source_path,
            output_path=dest_path,
            src_type=surface_type,
            dst_type=target,
            eps=1e-6,
            max_resistance=None,
            suitability_cap=1.0,
            output_nodata=None,
            force_dtype="float32",
            scale_range=(min_val, max_val) if target == "resistance" else None,
            scale_mode=mode,
        )
    except Exception as exc:
        return False, f"Conversion failed: {exc}"

    preview = cache.get("preview_rasters")
    if not isinstance(preview, dict):
        preview = {}

    if target == "resistance":
        cache["conversion_scale_mode"] = mode
        mode_label = "linear" if mode == "linear" else "exp"
        label = f"Converted cost surface [{mode_label} {format_range(min_val, max_val)}]"
        preview = {k: v for k, v in preview.items() if not k.lower().startswith("converted cost surface")}
        cache["analysis_cost_tif"] = dest_path
    else:
        label = "Converted conductance surface"
        preview = {k: v for k, v in preview.items() if not k.lower().startswith("converted conductance surface")}

    preview[label] = dest_path
    cache["preview_rasters"] = preview
    cache["conversion_target"] = target
    cache["conversion_path"] = dest_path
    cache["conversion_label"] = label
    cache["awaiting_conversion_choice"] = False
    cache["download_bundle_dirty"] = True
    cache["results_zip_path"] = None
    # Previously computed outputs refer to the old resistance values.
    # Invalidate downstream products and their preview layers.
    for stale_key in (
        "lcp_geojsonl", "lcp_index_csv", "lcp_unreachable_csv", "lcp_count_tif",
        "lcp_cost_matrix", "lcp_zip_path", "lcp_analysis_metric_tif", "lcp_crs_metadata",
        "kernel_surface_tif", "kernel_summary_json", "crk_analysis_cost_tif", "crk_crs_metadata",
        "circuit_pairs_csv", "circuit_pairs_df", "circuit_unplaced_csv",
        "circuit_summary_json", "circuit_current_density_tif", "circuit_analysis_cost_tif",
        "circuit_crs_metadata", "graph_metrics_csv", "graph_metrics_df",
        "graph_metric_name", "graph_edges_df", "graph_metric_ranges", "graph_probability_decay",
        "graph_summary_json",
    ):
        cache[stale_key] = None
    cache["preview_rasters"] = {
        key: path for key, path in cache.get("preview_rasters", {}).items()
        if not (key.startswith("Cumulative resistant kernel") or key == "Circuit current density"
                or "LCP Density" in key)
    }

    session = cache.get("session_info")
    if isinstance(session, dict):
        session["conversion_target"] = target
        session["conversion_path"] = dest_path
        session["conversion_completed_at"] = datetime.now(timezone.utc).isoformat()
        if target == "resistance":
            session["conversion_resistance_range"] = [
                float(cache.get("conversion_min_resistance", 1.0)),
                float(cache.get("conversion_max_resistance", 100.0)),
            ]
            session["conversion_scale_mode"] = mode

    if target == "resistance":
        mode_sentence = "linearly" if mode == "linear" else "with exponential decay"
        message = (
            "Converted the raster to a cost surface. "
            f"Resistance values were rescaled {mode_sentence} to {format_range(min_val, max_val)}"
        )
        # Keep conductance in sync with the latest cost scaling.
        conductance_refresh_note = ""
        conductance_path = conversion_output_path(cache, converted_dir, "conductance")
        try:
            convert_surface(
                input_path=dest_path,
                output_path=conductance_path,
                src_type="resistance",
                dst_type="conductance",
                eps=1e-6,
                max_resistance=None,
                suitability_cap=1.0,
                output_nodata=None,
                force_dtype="float32",
                scale_range=None,
                scale_mode="linear",
            )
            preview = cache.get("preview_rasters")
            if not isinstance(preview, dict):
                preview = {}
            preview = {
                k: v for k, v in preview.items()
                if not k.lower().startswith("converted conductance surface")
            }
            preview["Converted conductance surface"] = conductance_path
            cache["preview_rasters"] = preview
            conductance_refresh_note = " Conductance surface was refreshed to match the updated cost scaling."
        except Exception as exc:
            conductance_refresh_note = f" Conductance refresh failed: {exc}"
    else:
        message = "Converted the raster to a conductance surface."
        conductance_refresh_note = ""

    if announce_path:
        message = f"{message} Saved to the converted surfaces folder."
    if target == "resistance" and conductance_refresh_note:
        message = f"{message}{conductance_refresh_note}"

    return True, message


def _set_scale_mode(cache: Cache, mode: Optional[str]) -> Optional[str]:
    if not mode:
        return None
    mode_norm = mode.strip().lower()
    if mode_norm not in {"linear", "exponential"}:
        return None
    prev = cache.get("conversion_scale_mode", "linear")
    cache["conversion_scale_mode"] = mode_norm
    if mode_norm == prev:
        return f"Scaling mode remains {'exponential decay' if mode_norm == 'exponential' else 'linear'}."
    return f"Scaling mode set to {'exponential decay' if mode_norm == 'exponential' else 'linear'} for future cost conversions."


def process_conversion_chat(
    cache: Cache,
    converted_dir: str,
    chat_history: List[Dict[str, str]],
    message: str,
    surface_labels: Dict[str, str],
) -> Tuple[bool, bool]:
    msg = message.strip()
    awaiting = bool(cache.get("awaiting_conversion_choice"))

    keywords = (
        "convert", "range", "scale", "cost", "resistance", "conduct", "min", "max",
        "skip", "exp", "exponential", "decay", "linear", "adjust", "leave", "keep",
    )
    msg_lc = msg.lower()
    if not awaiting and not any(keyword in msg_lc for keyword in keywords):
        return False, False

    parsed = interpret_conversion_message(msg)
    action = parsed.get("action")

    if action == "none":
        if awaiting:
            chat_history.append({"role": "assistant", "content": conversion_prompt(cache, surface_labels)})
            return True, False
        return False, False

    if action == "skip":
        cache["awaiting_conversion_choice"] = False
        chat_history.append(
            {
                "role": "assistant",
                "content": "Okay, I'll keep the uploaded raster as-is. Ask for a conversion whenever you need it.",
            }
        )
        return True, False

    if action == "keep_current":
        cache["awaiting_conversion_choice"] = False
        target = cache.get("conversion_target")
        if target == "resistance" and cache.get("conversion_path"):
            min_val = float(cache.get("conversion_min_resistance", 1.0))
            max_val = float(cache.get("conversion_max_resistance", 100.0))
            chat_history.append(
                {
                    "role": "assistant",
                    "content": f"Okay. I'll keep using the current cost surface with resistance range {format_range(min_val, max_val)}.",
                }
            )
        else:
            chat_history.append(
                {
                    "role": "assistant",
                    "content": "Okay. I'll keep the current conversion settings unless you ask to change them.",
                }
            )
        return True, False

    if action == "ask_target":
        cache["awaiting_conversion_choice"] = True
        chat_history.append(
            {
                "role": "assistant",
                "content": "Should I convert it to a cost surface or to a conductance surface?",
            }
        )
        return True, False

    if action == "range_missing":
        cache["awaiting_conversion_choice"] = True
        chat_history.append(
            {
                "role": "assistant",
                "content": "Please provide both minimum and maximum resistance values, e.g. 'range 1-100'.",
            }
        )
        return True, False

    if action == "range":
        mode_msg = _set_scale_mode(cache, parsed.get("scale_mode"))
        ok, err_msg, min_val, max_val, _ = resolve_conversion_range(cache, parsed.get("min"), parsed.get("max"))
        if not ok:
            if mode_msg:
                chat_history.append({"role": "assistant", "content": mode_msg})
            chat_history.append({"role": "assistant", "content": err_msg})
            return True, False

        response = f"Resistance range set to {format_range(min_val, max_val)}."
        has_cost_surface = bool(cache.get("analysis_cost_tif"))
        if (
            (
                cache.get("conversion_target") == "resistance"
                and cache.get("conversion_path")
            )
            or cache.get("surface_type") == "resistance"
            or has_cost_surface
        ):
            success, conv_msg = execute_surface_conversion(cache, converted_dir, "resistance")
            if success:
                if mode_msg:
                    chat_history.append({"role": "assistant", "content": mode_msg})
                chat_history.append({"role": "assistant", "content": response})
                chat_history.append({"role": "assistant", "content": conv_msg})
                return True, True
            response = f"{response} Tried to update the converted surface but hit: {conv_msg}"
        messages = []
        if mode_msg:
            messages.append(mode_msg)
        messages.append(response)
        for text in messages:
            chat_history.append({"role": "assistant", "content": text})
        return True, False

    if action == "convert":
        target = parsed.get("target")
        mode_msg = _set_scale_mode(cache, parsed.get("scale_mode"))
        success, conv_msg = execute_surface_conversion(
            cache,
            converted_dir,
            target,
            min_override=parsed.get("min"),
            max_override=parsed.get("max"),
            scale_mode=parsed.get("scale_mode"),
        )
        if success:
            if mode_msg:
                chat_history.append({"role": "assistant", "content": mode_msg})
            chat_history.append({"role": "assistant", "content": conv_msg})
            if target == "resistance":
                min_val = float(cache.get("conversion_min_resistance", 1.0))
                max_val = float(cache.get("conversion_max_resistance", 100.0))
                chat_history.append(
                    {
                        "role": "assistant",
                        "content": f"The resistance range now spans {format_range(min_val, max_val)}. "
                        "Let me know if you want to adjust it.",
                    }
                )
            else:
                chat_history.append(
                    {
                        "role": "assistant",
                        "content": "You can also request a cost conversion whenever you're ready for cost-based analyses.",
                    }
                )
            return True, True

        if mode_msg:
            chat_history.append({"role": "assistant", "content": mode_msg})
        chat_history.append({"role": "assistant", "content": conv_msg})
        cache["awaiting_conversion_choice"] = True
        return True, False

    if action == "scale_mode":
        mode_msg = _set_scale_mode(cache, parsed.get("scale_mode"))
        if mode_msg:
            has_cost_surface = bool(cache.get("analysis_cost_tif"))
            if (
                (cache.get("conversion_target") == "resistance" and cache.get("conversion_path"))
                or cache.get("surface_type") == "resistance"
                or has_cost_surface
            ):
                success, conv_msg = execute_surface_conversion(
                    cache,
                    converted_dir,
                    "resistance",
                    scale_mode=parsed.get("scale_mode"),
                )
                if success:
                    chat_history.append({"role": "assistant", "content": mode_msg})
                    chat_history.append({"role": "assistant", "content": conv_msg})
                    return True, True
                chat_history.append({"role": "assistant", "content": f"{mode_msg} Tried to update the converted surface but hit: {conv_msg}"})
                return True, False
            chat_history.append({"role": "assistant", "content": mode_msg})
            return True, False
        chat_history.append(
            {
                "role": "assistant",
                "content": (
                    "I didn't catch that scaling mode. Say 'use linear scaling' or 'use exponential decay' to pick one."
                ),
            }
        )
        return True, False

    return False, False


__all__ = [
    "conversion_prompt",
    "process_conversion_chat",
    "execute_surface_conversion",
    "interpret_conversion_message",
    "resolve_conversion_range",
    "conversion_output_path",
    "format_range",
]


