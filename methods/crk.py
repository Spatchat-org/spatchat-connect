# crk.py
"""Cumulative resistant kernel analysis helpers.

This module implements a small, self-contained workflow for computing
cumulative resistant kernels over a resistance (cost) raster. It focuses on
simplicity and direct usability from scripts or interactive notebooks and does
not depend on the rest of the application stack.

Features
--------
* Parses human-friendly dispersal thresholds (e.g., ``"750 m"`` or ``"2 km"``).
* Prompts the user for a bandwidth when it is not supplied programmatically.
* Computes Gaussian kernels by default, with an optional Epanechnikov kernel.
* Writes an output GeoTIFF and returns the in-memory surface when requested.

Typical usage
-------------
>>> from methods import crk
>>> surface = crk.compute_cumulative_resistant_kernel(
...     "cost_surface.tif", nodes_df, crk.ask_for_bandwidth(),
... )

The helper also exposes a tiny CLI so the module can be invoked directly:

```
python -m methods.crk cost.tif nodes.csv --out kernel.tif
```
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Optional, Sequence
import argparse
from collections import Counter
import math
import os
import re
import sys

import numpy as np
import pandas as pd
import rasterio
try:
    from analysis_tracker import record_step as _record_analysis_step
except Exception:  # pragma: no cover - optional dependency
    _record_analysis_step = None

from rasterio.transform import Affine
from pyproj import CRS, Transformer
from skimage.graph import MCP, MCP_Geometric

__all__ = [
    "CRKParams",
    "parse_bandwidth_text",
    "ask_for_bandwidth",
    "compute_cumulative_resistant_kernel",
]


# ---------------------------------------------------------------------------
# Dataclass configuration
# ---------------------------------------------------------------------------

@dataclass
class CRKParams:
    """Configuration parameters for resistant kernel computation."""

    id_col: str = "node_id"
    lon_col: str = "longitude"
    lat_col: str = "latitude"
    nodes_crs: Optional[str] = "EPSG:4326"
    fully_connected: bool = True
    geometric: bool = True
    barrier_cost: float = 1e12
    use_float32: bool = True
    kernel: str = "gaussian"  # "gaussian" | "epanechnikov"
    cutoff: Optional[float] = None
    normalize: bool = False


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

_DISTANCE_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z]*)\s*$")

_UNIT_ALIASES = {
    "m": "m",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "km": "km",
    "kilometer": "km",
    "kilometers": "km",
    "kilometre": "km",
    "kilometres": "km",
}


def parse_bandwidth_text(text: str) -> float:
    """Parse a human-entered dispersal threshold and return meters.

    Parameters
    ----------
    text:
        User-provided string such as ``"1000"``, ``"850 m"`` or ``"1.2 km"``.

    Returns
    -------
    float
        The numeric value converted to meters.

    Raises
    ------
    ValueError
        If the value cannot be parsed or is non-positive.
    """

    if text is None:
        raise ValueError("Bandwidth text cannot be None.")
    txt = str(text).strip()
    if not txt:
        raise ValueError("Bandwidth cannot be empty.")

    m = _DISTANCE_RE.match(txt)
    if not m:
        raise ValueError(f"Unrecognized distance format: {text!r}")

    value = float(m.group(1))
    if value <= 0:
        raise ValueError("Bandwidth must be positive.")

    unit = m.group(2).lower()
    if unit and unit not in _UNIT_ALIASES:
        raise ValueError(f"Unknown distance unit: {unit!r}")

    norm_unit = _UNIT_ALIASES.get(unit, "m")
    if norm_unit == "km":
        value *= 1000.0
    return float(value)


def ask_for_bandwidth(
    prompt: str = "Enter dispersal threshold/bandwidth (e.g., '500 m' or '2 km'): ",
    *,
    input_fn: Optional[Callable[[str], str]] = None,
    print_fn: Optional[Callable[[str], None]] = None,
) -> float:
    """Prompt the user for a dispersal threshold and return it in meters.

    The helper keeps asking until a valid value is provided. ``input_fn`` and
    ``print_fn`` arguments make the function testable by injecting mocks.
    """

    if input_fn is None:
        input_fn = input  # type: ignore
    if print_fn is None:
        print_fn = print  # type: ignore

    while True:
        try:
            raw = input_fn(prompt)
        except EOFError:
            raise RuntimeError("No bandwidth provided (EOF).") from None
        except KeyboardInterrupt:
            raise RuntimeError("Bandwidth prompt cancelled by user.") from None

        try:
            return parse_bandwidth_text(raw)
        except ValueError as exc:
            print_fn(f"Invalid bandwidth: {exc}. Please try again.")


def _xy_to_rc(transform: Affine, x: float, y: float) -> tuple[int, int]:
    col, row = (~transform) * (x, y)
    r = int(math.floor(row + 0.5))
    c = int(math.floor(col + 0.5))
    return r, c


def _estimate_pixel_size(transform: Affine) -> float:
    """Return an average pixel size (map units)."""

    sx = math.hypot(transform.a, transform.b)
    sy = math.hypot(transform.d, transform.e)
    if sx == 0 and sy == 0:
        raise ValueError("Invalid transform; pixel size cannot be zero.")
    if sx == 0:
        return float(sy)
    if sy == 0:
        return float(sx)
    return float((sx + sy) / 2.0)


def _prepare_transformer(nodes_crs: Optional[str], raster_crs) -> Optional[Transformer]:
    if not nodes_crs:
        return None
    if raster_crs is None:
        raise ValueError("Cost raster lacks CRS; cannot reproject nodes.")
    nodes_crs_obj = CRS.from_user_input(nodes_crs)
    raster_crs_obj = CRS.from_user_input(raster_crs)
    if nodes_crs_obj == raster_crs_obj:
        return None
    return Transformer.from_crs(nodes_crs_obj, raster_crs_obj, always_xy=True)


def _collect_sources(
    nodes_df: pd.DataFrame,
    params: CRKParams,
    *,
    transform: Affine,
    raster_crs,
    shape: tuple[int, int],
    traversable_mask: Optional[np.ndarray] = None,
) -> tuple[list[tuple[int, int]], list[str], list[str]]:
    """Return valid raster source cells plus skipped node ids."""

    H, W = shape
    required_cols = (params.id_col, params.lon_col, params.lat_col)
    for col in required_cols:
        if col not in nodes_df.columns:
            raise ValueError(f"nodes_df must contain column '{col}'")

    transformer = _prepare_transformer(params.nodes_crs, raster_crs)

    sources_rc: list[tuple[int, int]] = []
    skipped: list[str] = []
    invalid_cell_ids: list[str] = []

    for row in nodes_df.itertuples(index=False):
        try:
            x = float(getattr(row, params.lon_col))
            y = float(getattr(row, params.lat_col))
        except Exception:
            skipped.append(str(getattr(row, params.id_col)))
            continue

        if transformer is not None:
            x, y = transformer.transform(x, y)

        r, c = _xy_to_rc(transform, x, y)
        if not (0 <= r < H and 0 <= c < W):
            skipped.append(str(getattr(row, params.id_col)))
            continue

        if traversable_mask is not None and not traversable_mask[r, c]:
            invalid_cell_ids.append(str(getattr(row, params.id_col)))
            continue

        sources_rc.append((r, c))

    if not sources_rc:
        raise ValueError("No valid nodes fall within the raster extent.")

    return sources_rc, skipped, invalid_cell_ids


def _kernel_from_costs(costs: np.ndarray, *, sigma: float, kernel: str, cutoff: Optional[float]) -> np.ndarray:
    """Compute kernel weights from a cost-distance array."""

    costs = np.asarray(costs, dtype=np.float64)
    out = np.zeros_like(costs, dtype=np.float64)

    mask = np.isfinite(costs)
    if not np.any(mask):
        return out

    valid = costs[mask]
    if kernel == "gaussian":
        weights = np.exp(-0.5 * (valid / sigma) ** 2)
        if cutoff is not None:
            weights = np.where(valid <= cutoff, weights, 0.0)
    elif kernel == "epanechnikov":
        bandwidth = cutoff if cutoff is not None else sigma
        weights = np.zeros_like(valid)
        inner = valid <= bandwidth
        if np.any(inner):
            weights[inner] = 1.0 - (valid[inner] ** 2) / (bandwidth ** 2)
            weights[inner] = np.maximum(weights[inner], 0.0)
    else:
        raise ValueError(f"Unsupported kernel '{kernel}'")

    out[mask] = weights
    return out


def _source_window(
    rc: tuple[int, int],
    shape: tuple[int, int],
    *,
    cutoff: Optional[float],
    min_cost: Optional[float],
) -> tuple[slice, slice, tuple[int, int]]:
    """Return a local raster window that safely contains the cutoff search area."""

    r, c = rc
    height, width = shape
    if cutoff is None or min_cost is None or not np.isfinite(min_cost) or min_cost <= 0:
        row_slice = slice(0, height)
        col_slice = slice(0, width)
        return row_slice, col_slice, rc

    radius = int(math.ceil(float(cutoff) / float(min_cost))) + 1
    r0 = max(0, r - radius)
    r1 = min(height, r + radius + 1)
    c0 = max(0, c - radius)
    c1 = min(width, c + radius + 1)
    row_slice = slice(r0, r1)
    col_slice = slice(c0, c1)
    return row_slice, col_slice, (r - r0, c - c0)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_cumulative_resistant_kernel(
    cost_tif: str,
    nodes_df: pd.DataFrame,
    bandwidth_meters: float,
    params: Optional[CRKParams] = None,
    *,
    out_tif: Optional[str] = None,
    return_surface: bool = True,
) -> dict:
    """Compute a cumulative resistant kernel surface.

    Parameters
    ----------
    cost_tif:
        Path to the resistance raster (GeoTIFF).
    nodes_df:
        DataFrame with node identifiers and coordinates.
    bandwidth_meters:
        Dispersal threshold/bandwidth in meters (converted from user input).
    params:
        Optional :class:`CRKParams` controlling computation details.
    out_tif:
        If provided, the cumulative kernel surface is written to this path.
    return_surface:
        When ``True`` (default), the in-memory array is included in the return
        dictionary.

    Returns
    -------
    dict
        Keys include ``surface`` (optional), ``bandwidth_m``, ``kernel`` and
        metadata about skipped/used nodes.
    """

    if params is None:
        params = CRKParams()

    if bandwidth_meters <= 0:
        raise ValueError("bandwidth_meters must be positive.")

    if not os.path.exists(cost_tif):
        raise FileNotFoundError(cost_tif)

    with rasterio.open(cost_tif) as ds:
        raster_crs = ds.crs
        transform = ds.transform
        nodata = ds.nodata
        height, width = ds.height, ds.width

        arr = ds.read(1, masked=True)
        arr = arr.astype(np.float32 if params.use_float32 else np.float64, copy=False)
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        arr = np.where(~np.isfinite(arr), np.nan, arr)
        traversable_mask = np.isfinite(arr)
        arr = np.where(~traversable_mask, float(params.barrier_cost), arr)

        pixel_size = _estimate_pixel_size(transform)
        # MCP_Geometric uses grid-step units (1 per axial move) rather than map units.
        # Scale resistance by pixel size so accumulated path cost is in resistance*meters.
        # With resistance ~= 1, user-entered bandwidth/cutoff in meters maps correctly.
        arr = arr * float(pixel_size)
        sigma = float(bandwidth_meters)

        # Warn if CRS is not projected in meters; kernels assume meter units.
        if raster_crs is not None:
            try:
                crs_obj = CRS.from_user_input(raster_crs)
            except Exception:
                crs_obj = None
            if crs_obj is not None and crs_obj.is_projected:
                axis_units = {info.unit_name.lower() for info in crs_obj.axis_info if info.unit_name}
                if axis_units and not any("metre" in u or "meter" in u for u in axis_units):
                    raise ValueError(
                        "Cost raster CRS is projected but not in meters; cannot interpret bandwidth."
                    )
            elif crs_obj is not None and not crs_obj.is_projected:
                raise ValueError(
                    "Cost raster must use a projected CRS in meters to interpret bandwidth inputs."
                )

        sources_rc, skipped_ids, invalid_cell_ids = _collect_sources(
            nodes_df,
            params,
            transform=transform,
            raster_crs=raster_crs,
            shape=(height, width),
            traversable_mask=traversable_mask,
        )
        if len(sources_rc) < 1:
            raise ValueError("No valid nodes fall within traversable raster cells.")

        source_counts = Counter(sources_rc)

        min_traversable_cost = float(np.nanmin(arr[traversable_mask])) if np.any(traversable_mask) else None
        surface = np.zeros((height, width), dtype=np.float64)
        solver_cls = MCP_Geometric if params.geometric else MCP

        search_limit = params.cutoff if params.cutoff is not None else None

        for rc, count in source_counts.items():
            row_slice, col_slice, local_rc = _source_window(
                rc,
                (height, width),
                cutoff=params.cutoff,
                min_cost=min_traversable_cost,
            )
            arr_view = arr[row_slice, col_slice]
            solver = solver_cls(arr_view, fully_connected=params.fully_connected)
            costs, _ = solver.find_costs(
                starts=[local_rc],
                ends=None,
                max_cost=search_limit,
            )
            kernel = _kernel_from_costs(costs, sigma=sigma, kernel=params.kernel, cutoff=params.cutoff)
            if count == 1:
                surface[row_slice, col_slice] += kernel
            else:
                surface[row_slice, col_slice] += kernel * float(count)

        if params.normalize:
            surface /= float(sum(source_counts.values()))

        result_surface = surface.astype(np.float32 if params.use_float32 else np.float64, copy=False)

        surface_path = None
        if out_tif:
            profile = ds.profile.copy()
            profile.update(dtype=rasterio.float32 if params.use_float32 else rasterio.float64, count=1, nodata=None)
            with rasterio.open(out_tif, "w", **profile) as dst:
                dst.write(result_surface.astype(profile["dtype"]), 1)
            surface_path = out_tif

    result = {
        "bandwidth_m": float(bandwidth_meters),
        "pixel_size": float(pixel_size),
        "kernel": params.kernel,
        "cutoff": params.cutoff,
        "normalize": params.normalize,
        "sources_used": int(sum(source_counts.values())),
        "sources_unique_cells": int(len(source_counts)),
        "sources_skipped": skipped_ids + invalid_cell_ids,
        "surface_path": surface_path,
    }

    if return_surface:
        result["surface"] = result_surface

    if _record_analysis_step:
        try:
            inputs = {
                "cost_tif": cost_tif,
                "nodes_count": int(len(nodes_df)),
                "bandwidth_meters": float(bandwidth_meters),
                "params": asdict(params),
            }
            outputs = {
                "surface_path": surface_path,
                "sources_used": int(sum(source_counts.values())),
                "sources_unique_cells": int(len(source_counts)),
                "sources_skipped": list(skipped_ids + invalid_cell_ids),
            }
            _record_analysis_step(
                kind="resistant_kernel",
                function="methods.crk.compute_cumulative_resistant_kernel",
                inputs=inputs,
                outputs=outputs,
                code_obj=compute_cumulative_resistant_kernel,
            )
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute cumulative resistant kernels")
    parser.add_argument("cost_tif", help="Path to the resistance (cost) raster GeoTIFF")
    parser.add_argument("nodes_csv", help="CSV file with nodes (must include longitude/latitude columns)")
    parser.add_argument("--bandwidth", "-b", help="Dispersal threshold (e.g., '500 m' or '1.2 km')")
    parser.add_argument("--kernel", choices=["gaussian", "epanechnikov"], default="gaussian")
    parser.add_argument("--cutoff", type=float, default=None, help="Optional cost-distance cutoff for kernels")
    parser.add_argument("--normalize", action="store_true", help="Normalize by number of source nodes")
    parser.add_argument("--id-col", default="node_id", help="Node identifier column name")
    parser.add_argument("--lon-col", default="longitude", help="Longitude/X column name")
    parser.add_argument("--lat-col", default="latitude", help="Latitude/Y column name")
    parser.add_argument("--nodes-crs", default="EPSG:4326", help="CRS of node coordinates")
    parser.add_argument("--out", dest="out_tif", default=None, help="Output GeoTIFF path for the cumulative kernel surface")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        nodes_df = pd.read_csv(args.nodes_csv)
    except Exception as exc:  # pragma: no cover - CLI convenience
        parser.error(f"Failed to read nodes CSV: {exc}")

    bandwidth_m = parse_bandwidth_text(args.bandwidth) if args.bandwidth else ask_for_bandwidth()

    params = CRKParams(
        id_col=args.id_col,
        lon_col=args.lon_col,
        lat_col=args.lat_col,
        nodes_crs=args.nodes_crs,
        kernel=args.kernel,
        cutoff=args.cutoff,
        normalize=args.normalize,
    )

    try:
        res = compute_cumulative_resistant_kernel(
            cost_tif=args.cost_tif,
            nodes_df=nodes_df,
            bandwidth_meters=bandwidth_m,
            params=params,
            out_tif=args.out_tif,
        )
    except Exception as exc:  # pragma: no cover - CLI convenience
        parser.error(str(exc))
        return 2

    if res.get("surface_path"):
        print(f"Wrote cumulative resistant kernel raster to {res['surface_path']}")
    print(f"Used {res['sources_used']} source nodes (skipped {len(res['sources_skipped'])}).")
    print(f"Bandwidth: {res['bandwidth_m']:.3f} meters; kernel: {res['kernel']}.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())




