"""Convert raster surfaces between suitability, resistance, and conductance.

The conversion logic follows the common ecological conventions:

* Resistance (or cost/friction) is treated as the canonical representation.
* Conductance is the reciprocal of resistance.
* Suitability is treated as conductance constrained to the ``[0, suitability_cap]``
  range (``1.0`` by default).

Input nodata cells are preserved as nodata in the output, and helper options are
available to control clipping and output dtype.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import rasterio


TYPE_ALIASES = {
    "suitability": "suitability",
    "habitat": "suitability",
    "suit": "suitability",
    "resistance": "resistance",
    "cost": "resistance",
    "costs": "resistance",
    "friction": "resistance",
    "conductance": "conductance",
    "conductivity": "conductance",
    "cond": "conductance",
}


@dataclass
class ConversionOptions:
    eps: float = 1e-6
    max_resistance: Optional[float] = None
    suitability_cap: Optional[float] = 1.0


def normalise_type(name: str) -> str:
    key = name.strip().lower()
    if key not in TYPE_ALIASES:
        raise ValueError(f"Unknown surface type '{name}'. Expected one of: {sorted(TYPE_ALIASES)}")
    return TYPE_ALIASES[key]


def to_resistance(data: np.ndarray, src_type: str, opts: ConversionOptions) -> np.ndarray:
    if src_type == "resistance":
        return data

    mask = np.full(data.shape, False, dtype=bool)
    working = data.astype(np.float64, copy=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.divide(1.0, working, out=np.zeros_like(working), where=np.abs(working) > opts.eps)
    mask |= working <= opts.eps

    inv[mask] = np.nan
    return inv


def from_resistance(resistance: np.ndarray, dst_type: str, opts: ConversionOptions) -> np.ndarray:
    res = resistance.astype(np.float64, copy=True)
    res[res <= opts.eps] = np.nan

    if opts.max_resistance is not None:
        res = np.where(res > opts.max_resistance, opts.max_resistance, res)

    if dst_type == "resistance":
        return res

    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.divide(1.0, res, out=np.zeros_like(res), where=~np.isnan(res))

    if dst_type == "conductance":
        return inv

    if dst_type == "suitability":
        if opts.suitability_cap is not None:
            inv = np.where(inv > opts.suitability_cap, opts.suitability_cap, inv)
        inv = np.where(inv < 0, 0, inv)
        return inv

    raise ValueError(f"Unsupported destination type: {dst_type}")


def convert_array(data: np.ndarray, src_type: str, dst_type: str, opts: ConversionOptions) -> np.ndarray:
    resistance = to_resistance(data, src_type, opts)
    if opts.max_resistance is not None:
        resistance = np.where(resistance > opts.max_resistance, opts.max_resistance, resistance)
    return from_resistance(resistance, dst_type, opts)


def _scale_resistance_linear(data: np.ndarray, target_range: Tuple[float, float]) -> np.ndarray:
    target_min, target_max = target_range
    if target_min <= 0 or target_max <= 0:
        raise ValueError("Resistance scaling range must use positive values.")
    if target_min >= target_max:
        raise ValueError("Resistance scaling range must have min < max.")

    scaled = data.copy()
    finite_mask = np.isfinite(scaled) & (scaled > 0)
    if not np.any(finite_mask):
        return scaled

    current_min = float(np.nanmin(scaled[finite_mask]))
    current_max = float(np.nanmax(scaled[finite_mask]))
    if current_max <= current_min:
        scaled[finite_mask] = target_min
        return scaled

    span = current_max - current_min
    scale = (target_max - target_min) / span
    scaled_values = target_min + (scaled - current_min) * scale
    scaled = np.where(finite_mask, np.clip(scaled_values, target_min, target_max), scaled)
    return scaled


def _suitability_to_resistance_exponential(
    suitability: np.ndarray,
    target_range: Tuple[float, float],
    *,
    eps: float,
) -> np.ndarray:
    target_min, target_max = target_range
    if target_min <= 0 or target_max <= 0:
        raise ValueError("Resistance scaling range must use positive values.")
    if target_min >= target_max:
        raise ValueError("Resistance scaling range must have min < max.")

    out = np.full_like(suitability, np.nan, dtype=np.float64)
    valid = np.isfinite(suitability)
    if not np.any(valid):
        return out

    ratio = target_max / target_min
    clamped = np.clip(suitability[valid], 0.0, 1.0)
    clamped = np.where(clamped < eps, eps, clamped)
    values = target_min * (ratio ** (1.0 - clamped))
    values = np.clip(values, target_min, target_max)
    out[valid] = values
    return out


def _apply_resistance_scaling(
    resistance: np.ndarray,
    target_range: Tuple[float, float],
    *,
    mode: str,
) -> np.ndarray:
    if mode == "linear":
        return _scale_resistance_linear(resistance, target_range)
    if mode == "exponential":
        # For exponential mode the expectation is that the resistance array has already been
        # generated via suitability-to-resistance mapping. If that's not the case, fall back.
        return _scale_resistance_linear(resistance, target_range)
    raise ValueError(f"Unsupported scale mode '{mode}'")


def convert_surface(
    input_path: str,
    output_path: str,
    src_type: str,
    dst_type: str,
    eps: float,
    max_resistance: Optional[float],
    suitability_cap: Optional[float],
    output_nodata: Optional[float],
    force_dtype: Optional[str],
    scale_range: Optional[Tuple[float, float]] = None,
    scale_mode: str = "linear",
) -> None:
    src_type_norm = normalise_type(src_type)
    dst_type_norm = normalise_type(dst_type)
    mode_norm = (scale_mode or "linear").strip().lower()
    if mode_norm not in {"linear", "exponential"}:
        raise ValueError(f"Unsupported scale mode '{scale_mode}'. Expected 'linear' or 'exponential'.")
    if mode_norm == "exponential" and dst_type_norm != "resistance":
        mode_norm = "linear"

    options = ConversionOptions(eps=eps, max_resistance=max_resistance, suitability_cap=suitability_cap)

    with rasterio.open(input_path) as src:
        profile = src.profile
        profile.update(count=src.count)

        if profile.get("dtype", "float32") not in ("float32", "float64"):
            profile.update(dtype="float32")

        nodata_in = src.nodata
        nodata_out = output_nodata if output_nodata is not None else nodata_in

        data = src.read(masked=True).astype(np.float64)
        original_mask = np.ma.getmaskarray(data)
        converted = np.full(data.shape, np.nan, dtype=np.float64)

        for idx in range(src.count):
            layer = data[idx]
            arr = layer.filled(np.nan)
            if (
                dst_type_norm == "resistance"
                and mode_norm == "exponential"
                and scale_range is not None
                and src_type_norm == "suitability"
            ):
                out = _suitability_to_resistance_exponential(arr, scale_range, eps=options.eps)
            else:
                out = convert_array(arr, src_type_norm, dst_type_norm, options)
            converted[idx] = out

        if dst_type_norm == "resistance" and scale_range is not None:
            if mode_norm == "exponential" and src_type_norm == "suitability":
                target_min, target_max = scale_range
                converted = np.clip(converted, target_min, target_max)
            else:
                converted = _apply_resistance_scaling(converted, scale_range, mode="linear")

        if nodata_out is None:
            nodata_out = np.nan

        out_dtype = force_dtype if force_dtype is not None else profile.get("dtype", "float32")
        if out_dtype not in ("float32", "float64"):
            out_dtype = "float32"

        profile.update(dtype=out_dtype, nodata=nodata_out)

        filled = np.where(original_mask | ~np.isfinite(converted), nodata_out, converted).astype(out_dtype)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(filled)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert raster surfaces between suitability, resistance (cost), and conductance.",
    )
    parser.add_argument("--input", required=True, help="Path to the input raster.")
    parser.add_argument("--output", required=True, help="Path to write the converted raster.")
    parser.add_argument(
        "--input-type",
        required=True,
        help="Type of the input surface (suitability, resistance/cost, conductance).",
    )
    parser.add_argument(
        "--output-type",
        required=True,
        help="Desired output surface type (suitability, resistance/cost, conductance).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="Small value to avoid division by zero during conversion.",
    )
    parser.add_argument(
        "--max-resistance",
        type=float,
        default=None,
        help="Optional upper cap applied to resistance values before conversion.",
    )
    parser.add_argument(
        "--suitability-cap",
        type=float,
        default=1.0,
        help="Optional cap applied to suitability outputs (set to None to disable).",
    )
    parser.add_argument(
        "--output-nodata",
        type=float,
        default=None,
        help="Override nodata value in the output raster (default: reuse input nodata).",
    )
    parser.add_argument(
        "--force-dtype",
        choices=["float32", "float64"],
        default=None,
        help="Force the datatype of the output raster (default: follow input).",
    )
    parser.add_argument(
        "--scale-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Optionally rescale resistance outputs to the provided min/max range.",
    )
    parser.add_argument(
        "--scale-mode",
        choices=["linear", "exponential"],
        default="linear",
        help="Scaling mode applied when generating resistance outputs (default: linear).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    convert_surface(
        input_path=args.input,
        output_path=args.output,
        src_type=args.input_type,
        dst_type=args.output_type,
        eps=args.eps,
        max_resistance=args.max_resistance,
        suitability_cap=args.suitability_cap,
        output_nodata=args.output_nodata,
        force_dtype=args.force_dtype,
        scale_range=tuple(args.scale_range) if args.scale_range else None,
        scale_mode=args.scale_mode,
    )


if __name__ == "__main__":
    main()
