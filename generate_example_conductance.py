#!/usr/bin/env python3
"""Generate a synthetic conductance surface for SpatChat demos.

This script creates a smooth conductance raster with several corridor-like
regions (high conductance) and barrier-like regions (low conductance) so the
resulting surface is interesting for connectivity tests.

Examples
--------
python generate_example_conductance.py --output-dir outputs/examples
"""

from __future__ import annotations

import argparse
import pathlib
import textwrap
from typing import Sequence

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from pyproj import Transformer

DEFAULT_OUTPUT_DIR = pathlib.Path("outputs/examples")


def _make_corridor_mask(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    *,
    ridge_y: float,
    width: float,
    strength: float,
) -> np.ndarray:
    """Smooth ridge aligned east-west that boosts conductance."""
    ridge = np.exp(-((grid_y - ridge_y) ** 2) / (2.0 * width ** 2))
    ridge *= strength
    return ridge.astype(np.float32)


def _make_barrier(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    radius: float,
    depth: float,
) -> np.ndarray:
    """Circular depression that lowers conductance (acts like a barrier)."""
    dist2 = (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2
    barrier = np.exp(-dist2 / (2.0 * radius ** 2))
    return (depth * barrier).astype(np.float32)


def _add_random_oases(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    *,
    rng: np.random.Generator,
    n: int,
    strength_range: Sequence[float] = (0.2, 0.6),
    radius_range: Sequence[float] = (0.05, 0.2),
) -> np.ndarray:
    """Add random bumps that act as local high-conductance oases."""
    out = np.zeros_like(grid_x, dtype=np.float32)
    for _ in range(int(n)):
        cx = rng.uniform(-1.0, 1.0)
        cy = rng.uniform(-1.0, 1.0)
        strength = rng.uniform(*strength_range)
        radius = rng.uniform(*radius_range)
        dist2 = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
        bump = strength * np.exp(-dist2 / (2.0 * radius ** 2))
        out += bump.astype(np.float32)
    return out


def build_conductance_surface(
    width: int,
    height: int,
    *,
    seed: int | None = 13,
) -> np.ndarray:
    """Create a synthetic conductance array scaled to [0, 1]."""
    rng = np.random.default_rng(seed)

    # Normalized coordinate grid in [-1, 1]
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    base = 0.35 + 0.15 * np.exp(-(grid_x ** 2 + grid_y ** 2) * 1.8)
    base += 0.05 * rng.normal(size=base.shape)

    # Add a couple of east-west corridors (higher conductance bands)
    corridors = (
        _make_corridor_mask(grid_x, grid_y, ridge_y=-0.5, width=0.1, strength=0.25)
        + _make_corridor_mask(grid_x, grid_y, ridge_y=0.3, width=0.15, strength=0.18)
    )

    # Add a diagonal barrier and a circular barrier region
    diag = 0.2 * np.exp(-((grid_y - 0.4 * grid_x) ** 2) / (2.0 * 0.08 ** 2))
    barriers = (
        diag.astype(np.float32)
        + _make_barrier(grid_x, grid_y, center_x=-0.4, center_y=0.6, radius=0.25, depth=0.35)
    )

    oases = _add_random_oases(grid_x, grid_y, rng=rng, n=6)

    conductance = base + corridors + oases - barriers
    conductance = np.clip(conductance, 0.02, None)
    conductance -= conductance.min()
    conductance /= conductance.max() + 1e-9
    return conductance.astype(np.float32)


def _write_raster(
    arr: np.ndarray,
    out_path: pathlib.Path,
    *,
    pixel_size: float = 120.0,
) -> rasterio.Affine:
    """Write the conductance array to GeoTIFF using a simple UTM grid."""
    transform = from_origin(500_000.0, 5_430_000.0, pixel_size, pixel_size)
    metadata = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32610",  # UTM Zone 10N - arbitrary but metric
        "transform": transform,
        "nodata": 0.0,
    }
    with rasterio.open(out_path, "w", **metadata) as dst:
        dst.write(arr, 1)
    return transform


def _sample_points(
    arr: np.ndarray,
    *,
    transform: rasterio.Affine,
    out_path: pathlib.Path,
    count: int,
    seed: int,
) -> None:
    """Sample random points and record their conductance values."""
    rng = np.random.default_rng(seed)

    pixel_width = transform.a
    pixel_height = abs(transform.e)
    west = transform.c
    north = transform.f
    height, width = arr.shape
    east = west + width * pixel_width
    south = north - height * pixel_height

    # Keep samples away from borders to make bilinear sampling stable.
    xs = rng.uniform(west + pixel_width, east - pixel_width, size=count)
    ys = rng.uniform(south + pixel_height, north - pixel_height, size=count)

    col = (xs - west) / pixel_width - 0.5
    row = (north - ys) / pixel_height - 0.5

    col0 = np.clip(np.floor(col).astype(int), 0, width - 1)
    col1 = np.clip(col0 + 1, 0, width - 1)
    row0 = np.clip(np.floor(row).astype(int), 0, height - 1)
    row1 = np.clip(row0 + 1, 0, height - 1)

    tx = col - col0
    ty = row - row0

    vals = (
        (1 - tx) * (1 - ty) * arr[row0, col0]
        + tx * (1 - ty) * arr[row0, col1]
        + (1 - tx) * ty * arr[row1, col0]
        + tx * ty * arr[row1, col1]
    )

    transformer = Transformer.from_crs("EPSG:32610", "EPSG:4326", always_xy=True)
    lons, lats = transformer.transform(xs, ys)

    df = pd.DataFrame(
        {
            "site_id": [f"site_{i+1:02d}" for i in range(count)],
            "longitude": np.round(lons, 6),
            "latitude": np.round(lats, 6),
            "easting": np.round(xs, 2),
            "northing": np.round(ys, 2),
            "conductance_at_site": np.round(vals, 4),
        }
    )
    df.to_csv(out_path, index=False)


def _maybe_write_preview(arr: np.ndarray, preview_path: pathlib.Path | None) -> None:
    if preview_path is None:
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    im = ax.imshow(arr, cmap="viridis", origin="upper")
    ax.set_title("Synthetic conductance")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8, label="Conductance")
    fig.savefig(preview_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__ or "").strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where generated files will be written (default: outputs/examples/).",
    )
    parser.add_argument(
        "--raster-name",
        default="spatchat_example_conductance.tif",
        help="Filename for the conductance GeoTIFF (default: spatchat_example_conductance.tif).",
    )
    parser.add_argument("--width", type=int, default=512, help="Raster width in pixels (default: 512)")
    parser.add_argument("--height", type=int, default=384, help="Raster height in pixels (default: 384)")
    parser.add_argument("--seed", type=int, default=13, help="Random seed controlling the oases placement")
    parser.add_argument("--pixel-size", type=float, default=120.0,
                        help="Pixel size in projected units (default: 120 meters)")
    parser.add_argument("--preview", type=pathlib.Path, default=None,
                        help="Optional path for a PNG preview of the surface")
    parser.add_argument(
        "--points-name",
        default="spatchat_example_conductance_points.csv",
        help="Filename for the sampled conductance points CSV (default: spatchat_example_conductance_points.csv).",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--points",
        type=pathlib.Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--points-count", type=int, default=14,
                        help="Number of sample points to generate (default: 14)")

    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raster_path = args.out if args.out is not None else output_dir / args.raster_name
    if args.points is not None:
        points_path: pathlib.Path | None = args.points
    else:
        points_name = (args.points_name or "").strip()
        points_path = None if not points_name else output_dir / points_name

    arr = build_conductance_surface(args.width, args.height, seed=args.seed)
    raster_path.parent.mkdir(parents=True, exist_ok=True)
    transform = _write_raster(arr, raster_path, pixel_size=args.pixel_size)

    if points_path is not None:
        points_path.parent.mkdir(parents=True, exist_ok=True)
        _sample_points(
            arr,
            transform=transform,
            out_path=points_path,
            count=max(1, args.points_count),
            seed=args.seed + 17,
        )

    if args.preview is not None:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        _maybe_write_preview(arr, args.preview)

    print(f"Conductance surface written to {raster_path}")
    if points_path is not None:
        print(f"Sample points written to {points_path}")
    if args.preview is not None:
        print(f"Preview saved to {args.preview}")


if __name__ == "__main__":
    main()





