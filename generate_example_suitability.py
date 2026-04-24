"""Generate a synthetic suitability raster and sampling points for SpatChat demos."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import rasterio
    from rasterio.transform import from_origin
except ImportError as exc:  # pragma: no cover - runtime dependency only
    raise SystemExit(
        "rasterio is required to generate the sample GeoTIFF. "
        "Install it in this environment before running this script."
    ) from exc


def _build_suitability_surface(shape: tuple[int, int], *, seed: int) -> np.ndarray:
    """Create a smooth suitability surface scaled to [0, 1]."""
    rng = np.random.default_rng(seed)
    height, width = shape
    grid_y = np.linspace(0.0, 1.0, height)
    grid_x = np.linspace(0.0, 1.0, width)
    xx, yy = np.meshgrid(grid_x, grid_y)

    base = 0.25 + 0.5 * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)

    oasis_count = rng.integers(2, 4)
    oasis = np.zeros_like(base)
    for _ in range(oasis_count):
        amp = rng.uniform(0.25, 0.45)
        cx = rng.uniform(0.1, 0.9)
        cy = rng.uniform(0.1, 0.9)
        sigma = rng.uniform(0.04, 0.12)
        oasis += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2))

    barrier_count = rng.integers(2, 4)
    barriers = np.zeros_like(base)
    for _ in range(barrier_count):
        depth = rng.uniform(0.2, 0.4)
        cx = rng.uniform(0.1, 0.9)
        cy = rng.uniform(0.1, 0.9)
        sigma_x = rng.uniform(0.05, 0.15)
        sigma_y = rng.uniform(0.05, 0.15)
        barriers -= depth * np.exp(-(((xx - cx) / sigma_x) ** 2 + ((yy - cy) / sigma_y) ** 2))

    trend = 0.2 * xx + 0.15 * yy
    noise = rng.normal(scale=0.03, size=(height, width))

    surface = base + oasis + barriers + trend + noise
    surface = np.clip(surface, 0.0, 1.0)
    return surface.astype("float32")


def _write_raster(
    path: Path,
    array: np.ndarray,
    *,
    cell_size: float,
    origin_lon: float,
    origin_lat: float,
) -> rasterio.Affine:
    transform = from_origin(origin_lon, origin_lat, cell_size, cell_size)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(array, 1)
    return transform


def _sample_points(
    path: Path,
    *,
    surface: np.ndarray,
    transform: rasterio.Affine,
    seed: int,
    count: int = 14,
) -> None:
    rng = np.random.default_rng(seed)
    height, width = surface.shape

    if height <= 1:
        row = np.zeros(count, dtype=np.float32)
    else:
        row = rng.uniform(0.0, height - 1.0, size=count)

    if width <= 1:
        col = np.zeros(count, dtype=np.float32)
    else:
        col = rng.uniform(0.0, width - 1.0, size=count)

    row0 = np.floor(row).astype(int)
    row1 = np.clip(row0 + 1, 0, height - 1)
    col0 = np.floor(col).astype(int)
    col1 = np.clip(col0 + 1, 0, width - 1)

    ty = row - row0
    tx = col - col0

    vals = (
        (1 - tx) * (1 - ty) * surface[row0, col0]
        + tx * (1 - ty) * surface[row0, col1]
        + (1 - tx) * ty * surface[row1, col0]
        + tx * ty * surface[row1, col1]
    )

    pixel_width = transform.a
    pixel_height = abs(transform.e)
    west = transform.c
    north = transform.f

    lon = west + (col + 0.5) * pixel_width
    lat = north - (row + 0.5) * pixel_height

    df = pd.DataFrame(
        {
            "site_id": [f"S{i+1:02d}" for i in range(count)],
            "longitude": np.round(lon, 6),
            "latitude": np.round(lat, 6),
            "suitability_at_site": np.round(vals, 4),
        }
    )
    df.sort_values("site_id").to_csv(path, index=False)


def generate(output_dir: Path, *, raster_name: str, points_name: str, seed: int) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raster_path = output_dir / raster_name
    points_path = output_dir / points_name

    surface = _build_suitability_surface((210, 230), seed=seed)

    origin_lon, origin_lat = -114.2, 47.85
    cell_size = 0.008
    transform = _write_raster(
        raster_path,
        surface,
        cell_size=cell_size,
        origin_lon=origin_lon,
        origin_lat=origin_lat,
    )

    _sample_points(points_path, surface=surface, transform=transform, seed=seed + 5)

    return raster_path, points_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate example suitability surface and sampling points.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/examples"),
        help="Directory where files will be written (default: outputs/examples/).",
    )
    parser.add_argument(
        "--raster-name",
        default="spatchat_example_suitability.tif",
        help="Filename for the generated suitability raster (default: spatchat_example_suitability.tif).",
    )
    parser.add_argument(
        "--points-name",
        default="spatchat_example_suitability_points.csv",
        help="Filename for the generated sampling points (default: spatchat_example_suitability_points.csv).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=314,
        help="Random seed for reproducible sampling.",
    )
    args = parser.parse_args(argv)

    raster_path, points_path = generate(
        args.output_dir,
        raster_name=args.raster_name,
        points_name=args.points_name,
        seed=args.seed,
    )
    print(f"Example suitability raster written to: {raster_path}")
    print(f"Example sampling points written to:   {points_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
