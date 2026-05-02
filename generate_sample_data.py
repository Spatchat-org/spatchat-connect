"""Generate sample raster and nodes CSV for local SpatChat testing."""

from __future__ import annotations

import argparse
import sys
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


def _build_cost_surface(shape: tuple[int, int], *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    height, width = shape
    grid_y = np.linspace(0.0, 1.0, height)
    grid_x = np.linspace(0.0, 1.0, width)
    xx, yy = np.meshgrid(grid_x, grid_y)

    base_level = rng.uniform(25.0, 55.0)
    slope_x = rng.uniform(15.0, 40.0)
    slope_y = rng.uniform(-15.0, 15.0)
    gradient = base_level + slope_x * xx + slope_y * yy

    hotspot_amp = rng.uniform(25.0, 60.0)
    hotspot_x = rng.uniform(0.15, 0.85)
    hotspot_y = rng.uniform(0.15, 0.85)
    hotspot_sigma = rng.uniform(0.015, 0.045)
    hotspot = hotspot_amp * np.exp(-((xx - hotspot_x) ** 2 + (yy - hotspot_y) ** 2) / (2.0 * hotspot_sigma**2))

    valley_amp = rng.uniform(15.0, 35.0)
    valley_x = rng.uniform(0.1, 0.9)
    valley_y = rng.uniform(0.1, 0.9)
    valley_sigma = rng.uniform(0.02, 0.05)
    valley = -valley_amp * np.exp(-((xx - valley_x) ** 2 + (yy - valley_y) ** 2) / (2.0 * valley_sigma**2))

    ridge_amp = rng.uniform(8.0, 18.0)
    ridge_orientation = rng.uniform(0, np.pi)
    ridge_offset = rng.uniform(-0.2, 0.2)
    ridge = ridge_amp * np.sin(6.0 * (xx * np.cos(ridge_orientation) + yy * np.sin(ridge_orientation) + ridge_offset))

    noise_scale = rng.uniform(2.0, 6.0)
    noise = rng.normal(scale=noise_scale, size=(height, width))

    surface = gradient + hotspot + valley + ridge + noise
    return surface.astype("float32")


def _write_raster(path: Path, array: np.ndarray, *, cell_size: float, origin_lon: float, origin_lat: float) -> None:
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


def _write_nodes(path: Path, *, bounds: tuple[float, float, float, float], count: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    min_lon, min_lat, max_lon, max_lat = bounds
    lon_vals = rng.uniform(min_lon + 0.01, max_lon - 0.01, size=count)
    lat_vals = rng.uniform(min_lat + 0.01, max_lat - 0.01, size=count)
    weights = rng.integers(1, 6, size=count)

    df = pd.DataFrame(
        {
            "node_id": [f"N{i+1:02d}" for i in range(count)],
            "longitude": np.round(lon_vals, 6),
            "latitude": np.round(lat_vals, 6),
            "weight": weights,
        }
    )
    df.sort_values("node_id").to_csv(path, index=False)


def generate(output_dir: Path, *, raster_name: str, nodes_name: str, seed: int) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raster_path = output_dir / raster_name
    nodes_path = output_dir / nodes_name

    raster = _build_cost_surface((220, 240), seed=seed)
    # Upper-left origin latitude > lower latitude because pixels extend south.
    origin_lon, origin_lat = -123.5, 45.75
    cell_size = 0.01
    _write_raster(raster_path, raster, cell_size=cell_size, origin_lon=origin_lon, origin_lat=origin_lat)

    width_deg = raster.shape[1] * cell_size
    height_deg = raster.shape[0] * cell_size
    bounds = (
        origin_lon,
        origin_lat - height_deg,
        origin_lon + width_deg,
        origin_lat,
    )
    _write_nodes(nodes_path, bounds=bounds, count=12, seed=seed + 1)

    return raster_path, nodes_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate sample connectivity inputs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/examples"),
        help="Directory where files will be written (default: outputs/examples/).",
    )
    parser.add_argument(
        "--raster-name",
        default="spatchat_example_cost_surface.tif",
        help="Filename for the generated raster (default: spatchat_example_cost_surface.tif).",
    )
    parser.add_argument(
        "--nodes-name",
        default="spatchat_example_cost_nodes.csv",
        help="Filename for the generated nodes table (default: spatchat_example_cost_nodes.csv).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2027,
        help="Random seed for reproducible sampling.",
    )
    args = parser.parse_args(argv)

    raster_path, nodes_path = generate(
        args.output_dir,
        raster_name=args.raster_name,
        nodes_name=args.nodes_name,
        seed=args.seed,
    )
    print(f"Sample cost raster written to: {raster_path}")
    print(f"Sample nodes CSV written to:   {nodes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
