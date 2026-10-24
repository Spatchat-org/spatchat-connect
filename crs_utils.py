"""Shared CRS and metric-grid preparation for Spatchat Connectivity.

Original uploads are never overwritten. Reprojection is only performed when
needed for a north-up, square-cell, metre-based analysis grid.
"""
import math
import os
import re
from typing import Optional, Tuple

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.warp import Resampling, calculate_default_transform, reproject


def _parse_epsg_literal(s):
    s = str(s).strip()
    m = re.search(r'(?i)\bepsg\s*:\s*(\d{4,6})\b', s) or re.match(r'^\s*(\d{4,6})\s*$', s)
    return int(m.group(1)) if m else None


def _parse_utm_any(s):
    """Parse WGS84 UTM zone with hemisphere or MGRS latitude band.

    N means north in either convention. Bare S is ambiguous, so explicit
    'South' is required for the southern hemisphere; 'band S' means MGRS.
    """
    txt = str(s).strip()
    pattern = (r"(?i)^(?:(?:wgs\s*84\s*[/,-]?\s*)?utm\s*)?"
               r"(?:zone\s*)?(\d{1,2})\s*([A-Za-z]+)?$")
    m = re.fullmatch(pattern, txt)
    if not m:
        # Explicit MGRS latitude-band notation, e.g. 'UTM zone 17 band S'
        m_band = re.fullmatch(
            r"(?i)(?:utm\s*)?(?:zone\s*)?(\d{1,2})\s*(?:latitude\s*)?band\s*([C-HJ-NP-X])",
            txt,
        )
        if not m_band:
            return None
        zone = int(m_band.group(1))
        suffix = m_band.group(2).upper()
        explicit_band = True
    else:
        zone = int(m.group(1))
        suffix = (m.group(2) or "").upper()
        explicit_band = False
    if not 1 <= zone <= 60:
        raise ValueError("UTM zone must be between 1 and 60.")
    if suffix in ("NORTH", "NORTHERN"):
        north = True
    elif suffix in ("SOUTH", "SOUTHERN"):
        north = False
    elif not suffix:
        raise ValueError("Specify a UTM hemisphere or latitude band, e.g. '17N', '17M', or '17 South'.")
    elif len(suffix) == 1 and suffix in "CDEFGHJKLMNPQRSTUVWX":
        if suffix == "S" and not explicit_band:
            raise ValueError("'17S' is ambiguous: use '17 South' for Southern Hemisphere or '17 band S' for MGRS latitude band S.")
        north = suffix >= "N"
    else:
        raise ValueError("Unrecognized UTM suffix. Use a hemisphere or MGRS latitude band C-X (excluding I and O).")
    return (32600 if north else 32700) + zone


def parse_crs_input(user_text: str) -> int:
    code = _parse_epsg_literal(user_text) or _parse_utm_any(user_text)
    if not code:
        raise ValueError("Invalid CRS. Supply an EPSG code such as 'EPSG:32610'.")
    CRS.from_epsg(code)
    return code


def utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    if not (math.isfinite(lon) and math.isfinite(lat) and -180 <= lon <= 180 and -80 <= lat <= 84):
        raise ValueError('Automatic UTM requires a location between 80 S and 84 N.')
    zone = max(1, min(60, int(math.floor((lon + 180) / 6) + 1)))
    return (32600 if lat >= 0 else 32700) + zone


def centroid_lonlat_from_bounds(bounds, src_crs) -> Optional[Tuple[float, float]]:
    if src_crs is None or bounds is None:
        return None
    try:
        cx = (float(bounds.left) + float(bounds.right)) / 2
        cy = (float(bounds.bottom) + float(bounds.top)) / 2
        lon, lat = Transformer.from_crs(src_crs, 'EPSG:4326', always_xy=True).transform(cx, cy)
        if math.isfinite(lon) and math.isfinite(lat):
            return float(lon), float(lat)
    except Exception:
        pass
    return None


def needs_meter_projection(crs) -> bool:
    if crs is None:
        raise ValueError('Raster has no CRS. Assign the correct CRS before analysis.')
    obj = CRS.from_user_input(crs)
    if not obj.is_projected:
        return True
    units = [axis.unit_name.lower() for axis in obj.axis_info if axis.unit_name]
    return not (len(units) >= 2 and all('metre' in u or 'meter' in u for u in units[:2]))


def choose_meter_crs_for_raster(raster_path: str, centroid_hint=None) -> Optional[int]:
    if not raster_path or not os.path.exists(raster_path):
        return None
    with rasterio.open(raster_path) as ds:
        if ds.crs is None:
            raise ValueError('Raster has no CRS. Assign its CRS before connectivity analysis.')
        if not needs_meter_projection(ds.crs):
            return None
        ll = centroid_hint if centroid_hint is not None else centroid_lonlat_from_bounds(ds.bounds, ds.crs)
        if ll is None:
            raise ValueError('Could not determine a suitable projected CRS from raster bounds.')
        return utm_epsg_from_lonlat(*map(float, ll))


def _metric_square_north_up(ds, tolerance=1e-6):
    if needs_meter_projection(ds.crs):
        return False
    t = ds.transform
    return (t.a > 0 and t.e < 0 and abs(t.b) <= tolerance and abs(t.d) <= tolerance
            and abs(abs(t.a) - abs(t.e)) <= tolerance * max(abs(t.a), abs(t.e), 1))


def reproject_raster_to_crs(src_path: str, dst_path: str, dst_crs, *,
                            resampling=Resampling.nearest, resolution_m=None) -> None:
    """Reproject without interpolating hard barriers by default."""
    if os.path.abspath(src_path) == os.path.abspath(dst_path):
        raise ValueError('Destination must differ from the source raster.')
    with rasterio.open(src_path) as src:
        if src.crs is None:
            raise ValueError('Source raster has no CRS.')
        target = CRS.from_user_input(dst_crs)
        if resolution_m is None:
            approx_transform, _, _ = calculate_default_transform(
                src.crs, target, src.width, src.height, *src.bounds)
            resolution_m = min(abs(approx_transform.a), abs(approx_transform.e))
        resolution_m = float(resolution_m)
        if not math.isfinite(resolution_m) or resolution_m <= 0:
            raise ValueError('Invalid target resolution.')
        transform, width, height = calculate_default_transform(
            src.crs, target, src.width, src.height, *src.bounds,
            resolution=(resolution_m, resolution_m))
        if width * height > 150_000_000:
            raise ValueError('Projected raster would exceed 150 million cells; choose a coarser resolution.')
        profile = src.profile.copy()
        nodata = src.nodata
        if nodata is None:
            # Floating-point rasters can safely carry NaN as a NoData marker.
            profile.update(dtype='float32', nodata=np.nan)
            nodata = np.nan
        profile.update(driver='GTiff', crs=target, transform=transform,
                       width=width, height=height, nodata=nodata, compress='lzw')
        os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
        with rasterio.open(dst_path, 'w', **profile) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(source=rasterio.band(src, band_idx),
                          destination=rasterio.band(dst, band_idx),
                          src_transform=src.transform, src_crs=src.crs,
                          src_nodata=src.nodata, dst_transform=transform,
                          dst_crs=target, dst_nodata=nodata,
                          resampling=resampling)


def prepare_metric_analysis_raster(src_path: str, output_dir: str, *,
                                   resampling=Resampling.nearest,
                                   resolution_m=None):
    """Return (analysis_path, metadata). Preserve already-suitable input rasters.

    Use nearest by default for cost/resistance and barrier surfaces. For truly
    continuous surfaces, callers can explicitly request bilinear resampling.
    """
    if not os.path.isfile(src_path):
        raise FileNotFoundError(src_path)
    with rasterio.open(src_path) as ds:
        if ds.crs is None:
            raise ValueError('Raster has no CRS; identify its CRS before running connectivity analyses.')
        original_crs = str(ds.crs)
        already_suitable = _metric_square_north_up(ds)
        target_crs = ds.crs if not needs_meter_projection(ds.crs) else CRS.from_epsg(
            choose_meter_crs_for_raster(src_path))
        source_resolution = tuple(float(x) for x in ds.res)
    if already_suitable and resolution_m is None:
        return src_path, {'original_crs': original_crs, 'analysis_crs': original_crs,
                          'reprojected': False, 'analysis_path': src_path,
                          'source_resolution': source_resolution}
    import hashlib
    key = hashlib.sha256((os.path.abspath(src_path) + str(os.stat(src_path).st_mtime_ns)
                          + str(target_crs) + str(resolution_m) + str(resampling)).encode()).hexdigest()[:16]
    os.makedirs(output_dir, exist_ok=True)
    dst_path = os.path.join(output_dir, f'analysis_metric_{key}.tif')
    if not os.path.isfile(dst_path):
        reproject_raster_to_crs(src_path, dst_path, target_crs,
                                resampling=resampling, resolution_m=resolution_m)
    return dst_path, {'original_crs': original_crs, 'analysis_crs': str(target_crs),
                      'reprojected': True, 'analysis_path': dst_path,
                      'source_resolution': source_resolution,
                      'resampling': str(resampling)}

# -----------------------------------------------------------------------------
# Node CSV CRS support (same module as raster CRS utilities)
# -----------------------------------------------------------------------------


def validate_metric_analysis_raster(path: str) -> None:
    """Validate the metric square-cell grid required by connectivity methods."""
    with rasterio.open(path) as ds:
        if ds.crs is None:
            raise ValueError("Analysis raster has no CRS.")
        if needs_meter_projection(ds.crs):
            raise ValueError("Analysis raster must have a projected metric CRS.")
        if not _metric_square_north_up(ds):
            raise ValueError(
                "Analysis raster must have north-up, square metric cells. "
                "Run prepare_metric_analysis_raster first."
            )
