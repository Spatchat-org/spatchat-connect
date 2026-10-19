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
    txt = str(s).strip()
    m = re.search(r'(?i)\b(?:utm|zone)\b[^0-9]*?(\d{1,2})\s*([A-Za-z])?', txt)
    if not m:
        m = re.search(r'\b(\d{1,2})\s*([C-HJ-NP-Xc-hj-np-x])\b', txt)
    if not m:
        return None
    zone, band = int(m.group(1)), (m.group(2) or '').upper()
    if not 1 <= zone <= 60:
        return None
    # N/S are ambiguous between hemisphere suffixes and latitude bands.
    if band in ('N', 'S'):
        raise ValueError("Ambiguous UTM N/S suffix. Use an explicit EPSG code (e.g. EPSG:32617 or EPSG:32717).")
    hemi = 'N' if not band or band >= 'N' else 'S'
    return (32600 if hemi == 'N' else 32700) + zone


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

def suggest_node_columns(df):
    """Return (x_column, y_column, confident_wgs84).

    Numeric ranges alone never establish a CRS. Only explicitly named
    geographic coordinate columns can be automatically accepted as WGS84.
    """
    import pandas as pd
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None, None, False
    names = {str(c).strip().lower(): c for c in df.columns}
    for a, b in (("longitude", "latitude"), ("lon", "lat"), ("long", "lat")):
        if a in names and b in names:
            x, y = names[a], names[b]
            xx = pd.to_numeric(df[x], errors="coerce")
            yy = pd.to_numeric(df[y], errors="coerce")
            valid = (xx.notna().all() and yy.notna().all()
                     and xx.between(-180, 180).all() and yy.between(-90, 90).all())
            return x, y, bool(valid)
    for a, b in (("x", "y"), ("easting", "northing"), ("east", "north"),
                 ("utm_x", "utm_y"), ("longitude", "latitude"), ("lon", "lat")):
        if a in names and b in names:
            return names[a], names[b], False
    numeric = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().all()]
    return (numeric[0] if numeric else None,
            numeric[1] if len(numeric) > 1 else None, False)


def prepare_node_coordinates(df, x_col, y_col, source_crs, id_col=None):
    """Return a copy with validated WGS84 longitude/latitude and unique node_id.

    Preserve uploaded columns; original coordinate values are never overwritten
    except when their names are already longitude/latitude (then values are
    retained in source_longitude/source_latitude if reprojection is required).
    """
    import pandas as pd
    import numpy as np
    if x_col not in df.columns or y_col not in df.columns or x_col == y_col:
        raise ValueError("Select two distinct, existing X and Y columns.")
    if id_col is not None and id_col not in df.columns:
        raise ValueError("Selected node ID column is missing.")
    xx = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    yy = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    if not (np.isfinite(xx).all() and np.isfinite(yy).all()):
        raise ValueError("X and Y columns must contain finite numeric values in every row.")
    crs = CRS.from_user_input(parse_crs_input(source_crs) if isinstance(source_crs, str) else source_crs)
    if not crs.is_geographic and not crs.is_projected:
        raise ValueError("Use a geographic or projected coordinate reference system.")
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(xx, yy)
    lon, lat = np.asarray(lon), np.asarray(lat)
    if not (np.isfinite(lon).all() and np.isfinite(lat).all()
            and np.all((lon >= -180) & (lon <= 180))
            and np.all((lat >= -90) & (lat <= 90))):
        raise ValueError("Coordinates transform outside valid longitude/latitude bounds. Check the CRS and X/Y selection.")
    out = df.copy()
    if crs != CRS.from_epsg(4326):
        for col in ("longitude", "latitude"):
            if col in out.columns:
                original_name = f"source_{col}"
                if original_name not in out.columns:
                    out[original_name] = out[col]
    out["longitude"] = lon
    out["latitude"] = lat
    if id_col is None:
        if "node_id" in out.columns:
            ids = out["node_id"].astype(str).str.strip()
            if ids.ne("").all() and ids.is_unique and not ids.str.lower().isin({"nan", "none", "null"}).any():
                return out
        width = max(2, len(str(len(out))))
        out["node_id"] = [f"node_{i:0{width}d}" for i in range(1, len(out) + 1)]
    else:
        ids = out[id_col].astype("string").str.strip()
        if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
            raise ValueError("Node identifiers must be nonempty and unique.")
        if id_col != "node_id" and "node_id" in out.columns:
            out["source_node_id"] = out["node_id"]
        out["node_id"] = ids.astype(str)
    return out
