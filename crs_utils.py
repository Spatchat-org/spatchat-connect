# crs_utils.py
import math
import os
import re
from typing import Optional, Tuple

import rasterio
from pyproj import CRS, Transformer
from rasterio.warp import Resampling, calculate_default_transform, reproject

def _parse_epsg_literal(s):
    s = str(s).strip()
    m = re.search(r'(?i)\bepsg\s*:\s*(\d{4,6})\b', s) or re.match(r'^\s*(\d{4,6})\s*$', s)
    return int(m.group(1)) if m else None

def _parse_utm_any(s):
    txt = str(s).strip()
    patterns = [
        r'(?i)\butm\b[^0-9]*?(\d{1,2})\s*([A-Za-z])?',
        r'(?i)\bzone\s*(\d{1,2})\s*([A-Za-z])?',
        r'\b(\d{1,2})\s*([C-HJ-NP-Xc-hj-np-x])\b',
        r'\b(\d{1,2})\s*([NnSs])\b',
    ]
    import re as _re
    m=None
    for p in patterns:
        m=_re.search(p, txt)
        if m: break
    if not m: return None
    zone=int(m.group(1)); band=(m.group(2) or '').upper()
    if band in ('N','S'):
        hemi = 'N' if band=='N' else 'S'
    elif band:
        hemi = 'N' if band>='N' else 'S'
    else:
        hemi = 'N'
    return (32600 if hemi=='N' else 32700) + zone

def parse_crs_input(user_text: str) -> int:
    code = _parse_epsg_literal(user_text) or _parse_utm_any(user_text)
    if not code:
        raise ValueError("Invalid CRS. Try 'EPSG:32610', '32610', 'UTM 10T', or 'zone 10N'.")
    return code


def utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    zone = int(math.floor((float(lon) + 180.0) / 6.0) + 1)
    zone = max(1, min(60, zone))
    return (32600 if float(lat) >= 0.0 else 32700) + zone


def centroid_lonlat_from_bounds(bounds, src_crs) -> Optional[Tuple[float, float]]:
    if src_crs is None or bounds is None:
        return None
    try:
        cx = (float(bounds.left) + float(bounds.right)) / 2.0
        cy = (float(bounds.bottom) + float(bounds.top)) / 2.0
        to_wgs = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
        lon, lat = to_wgs.transform(cx, cy)
        if not (math.isfinite(lon) and math.isfinite(lat)):
            return None
        return float(lon), float(lat)
    except Exception:
        return None


def needs_meter_projection(crs) -> bool:
    if crs is None:
        return False
    try:
        crs_obj = CRS.from_user_input(crs)
    except Exception:
        return False
    if not crs_obj.is_projected:
        return True
    axis_units = {info.unit_name.lower() for info in crs_obj.axis_info if info.unit_name}
    if not axis_units:
        return False
    return not any("metre" in u or "meter" in u for u in axis_units)


def choose_meter_crs_for_raster(
    raster_path: str,
    centroid_hint: Optional[Tuple[float, float]] = None,
) -> Optional[int]:
    if not raster_path or not os.path.exists(raster_path):
        return None
    with rasterio.open(raster_path) as ds:
        if ds.crs is None:
            return None
        if not needs_meter_projection(ds.crs):
            return None
        lon = lat = None
        if centroid_hint is not None:
            lon, lat = centroid_hint
        else:
            ll = centroid_lonlat_from_bounds(ds.bounds, ds.crs)
            if ll:
                lon, lat = ll
        if lon is None or lat is None:
            return None
        return utm_epsg_from_lonlat(float(lon), float(lat))


def reproject_raster_to_crs(src_path: str, dst_path: str, dst_crs) -> None:
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,
        )
        profile = src.profile.copy()
        profile.update(
            crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
        )
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        with rasterio.open(dst_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src.nodata,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    dst_nodata=src.nodata,
                    resampling=Resampling.bilinear,
                )
