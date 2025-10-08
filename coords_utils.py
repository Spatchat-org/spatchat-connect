# coords_utils.py
import re
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from crs_utils import parse_crs_input

try:
    from pyproj import Transformer
except Exception:  # pragma: no cover
    Transformer = None  # type: ignore


# ------------------------------------------------------------
# Basic range checks (kept for consistency with Home Range)
# ------------------------------------------------------------
def looks_like_latlon(df: pd.DataFrame, x_col: str, y_col: str):
    """Return 'lonlat' or 'latlon' if columns look like degrees; else None."""
    try:
        x_vals = pd.to_numeric(df[x_col], errors="coerce")
        y_vals = pd.to_numeric(df[y_col], errors="coerce")
        if x_vals.between(-180, 180).all() and y_vals.between(-90, 90).all():
            return "lonlat"
        if x_vals.between(-90, 90).all() and y_vals.between(-180, 180).all():
            return "latlon"
    except Exception:
        return None
    return None


def looks_invalid_latlon(df: pd.DataFrame, lat_col: str, lon_col: str) -> bool:
    """True if labeled lat/lon columns are out of bounds."""
    try:
        lat = pd.to_numeric(df[lat_col], errors="coerce")
        lon = pd.to_numeric(df[lon_col], errors="coerce")
        return not (lat.between(-90, 90).all() and lon.between(-180, 180).all())
    except Exception:
        return True


# ------------------------------------------------------------
# Text level parsing (kept for parity; used elsewhere)
# ------------------------------------------------------------
def parse_levels_from_text(text: str):
    """Extract % levels (clamp 100→99) with sane defaults."""
    levels = [int(val) for val in re.findall(r'\b([1-9][0-9]?|100)\b', text)]
    levels = [min(l, 99) for l in levels]
    return sorted(set(levels or [95]))


# ------------------------------------------------------------
# Connectivity-specific helpers
# ------------------------------------------------------------
_LON_NAMES = {"longitude", "lon", "long", "x"}
_LAT_NAMES = {"latitude", "lat", "y"}

def detect_lonlat_columns(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    """
    Heuristic detection of lon/lat columns in degrees.
    1) Exact/common name matches
    2) Value-range check on all numeric columns (tries both orders)
    Returns (lon_col, lat_col) or None.
    """
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}

    # 1) Name-based
    lon_col = next((low[c] for c in _LON_NAMES if c in low), None)
    lat_col = next((low[c] for c in _LAT_NAMES if c in low), None)
    if lon_col and lat_col:
        try:
            lon_ok = pd.to_numeric(df[lon_col], errors="coerce").between(-180, 180).all()
            lat_ok = pd.to_numeric(df[lat_col], errors="coerce").between(-90, 90).all()
            if lon_ok and lat_ok:
                return lon_col, lat_col
        except Exception:
            pass  # fall through to value-based

    # 2) Value-based over numeric candidates
    numeric_cols: List[str] = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() >= 0.9:  # mostly numeric
            numeric_cols.append(c)

    for i, xc in enumerate(numeric_cols):
        for yc in numeric_cols[i+1:]:
            tag = looks_like_latlon(df, xc, yc)
            if tag == "lonlat":
                return xc, yc
            if tag == "latlon":
                return yc, xc
    return None


def ensure_wgs84_longlat(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    crs_text: Optional[str] = None
) -> pd.DataFrame:
    """
    Standardize to columns named 'longitude'/'latitude' in WGS84.
    - If (x,y) already look like lon/lat in degrees, just rename/copy.
    - Else require a CRS (e.g., 'EPSG:32633', '32633', 'UTM 33N') and reproject.
    Raises ValueError if CRS is missing or pyproj is unavailable.
    """
    out = df.copy()

    # Case 1: already lon/lat (degrees)
    tag = looks_like_latlon(out, x_col, y_col)
    if tag == "lonlat":
        out["longitude"] = pd.to_numeric(out[x_col], errors="coerce")
        out["latitude"]  = pd.to_numeric(out[y_col], errors="coerce")
        return out
    if tag == "latlon":
        out["longitude"] = pd.to_numeric(out[y_col], errors="coerce")
        out["latitude"]  = pd.to_numeric(out[x_col], errors="coerce")
        return out

    # Case 2: need to transform projected X/Y → WGS84
    if Transformer is None:
        raise ValueError("pyproj is required to transform projected coordinates to WGS84.")

    if not crs_text or not str(crs_text).strip():
        raise ValueError(
            "Coordinates do not look like lon/lat. Please specify a CRS/UTM zone "
            "(e.g., 'EPSG:32610', '32610', 'UTM 10T')."
        )

    epsg_src = parse_crs_input(crs_text)  # may raise ValueError with a friendly tip
    try:
        transformer = Transformer.from_crs(epsg_src, 4326, always_xy=True)
        x = pd.to_numeric(out[x_col], errors="coerce").values
        y = pd.to_numeric(out[y_col], errors="coerce").values
        lon, lat = transformer.transform(x, y)
        out["longitude"] = lon
        out["latitude"]  = lat
    except Exception as e:
        raise ValueError(f"Failed to convert coordinates: {e}")

    # Final bounds check
    if looks_invalid_latlon(out, "latitude", "longitude"):
        raise ValueError("Reprojected coordinates are out of bounds; check CRS and X/Y selection.")
    return out
