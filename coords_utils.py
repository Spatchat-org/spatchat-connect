"""Point coordinate helpers for Spatchat Connectivity (Home Range-style architecture)."""
import re
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from crs_utils import parse_crs_input


def looks_like_latlon(df: pd.DataFrame, x_col: str, y_col: str):
    """Check degree ranges, not a definitive CRS determination."""
    try:
        x = pd.to_numeric(df[x_col], errors="coerce")
        y = pd.to_numeric(df[y_col], errors="coerce")
        if not (x.notna().all() and y.notna().all()):
            return None
        if x.between(-180, 180).all() and y.between(-90, 90).all():
            return "lonlat"
        if x.between(-90, 90).all() and y.between(-180, 180).all():
            return "latlon"
    except (KeyError, TypeError, ValueError):
        pass
    return None


def looks_invalid_latlon(df: pd.DataFrame, lat_col: str, lon_col: str) -> bool:
    try:
        lat = pd.to_numeric(df[lat_col], errors="coerce")
        lon = pd.to_numeric(df[lon_col], errors="coerce")
        return not (lat.notna().all() and lon.notna().all()
                    and lat.between(-90, 90).all() and lon.between(-180, 180).all())
    except (KeyError, TypeError, ValueError):
        return True


def parse_levels_from_text(text: str):
    levels = [int(v) for v in re.findall(r"\b([1-9][0-9]?|100)\b", text)]
    return sorted(set(min(v, 99) for v in (levels or [95])))


def detect_lonlat_columns(df: pd.DataFrame):
    x, y, confident = suggest_node_columns(df)
    return (x, y) if confident else None


def ensure_wgs84_longlat(df, x_col, y_col, crs_text=None):
    """Honor explicitly supplied CRS even if coordinates resemble degrees."""
    if crs_text is None or not str(crs_text).strip():
        if looks_like_latlon(df, x_col, y_col) != "lonlat":
            raise ValueError("Specify the source CRS before transforming points.")
        crs_text = "EPSG:4326"
    return prepare_node_coordinates(df, x_col, y_col, crs_text)


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
