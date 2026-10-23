"""Folium map helpers; input coordinates must be WGS84 lon/lat."""
import folium
import numpy as np
import pandas as pd


def render_empty_map():
    m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)
    folium.TileLayer('CartoDB positron', attr='CartoDB').add_to(m)
    folium.TileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
                     attr='OpenTopoMap', name='Topographic').add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m._repr_html_()


def fit_map_to_bounds(m, df, lat_col=None, lon_col=None):
    if df is None or df.empty:
        return m
    cols = {str(c).lower(): c for c in df.columns}
    if lat_col is None or lon_col is None:
        for a, b in [('latitude', 'longitude'), ('lat', 'lon'), ('y', 'x')]:
            if a in cols and b in cols:
                lat_col, lon_col = cols[a], cols[b]
                break
    if lat_col not in df.columns or lon_col not in df.columns:
        return m
    lat = pd.to_numeric(df[lat_col], errors='coerce')
    lon = pd.to_numeric(df[lon_col], errors='coerce')
    valid = lat.between(-90, 90) & lon.between(-180, 180) & np.isfinite(lat) & np.isfinite(lon)
    if valid.any():
        m.fit_bounds([[float(lat[valid].min()), float(lon[valid].min())],
                      [float(lat[valid].max()), float(lon[valid].max())]])
    return m
