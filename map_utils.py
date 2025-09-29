# map_utils.py
import folium, numpy as np

def render_empty_map():
    m = folium.Map(location=[0,0], zoom_start=2, control_scale=True)
    folium.TileLayer("OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB positron", attr='CartoDB').add_to(m)
    folium.TileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", attr="OpenTopoMap", name="Topographic").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m._repr_html_()

def fit_map_to_bounds(m, df):
    min_lat, max_lat = df['latitude'].min(), df['latitude'].max()
    min_lon, max_lon = df['longitude'].min(), df['longitude'].max()
    if np.isfinite([min_lat, max_lat, min_lon, max_lon]).all():
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
    return m
