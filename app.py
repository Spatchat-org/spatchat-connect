# app.py

import os
import shutil
import sys
from typing import Optional, Tuple

import gradio as gr
import pandas as pd

# Optional deps for raster preview (bounds only). If rasterio is missing, preview will show empty map with a note.
try:
    import rasterio
except Exception:  # pragma: no cover
    rasterio = None

import folium

print("Starting SpatChat: Connectivity Analysis")

# --------------------------------------------------------------------------------------
# In-memory cache (replace with your storage.py patterns later)
# --------------------------------------------------------------------------------------
CACHED = {
    "raster_path": None,
    "raster_bounds": None,   # (minx, miny, maxx, maxy) in raster CRS
    "raster_crs": None,      # proj string or EPSG code
    "nodes_df": None,        # pandas DataFrame with detected lon/lat columns
}

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _render_empty_map(msg: Optional[str] = None) -> str:
    m = folium.Map(location=[20, 0], zoom_start=2, control_scale=True, tiles="CartoDB positron")
    folium.LayerControl().add_to(m)
    if msg:
        folium.map.Marker([0, 0], popup=msg).add_to(m)
    return m._repr_html_()


def _bounds_center(bounds: Tuple[float, float, float, float]) -> Tuple[float, float]:
    minx, miny, maxx, maxy = bounds
    return ( (miny + maxy) / 2.0, (minx + maxx) / 2.0 )  # lat, lon


def _detect_nodes_lonlat(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    # Try common variants
    cols = {c.lower(): c for c in df.columns}
    for lon_key, lat_key in [("longitude", "latitude"), ("lon", "lat"), ("x", "y")]:
        if lon_key in cols and lat_key in cols:
            lon_col, lat_col = cols[lon_key], cols[lat_key]
            try:
                lon_ok = pd.to_numeric(df[lon_col], errors="coerce").between(-180, 180).all()
                lat_ok = pd.to_numeric(df[lat_col], errors="coerce").between(-90, 90).all()
                if lon_ok and lat_ok:
                    return lon_col, lat_col
            except Exception:
                pass
    return None


def _build_preview_map() -> str:
    # If we have raster bounds, center on them and draw a rectangle.
    if CACHED["raster_bounds"]:
        lat, lon = _bounds_center(CACHED["raster_bounds"])
        m = folium.Map(location=[lat, lon], zoom_start=6, control_scale=True, tiles="CartoDB positron")
        b = CACHED["raster_bounds"]
        rect = folium.Rectangle(
            bounds=[(b[1], b[0]), (b[3], b[2])],  # (lat, lon) pairs
            fill=False,
            weight=2,
        )
        rect.add_to(m)
        folium.Popup("Raster extent").add_to(rect)
    else:
        m = folium.Map(location=[20, 0], zoom_start=2, control_scale=True, tiles="CartoDB positron")

    # If we have nodes, plot them.
    if isinstance(CACHED.get("nodes_df"), pd.DataFrame) and not CACHED["nodes_df"].empty:
        guess = _detect_nodes_lonlat(CACHED["nodes_df"])
        if guess:
            lon_col, lat_col = guess
            pts = CACHED["nodes_df"][[lat_col, lon_col]].dropna().values.tolist()
            for lat, lon in pts:
                try:
                    folium.CircleMarker(location=[float(lat), float(lon)], radius=3).add_to(m)
                except Exception:
                    continue
            folium.LayerControl().add_to(m)
        else:
            folium.Marker([0, 0], popup="Nodes CSV loaded, but lat/lon not detected.").add_to(m)

    return m._repr_html_()


def _help_text() -> str:
    return (
        "Connectivity room ready. Upload a raster (required) and optional nodes CSV.\n\n"
        "You can then ask for: \n"
        "• Cumulative resistant kernel (choose kernel and σ in cost units)\n"
        "• Least‑cost paths / cost distance / corridors\n"
        "• Circuit-style current density\n\n"
        "Examples:\n"
        "• I want a resistant kernel with σ=500 and cutoff=5000\n"
        "• Compute least-cost paths between nodes\n"
        "• Run circuit current density, normalize by source weight\n"
    )

# --------------------------------------------------------------------------------------
# Upload handlers
# --------------------------------------------------------------------------------------

def handle_raster_upload(file):
    """Cache raster on disk, read bounds, and refresh preview map."""
    if file is None:
        return (
            [{"role": "assistant", "content": "No raster selected."}],
            _render_empty_map("No raster selected"),
        )

    # Save
    dst = os.path.join(UPLOAD_DIR, os.path.basename(file))
    try:
        shutil.copy(file, dst)
        CACHED["raster_path"] = dst
    except Exception as e:
        return (
            [{"role": "assistant", "content": f"Failed to save raster: {e}"}],
            _render_empty_map("Raster save error"),
        )

    # Read basic metadata
    if rasterio is None:
        CACHED["raster_bounds"] = None
        CACHED["raster_crs"] = None
        msg = (
            "Raster uploaded. (Install rasterio to see extent preview.)\n\n" + _help_text()
        )
        return ([{"role": "assistant", "content": msg}], _build_preview_map())

    try:
        with rasterio.open(dst) as ds:
            CACHED["raster_bounds"] = ds.bounds  # (minx, miny, maxx, maxy)
            CACHED["raster_crs"] = str(ds.crs) if ds.crs else None
    except Exception as e:
        CACHED["raster_bounds"] = None
        CACHED["raster_crs"] = None
        return (
            [{"role": "assistant", "content": f"Raster uploaded, but metadata read failed: {e}"}],
            _build_preview_map(),
        )

    info = "\n".join([
        "Raster uploaded.",
        f"• Bounds: {CACHED['raster_bounds']}",
        f"• CRS: {CACHED['raster_crs']}",
        "",
        _help_text(),
    ])
    return ([{"role": "assistant", "content": info}], _build_preview_map())


def handle_nodes_upload(file, chat_history):
    """Cache nodes CSV in memory, detect lon/lat, and refresh preview map."""
    chat_history = list(chat_history)

    if file is None:
        CACHED["nodes_df"] = None
        chat_history.append({"role": "assistant", "content": "Nodes cleared."})
        return chat_history, _build_preview_map()

    try:
        df = pd.read_csv(file)
        CACHED["nodes_df"] = df
    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"Failed to read nodes CSV: {e}"})
        return chat_history, _build_preview_map()

    note = "Nodes CSV uploaded. "
    guess = _detect_nodes_lonlat(CACHED["nodes_df"]) or (None, None)
    if all(guess):
        note += f"Detected lon/lat columns: {guess[0]} / {guess[1]}."
    else:
        note += (
            "Lat/Lon columns not detected. Include 'longitude'/'latitude' (or 'lon'/'lat')."
        )

    chat_history.append({"role": "assistant", "content": note})
    return chat_history, _build_preview_map()


# --------------------------------------------------------------------------------------
# Chat handler (placeholder)
# --------------------------------------------------------------------------------------

def handle_chat(chat_history, user_message):
    chat_history = list(chat_history)
    msg = user_message.strip()

    if not msg:
        chat_history.append({"role": "assistant", "content": "How can I help?"})
        return chat_history, gr.update(), gr.update(visible=False)

    # TODO: plug in your ask_llm() + tool routing for kernels/LCP/circuit. For now, echo help.
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": _help_text()})

    # Map remains as-is for now
    return chat_history, gr.update(value=_build_preview_map()), gr.update(visible=False)


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
with gr.Blocks(title="SpatChat: Connectivity Analysis") as demo:
    gr.Image(
        value="logo_long1.png",
        show_label=False,
        show_download_button=False,
        show_share_button=False,
        type="filepath",
        elem_id="logo-img"
    )
    gr.HTML(
        """
    <style>
    #logo-img img {
        height: 90px;
        margin: 10px 50px 10px 10px;
        border-radius: 6px;
    }
    </style>
    """
    )
    gr.Markdown("## 🌐 SpatChat: Connectivity Analysis {connectivity} 🦌🦅🦝")
    gr.HTML(
        """
    <div style="margin-top: -10px; margin-bottom: 15px;">
      <input type="text" value="https://spatchat.org/browse/?room=connectivity" id="shareLink" readonly style="width: 50%; padding: 5px; background-color: #f8f8f8; color: #222; font-weight: 500; border: 1px solid #ccc; border-radius: 4px;">
      <button onclick="navigator.clipboard.writeText(document.getElementById('shareLink').value)" style="padding: 5px 10px; background-color: #007BFF; color: white; border: none; border-radius: 4px; cursor: pointer;">
        📋 Copy Share Link
      </button>
      <div style="margin-top: 10px; font-size: 14px;">
        <b>Share:</b>
        <a href="https://twitter.com/intent/tweet?text=Checkout+Spatchat!&url=https://spatchat.org/browse/?room=connectivity" target="_blank">🐦 Twitter</a> |
        <a href="https://www.facebook.com/sharer/sharer.php?u=https://spatchat.org/browse/?room=connectivity" target="_blank">📘 Facebook</a>
      </div>
    </div>
    """
    )
    gr.Markdown(
        """
        <div style="font-size: 14px;">
        © 2025 Ho Yi Wan & Logan Hysen. All rights reserved.<br>
        If you use Spatchat in research, please cite:<br>
        <b>Wan, H.Y.</b> & <b>Hysen, L.</b> (2025). <i>SpatChat: Connectivity Analysis.</i>
        </div>
    """
    )

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="SpatChat",
                show_label=True,
                type="messages",
                value=[{"role": "assistant", "content": "Welcome! Upload a raster (GeoTIFF) to begin. Optionally add a nodes CSV with longitude/latitude."}]
                height=400
            )
            user_input = gr.Textbox(
                label="Ask Spatchat",
                placeholder="Type commands...",
                lines=1
            )

            # NEW: Separate uploaders for raster and nodes
            raster_input = gr.File(
                label="Upload Raster (GeoTIFF .tif/.tiff; required)",
                file_types=[".tif", ".tiff"],
            )
            nodes_input = gr.File(
                label="Upload Nodes CSV (optional)",
                file_types=[".csv"],
                interactive=True,
            )
        with gr.Column(scale=3):
            map_output = gr.HTML(label="Map Preview", value=_render_empty_map(), show_label=False)
            download_btn = gr.DownloadButton(
                "📥 Download Results",
                value=None,
                visible=False
            )

    # Queue
    demo.queue(max_size=16)

    # Wire events
    raster_input.change(
        fn=handle_raster_upload,
        inputs=raster_input,
        outputs=[chatbot, map_output]
    )
    nodes_input.change(
        fn=handle_nodes_upload,
        inputs=[nodes_input, chatbot],
        outputs=[chatbot, map_output]
    )
    user_input.submit(
        fn=handle_chat,
        inputs=[chatbot, user_input],
        outputs=[chatbot, map_output, download_btn]
    )
    user_input.submit(lambda *args: "", inputs=None, outputs=user_input)

# HF Spaces-friendly launch
demo.launch(ssr_mode=False)
