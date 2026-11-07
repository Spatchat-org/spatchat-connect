# results_plots.py
"""Datasheet and Plots tab contents for the connectivity room, built from the session's outputs."""
import os

import numpy as np
import pandas as pd
import rasterio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_SKIP_COLS = {"x", "y", "lat", "lon", "latitude", "longitude", "id", "index", "node_id", "source", "target",
              "source_x", "source_y", "target_x", "target_y"}
_MAX_RASTER_PX = 700


def _signature(output_dir: str):
    """Cheap fingerprint of the result files, so figures are only rebuilt when something changed."""
    sig = []
    for root, dirs, files in os.walk(output_dir):
        dirs[:] = [d for d in dirs if d not in ("converted", "_plots")]
        for fn in files:
            if fn.lower().endswith((".tif", ".tiff", ".csv")):
                full = os.path.join(root, fn)
                try:
                    sig.append((full, os.path.getmtime(full), os.path.getsize(full)))
                except OSError:
                    pass
    return tuple(sorted(sig))


def _plot_raster(path: str, dest: str) -> bool:
    with rasterio.open(path) as src:
        scale = max(src.width, src.height) / _MAX_RASTER_PX
        scale = max(scale, 1.0)
        out_shape = (max(1, int(src.height / scale)), max(1, int(src.width / scale)))
        data = src.read(1, out_shape=out_shape, masked=True).astype("float64")
        bounds = src.bounds
    if data.count() == 0:
        return False
    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(data, cmap="viridis", extent=(bounds.left, bounds.right, bounds.bottom, bounds.top))
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(os.path.splitext(os.path.basename(path))[0].replace("_", " "))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(dest, dpi=120)
    plt.close(fig)
    return True


def _plot_table(path: str, dest_dir: str) -> list:
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and str(c).lower() not in _SKIP_COLS
           and not str(c).lower().endswith(("_x", "_y"))]
    paths = []
    stem = os.path.splitext(os.path.basename(path))[0]
    for col in num[:3]:
        vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals) < 2:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.hist(vals, bins=min(30, max(5, len(vals) // 2)), color="#fa6a98", edgecolor="white")
        ax.set_xlabel(str(col))
        ax.set_ylabel("Count")
        ax.set_title(f"{stem.replace('_', ' ')}: {col}")
        fig.tight_layout()
        dest = os.path.join(dest_dir, f"{stem}_{col}.png".replace("/", "_"))
        fig.savefig(dest, dpi=120)
        plt.close(fig)
        paths.append(dest)
    return paths


def _plot_nodes(nodes_df, dest: str) -> bool:
    if nodes_df is None or getattr(nodes_df, "empty", True):
        return False
    lon = next((c for c in nodes_df.columns if str(c).lower() in ("longitude", "lon", "x")), None)
    lat = next((c for c in nodes_df.columns if str(c).lower() in ("latitude", "lat", "y")), None)
    if lon is None or lat is None:
        return False
    fig, ax = plt.subplots(figsize=(6.5, 5))
    weight = next((c for c in nodes_df.columns if str(c).lower() == "weight"), None)
    sc = ax.scatter(nodes_df[lon], nodes_df[lat], c=nodes_df[weight] if weight else "#fa6a98", s=70,
                    cmap="viridis", edgecolor="white")
    if weight:
        fig.colorbar(sc, ax=ax, shrink=0.8, label="weight")
    idc = next((c for c in nodes_df.columns if str(c).lower() in ("node_id", "id")), None)
    if idc is not None and len(nodes_df) <= 40:
        for _, r in nodes_df.iterrows():
            ax.annotate(str(r[idc]), (r[lon], r[lat]), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_title(f"Nodes ({len(nodes_df)})")
    ax.set_xlabel(str(lon))
    ax.set_ylabel(str(lat))
    fig.tight_layout()
    fig.savefig(dest, dpi=120)
    plt.close(fig)
    return True


def build_figures(cache: dict) -> list:
    """PNG paths for the nodes plus every result raster/table in the session's output directory."""
    output_dir = (cache or {}).get("output_dir")
    nodes_df = (cache or {}).get("nodes_df")
    if not output_dir or not os.path.isdir(output_dir):
        return []
    sig = (_signature(output_dir), None if nodes_df is None else len(nodes_df))
    if cache.get("plots_sig") == sig and cache.get("plots_paths"):
        return [p for p in cache["plots_paths"] if os.path.exists(p)]

    plots_dir = os.path.join(output_dir, "_plots")
    os.makedirs(plots_dir, exist_ok=True)
    for fn in os.listdir(plots_dir):
        try:
            os.remove(os.path.join(plots_dir, fn))
        except OSError:
            pass

    paths = []
    dest = os.path.join(plots_dir, "nodes.png")
    if _plot_nodes(nodes_df, dest):
        paths.append(dest)
    for root, dirs, files in os.walk(output_dir):
        dirs[:] = [d for d in dirs if d not in ("converted", "_plots")]
        for fn in sorted(files):
            full = os.path.join(root, fn)
            low = fn.lower()
            try:
                if low.endswith((".tif", ".tiff")):
                    d = os.path.join(plots_dir, os.path.splitext(fn)[0] + ".png")
                    if _plot_raster(full, d):
                        paths.append(d)
                elif low.endswith(".csv") and "nodes" not in low and "matrix" not in low:
                    paths.extend(_plot_table(full, plots_dir))
            except Exception as exc:
                print(f"[connect] could not plot {fn}: {exc}")
    cache["plots_sig"] = sig
    cache["plots_paths"] = paths
    return paths
