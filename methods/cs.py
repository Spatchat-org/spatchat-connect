"""cs.py

Circuit-theory connectivity analysis utilities.

This module implements a lightweight, dependency-free (aside from SciPy) solver
for *effective resistance* between focal nodes placed on a resistance raster.
The API mirrors high-level Circuitscape usage: provide a cost (resistance)
raster and a table of focal nodes, and the solver injects one ampere of current
between every unordered pair to measure the resulting voltage drop.  The
voltage drop is exactly the effective resistance for the pair, which can be
used as a circuit-theory analogue to least-cost distance.

The implementation purposely keeps the feature set compact so that it can run
in memory-restricted developer environments:

* The raster window is cropped to the convex hull of the requested nodes with a
  configurable pixel margin.  This avoids materialising a full-world raster
  when only a subsection is required.
* The solver constructs the sparse graph Laplacian directly from the raster
  pixels (using 4 or 8 connectivity) and factorises the reduced Laplacian once.
  Each node pair is then solved by re-using that factorisation, making the
  per-pair solve very fast.
* All outputs are returned as plain Python / pandas objects.  If an ``outdir``
  is supplied they are also written to CSV files for downstream tooling.

Example
-------
>>> from methods.cs import compute_effective_resistances
>>> result = compute_effective_resistances(
...     cost_tif="data/cost.tif",
...     nodes_df=nodes_df,
...     outdir="outputs",
... )
>>> result["pairs"].head()
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Optional, Sequence
import math
import os

import numpy as np
import pandas as pd
import rasterio

try:
    from analysis_tracker import record_step as _record_analysis_step
except Exception:  # pragma: no cover - optional dependency
    _record_analysis_step = None


from rasterio.windows import Window
from rasterio.transform import Affine
from pyproj import CRS, Transformer
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu


@dataclass
class CircuitTheoryParams:
    """Configuration parameters for :func:`compute_effective_resistances`."""

    id_col: str = "node_id"
    lon_col: str = "longitude"
    lat_col: str = "latitude"
    nodes_crs: Optional[str] = "EPSG:4326"
    fully_connected: bool = True  # 8-neighbourhood if True else 4-neighbourhood
    barrier_cost: float = 1e12    # cost threshold to treat as impassable
    crop_margin: int = 128        # extra pixels around the convex hull of nodes
    min_conductance: float = 1e-9  # lower bound to avoid zero division
    write_current_density: bool = True


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _xy_to_rc(transform: Affine, x: float, y: float) -> tuple[int, int]:
    col, row = (~transform) * (x, y)
    return int(math.floor(row + 0.5)), int(math.floor(col + 0.5))


def _estimate_pixel_size(transform: Affine) -> float:
    sx = math.hypot(transform.a, transform.b)
    sy = math.hypot(transform.d, transform.e)
    if sx == 0 and sy == 0:
        raise ValueError("Invalid transform; pixel size cannot be zero.")
    if sx == 0:
        return float(sy)
    if sy == 0:
        return float(sx)
    return float((sx + sy) / 2.0)


def _project_nodes(
    df: pd.DataFrame,
    params: CircuitTheoryParams,
    raster_crs: Optional[CRS],
) -> Sequence[tuple[str, float, float]]:
    """Project node coordinates to raster CRS if necessary."""

    id_col = params.id_col
    lon_col = params.lon_col
    lat_col = params.lat_col

    if params.nodes_crs and raster_crs:
        transformer = Transformer.from_crs(
            params.nodes_crs, raster_crs, always_xy=True
        )
        coords = [
            (
                str(row[id_col]),
                *transformer.transform(row[lon_col], row[lat_col]),
            )
            for _, row in df.iterrows()
        ]
    else:
        coords = [
            (str(row[id_col]), float(row[lon_col]), float(row[lat_col]))
            for _, row in df.iterrows()
        ]
    return coords


def _prepare_window(
    rc_points: Sequence[tuple[int, int]],
    raster_shape: tuple[int, int],
    margin: int,
) -> Window:
    rows = [rc[0] for rc in rc_points]
    cols = [rc[1] for rc in rc_points]
    rmin = max(0, min(rows) - margin)
    rmax = min(raster_shape[0], max(rows) + margin + 1)
    cmin = max(0, min(cols) - margin)
    cmax = min(raster_shape[1], max(cols) + margin + 1)
    return Window.from_slices((int(rmin), int(rmax)), (int(cmin), int(cmax)))


def _edge_conductance(cost_a: float, cost_b: float, step_length: float, min_cond: float) -> float:
    if not np.isfinite(cost_a) or not np.isfinite(cost_b):
        return 0.0
    denom = (cost_a + cost_b) * step_length
    if denom <= 0:
        return 0.0
    return max(2.0 / denom, min_cond)


def _unique_neighbour_offsets(fully_connected: bool, pixel_size: float) -> list[tuple[int, int, float]]:
    offsets = [(1, 0, float(pixel_size)), (0, 1, float(pixel_size))]
    if fully_connected:
        diagonal = float(pixel_size) * math.sqrt(2.0)
        offsets.extend([(1, 1, diagonal), (1, -1, diagonal)])
    return offsets


def _enumerate_edges(
    cost_array: np.ndarray,
    valid_mask: np.ndarray,
    index_map: dict[tuple[int, int], int],
    fully_connected: bool,
    pixel_size: float,
    min_cond: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = cost_array.shape
    u_idx: list[int] = []
    v_idx: list[int] = []
    conductance: list[float] = []
    for (r, c), idx in index_map.items():
        cost_here = float(cost_array[r, c])
        for dr, dc, step_length in _unique_neighbour_offsets(fully_connected, pixel_size):
            nr = r + dr
            nc = c + dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                continue
            if not valid_mask[nr, nc]:
                continue
            nidx = index_map.get((nr, nc))
            if nidx is None:
                continue
            g = _edge_conductance(cost_here, float(cost_array[nr, nc]), step_length, min_cond)
            if g <= 0:
                continue
            u_idx.append(int(idx))
            v_idx.append(int(nidx))
            conductance.append(float(g))
    return (
        np.asarray(u_idx, dtype=np.int32),
        np.asarray(v_idx, dtype=np.int32),
        np.asarray(conductance, dtype=np.float64),
    )


def _build_laplacian(
    cost_array: np.ndarray,
    valid_mask: np.ndarray,
    fully_connected: bool,
    pixel_size: float,
    min_cond: float,
) -> tuple[csc_matrix, dict[tuple[int, int], int], np.ndarray, tuple[np.ndarray, np.ndarray]]:
    h, w = cost_array.shape
    coords = np.argwhere(valid_mask)
    index_map: dict[tuple[int, int], int] = {
        (int(r), int(c)): int(i) for i, (r, c) in enumerate(coords)
    }
    n = coords.shape[0]
    if n == 0:
        raise ValueError("No valid cells to construct circuit graph.")

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    diag_accum = np.zeros(n, dtype=np.float64)
    edge_u, edge_v, edge_g = _enumerate_edges(cost_array, valid_mask, index_map, fully_connected, pixel_size, min_cond)
    for idx, nidx, g in zip(edge_u, edge_v, edge_g):
        rows.extend((idx, nidx))
        cols.extend((nidx, idx))
        data.extend((-g, -g))
        diag_accum[idx] += g
        diag_accum[nidx] += g

    rows.extend(range(n))
    cols.extend(range(n))
    data.extend(diag_accum.tolist())

    laplacian = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsc()
    edge_data = np.column_stack((edge_u, edge_v)).astype(np.int32, copy=False)
    return laplacian, index_map, coords.astype(np.int32, copy=False), [edge_data, edge_g]


def _factorise_reduced_laplacian(laplacian: csc_matrix) -> tuple[np.ndarray, splu]:
    n = laplacian.shape[0]
    if n < 2:
        raise ValueError("Need at least two valid cells for circuit analysis.")
    mask = np.ones(n, dtype=bool)
    mask[0] = False
    reduced = laplacian[mask][:, mask]
    solver = splu(reduced)
    return mask, solver


def _restrict_to_node_components(
    laplacian: csc_matrix,
    coords: np.ndarray,
    edges: tuple[np.ndarray, np.ndarray],
    placed_indices: Sequence[int],
) -> tuple[csc_matrix, np.ndarray, tuple[np.ndarray, np.ndarray], dict[int, int]]:
    if laplacian.shape[0] == 0:
        return laplacian, coords, edges, {}

    n_components, labels = connected_components(laplacian, directed=False, return_labels=True)
    if n_components <= 1:
        identity = {int(i): int(i) for i in range(laplacian.shape[0])}
        return laplacian, coords, edges, identity

    keep_components = {int(labels[int(idx)]) for idx in placed_indices}
    keep_mask = np.isin(labels, list(keep_components))
    kept_indices = np.flatnonzero(keep_mask)
    if kept_indices.size == 0:
        return laplacian, coords, edges, {}

    reindex = {int(old): int(new) for new, old in enumerate(kept_indices)}
    reduced_laplacian = laplacian[keep_mask][:, keep_mask].tocsc()
    reduced_coords = coords[keep_mask]

    edge_nodes, edge_g = edges
    if edge_nodes.size == 0:
        reduced_edges = (
            np.empty((0, 2), dtype=np.int32),
            np.empty((0,), dtype=np.float64),
        )
    else:
        edge_keep = keep_mask[edge_nodes[:, 0]] & keep_mask[edge_nodes[:, 1]]
        kept_edge_nodes = edge_nodes[edge_keep]
        kept_edge_g = edge_g[edge_keep]
        if kept_edge_nodes.size:
            remapped = np.empty_like(kept_edge_nodes)
            remapped[:, 0] = np.array([reindex[int(idx)] for idx in kept_edge_nodes[:, 0]], dtype=np.int32)
            remapped[:, 1] = np.array([reindex[int(idx)] for idx in kept_edge_nodes[:, 1]], dtype=np.int32)
        else:
            remapped = np.empty((0, 2), dtype=np.int32)
        reduced_edges = (remapped, kept_edge_g)

    return reduced_laplacian, reduced_coords, reduced_edges, reindex


def _slice_component(
    laplacian: csc_matrix,
    coords: np.ndarray,
    edges: tuple[np.ndarray, np.ndarray],
    component_mask: np.ndarray,
) -> tuple[csc_matrix, np.ndarray, tuple[np.ndarray, np.ndarray], dict[int, int]]:
    kept_indices = np.flatnonzero(component_mask)
    reindex = {int(old): int(new) for new, old in enumerate(kept_indices)}
    sub_laplacian = laplacian[component_mask][:, component_mask].tocsc()
    sub_coords = coords[component_mask]
    edge_nodes, edge_g = edges
    if edge_nodes.size == 0:
        return (
            sub_laplacian,
            sub_coords,
            (np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float64)),
            reindex,
        )
    edge_keep = component_mask[edge_nodes[:, 0]] & component_mask[edge_nodes[:, 1]]
    kept_edge_nodes = edge_nodes[edge_keep]
    kept_edge_g = edge_g[edge_keep]
    if kept_edge_nodes.size:
        remapped = np.empty_like(kept_edge_nodes)
        remapped[:, 0] = np.array([reindex[int(idx)] for idx in kept_edge_nodes[:, 0]], dtype=np.int32)
        remapped[:, 1] = np.array([reindex[int(idx)] for idx in kept_edge_nodes[:, 1]], dtype=np.int32)
    else:
        remapped = np.empty((0, 2), dtype=np.int32)
    return sub_laplacian, sub_coords, (remapped, kept_edge_g), reindex


def _effective_resistance(
    solver: splu,
    mask: np.ndarray,
    u_idx: int,
    v_idx: int,
) -> float:
    potentials = _solve_potentials(solver, mask, u_idx, v_idx)
    return float(potentials[u_idx] - potentials[v_idx])


def _solve_potentials(
    solver: splu,
    mask: np.ndarray,
    u_idx: int,
    v_idx: int,
) -> np.ndarray:
    if u_idx == v_idx:
        return np.zeros(mask.size, dtype=np.float64)
    n = mask.size
    b_full = np.zeros(n, dtype=np.float64)
    b_full[u_idx] = 1.0
    b_full[v_idx] = -1.0
    b_reduced = b_full[mask]
    potentials_reduced = solver.solve(b_reduced)
    potentials = np.zeros(n, dtype=np.float64)
    potentials[mask] = potentials_reduced
    return potentials


def _accumulate_current_density(
    current_density: np.ndarray,
    potentials: np.ndarray,
    coords: np.ndarray,
    edges: Sequence[np.ndarray],
) -> None:
    if current_density.size == 0:
        return
    if not edges or len(edges) != 2:
        return
    edge_nodes, conductance = edges
    if edge_nodes.size == 0 or conductance.size == 0:
        return
    u_idx = edge_nodes[:, 0]
    v_idx = edge_nodes[:, 1]
    currents = np.abs(conductance * (potentials[u_idx] - potentials[v_idx]))
    valid = currents > 0
    if not np.any(valid):
        return
    currents = currents[valid] * 0.5
    u_coords = coords[u_idx[valid]]
    v_coords = coords[v_idx[valid]]
    np.add.at(current_density, (u_coords[:, 0], u_coords[:, 1]), currents)
    np.add.at(current_density, (v_coords[:, 0], v_coords[:, 1]), currents)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_effective_resistances(
    cost_tif: str,
    nodes_df: pd.DataFrame,
    params: Optional[CircuitTheoryParams] = None,
    outdir: Optional[str] = None,
) -> dict:
    """Compute pairwise effective resistance between nodes on a resistance grid.

    Parameters
    ----------
    cost_tif:
        Path to a raster dataset whose first band encodes resistance / cost.
    nodes_df:
        DataFrame containing at least the id/longitude/latitude columns defined
        in :class:`CircuitTheoryParams`.
    params:
        Optional override for configuration parameters.
    outdir:
        If provided, outputs ``circuit_pairs.csv``,
        ``circuit_unplaced_nodes.csv``, ``circuit_disconnected_pairs.csv`` and, when enabled,
        ``circuit_current_density.tif`` to this directory.

    Returns
    -------
    dict
        A dictionary with the following keys:

        ``pairs``
            pandas.DataFrame with columns ``source``, ``target`` and
            ``effective_resistance``.
        ``unplaced_nodes``
            pandas.DataFrame listing nodes that could not be placed on the
            raster (with reasons).
        ``laplacian_size``
            Number of valid raster cells considered in the circuit graph.
    """

    params = params or CircuitTheoryParams()
    if not os.path.exists(cost_tif):
        raise FileNotFoundError(cost_tif)
    if nodes_df.empty:
        raise ValueError("nodes_df is empty; at least two nodes are required.")
    for col in (params.id_col, params.lon_col, params.lat_col):
        if col not in nodes_df.columns:
            raise ValueError(f"nodes_df must contain column '{col}'")

    with rasterio.open(cost_tif) as ds:
        raster_crs = ds.crs
        raster_shape = (ds.height, ds.width)
        global_transform = ds.transform
        raster_profile = ds.profile.copy()
        pixel_size = _estimate_pixel_size(global_transform)

        projected = _project_nodes(nodes_df, params, raster_crs)
        node_rcs = []
        unplaced_records = []
        for node_id, x, y in projected:
            global_r, global_c = _xy_to_rc(global_transform, x, y)
            if not (0 <= global_r < raster_shape[0] and 0 <= global_c < raster_shape[1]):
                unplaced_records.append(
                    {"node_id": node_id, "reason": "outside_raster"}
                )
                continue
            node_rcs.append((node_id, global_r, global_c))

        if len(node_rcs) < 2:
            raise ValueError(
                "Need at least two nodes within the raster extent to run circuit analysis."
            )

        rc_points = [(r, c) for _, r, c in node_rcs]
        window = _prepare_window(rc_points, raster_shape, params.crop_margin)
        arr = ds.read(1, window=window, masked=True)
        window_transform = ds.window_transform(window)

    arr_filled = np.array(arr.filled(np.nan), dtype=np.float64)
    valid_mask = np.isfinite(arr_filled) & (arr_filled < params.barrier_cost)

    if not np.any(valid_mask):
        raise ValueError("No traversable cells within the cropped raster window.")

    placed_records = []

    row_off = int(window.row_off)
    col_off = int(window.col_off)

    for node_id, global_r, global_c in node_rcs:
        local_r = global_r - row_off
        local_c = global_c - col_off
        if (
            local_r < 0
            or local_r >= arr_filled.shape[0]
            or local_c < 0
            or local_c >= arr_filled.shape[1]
        ):
            unplaced_records.append(
                {"node_id": node_id, "reason": "outside_window"}
            )
            continue
        if not valid_mask[local_r, local_c]:
            unplaced_records.append(
                {"node_id": node_id, "reason": "invalid_or_barrier"}
            )
            continue
        placed_records.append(
            {
                "node_id": node_id,
                "local_rc": (int(local_r), int(local_c)),
                "index": None,
            }
        )

    if len(placed_records) < 2:
        raise ValueError(
            "Need at least two nodes placed on valid cells to run circuit analysis."
        )

    laplacian, index_map, coords, edges = _build_laplacian(
        arr_filled, valid_mask, params.fully_connected, pixel_size, params.min_conductance
    )
    for record in placed_records:
        local_r, local_c = record["local_rc"]
        record["index"] = index_map[(int(local_r), int(local_c))]

    placed_indices = [int(record["index"]) for record in placed_records]
    laplacian, coords, edges, reindex = _restrict_to_node_components(
        laplacian,
        coords,
        edges,
        placed_indices,
    )
    if not reindex:
        raise ValueError("No connected traversable component contains the placed nodes.")
    for record in placed_records:
        old_index = int(record["index"])
        if old_index not in reindex:
            raise ValueError("At least one node falls outside the connected component used for circuit analysis.")
        record["index"] = reindex[old_index]

    pair_rows = []
    disconnected_rows = []
    current_density = np.zeros(arr_filled.shape, dtype=np.float64) if params.write_current_density else None
    n_components, labels = connected_components(laplacian, directed=False, return_labels=True)
    placed_by_component: dict[int, list[dict]] = {}
    for record in placed_records:
        comp = int(labels[int(record["index"])])
        placed_by_component.setdefault(comp, []).append(record)

    component_masks = {comp: labels == comp for comp in placed_by_component}
    for comp, records in placed_by_component.items():
        if len(records) < 2:
            continue
        sub_laplacian, sub_coords, sub_edges, sub_reindex = _slice_component(
            laplacian,
            coords,
            edges,
            component_masks[comp],
        )
        mask, solver = _factorise_reduced_laplacian(sub_laplacian)
        for a, b in combinations(records, 2):
            a_idx = sub_reindex[int(a["index"])]
            b_idx = sub_reindex[int(b["index"])]
            potentials = _solve_potentials(solver, mask, a_idx, b_idx)
            res = float(potentials[a_idx] - potentials[b_idx])
            pair_rows.append(
                {
                    "source": a["node_id"],
                    "target": b["node_id"],
                    "effective_resistance": res,
                }
            )
            if current_density is not None:
                _accumulate_current_density(current_density, potentials, sub_coords, sub_edges)

    component_items = sorted(placed_by_component.items(), key=lambda item: item[0])
    for idx, (_, left_records) in enumerate(component_items):
        for _, right_records in component_items[idx + 1:]:
            for a in left_records:
                for b in right_records:
                    disconnected_rows.append(
                        {
                            "source": a["node_id"],
                            "target": b["node_id"],
                            "reason": "disconnected_components",
                        }
                    )

    pairs_df = pd.DataFrame(pair_rows)
    unplaced_df = pd.DataFrame(unplaced_records)
    disconnected_df = pd.DataFrame(disconnected_rows)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        pairs_path = os.path.join(outdir, "circuit_pairs.csv")
        pairs_df.to_csv(pairs_path, index=False)
        if not unplaced_df.empty:
            unplaced_path = os.path.join(outdir, "circuit_unplaced_nodes.csv")
            unplaced_df.to_csv(unplaced_path, index=False)
        else:
            unplaced_path = None
        if not disconnected_df.empty:
            disconnected_path = os.path.join(outdir, "circuit_disconnected_pairs.csv")
            disconnected_df.to_csv(disconnected_path, index=False)
        else:
            disconnected_path = None
        if current_density is not None:
            current_density_path = os.path.join(outdir, "circuit_current_density.tif")
            profile = raster_profile.copy()
            profile.update(
                driver="GTiff",
                height=int(current_density.shape[0]),
                width=int(current_density.shape[1]),
                count=1,
                dtype="float32",
                transform=window_transform,
                nodata=0.0,
                compress="LZW",
            )
            with rasterio.open(current_density_path, "w", **profile) as dst:
                dst.write(current_density.astype(np.float32, copy=False), 1)
        else:
            current_density_path = None
    else:
        pairs_path = None
        unplaced_path = None
        disconnected_path = None
        current_density_path = None

    result = {
        "pairs": pairs_df,
        "unplaced_nodes": unplaced_df,
        "disconnected_pairs": disconnected_df,
        "laplacian_size": laplacian.shape[0],
        "pairs_csv": pairs_path,
        "unplaced_csv": unplaced_path,
        "disconnected_csv": disconnected_path,
        "current_density_array": None if outdir else current_density,
        "current_density_tif": current_density_path,
        "window_transform": window_transform,
    }

    if _record_analysis_step:
        try:
            inputs = {
                "cost_tif": cost_tif,
                "nodes_count": int(len(nodes_df)),
                "params": asdict(params),
                "outdir": outdir,
            }
            outputs = {
                "pairs_count": int(len(pairs_df)),
                "unplaced_count": int(len(unplaced_df)),
                "disconnected_pairs_count": int(len(disconnected_df)),
                "pairs_csv": pairs_path,
                "unplaced_csv": unplaced_path,
                "disconnected_csv": disconnected_path,
                "current_density_tif": current_density_path,
                "laplacian_size": int(laplacian.shape[0]),
            }
            _record_analysis_step(
                kind="circuit_effective_resistance",
                function="methods.cs.compute_effective_resistances",
                inputs=inputs,
                outputs=outputs,
                code_obj=compute_effective_resistances,
            )
        except Exception:
            pass

    return result

