"""Graph-theory workflow helpers."""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from methods.gt import GraphAnalysisParams, run_graph_theory_analysis

Cache = Dict[str, Any]


def _build_probability_decay_graph(edges: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if "cost" not in edges.columns:
        return edges.copy(), {"mode": "none", "edge_count_total": int(len(edges)), "edge_count_kept": int(len(edges))}
    decayed = edges.copy()
    costs = pd.to_numeric(decayed["cost"], errors="coerce")
    valid_costs = costs[np.isfinite(costs) & (costs > 0)]
    if valid_costs.empty:
        decayed["probability"] = 1.0
        return decayed, {"mode": "flat", "edge_count_total": int(len(decayed)), "edge_count_kept": int(len(decayed)), "probability_threshold": 0.0}
    decay_scale = float(valid_costs.median())
    probability = np.exp(-costs / max(decay_scale, 1e-12))
    decayed["probability"] = probability
    probability_threshold = min(0.2, max(0.05, float(probability.median())))
    kept = decayed[probability >= probability_threshold].copy()
    if kept.empty:
        kept = decayed.nlargest(min(10, len(decayed)), "probability").copy()
        probability_threshold = float(kept["probability"].min())
    summary = {"mode": "exponential_decay", "decay_scale_cost": decay_scale, "probability_threshold": float(probability_threshold), "edge_count_total": int(len(decayed)), "edge_count_kept": int(len(kept))}
    return kept, summary


def _metric_ranges(node_metrics: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for metric in ("betweenness", "closeness", "pagerank", "degree", "clustering"):
        if metric not in node_metrics.columns:
            continue
        vals = pd.to_numeric(node_metrics[metric], errors="coerce")
        valid = vals[np.isfinite(vals)]
        if not valid.empty:
            out[metric] = {"min": float(valid.min()), "max": float(valid.max())}
    return out


def run_graph_model(cache: Cache, output_dir: str, job_id: str, *, ensure_lcp: Optional[Callable[[Cache, str], Tuple[bool, str, Optional[str]]]] = None) -> Tuple[bool, str, Optional[str]]:
    lcp_index_csv = cache.get("lcp_index_csv")
    prep_messages: list[str] = []
    if (not lcp_index_csv or not os.path.exists(lcp_index_csv)) and ensure_lcp is not None:
        prep_ok, prep_msg, _ = ensure_lcp(cache, job_id)
        if prep_msg:
            prep_messages.append(prep_msg)
        if not prep_ok:
            combined = " ".join(m.strip() for m in prep_messages if m).strip()
            return False, combined or "Graph model could not prepare its least-cost network.", None
        lcp_index_csv = cache.get("lcp_index_csv")
    if not lcp_index_csv or not os.path.exists(lcp_index_csv):
        return False, "Graph model needs a least-cost network, but I couldn't prepare one from the current inputs.", None
    nodes_df = cache.get("nodes_df")
    if not isinstance(nodes_df, pd.DataFrame) or nodes_df.empty:
        return False, "Upload a nodes CSV before running the graph model.", None
    try:
        edges = pd.read_csv(lcp_index_csv, dtype={"source": "string", "target": "string"})
    except Exception as exc:
        return False, f"Graph model failed while reading the LCP edge table: {exc}", None
    if edges.empty:
        return False, "The LCP edge table is empty, so there is no graph to analyze.", None
    # Reject missing IDs rather than constructing a graph node literally called 'nan'.
    edges = edges.dropna(subset=["source", "target"]).copy()
    edges["source"] = edges["source"].str.strip()
    edges["target"] = edges["target"].str.strip()
    edges = edges[(edges["source"] != "") & (edges["target"] != "")].copy()
    if edges.empty:
        return False, "The LCP edge table has no valid source/target identifiers.", None
    filtered_edges, decay_summary = _build_probability_decay_graph(edges)
    if filtered_edges.empty:
        return False, "Graph model failed because probability decay removed every edge.", None
    params = GraphAnalysisParams(
        distance_col="cost" if "cost" in filtered_edges.columns else None,
        strength_col="probability" if "probability" in filtered_edges.columns else None,
    )
    if "node_id" not in nodes_df.columns:
        return False, "Prepared nodes are missing node_id; confirm coordinates and upload again.", None
    node_ids = nodes_df["node_id"].dropna().astype(str).str.strip()
    node_ids = node_ids[node_ids != ""].drop_duplicates().tolist()
    try:
        result = run_graph_theory_analysis(filtered_edges, params=params, all_node_ids=node_ids)
    except Exception as exc:
        return False, f"Graph model failed: {exc}", None
    node_metrics = result.get("node_metrics")
    if not isinstance(node_metrics, pd.DataFrame) or node_metrics.empty:
        return False, "Graph model finished, but no node metrics were produced.", None
    node_metrics = node_metrics.reset_index()
    if "node" in node_metrics.columns and "node_id" not in node_metrics.columns:
        node_metrics = node_metrics.rename(columns={"node": "node_id"})
    elif "index" in node_metrics.columns and "node_id" not in node_metrics.columns:
        node_metrics = node_metrics.rename(columns={"index": "node_id"})
    lon_col, lat_col = cache.get("nodes_lon_col"), cache.get("nodes_lat_col")
    if (lon_col in nodes_df.columns and lat_col in nodes_df.columns and "node_id" in nodes_df.columns and "node_id" in node_metrics.columns):
        coords_df = nodes_df[["node_id", lon_col, lat_col]].copy().rename(columns={lon_col: "longitude", lat_col: "latitude"})
        coords_df["node_id"] = coords_df["node_id"].astype("string")
        node_metrics["node_id"] = node_metrics["node_id"].astype("string")
        # Ensure a single coordinate row per node, avoiding accidental one-to-many joins.
        coords_df = coords_df.drop_duplicates(subset=["node_id"], keep="first")
        node_metrics = node_metrics.drop(columns=["longitude", "latitude"], errors="ignore")
        node_metrics = node_metrics.merge(coords_df, on="node_id", how="left", validate="one_to_one")
    metric_ranges = _metric_ranges(node_metrics)
    metric_preference = ("betweenness", "pagerank", "closeness", "degree", "clustering")
    selected_metric = next((name for name in metric_preference if name in node_metrics.columns and (bounds := metric_ranges.get(name)) and not math.isclose(bounds["min"], bounds["max"], rel_tol=0.0, abs_tol=1e-12)), None)
    if selected_metric is None:
        selected_metric = next((name for name in metric_preference if name in node_metrics.columns), None)
    if selected_metric is None:
        return False, "Graph model finished, but there was no plottable graph metric to display.", None
    job_dir = os.path.join(output_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)
    metrics_csv = os.path.join(job_dir, "graph_node_metrics.csv")
    summary_json = os.path.join(job_dir, "graph_summary.json")
    node_metrics.to_csv(metrics_csv, index=False)
    summary = {"graph_metrics_csv": metrics_csv, "selected_metric": selected_metric, "metrics_computed": list(result.get("metrics") or ()), "component_count": result.get("component_count"), "metric_ranges": metric_ranges, "probability_decay": decay_summary, "generated_at": datetime.now(timezone.utc).isoformat()}
    try:
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    except Exception:
        pass
    cache["graph_metrics_csv"] = metrics_csv
    cache["graph_metrics_df"] = node_metrics
    cache["graph_metric_name"] = selected_metric
    cache["graph_edges_df"] = filtered_edges.copy()
    cache["graph_metric_ranges"] = metric_ranges
    cache["graph_probability_decay"] = decay_summary
    cache["graph_summary_json"] = summary_json if os.path.exists(summary_json) else None
    cache["download_bundle_dirty"] = True
    cache["results_zip_path"] = None
    component_count = result.get("component_count")
    component_text = f" across {component_count} connected components" if component_count is not None else ""
    message = " ".join(m.strip() for m in prep_messages if m).strip()
    final_message = (f"Graph model complete. Added graph edges plus a node layer colored by {selected_metric}{component_text}. " f"Probability decay kept {decay_summary.get('edge_count_kept', 0)} of {decay_summary.get('edge_count_total', 0)} links. " "The existing LCP layers remain on the map.")
    if message:
        final_message = f"{message} {final_message}"
    return True, final_message, metrics_csv


__all__ = ["run_graph_model"]
