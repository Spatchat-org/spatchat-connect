"""Graph theory utilities for connectivity analyses.

This module provides a convenience function that takes an edge list and
computes a small suite of graph-theoretic metrics that are commonly used in
connectivity workflows (degree, betweenness, closeness, PageRank, clustering,
and component information).  The function returns the metrics as a tidy
``pandas.DataFrame`` so the caller can directly join the results back to a node
table or serialize them to disk.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence

import networkx as nx
import numpy as np
import pandas as pd

try:
    from analysis_tracker import record_step as _record_analysis_step
except Exception:  # pragma: no cover - optional dependency
    _record_analysis_step = None



__all__ = ["GraphAnalysisParams", "run_graph_theory_analysis"]


@dataclass(slots=True)
class GraphAnalysisParams:
    """Configuration for :func:`run_graph_theory_analysis`.

    Parameters
    ----------
    source_col:
        Name of the column that contains the source node identifiers.
    target_col:
        Name of the column that contains the target node identifiers.
    weight_col:
        Optional name of a column containing edge weights (treated as
        distances / costs).  Values that cannot be coerced to ``float`` are
        ignored when building the graph.
    directed:
        If ``True`` a :class:`networkx.DiGraph` is used; otherwise an
        undirected :class:`networkx.Graph` is built.
    metrics:
        Sequence of metric names to compute.  Supported values are ``"degree"``,
        ``"betweenness"``, ``"closeness"``, ``"pagerank"``, ``"clustering"`` and
        ``"components"``.  Unknown entries are ignored.
    """

    source_col: str = "source"
    target_col: str = "target"
    weight_col: Optional[str] = None
    distance_col: Optional[str] = None
    strength_col: Optional[str] = None
    directed: bool = False
    metrics: Sequence[str] = field(
        default_factory=lambda: (
            "degree",
            "betweenness",
            "closeness",
            "pagerank",
            "clustering",
            "components",
        )
    )


def _coerce_weight(value: object) -> Optional[float]:
    """Safely coerce *value* to ``float``.

    Returns ``None`` when the value cannot be interpreted as a finite number.
    """

    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _prepare_graph(edges: pd.DataFrame, params: GraphAnalysisParams) -> nx.Graph:
    """Build a NetworkX graph from an edge table."""

    if params.source_col not in edges.columns or params.target_col not in edges.columns:
        raise ValueError(
            "Edge table must contain source and target columns"
        )

    graph_cls = nx.DiGraph if params.directed else nx.Graph
    G: nx.Graph = graph_cls()

    distance_col = params.distance_col or params.weight_col
    strength_col = params.strength_col or params.weight_col
    use_distance = distance_col and distance_col in edges.columns
    use_strength = strength_col and strength_col in edges.columns

    if use_distance or use_strength:
        records: list[tuple[object, object, dict[str, float]]] = []
        needed_cols = [params.source_col, params.target_col]
        if use_distance:
            needed_cols.append(distance_col)
        if use_strength and strength_col not in needed_cols:
            needed_cols.append(strength_col)
        for row in edges[needed_cols].itertuples(index=False, name=None):
            src, dst = row[0], row[1]
            if pd.isna(src) or pd.isna(dst) or not str(src).strip() or not str(dst).strip():
                continue
            src, dst = str(src), str(dst)
            attrs: dict[str, float] = {}
            idx = 2
            if use_distance:
                distance_val = _coerce_weight(row[idx])
                idx += 1
                if distance_val is not None:
                    attrs["distance"] = distance_val
            if use_strength:
                strength_pos = idx if use_distance and strength_col != distance_col else 2
                strength_val = _coerce_weight(row[strength_pos])
                if strength_val is not None:
                    attrs["strength"] = strength_val
            records.append((src, dst, attrs))
        if records:
            G.add_edges_from(records)
        else:
            G.add_edges_from(
                ((str(src), str(dst)) for src, dst in edges[[params.source_col, params.target_col]].itertuples(index=False, name=None) if pd.notna(src) and pd.notna(dst) and str(src).strip() and str(dst).strip())
            )
    else:
        G.add_edges_from(
            ((str(src), str(dst)) for src, dst in edges[[params.source_col, params.target_col]].itertuples(index=False, name=None) if pd.notna(src) and pd.notna(dst) and str(src).strip() and str(dst).strip())
        )

    return G


def run_graph_theory_analysis(
    edges: pd.DataFrame, params: Optional[GraphAnalysisParams] = None
) -> dict[str, object]:
    """Compute graph-theory metrics for an edge list.

    Parameters
    ----------
    edges:
        Edge list where each row defines a connection between two nodes.  The
        table **must** contain the columns defined by ``params.source_col`` and
        ``params.target_col``.  The optional ``params.weight_col`` is interpreted
        as a distance/cost weight (smaller = stronger connection).
    params:
        Optional :class:`GraphAnalysisParams` instance that customises column
        names, whether the graph is directed, and which metrics to compute.

    Returns
    -------
    dict
        Dictionary with the following keys:

        ``graph``
            The :mod:`networkx` graph instance that was analysed.
        ``node_metrics``
            ``pandas.DataFrame`` indexed by node identifier with the metrics
            that were computed.
        ``metrics``
            Tuple of metric names (in the order they were requested) that were
            successfully evaluated.
        ``component_count``
            Number of connected (or weakly connected) components when the
            ``"components"`` metric is requested.
    """

    if edges.empty:
        raise ValueError("Edge table is empty; nothing to analyse")

    params = params or GraphAnalysisParams()
    metrics = tuple(dict.fromkeys(params.metrics))  # preserve order, drop duplicates

    G = _prepare_graph(edges, params)
    if G.number_of_nodes() == 0:
        raise ValueError("Graph has no nodes after filtering invalid edges")

    node_df = pd.DataFrame(index=pd.Index(sorted(G.nodes()), name="node"))
    available_metrics: list[str] = []
    distance_attr = None
    if (params.distance_col and params.distance_col in edges.columns) or (params.weight_col and params.weight_col in edges.columns):
        distance_attr = "distance"
    strength_attr = None
    if (params.strength_col and params.strength_col in edges.columns) or (params.weight_col and params.weight_col in edges.columns):
        strength_attr = "strength"

    for metric in metrics:
        if metric == "degree":
            if params.directed:
                node_df["out_degree"] = pd.Series(dict(G.out_degree(weight=strength_attr)))
                node_df["in_degree"] = pd.Series(dict(G.in_degree(weight=strength_attr)))
            node_df["degree"] = pd.Series(dict(G.degree(weight=strength_attr)))
            available_metrics.append(metric)
        elif metric == "betweenness":
            node_df["betweenness"] = pd.Series(
                nx.betweenness_centrality(G, weight=distance_attr, normalized=True)
            )
            available_metrics.append(metric)
        elif metric == "closeness":
            node_df["closeness"] = pd.Series(
                nx.closeness_centrality(G, distance=distance_attr)
            )
            available_metrics.append(metric)
        elif metric == "pagerank":
            node_df["pagerank"] = pd.Series(
                nx.pagerank(G, weight=strength_attr)
            )
            available_metrics.append(metric)
        elif metric == "clustering":
            if params.directed:
                # Use an undirected view for clustering.
                undirected = G.to_undirected()
                node_df["clustering"] = pd.Series(nx.clustering(undirected, weight=strength_attr))
            else:
                node_df["clustering"] = pd.Series(nx.clustering(G, weight=strength_attr))
            available_metrics.append(metric)
        elif metric == "components":
            if params.directed:
                components = list(nx.weakly_connected_components(G))
            else:
                components = list(nx.connected_components(G))
            comp_id_map = {}
            comp_size_map = {}
            for comp_id, nodes in enumerate(components):
                size = len(nodes)
                for node in nodes:
                    comp_id_map[node] = comp_id
                    comp_size_map[node] = size
            node_df["component_id"] = pd.Series(comp_id_map)
            node_df["component_size"] = pd.Series(comp_size_map)
            available_metrics.append(metric)
        # Unknown metrics are ignored silently so the caller can request
        # experimental names without breaking backwards compatibility.

    result = {
        "graph": G,
        "node_metrics": node_df.sort_index(),
        "metrics": tuple(available_metrics),
    }
    if "components" in available_metrics:
        result["component_count"] = int(node_df["component_id"].nunique())

    if _record_analysis_step:
        try:
            inputs = {
                "edges_count": int(len(edges)),
                "params": asdict(params),
            }
            outputs = {
                "nodes_analyzed": int(G.number_of_nodes()),
                "edges_used": int(G.number_of_edges()),
                "metrics_computed": tuple(available_metrics),
            }
            _record_analysis_step(
                kind="graph_theory_analysis",
                function="methods.gt.run_graph_theory_analysis",
                inputs=inputs,
                outputs=outputs,
                code_obj=run_graph_theory_analysis,
            )
        except Exception:
            pass

    return result


