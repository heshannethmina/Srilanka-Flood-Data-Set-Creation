"""River graph construction (PROJECT_PROPOSAL.md §7.2).

Two relations are kept **separate** and never symmetrised:
  relation 0 — flow    : 35 directed upstream → downstream edges
  relation 1 — spatial : 204 distance-decayed k-NN edges, bidirectional
  relation 2 — self    : self-loops, so a node always retains its own state
                         (essential in flow-only mode, where headwater nodes
                          have no incoming edges at all)

Edge attributes are `[weight, distance_km/100, Δelevation/100, is_flow]` with
Δelevation = elevation(src) − elevation(dst), i.e. the drop the water falls.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd

REL_FLOW, REL_SPATIAL, REL_SELF = 0, 1, 2
N_RELATIONS = 3


@dataclass
class Graph:
    edge_index: np.ndarray   # [2, E] int64, row 0 = src, row 1 = dst
    edge_attr: np.ndarray    # [E, 4] float32
    edge_rel: np.ndarray     # [E] int64
    n_nodes: int

    def summary(self) -> str:
        c = np.bincount(self.edge_rel, minlength=N_RELATIONS)
        return (f"Graph({self.n_nodes} nodes, {self.edge_index.shape[1]} edges: "
                f"{c[REL_FLOW]} flow, {c[REL_SPATIAL]} spatial, {c[REL_SELF]} self)")


def build_graph(edges: pd.DataFrame, nodes: pd.DataFrame,
                node_ids: Sequence[str], mode: str = "both",
                self_loops: bool = True) -> Graph:
    nidx = {n: i for i, n in enumerate(node_ids)}
    elev = (nodes.set_index("node_id").reindex(node_ids)["elevation_m"]
            .fillna(0.0).to_numpy(np.float32))

    keep = {"both": {"flow", "spatial"}, "flow": {"flow"},
            "spatial": {"spatial"}, "none": set()}[mode]

    src: List[int] = []
    dst: List[int] = []
    attr: List[List[float]] = []
    rel: List[int] = []

    e = edges[edges["edge_type"].isin(keep)]
    for r in e.itertuples(index=False):
        if r.src not in nidx or r.dst not in nidx:
            continue
        s, d = nidx[r.src], nidx[r.dst]
        is_flow = 1.0 if r.edge_type == "flow" else 0.0
        src.append(s)
        dst.append(d)
        attr.append([float(r.weight), float(r.distance_km) / 100.0,
                     float(elev[s] - elev[d]) / 100.0, is_flow])
        rel.append(REL_FLOW if is_flow else REL_SPATIAL)

    if self_loops:
        for i in range(len(node_ids)):
            src.append(i)
            dst.append(i)
            attr.append([1.0, 0.0, 0.0, 0.0])
            rel.append(REL_SELF)

    if not src:      # graph_mode='none' with self_loops=False
        return Graph(np.zeros((2, 0), np.int64), np.zeros((0, 4), np.float32),
                     np.zeros((0,), np.int64), len(node_ids))

    return Graph(
        edge_index=np.array([src, dst], dtype=np.int64),
        edge_attr=np.array(attr, dtype=np.float32),
        edge_rel=np.array(rel, dtype=np.int64),
        n_nodes=len(node_ids),
    )
