"""
Step 4 - Build the graph structure for the spatiotemporal GNN.

Edge types:
  * flow    : directed upstream -> downstream river-topology edge (from `downstream_of`)
  * spatial : k-nearest-neighbour edges (haversine) capturing shared meteorology,
              added in both directions with a distance-decayed weight.

Output: data/processed/edges.csv, data/processed/nodes.csv
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import config as C
from nodes import as_records


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def main(k=4, spatial_scale_km=40.0):
    nodes = as_records()
    ndf = pd.DataFrame(nodes)

    # enrich with snapped elevation if available
    meta_path = os.path.join(C.PROC_DIR, "nodes_meta.csv")
    if os.path.exists(meta_path):
        meta = pd.read_csv(meta_path)[["node_id", "elevation_m", "snap_lat", "snap_lon"]]
        ndf = ndf.merge(meta, on="node_id", how="left")
    ndf.to_csv(os.path.join(C.PROC_DIR, "nodes.csv"), index=False)

    idx = {r["node_id"]: i for i, r in enumerate(nodes)}
    edges = []
    lat = {r["node_id"]: r["lat"] for r in nodes}
    lon = {r["node_id"]: r["lon"] for r in nodes}

    # 1) directed flow edges: parent (upstream) -> child (this node)
    for r in nodes:
        parent = r["downstream_of"]
        if parent and parent in idx:
            dist = haversine(lat[parent], lon[parent], r["lat"], r["lon"])
            edges.append({"src": parent, "dst": r["node_id"], "edge_type": "flow",
                          "weight": 1.0, "distance_km": round(dist, 2)})

    # 2) spatial kNN edges (bidirectional, distance-decayed)
    ids = [r["node_id"] for r in nodes]
    coords = np.array([[lat[i], lon[i]] for i in ids])
    for a in range(len(ids)):
        dists = haversine(coords[a, 0], coords[a, 1], coords[:, 0], coords[:, 1])
        order = np.argsort(dists)
        added = 0
        for b in order:
            if b == a:
                continue
            edges.append({"src": ids[a], "dst": ids[b], "edge_type": "spatial",
                          "weight": round(float(np.exp(-dists[b] / spatial_scale_km)), 4),
                          "distance_km": round(float(dists[b]), 2)})
            added += 1
            if added >= k:
                break

    edf = pd.DataFrame(edges).drop_duplicates(subset=["src", "dst", "edge_type"])
    edf.to_csv(os.path.join(C.PROC_DIR, "edges.csv"), index=False)
    n_flow = (edf.edge_type == "flow").sum()
    n_spat = (edf.edge_type == "spatial").sum()
    print(f"Graph: {len(ndf)} nodes, {len(edf)} edges ({n_flow} flow, {n_spat} spatial)")
    print(f"-> data/processed/nodes.csv, data/processed/edges.csv")


if __name__ == "__main__":
    main()
