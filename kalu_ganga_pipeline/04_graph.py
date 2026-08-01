"""
Step 4 - Build the river-topology graph for the 5 Kalu Ganga nodes.

Edge types:
  * flow    : directed upstream -> downstream (from downstream_of links)
  * spatial : k-nearest-neighbour edges (haversine distance, bidirectional,
              distance-decayed weight)

Output: data/processed/nodes.csv
        data/processed/edges.csv
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C
from nodes import as_records


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def main(k=3, spatial_scale_km=30.0):
    """
    k               : spatial kNN neighbours per node
    spatial_scale_km: distance decay scale for spatial edge weights
    """
    nodes = as_records()
    ndf   = pd.DataFrame(nodes)

    meta_path = os.path.join(C.PROC_DIR, "nodes_meta.csv")
    if os.path.exists(meta_path):
        meta = pd.read_csv(meta_path)[["node_id", "elevation_m", "snap_lat", "snap_lon"]]
        ndf  = ndf.merge(meta, on="node_id", how="left")
    ndf.to_csv(os.path.join(C.PROC_DIR, "nodes.csv"), index=False)

    idx  = {r["node_id"]: i for i, r in enumerate(nodes)}
    lat  = {r["node_id"]: r["lat"]  for r in nodes}
    lon  = {r["node_id"]: r["lon"]  for r in nodes}
    ids  = [r["node_id"] for r in nodes]
    edges = []

    # 1) directed flow edges (upstream -> downstream)
    for r in nodes:
        parent = r["downstream_of"]
        if parent and parent in idx:
            dist = haversine(lat[parent], lon[parent], r["lat"], r["lon"])
            edges.append({"src": parent, "dst": r["node_id"], "edge_type": "flow",
                          "weight": 1.0, "distance_km": round(dist, 2)})

    # 2) spatial kNN edges (bidirectional, distance-decayed)
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

    edf    = pd.DataFrame(edges).drop_duplicates(subset=["src", "dst", "edge_type"])
    edf.to_csv(os.path.join(C.PROC_DIR, "edges.csv"), index=False)
    n_flow = (edf.edge_type == "flow").sum()
    n_spat = (edf.edge_type == "spatial").sum()
    print(f"Graph: {len(ndf)} nodes, {len(edf)} edges  ({n_flow} flow + {n_spat} spatial)")
    print("Flow edges:")
    for _, e in edf[edf.edge_type == "flow"].iterrows():
        print(f"  {e.src} -> {e.dst}  ({e.distance_km:.1f} km)")
    print("-> data/processed/nodes.csv, data/processed/edges.csv")


if __name__ == "__main__":
    main()
