"""
Kalu Ganga river nodes (5 monitoring points).

River flows: Balangoda -> Ratnapura -> Bulathsinhala -> Kalutara (mouth)
             Agalawatta is a tributary joining between Bulathsinhala and Kalutara.

All nodes are in the 'Kalu' basin, wet zone (SW monsoon dominant).
"""

# node_id, name, basin, lat, lon, position, downstream_of, zone
NODES = [
    # ---- Kalu River main stem (upstream -> outlet) ----
    ("KAL_BAL", "Balangoda (upper Kalu)",  "Kalu", 6.652, 80.698, "upstream",   None,      "wet"),
    ("KAL_RAT", "Ratnapura",               "Kalu", 6.683, 80.400, "mid",        "KAL_BAL", "wet"),
    ("KAL_BUL", "Bulathsinhala",           "Kalu", 6.660, 80.190, "mid",        "KAL_RAT", "wet"),
    ("KAL_AGA", "Agalawatta",              "Kalu", 6.535, 80.150, "mid",        "KAL_BUL", "wet"),
    ("KAL_KLT", "Kalutara (Kalu mouth)",   "Kalu", 6.583, 79.960, "outlet",     "KAL_AGA", "wet"),
]

COLUMNS = ["node_id", "name", "basin", "lat", "lon", "position", "downstream_of", "zone"]


def as_records():
    return [dict(zip(COLUMNS, n)) for n in NODES]


if __name__ == "__main__":
    import csv, os
    out = os.path.join(os.path.dirname(__file__), "data", "processed", "nodes.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(as_records())
    print(f"Wrote {len(NODES)} Kalu Ganga nodes -> {out}")
    for n in NODES:
        print(f"  {n[0]:10s}  {n[1]:30s}  lat={n[3]}  lon={n[4]}")
