"""
Node network for the Sri Lanka flood-forecasting dataset.

Each node is a monitoring point placed along a major flood-prone river basin.
Nodes carry a basin id, an approximate hydrological position (upstream / mid /
downstream / outlet), and a `downstream_of` link giving the directed
river-topology edge used to build the spatiotemporal graph.

Coordinates are hand-placed from public gazetteer / river-course knowledge and
snapped by the Open-Meteo APIs to their nearest reanalysis grid cell (the
returned lat/lon and elevation are stored during download).

Zone: 'wet' (SW monsoon, May-Sep dominated), 'dry' (NE monsoon, Dec-Feb),
'intermediate'. Used only as a static contextual feature / for stratification.
"""

# node_id, name, basin, lat, lon, position, downstream_of, zone
NODES = [
    # ---------------- Kelani River basin (Colombo) ----------------
    ("KEL_UP",  "Norwood (upper Kelani)",   "Kelani",        6.831, 80.616, "upstream",   None,      "wet"),
    ("KEL_KIT", "Kitulgala",                "Kelani",        6.989, 80.412, "upstream",   "KEL_UP",  "wet"),
    ("KEL_HAN", "Hanwella",                 "Kelani",        6.902, 80.082, "mid",        "KEL_KIT", "wet"),
    ("KEL_KAD", "Kaduwela",                 "Kelani",        6.933, 79.983, "mid",        "KEL_HAN", "wet"),
    ("KEL_COL", "Colombo (Kelani mouth)",   "Kelani",        6.970, 79.873, "outlet",     "KEL_KAD", "wet"),

    # ---------------- Kalu River basin (Ratnapura / Kalutara) ----------------
    ("KAL_BAL", "Balangoda (upper Kalu)",   "Kalu",          6.652, 80.698, "upstream",   None,      "wet"),
    ("KAL_RAT", "Ratnapura",                "Kalu",          6.683, 80.400, "mid",        "KAL_BAL", "wet"),
    ("KAL_BUL", "Bulathsinhala",            "Kalu",          6.660, 80.190, "mid",        "KAL_RAT", "wet"),
    ("KAL_KLT", "Kalutara (Kalu mouth)",    "Kalu",          6.583, 79.960, "outlet",     "KAL_BUL", "wet"),

    # ---------------- Gin Ganga (Galle) ----------------
    ("GIN_NEL", "Neluwa (upper Gin)",       "Gin",           6.423, 80.293, "upstream",   None,      "wet"),
    ("GIN_BAD", "Baddegama",                "Gin",           6.171, 80.190, "mid",        "GIN_NEL", "wet"),
    ("GIN_GAL", "Galle (Gin mouth)",        "Gin",           6.089, 80.221, "outlet",     "GIN_BAD", "wet"),

    # ---------------- Nilwala Ganga (Matara) ----------------
    ("NIL_PIT", "Pitabeddara (upper Nilwala)", "Nilwala",    6.300, 80.552, "upstream",   None,      "wet"),
    ("NIL_AKU", "Akuressa",                 "Nilwala",       6.101, 80.481, "mid",        "NIL_PIT", "wet"),
    ("NIL_MAT", "Matara (Nilwala mouth)",   "Nilwala",       5.949, 80.550, "outlet",     "NIL_AKU", "wet"),

    # ---------------- Attanagalu Oya (Gampaha) ----------------
    ("ATT_NIT", "Nittambuwa (upper)",       "AttanagaluOya", 7.141, 80.098, "upstream",   None,      "wet"),
    ("ATT_GAM", "Gampaha",                  "AttanagaluOya", 7.092, 79.994, "mid",        "ATT_NIT", "wet"),
    ("ATT_JAE", "Ja-Ela (mouth)",           "AttanagaluOya", 7.074, 79.892, "outlet",     "ATT_GAM", "wet"),

    # ---------------- Maha Oya ----------------
    ("MHO_GIR", "Giriulla (upper Maha Oya)", "MahaOya",      7.331, 80.121, "upstream",   None,      "intermediate"),
    ("MHO_MOU", "Maha Oya mouth (Kochchikade)", "MahaOya",   7.301, 79.833, "outlet",     "MHO_GIR", "wet"),

    # ---------------- Deduru Oya (Kurunegala / Chilaw) ----------------
    ("DED_KUR", "Kurunegala (upper Deduru)", "DeduruOya",    7.487, 80.362, "upstream",   None,      "intermediate"),
    ("DED_CHI", "Chilaw (Deduru mouth)",    "DeduruOya",     7.576, 79.797, "outlet",     "DED_KUR", "intermediate"),

    # ---------------- Mahaweli River (largest basin) ----------------
    ("MAH_NUW", "Nuwara Eliya (upper Mahaweli)", "Mahaweli", 6.949, 80.782, "upstream",   None,      "wet"),
    ("MAH_PER", "Peradeniya / Kandy",       "Mahaweli",      7.271, 80.598, "mid",        "MAH_NUW", "wet"),
    ("MAH_MHY", "Mahiyanganaya",            "Mahaweli",      7.331, 81.001, "mid",        "MAH_PER", "intermediate"),
    ("MAH_MAN", "Manampitiya (Polonnaruwa)", "Mahaweli",     7.921, 81.101, "downstream", "MAH_MHY", "dry"),

    # ---------------- Gal Oya / Batticaloa (East) ----------------
    ("GAL_ING", "Inginiyagala (Gal Oya reservoir)", "GalOya", 7.200, 81.500, "upstream",  None,      "dry"),
    ("GAL_AMP", "Ampara",                   "GalOya",        7.291, 81.671, "mid",        "GAL_ING", "dry"),
    ("BAT_VAL", "Valachchenai",             "Batticaloa",    7.921, 81.531, "mid",        None,      "dry"),
    ("BAT_BAT", "Batticaloa (lagoon)",      "Batticaloa",    7.717, 81.700, "outlet",     "BAT_VAL", "dry"),

    # ---------------- Walawe (Embilipitiya / Hambantota) ----------------
    ("WAL_EMB", "Embilipitiya (upper Walawe)", "Walawe",     6.343, 80.851, "upstream",   None,      "intermediate"),
    ("WAL_AMB", "Ambalantota (Walawe mouth)", "Walawe",      6.121, 81.021, "outlet",     "WAL_EMB", "dry"),

    # ---------------- Kirindi Oya (Tissamaharama) ----------------
    ("KIR_TIS", "Tissamaharama",            "KirindiOya",    6.281, 81.281, "outlet",     None,      "dry"),

    # ---------------- Malwathu Oya (Anuradhapura / Mannar) ----------------
    ("MAL_ANU", "Anuradhapura (upper Malwathu)", "MalwathuOya", 8.351, 80.401, "upstream", None,     "dry"),
    ("MAL_MAN", "Mannar (Malwathu mouth)",  "MalwathuOya",   8.951, 79.951, "outlet",     "MAL_ANU", "dry"),

    # ---------------- Yan Oya / Kala Oya (dry-zone tanks) ----------------
    ("YAN_HOR", "Horowpothana (Yan Oya)",   "YanOya",        8.551, 80.881, "mid",        None,      "dry"),
    ("KLA_ELA", "Eluwankulama (Kala Oya)",  "KalaOya",       8.151, 79.851, "outlet",     None,      "dry"),

    # ---------------- Additional wet-zone flood hotspots ----------------
    ("KEL_AVI", "Avissawella",              "Kelani",        6.954, 80.211, "mid",        "KEL_HAN", "wet"),
    ("KAL_AGA", "Agalawatta",               "Kalu",          6.535, 80.150, "mid",        "KAL_BUL", "wet"),
    ("GIN_THA", "Thawalama",                "Gin",           6.331, 80.331, "upstream",   "GIN_NEL", "wet"),
    ("NIL_URA", "Urubokka",                 "Nilwala",       6.231, 80.601, "upstream",   "NIL_PIT", "wet"),
    ("MAH_VIC", "Victoria reservoir (Teldeniya)", "Mahaweli", 7.241, 80.781, "mid",       "MAH_PER", "wet"),
    ("MAH_POL", "Polonnaruwa town",         "Mahaweli",      7.940, 81.001, "downstream", "MAH_MHY", "dry"),
    ("WAL_UDA", "Udawalawe reservoir",      "Walawe",        6.441, 80.881, "upstream",   "WAL_EMB", "intermediate"),
    ("ATT_VEY", "Veyangoda",                "AttanagaluOya", 7.155, 80.061, "upstream",   "ATT_NIT", "wet"),
    ("DED_WAR", "Wariyapola",               "DeduruOya",     7.621, 80.231, "upstream",   "DED_KUR", "intermediate"),

    # ---------------- Colombo metro canal / low-lying (urban flooding) ----------------
    ("COL_KOT", "Kotte (Diyawanna low-lying)", "ColomboMetro", 6.888, 79.901, "outlet",   None,      "wet"),
    ("COL_KOL", "Kolonnawa",                "ColomboMetro",  6.933, 79.891, "outlet",     "COL_KOT", "wet"),

    # ---------------- Kalani tributary + Kelani upper add ----------------
    ("KEL_RUW", "Ruwanwella",               "Kelani",        7.041, 80.251, "mid",        "KEL_KIT", "wet"),

    # ---------------- Additional east/north ----------------
    ("MAH_KAN", "Kandy city (Mahaweli mid)", "Mahaweli",     7.291, 80.635, "mid",        "MAH_PER", "wet"),
    ("BAT_ERA", "Eravur",                   "Batticaloa",    7.771, 81.601, "mid",        "BAT_VAL", "dry"),
]

COLUMNS = ["node_id", "name", "basin", "lat", "lon", "position", "downstream_of", "zone"]


def as_records():
    return [dict(zip(COLUMNS, n)) for n in NODES]


if __name__ == "__main__":
    import csv, os
    out = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "nodes.csv")
    out = os.path.abspath(out)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(as_records())
    print(f"Wrote {len(NODES)} nodes -> {out}")
