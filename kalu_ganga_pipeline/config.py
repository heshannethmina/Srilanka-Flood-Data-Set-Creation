"""
Central configuration for the Kalu Ganga flood dataset build.

Scoped to the 5 Kalu River basin nodes:
  KAL_BAL  Balangoda    (upper)
  KAL_RAT  Ratnapura    (mid)
  KAL_BUL  Bulathsinhala(mid)
  KAL_AGA  Agalawatta   (mid)
  KAL_KLT  Kalutara     (outlet)

Data sources (all free, no API key):
  - NASA POWER      : daily weather + soil wetness
  - Open-Meteo ERA5 : high-res precipitation
  - GloFAS Flood API: river discharge
  - MS Planetary Computer STAC (sentinel-1-rtc): SAR imagery
"""
import os

HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = HERE                                          # kalu_ganga_pipeline/ is the root
RAW_DIR  = os.path.join(ROOT, "data", "raw")
PROC_DIR = os.path.join(ROOT, "data", "processed")

# ── Temporal coverage ────────────────────────────────────────────────────────
START_DATE = "2003-01-01"
END_DATE   = "2025-12-31"

# ── API endpoints (no API key required) ──────────────────────────────────────
POWER_URL   = "https://power.larc.nasa.gov/api/temporal/daily/point"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FLOOD_URL   = "https://flood-api.open-meteo.com/v1/flood"

# NASA POWER parameters -> friendly column names
POWER_PARAMS = {
    "PRECTOTCORR":       "precipitation_sum",
    "T2M":               "temperature_2m_mean",
    "T2M_MAX":           "temperature_2m_max",
    "T2M_MIN":           "temperature_2m_min",
    "RH2M":              "relative_humidity_2m",
    "WS10M":             "windspeed_10m_mean",
    "WS10M_MAX":         "windspeed_10m_max",
    "ALLSKY_SFC_SW_DWN": "shortwave_radiation",
    "GWETTOP":           "soil_wet_top",
    "GWETROOT":          "soil_wet_root",
    "GWETPROF":          "soil_wet_profile",
}
POWER_FILL  = -999.0
FLOOD_VARS  = ["river_discharge"]

# ── Label / target configuration ─────────────────────────────────────────────
SEVERITY_PERCENTILES = {
    "moderate": 0.90,   # frequent high flow
    "high":     0.98,   # ~2-year event
    "severe":   0.995,  # ~5-10 year event
}
PRIMARY_SEVERITY     = "high"
FLOOD_HORIZONS_DAYS  = [1, 2, 3]

# ── Train / Val / Test split ──────────────────────────────────────────────────
TRAIN_END     = "2017-12-31"   # train  <= 2017
VAL_END       = "2020-12-31"   # val    2018-2020  |  test 2021-2025
HOLDOUT_BASIN = None           # no basin holdout (single-basin dataset)

# ── Networking ────────────────────────────────────────────────────────────────
REQUEST_TIMEOUT  = 120
MAX_RETRIES      = 14
SLEEP_BETWEEN    = 2.0
