"""Central configuration for the Sri Lanka flood dataset build."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RAW_DIR = os.path.join(ROOT, "data", "raw")
PROC_DIR = os.path.join(ROOT, "data", "processed")

# --- Temporal coverage -------------------------------------------------------
START_DATE = "2003-01-01"
END_DATE = "2024-12-31"

# --- API endpoints (no API key required) -------------------------------------
POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"  # NASA POWER daily meteorology
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"       # ERA5-Land (high-res precip)
FLOOD_URL = "https://flood-api.open-meteo.com/v1/flood"             # GloFAS reanalysis (river discharge)

# NASA POWER parameters -> friendly column names used throughout the pipeline.
# POWER is fully open (no key), daily, from 1981, and includes surface/root/profile
# soil wetness (GWET*), so no separate hourly soil-moisture feed is needed.
POWER_PARAMS = {
    "PRECTOTCORR":        "precipitation_sum",       # mm/day (bias-corrected precip)
    "T2M":                "temperature_2m_mean",     # deg C
    "T2M_MAX":            "temperature_2m_max",       # deg C
    "T2M_MIN":            "temperature_2m_min",       # deg C
    "RH2M":               "relative_humidity_2m",     # %
    "WS10M":              "windspeed_10m_mean",       # m/s
    "WS10M_MAX":          "windspeed_10m_max",        # m/s
    "ALLSKY_SFC_SW_DWN":  "shortwave_radiation",      # kWh/m2/day
    "GWETTOP":            "soil_wet_top",             # 0-1 surface soil wetness
    "GWETROOT":           "soil_wet_root",            # 0-1 root-zone soil wetness
    "GWETPROF":           "soil_wet_profile",         # 0-1 profile soil wetness
}
POWER_FILL = -999.0        # POWER missing-value sentinel

FLOOD_VARS = ["river_discharge"]

# --- Label / target configuration -------------------------------------------
# Flood state defined by GloFAS discharge exceeding a per-node return-period
# threshold, estimated from the historical daily-discharge distribution.
# Percentiles used as return-period proxies (climatological exceedance):
SEVERITY_PERCENTILES = {
    "moderate": 0.90,   # ~ frequent high flow
    "high":     0.98,   # ~ 2-year-ish event
    "severe":   0.995,  # ~ 5-10 year event
}
PRIMARY_SEVERITY = "high"          # threshold used for the main binary target
FLOOD_HORIZONS_DAYS = [1, 2, 3]    # predict flood within next t+1 (24h), t+2 (48h), t+3 (72h)

# --- Split configuration -----------------------------------------------------
TRAIN_END = "2017-12-31"           # temporal split: train <= 2017
VAL_END = "2020-12-31"             # val 2018-2020, test 2021-2024
HOLDOUT_BASIN = "Gin"              # entire basin held out for spatial generalization test

# --- Networking --------------------------------------------------------------
REQUEST_TIMEOUT = 120
MAX_RETRIES = 14                   # patient enough to wait out an hourly-quota reset
SLEEP_BETWEEN = 2.0                # polite delay between node requests (seconds)
