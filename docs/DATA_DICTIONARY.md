# Data Dictionary — `flood_dataset.parquet`

One row per **(node_id, date)**. Units in brackets.

## Identifiers & static context
| Column | Type | Description |
|---|---|---|
| `node_id` | str | Monitoring-point id (e.g. `KAL_RAT` = Ratnapura on the Kalu). |
| `date` | date | UTC calendar day. |
| `basin` | str | River basin the node belongs to. |
| `zone` | str | Climate zone: `wet` / `dry` / `intermediate`. |
| `position` | str | Hydrological position: `upstream` / `mid` / `downstream` / `outlet`. |
| `elevation_m` | float | Grid-cell elevation from Copernicus DEM (via Open-Meteo) [m]. |
| `drainage_proxy_q` | float | Mean train-period discharge — proxy for drainage area [m³/s]. |
| `log_drainage_proxy` | float | `log1p(drainage_proxy_q)`. |

## Dynamic meteorological inputs (NASA POWER, daily)
| Column | POWER param | Description |
|---|---|---|
| `precipitation_sum` | (ERA5-Land) | **Primary** daily precipitation [mm], high-res **ERA5-Land 0.1°** (Open-Meteo). Falls back to POWER where ERA5 missing. |
| `precip_era5` | (ERA5-Land) | Raw ERA5-Land daily precipitation [mm]. |
| `precipitation_sum_power` | PRECTOTCORR | POWER 0.5° daily precipitation [mm] (kept for reference/comparison). |
| `temperature_2m_mean/max/min` | T2M / T2M_MAX / T2M_MIN | 2 m air temperature [°C]. |
| `relative_humidity_2m` | RH2M | 2 m relative humidity [%]. |
| `windspeed_10m_mean` | WS10M | Mean 10 m wind speed [m/s]. |
| `windspeed_10m_max` | WS10M_MAX | Max 10 m wind speed [m/s]. |
| `shortwave_radiation` | ALLSKY_SFC_SW_DWN | All-sky surface shortwave down [kWh/m²/day]. |

## Soil-wetness inputs (NASA POWER, daily)
| Column | POWER param | Description |
|---|---|---|
| `soil_wet_top` | GWETTOP | Surface soil wetness (0–1, saturation fraction). |
| `soil_wet_root` | GWETROOT | Root-zone soil wetness (0–1). |
| `soil_wet_profile` | GWETPROF | Profile soil wetness (0–1). |
| `soil_wet_{top,root,profile}_anom` | — | Anomaly vs train-period mean. |

## Engineered antecedent-rainfall features
| Column | Type | Description |
|---|---|---|
| `precip_sum_{2,3,5,7,10,15,30}d` | float | Trailing rainfall accumulation over N days [mm]. |
| `precip_max_{3,7}d` | float | Max daily rainfall in trailing window [mm]. |
| `api_k090` | float | Antecedent Precipitation Index, decay k=0.90. |
| `wetdays_7d` | float | Count of days with >1 mm rain in last 7 days. |

## Hydrological features (GloFAS discharge)
| Column | Type | Description |
|---|---|---|
| `river_discharge` / `discharge` | float | Daily mean river discharge [m³/s]. |
| `log_discharge` | float | `log1p(discharge)`. |
| `discharge_rise_1d/3d` | float | Change vs 1 / 3 days earlier [m³/s]. |
| `discharge_mean_3d/7d` | float | Trailing mean discharge [m³/s]. |
| `q_clim_mean`, `q_clim_std` | float | Day-of-year climatology (train only) [m³/s]. |
| `discharge_anom` | float | `discharge − q_clim_mean`. |
| `discharge_zscore` | float | Standardized anomaly. |
| `discharge_pctl` | float | Empirical percentile vs train distribution (0–1). |

## Labels & thresholds
| Column | Type | Description |
|---|---|---|
| `flood_moderate/high/severe` | int8 | 1 if discharge ≥ 90 / 98 / 99.5th pctl threshold. |
| `thr_moderate/high/severe` | float | Per-node discharge thresholds [m³/s]. |
| `flood_state` | int8 | **Primary** flood state (= `flood_high`). |
| `label_confidence` | float | 0.5–1.0; lower inside ±15% band around the threshold; 0 if discharge missing. |

## Prediction targets (future-looking)
| Column | Type | Description |
|---|---|---|
| `target_flood_1d/2d/3d` | float | 1 if flood_state occurs within next 24 / 48 / 72 h. **Primary = `target_flood_1d`.** |
| `target_onset_1d/2d/3d` | float | 1 if a *new* flood begins within the window (flood ahead & not flooding now). |
| `target_next1d_discharge` | float | Discharge on day t+1 [m³/s] (regression). |
| `target_next3d_max_zscore` | float | Max discharge z-score over t+1..t+3 (regression). |

## Splits & bookkeeping
| Column | Type | Description |
|---|---|---|
| `event_id` | str/NA | Flood-episode id (merged consecutive flood days); NA on non-flood days. |
| `split_temporal` | str | `train` (≤2017) / `val` (2018–2020) / `test` (2021–2024). |
| `split_basin_holdout` | str | `test` if basin = Gin, else `train`. |
| `valid_sample` | int8 | 1 if ≥30-day feature warmup **and** target defined **and** discharge present. Filter on this for training. |

## Companion files
- `nodes.csv` — node_id, name, basin, lat/lon, position, downstream_of, zone, elevation.
- `edges.csv` — `src, dst, edge_type ∈ {flow, spatial}, weight, distance_km`.
- `events.csv` — event_id, node_id, basin, start, end, duration_days, peak_discharge.
- `thresholds.json` — per-node severity thresholds + annual-max reference level.
