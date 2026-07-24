# Sources, Attribution & Licensing

All data is derived from **public, free-access** products retrieved through the
Open-Meteo APIs (no API key required for non-commercial research use).

## Primary sources

| Product | Provider | Accessed via | License / terms |
|---|---|---|---|
| **NASA POWER** daily meteorology (rainfall, temperature, humidity, wind, radiation, soil wetness) | NASA Langley Research Center (POWER project; MERRA-2 / GEOS) | POWER REST API | Free & open, no key; NASA data are not copyrighted. Attribution requested. |
| **GloFAS** river discharge reanalysis | Copernicus Emergency Management Service (CEMS) / ECMWF | Open-Meteo Flood API | Copernicus free & open; attribution required. Contains modified Copernicus Emergency Management Service Information. |
| Elevation | NASA POWER grid metadata | POWER REST API | Free & open. |
| Discharge delivery layer | **Open-Meteo** | https://open-meteo.com | Data under CC-BY 4.0; free for non-commercial use. |

## Required attribution (cite in any publication)

> Weather and soil-wetness variables: **NASA POWER Project**, NASA Langley
> Research Center (LaRC), sourced from the MERRA-2 / GEOS assimilation.
> River discharge: **GloFAS, Copernicus Emergency Management Service (CEMS)**,
> delivered via **Open-Meteo** (CC-BY 4.0).
> "These data were obtained from the NASA Langley Research Center POWER Project."

## Documented flood-event references (label validation)

The historical flood episodes used in `05_validate.py` are drawn from public
disaster reporting (Sri Lanka **Disaster Management Centre**, **ReliefWeb**,
**EM-DAT**, and contemporaneous news archives). Dates are approximate episode
windows, used only to confirm that the discharge-derived labels activate during
real events — they are not redistributed here.

## Not used (and why)

- **Kaggle "Sri Lanka flood risk" synthetic dataset** — synthetic; unsuitable as
  a scientific label source (prototyping only).
- **Rain-gauge / Irrigation Department discharge** — not publicly downloadable
  without institutional access; would upgrade the dataset to *physics-guided*
  (see roadmap in the dataset card).

## Compliance notes
- This dataset redistributes **derived features and labels**, not raw Copernicus
  files. Downstream users must retain the attribution above.
- For **commercial** use, verify Open-Meteo and Copernicus commercial terms and
  consider pulling ERA5/GloFAS directly from the Copernicus CDS/EWDS.
