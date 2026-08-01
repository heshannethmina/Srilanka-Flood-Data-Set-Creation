# Validation & EDA Report — Kalu Ganga Flood Dataset

Nodes: ['KAL_AGA', 'KAL_BAL', 'KAL_BUL', 'KAL_KLT', 'KAL_RAT']  |  2003-01-01 -> 2025-12-31

## 1. Dataset size & class balance

- Total node-days      : **42,005**
- Valid supervised rows: **41,850**
- Flood-state prevalence ('high'): **1.80%**
- Positive rate target_flood_1d (valid): **1.80%**
- Positive rate target_flood_2d (valid): **2.13%**
- Positive rate target_flood_3d (valid): **2.43%**

## 2. Split sizes (valid samples)

  - test: 9,125
  - train: 27,245
  - val: 5,480

## 3. Per-node flood statistics

| Node | Position | Flood-days | Node-days | Rate % | Thr_high m3/s |
|---|---|---|---|---|---|
| KAL_AGA | mid | 165 | 8401 | 1.96 | 5.4 |
| KAL_BAL | upstream | 145 | 8401 | 1.73 | 39.1 |
| KAL_BUL | mid | 152 | 8401 | 1.81 | 6.7 |
| KAL_KLT | outlet | 156 | 8401 | 1.86 | 632.1 |
| KAL_RAT | mid | 137 | 8401 | 1.63 | 157.5 |

## 4. Label validation vs documented Kalu Ganga flood events

Detection window widened to ±4 days (downstream discharge lags rain).

| Event | Fired@high | Fired@moderate | Max pctl | Detected |
|---|---|---|---|---|
| May 2003 Ratnapura floods | 0/5 | 0/5 | 0.655 | ⚠️ |
| Jun 2014 Kalu floods | 0/5 | 0/5 | 0.594 | ⚠️ |
| May 2016 SW monsoon floods | 5/5 | 5/5 | 1.000 | ✅ |
| May 2017 SW floods (Ratnapura/Kalutara) | 0/5 | 0/5 | 0.896 | ⚠️ |
| May 2018 SW floods | 0/5 | 1/5 | 0.901 | ◧ |
| Dec 2019 Kalu floods | 0/5 | 5/5 | 0.955 | ◧ |
| Nov 2021 Western / Kalu floods | 0/5 | 3/5 | 0.926 | ◧ |
| Jun 2024 SW / Kalu floods | 4/5 | 5/5 | 0.992 | ✅ |

**Detection: 2/8 at high threshold, 5/8 at moderate threshold.**

## 5. Feature missingness (top columns with gaps)

| Column | % missing |
|---|---|
| event_id | 98.2 |
| discharge_rise_3d | 0.04 |
| discharge_rise_1d | 0.01 |
| target_flood_1d | 0.01 |
| target_onset_1d | 0.01 |
| target_onset_2d | 0.01 |
| target_flood_2d | 0.01 |
| target_flood_3d | 0.01 |
| target_onset_3d | 0.01 |
| target_next1d_discharge | 0.01 |