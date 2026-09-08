# Sri Lankan flood literature — what the papers actually say

Compiled 2026-09-08 from the downloaded PDFs in `Research_Papers.zip`. Every
number below was read from the paper itself, not from an abstract, a search
snippet or recollection. Pages actually read are recorded per source in §7.

**Why this document exists.** The proposal's reference lists were assembled at
design time and describe what these papers *claim*. This records what they
**measure**, which in three cases is very different — and that difference is the
strongest argument this project has.

---

## 1. The headline: every Sri Lankan ML flood paper reports a metric that hides its failure

| Paper | Headline claim | What the paper's own numbers show |
|---|---|---|
| **#07** Saubhagya et al. 2025 | "80%, 80%, 100% accuracy for 1–3-day-ahead" | Computed on **5 events in total** — 3 Minor + 2 Critical across 2017–2019 |
| **#22** Thilakarathne & Premachandra 2017 | "91.7% accuracy … can be used to predict flood incidents in any region in Sri Lanka" | **Mean recall 0.404.** Two of ten folds have precision = 0.000 *and* recall = 0.000 |
| **#21** Mahaganapathy et al. 2023 | "99% f1 score" | SMOTE applied; no temporal split stated anywhere in the abstract |

This is not a marginal observation. It is the same failure your project measured
from the other direction, and it is now documented in three independent local
studies. §2–§4 give the arithmetic.

---

## 2. #07 — Saubhagya et al. (2025), your closest competitor

*A Fusion of Deep Learning and Time Series Regression for Flood Forecasting: An
Application to the Ratnapura Area Based on the Kalu River Basin in Sri Lanka.*
Forecasting **7(2), 29**. Univ. of Colombo + Deakin. Funded by Univ. of Colombo
grant AP/3/2019/CG/30. Read in full (24 pp).

### What they built

- **One target station**: Ratnapura (6°40′43.2″N, 80°23′49.8″E), Kalu River Basin.
- **6 predictors**: water level at Ratnapura + 4 neighbouring gauges (Dela,
  Ellagawa, Putupaula, Millakanda) + rainfall at Ratnapura.
- **Data: 1 Jan 2014 – 31 Dec 2019** (6 years), daily. WL from Dept. of
  Irrigation, rainfall from Dept. of Meteorology.
- **Lookback 10 days**, horizons 1–3 days. Feature selection by Granger
  causality + cross-correlation.
- **Model A** = Vanilla Bi-LSTM (1 hidden layer, 100 neurons, Adam, lr 0.01, 100
  epochs, MAE loss). **Model B** = Model A's WL forecast + forecast rainfall
  combined by GLS linear regression, then ARIMA fitted to the residuals.
- **Target is water level (regression)**, thresholded *afterwards* into DoI risk
  classes: <16.5 m MSL No Flood; <19.5 Alert; <21 Minor; <21.75 Major; ≥21.75
  Critical.

### Their protocol is sound — do not accuse them of leakage

They use an **expanding-window temporal split**: train 2014–2015 → test 2016;
train 2014–2016 → test 2017; train 2014–2017 → test 2018; train 2014–2018 → test
2019. Then a genuinely held-out 2020 with no retraining. That is a correct
protocol for autocorrelated series, and it is the *strongest* methodology in the
local literature. The problem is elsewhere.

### The evidence base is five events

Counted from their own confusion matrices (Table 5a/b, Table 7, Table 9):

| Year | No Flood | Alert | Minor | Major | Critical |
|---|---|---|---|---|---|
| 2016 | 357 | 7 | 0 | 0 | 0 |
| 2017 | 348 | 12 | 1 | 0 | **2** |
| 2018 | 353 | 7 | 2 | 0 | 0 |
| 2019 | 357 | 5 | 0 | 0 | 0 |
| 2020 | 360 | 4 | 0 | 0 | 0 |

"Actionable" = Minor + Critical. Across the 2017–2019 evaluation window that is
**3 Minor + 2 Critical = 5 events**. The abstract's "accuracies of 80%, 80% and
100%" are therefore 4/5, 4/5 and 5/5. **Major level never occurs at all** in six
years of data, so that class is untested.

### Their overall accuracy sits at or below the majority-class rate

Table 8 reports overall forecasting accuracy. Compare against always predicting
"No Flood":

| Year | Majority-class rate | Their p1 | p2 | p3 |
|---|---|---|---|---|
| 2017 | 348/363 = **95.9%** | 96% | 94% | 89% |
| 2018 | 353/362 = **97.5%** | 92% | 87% | 81% |
| 2019 | 357/362 = **98.6%** | 92% | 83% | 69% |
| 2020 | 360/364 = **98.9%** | 94% | 86% | 72% |

In every year except 2017 at one day ahead, **the reported accuracy is below what
a model that never predicts a flood would score.** (Fair caveat: their accuracy
is over five classes, not binary, so this is not a strict like-for-like — but the
majority-class rate is the correct reference point for any accuracy figure on
this data, and it is not reported anywhere in the paper.)

### The two sentences you should quote

From their limitations:

> "The fitted model in this study uses data from only a few nearby WL gauging
> stations that are currently in function. **A better performance could have been
> achieved if data from a higher number of nearby WL gauging stations were
> available.**"

That is an explicit statement of the gap your 51-node, 16-basin network fills.

And from their conclusions, the methodological disagreement you must answer:

> "Obtaining flood risk levels based on WL forecasts produces more precise
> warnings since **direct flood risk forecasting leads to an imbalance
> classification problem.**"

They argue *against* direct classification — which is exactly what your project
does. This is a real, reasoned position, not an oversight, and your paper has to
engage it rather than ignore it. Your answer is available and defensible:
calibrated probability + PR-AUC + event detection at a capped false-alarm ratio
handles imbalance directly and *reports* the failure mode, whereas the
regress-then-threshold detour hides it behind an accuracy figure computed on five
events. But you have to make that argument explicitly.

### Their limitations, verbatim

No soil moisture, no river velocity; significant missingness in WL data;
insufficient compute; upstream stations Dela and Malwala have long missing
periods, which prevented building one model containing both.

---

## 3. #22 — Thilakarathne & Premachandra (2017)

*Predicting Floods in North Central Province of Sri Lanka using Machine Learning
and Data Mining Methods.* SLAAI 2017, printed pp. 44–49. Read in full (6 pp).

- **Monthly** resolution, Anuradhapura district, Jan 1976 – Dec 2015. 480 records.
- **51 flood-type disaster records** from DesInventar.
- Two stages: ARIMA forecasts monthly rainfall / min / max temperature → ANN
  binary classifier (1 hidden layer, **2 neurons**, lr 0.70) predicts flood in a
  future month. Built in Azure ML Studio; deployed as a web API.
- **10-fold cross-validation.**

### The reported table refutes the reported conclusion

Table 3, reproduced exactly:

| Fold | Accuracy | Precision | Recall | AUC |
|---|---|---|---|---|
| 0 | 0.854 | 1.000 | 0.364 | 0.958 |
| 1 | 0.917 | 0.500 | 0.250 | 0.747 |
| 2 | 0.875 | 0.800 | 0.444 | 0.923 |
| 3 | 0.958 | 1.000 | 0.714 | 0.951 |
| 4 | 0.958 | 1.000 | 0.600 | 0.944 |
| 5 | 0.979 | 0.500 | 1.000 | 0.979 |
| 6 | 0.917 | **0.000** | **0.000** | 0.815 |
| 7 | 0.917 | 0.333 | 0.333 | 0.911 |
| 8 | 0.938 | 0.500 | 0.333 | 0.889 |
| 9 | 0.854 | **0.000** | **0.000** | 0.948 |
| **Mean** | **0.917** | 0.563 | **0.404** | 0.907 |

The paper's conclusion — *"performs with a 91.7% accuracy, proving that this
model can be used to predict the occurrence of flood incidents in any region in
Sri Lanka"* — quotes only the accuracy column. **The model misses ~60% of
floods, and in two folds detects nothing at all** while still scoring 0.917 and
0.854 accuracy.

### Two further problems worth noting

1. **10-fold CV on a monthly time series is a random split** — precisely the
   protocol Roberts et al. (2017) and Kapoor & Narayanan (2023) identify as
   invalid for temporally structured data, and precisely what your RQ2 measures.
2. **The weather model it depends on is not usable.** Their rainfall RMSE is
   115.58 mm against a series mean of 111.71 mm — error larger than the mean.
   Table 2 shows January actual 15.8 mm vs. predicted 189.79; July actual 0.0 vs.
   predicted 32.47. The flood classifier is fed these forecasts and still reports
   91.7%.

---

## 4. #21 — Mahaganapathy et al. (2023)

*Flood Prediction Using Machine Learning Based on Metrological and Topographical
Features of Kalu Ganga River Basin, Sri Lanka.* ASURS 2023 abstract book, p. 28.
**Conference abstract only** — read in full (1 p); no full paper exists to check.

- Six catchments: Kalawana, Ayagama, Kuruwita, Pelmadulla, Elapatha, Kahawatta.
- Sources: Dept. of Meteorology, **OpenMeteo API**, DesInventar (DMIS) — note the
  OpenMeteo overlap with your own data pipeline.
- KNN imputation; outlier detection by moving average and percentile; **SMOTE**
  for imbalance; PCA vs. wrapper backward feature selection.
- SVM, logistic regression, naive Bayes, decision tree; stacking, blending,
  bagging, boosting. **Bagging + decision tree reaches 99% F1.**

**Read this cautiously and say so.** A 99% F1 on a rare-event flood problem,
with SMOTE and no stated temporal split, is the signature of resampling applied
before splitting — the textbook leakage case in Kapoor & Narayanan's taxonomy.
But this is an abstract; the split is simply not described. Cite it as *"reports
99% F1; the abstract does not state the splitting protocol"* and do not assert
leakage as fact.

---

## 5. Verified facts you can now cite

### 5.1 National hazard trends — #10, Partheepan et al., *Discover Geoscience* 4:168 (2026)

Mann-Kendall + Sen's slope + Pettitt change-point over 1975–2025. Read pp. 1–18.

| Finding | Value |
|---|---|
| Cyclonic disturbance frequency | 1.3 → 4.1 systems/yr (**+216%**), Sen's slope +0.73/decade, *p* < 0.001 |
| Extreme rainfall intensity | **+12.2 mm/decade** (95% CI 9.4–14.8) |
| Events > 100 mm/24 h | 3.2/yr (1975–90) → 6.8/yr (2010–25), **+112%**, IRR 2.12 (95% CI 1.89–2.38, *p* < 0.001) |
| Major flood events (≥100k affected or ≥USD 100M) | 0.4/yr → 1.8/yr, **4.5-fold**, χ² = 18.6, *p* < 0.001, controlling for population |
| Statistical change-point | **1998** (Pettitt, *p* < 0.05; posterior > 0.85) |
| 50-year national totals | 148 major floods · 114 landslides · 126 cyclonic disturbances · 20.7 M cumulative affected · USD 8,880 M |
| Peak risk month | **November** (22 flood events/yr avg), then October (18), December (16), June (14) |

**Cyclone Ditwah (Nov 2025)** — the event your 2025 image data covers:
620+ deaths, ~1,100,000 displaced, **21 districts**, ~USD 4,100 M. Deadliest
cyclone-related event since the 2004 tsunami. Attributed analysis: climate change
intensified its rainfall by **28–160%**, converting a 1-in-50-year event into a
1-in-30-year one.

Climatic zones used in the paper match your `zone` node feature exactly: wet
(>2,500 mm), intermediate (1,250–2,500), dry (<1,250), arid.

### 5.2 The lead-time numbers — and a correction you must make

An "8–16 h response delays reduced to 2–4 h" figure was circulated during the
literature search for this project. **Do not use it.** It came from a web search
snippet describing a proposed "Multi-Basin Living Laboratory" initiative, not
from any source in this collection; I read all 7 pages of #25 (the DoI briefing
the index attached to that entry) and it contains no response-time figure at all.

Checked 2026-09-08: the figure does **not** appear in `PROJECT_BRIEF.md` or
`PROJECT_PROPOSAL.md`, so nothing in the repository needs correcting — this note
exists to stop the number being adopted later.

The real, citable numbers are #10's Table 8:

| Era | Forecast lead time | Evacuation completion | Gap | Assessment |
|---|---|---|---|---|
| 1975–1990 | 12–24 h | 15–20 h | 6–12 h | Manageable |
| 1990–2002 | 24–48 h | 18–25 h | 12–24 h | Widening |
| 2003–2015 | 72–120 h | 20–30 h | 42–100 h | Critical bottleneck |
| 2015–2025 | 120–168 h | 18–36 h | 84–150 h | Maximum divergence |

⚠️ **Internal inconsistency in that paper:** its abstract and §5.4 both say the
gap widened to "36–54 h (2025)", while Table 8 says 84–150 h. Table 8 is
arithmetically consistent with its own lead-time and evacuation columns; the
36–54 h text figure is not. Quote Table 8, and note the discrepancy if you rely
on the number.

**This finding cuts against a naïve framing of your project, so handle it
deliberately.** Forecast lead time already improved 6-fold; evacuation time did
not move. The paper's conclusion is that the binding constraint is institutional,
not meteorological — they report the correlation between forecast accuracy and
reduced casualties is weak (R² < 0.4). A paper claiming "better forecasts save
lives in Sri Lanka" now has a 2026 citation arguing otherwise.

Your defensible position: the gap is *warning-to-action*, and what breaks that
chain is generic, uncalibrated, high-false-alarm warnings. #10 §5.4 says extended
lead times "create complacency: forecasts issued 5+ days in advance are discounted
as uncertain speculation." That is a **calibration and false-alarm problem** —
exactly what your ECE/Brier reporting and matched-FAR event detection address.
Lead time is not your contribution; trustworthiness of the warning is.

### 5.3 Operational practice — #25, Dept. of Irrigation (15 May 2024)

Briefing by Eng. S.P.C. Sugeeshwara, Director of Irrigation (Hydrology & Disaster
Management). Read in full (7 pp).

- **12 basins under highest flood threat in the SW monsoon**: Kala Oya, Deduru
  Oya, Maha Oya, Attanagalu Oya, **Kelani Ganga**, **Kalu Ganga**, Benthota
  Ganga, **Gin Ganga**, **Nilvala Ganga**, Kirama Oya, Uruboku Oya, Mahaweli
  Ganga Upstream.
- **9 flood-vulnerable districts**: Colombo, Gampaha, Ratnapura, Kalutara,
  Kegalle, Galle, Matara, K'gala, Puttalam.
- Five monitoring systems: manual; semi-automated (irrigation.gov.lk real-time
  WL); Hydro-Meteorological Information System (**not open to the public**);
  Rivernet (rivernet.lk); Wari Soba.
- **Warnings are issued on**: (1) DoM short- and medium-range forecasts, (2) the
  **soil moisture situation of the catchment**, (3) water levels of rivers and
  upstream reservoirs. Basin-wise warnings additionally use observed rainfall,
  observed river water level, **river trend**, and flood model results.

**This validates your feature set directly.** Operational practice keys on
rainfall, soil moisture, river level and *river trend* — which is precisely
`precipitation_sum`, `soil_wet_*`, `discharge`/`discharge_pctl` and
`discharge_rise_1d`/`_3d`. Your 33 channels mirror what the warning service
actually uses. Gin and Kelani both being on the priority list also justifies your
Gin-basin holdout and the Kelani SAR series as operationally relevant choices
rather than arbitrary ones.

### 5.4 Kelani thresholds and a 2008 call for SAR — #23, Gunasekara (2008)

*Flood Hazard Mapping in Lower Reach of Kelani River.* ENGINEER **41(5),
149–154**, Institution of Engineers Sri Lanka. Read in full (6 pp).

- HEC-RAS 1D steady flow + HEC-GeoRAS; Glencourse → Nagalagam Street, ~55 km,
  1,200 km², Colombo / Gampaha / Kegalle districts.
- **Official Kelani flood thresholds at the Nagalagam Street (Colombo) gauge:
  5.0–7.0 ft = minor flood; > 7.0 ft = major flood; > 9.0 ft = dangerous.**
- Gumbel peak discharge at Glencourse (35-year record): 10-yr 3,128 m³/s; 20-yr
  3,990; 50-yr 4,630.
- Inundated area 63 / 77 / 94 km²; buildings affected 9,010 / 14,064 / 17,005 for
  10/20/50-year return periods.
- **"Kelani is one river, where there has been a flood forecasting system.
  Forecasting of water levels in the river is done by means of five upstream
  gauges."** — an 18-year-old precedent for using upstream information, and the
  closest thing in the local literature to your flow-edge premise.
- Limitation, verbatim: *"A microwave image of the study area during the flood
  season definitely would have been very useful in the flood extent verification,
  which unfortunately was not available during the study."* A 2008 request for
  exactly the Sentinel-1 SAR your Branch B provides.

### 5.5 Independent event ground truth — #26a / #26b (IFRC DREF)

Both events fall inside your **test block**, so they are independent corroboration
for test-period episodes.

| | MDRLK019 | MDRLK020 |
|---|---|---|
| Event date | **02-06-2024** (SW monsoon) | **12-10-2024** (inter-monsoon) |
| Glide | FL-2024-000077-LKA | FL-2024-000189-LKA |
| Affected | **253,581** people / 66,906 families, 13 districts | **154,782** people / 39,522 families |
| Worst hit | Ratnapura, Gampaha, Matara, Kalutara, Colombo, Galle, Puttalam | Gampaha 78,281 · Colombo 60,233 · Puttalam 8,902 |
| Rivers named | Nilwala, Gin, Kalu, Attanagalu, Kelani | Attanagalu, Kelani, Gin, Kalu |
| Notes | 400 mm rainfall recorded; Puttalam hit first on 20 May (42,546 affected) | DoI early warning 11 Oct 14:00; Attanagalu Oya at primary flood level by morning 12 Oct; max rainfall 117 mm Eheliyagoda, 114 mm Halvatura, 105.5 mm Hanwella |

---

## 6. What this changes for the paper

**Confirmed novel in the Sri Lankan literature** — nothing I read uses any of:

- a river-network **graph** over gauges (closest is #23's "five upstream gauges",
  used in an operational rule, not a model);
- a **national multi-basin node network** (all studies are one station, one basin,
  or one district);
- **block-temporal + basin-holdout + event-level** protocols side by side;
- **calibration** reporting — no ECE, no Brier, no reliability diagram anywhere;
- **event-level pre-onset detection** with lead time as the reported metric;
- **released data and code**.

**Scale comparison for your table:**

| Study | Spatial units | Period | Resolution | Flood events evaluated |
|---|---|---|---|---|
| #07 Saubhagya 2025 | 1 target + 4 upstream | 2014–2019 | daily | **5** actionable |
| #22 Thilakarathne 2017 | 1 district | 1976–2015 | **monthly** | 51 records |
| #21 Mahaganapathy 2023 | 6 catchments | not stated | not stated | not stated |
| **This project** | **51 nodes / 16 basins** | **2003–2024** | **daily** | **1,469 episodes** |

**Three things to add to the paper:**

1. A related-work paragraph that reports #07's five-event evidence base and #22's
   0.404 recall *as measured facts with page references*, not as criticism. Let
   the arithmetic make the argument.
2. An explicit response to #07's regress-then-threshold position (§2 above).
3. A motivation section built on #10's verified trends, and a lead-time framing
   that concedes #10's finding rather than colliding with it (§5.2).

---

## 7. Provenance — exactly what was read

| # | Source | Pages read | Coverage |
|---|---|---|---|
| 07 | Saubhagya et al. 2025, *Forecasting* 7(2):29 | 1–24 | **complete** |
| 22 | Thilakarathne & Premachandra, SLAAI 2017 | 52–57 of 77 | **complete paper** (pp. 44–49 printed) |
| 25 | Dept. of Irrigation briefing, 15 May 2024 | 1–7 | **complete** |
| 23 | Gunasekara 2008, ENGINEER 41(5) | 1–6 | **complete** |
| 21 | Mahaganapathy et al., ASURS 2023 | 46 of 77 | **complete** (abstract is the whole item) |
| 10 | Partheepan et al. 2026, *Discover Geoscience* 4:168 | 1–18 of 25 | abstract → §6.1; **not read**: §6.2–§9, SL-RISK framework, references |
| 26a | IFRC DREF MDRLK019 | 1–3 of 21 | event description; **not read**: response plan, budget |
| 26b | IFRC DREF MDRLK020 | 1–3 of 25 | event description; **not read**: response plan, budget |
| 26c | IFRC Sri Lanka annual report 2024 | **0 of 22** | not read |

**Not in the collection** (source links only, no PDF): #08 Inzam et al. (Kelani
ML, ICIET 2022), #09 Makumbura et al. (*Results in Engineering* 27:105975, 2025 —
Kelani basin, data-driven vs. process-based), #24 Samarathunga concept paper. #09
is the one worth chasing: it is recent, peer-reviewed, and on your primary basin.

**Non-Sri Lankan papers in the collection remain unread**, except #01
(Kirschstein & Sun) pp. 1–6.
