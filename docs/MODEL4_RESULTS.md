# Model 4 result comparison — 10 September 2026

**Model 4 completed successfully, but it did not improve on the matched Model 2 control or LightGBM.** The best DNN point estimate in this run is the Model 2 control. Model 4 offers a smaller network, consistent multi-horizon outputs, and improvement over the percentile rule; it does not meet the intended accuracy upgrade.

Source: `C:\Users\hesha\OneDrive\Desktop\runs (1).zip`. Kaggle commit `503bb00719ddcc9ad72107b9c6a6c8fab9e28561`. The archive reports complete with no pending work. All seven archived source hashes were verified; 24 head-level AP/Brier results and 48 threshold contingency tables were recomputed from the predictions. All methods share identical test targets, dates, nodes and calibration rows. Rebuilt target disagreements: zero.

## Matched comparison

All rows below use 74,358 node-day test origins from 2021–2024. Training is 2003–2017, checkpoint selection 2018–2019 and calibration 2020. AP is tie-aware average precision (reported as PR-AUC in the output); higher is better.

| Method | 24h AP | 48h AP | 72h AP | Onset AP |
|---|---:|---:|---:|---:|
| LightGBM | 0.8606 | 0.7949 | 0.7451 | 0.2061 |
| Model 2 control (3 seeds) | 0.8554 | 0.8003 | 0.7605 | 0.2225 |
| Model 4 (3 seeds) | 0.8428 | 0.7787 | 0.7301 | 0.2138 |
| Model 4 current-day only (1 seed) | 0.8277 | 0.7657 | 0.7144 | 0.1821 |
| Discharge-percentile rule | 0.8176 | 0.7536 | 0.7115 | 0.1338 |
| Persistence | 0.5962 | 0.5176 | 0.4588 | 0.0053 |

Model 2 has the highest AP for 48h, 72h and onset. LightGBM has the highest 24h AP. Model 4 trails Model 2 at every head, though its onset AP exceeds LightGBM’s point estimate. Intervals were saved for 24h contrasts only; no significance claim follows from the other rankings.

## Next-day probability and decision quality

Brier and ECE are after the predeclared 2020 calibration (lower is better). F1/recall/FAR use each method’s own calibration-F1 threshold. These are different operating points, not matched test FAR.

| Method | Brier | ECE | F1 | Recall | FAR |
|---|---:|---:|---:|---:|---:|
| LightGBM | 0.0077 | 0.0044 | 0.7888 | 0.7634 | 0.1841 |
| Model 2 control (3 seeds) | 0.0077 | 0.0042 | 0.7856 | 0.7121 | 0.1241 |
| Model 4 (3 seeds) | 0.0091 | 0.0073 | 0.7799 | 0.7204 | 0.1497 |
| Model 4 current-day only (1 seed) | 0.0084 | 0.0048 | 0.7710 | 0.7746 | 0.2326 |

## Paired uncertainty

Difference = Model 4 AP minus comparator AP. The supplied 95% intervals use 500 resamples of 28-day blocks with all nodes retained together. They condition on fitted models, not training-seed uncertainty, and do not remove bias from prior exposure to these test years.

| Comparator | AP difference | 95% interval |
|---|---:|---:|
| Model 2 control (3 seeds) | -0.0126 | [-0.0250, -0.0013] |
| LightGBM | -0.0178 | [-0.0286, -0.0055] |
| Model 4 current-day only (1 seed) | +0.0151 | [+0.0007, +0.0253] |
| Discharge-percentile rule | +0.0252 | [+0.0065, +0.0590] |

Both Model 4–Model 2 and Model 4–LightGBM intervals lie below zero. The evidence here favours both competitors on the primary target. There is no saved Model 2–LightGBM interval, so their small gap is a point-estimate ordering only.

The current-day ablation comparison mixes three seeds with one. At the matched seed 0, Model 4 AP is 0.842861 versus 0.827711 for current-day only (+0.015150), which is promising but still a one-seed architectural contrast. More matched seeds are needed to attribute a reliable gain to history.

## Calibration diagnosis

Only 50 of 18,513 calibration origins are positive for 24h (0.2701%), versus 1,695 of 74,358 test origins (2.2795%): test prevalence is 8.44 times higher. The calibration period contains just 29 onset positives. This is a large temporal distribution difference.

| Method | Raw 24h Brier | Calibrated 24h Brier | Raw ECE | Calibrated ECE |
|---|---:|---:|---:|---:|
| Model 4 (3 seeds) | 0.007454 | 0.009136 | 0.002693 | 0.007321 |
| Model 2 control (3 seeds) | 0.007177 | 0.007721 | 0.002014 | 0.004235 |
| LightGBM | 0.007157 | 0.007724 | 0.002963 | 0.004386 |

For Model 4, calibration moves mean next-day probability from 2.2990% to 1.5474%, below the observed 2.2795%. The observed Brier/ECE deterioration is consistent with a calibrator that transfers poorly from the unusually quiet year; this does not establish prevalence shift as the sole cause. The mapping also pools three flood horizons. Monotone calibration leaves AP essentially unchanged, so calibration does not explain the AP deficit.

These raw scores are diagnostics, not a post-test replacement of the registered calibrated headline. A revised calibration policy must be selected on development periods and evaluated on a new locked holdout.

## Early warning remains weak

Model 4 onset AP is 0.2138, versus Model 2 0.2225 and LightGBM 0.2061. At its calibration-F1 onset threshold, Model 4 detects 141/392 onset-labelled node-days (35.97% recall), with 436 false alarms out of 577 alarms (75.56% FAR). These counts are node-days, not the merged event catalogue.

No nonempty Model 4 onset threshold met FAR ≤ 0.231 on calibration. The saved policy therefore raises no onset alarms. The separate flood-24h constrained policy warns 0/355 eligible event starts; the flood-72h policy warns 2/355. These results apply to those specific stringent thresholds, not every possible operating point. Model 2’s onset policy warns 5/355 events, with test node-day FAR 0.375 despite meeting the cap on calibration. Neither model demonstrates satisfactory low-false-alarm early warning here.

## Efficiency, seeds and subgroups

Model 4 has 160,944 parameters per independently seeded network versus 760,118 for the Model 2 control: 78.8% fewer (4.72× smaller). Both headline ensembles use three seeds; each Model 4 seed contains four shared-weight members. Parameter count is not a measured end-to-end latency comparison.

Raw 24h seed AP: Model 4 = 0.842861, 0.832831, 0.837058 (mean 0.837584, sample SD 0.005036); Model 2 = 0.838889, 0.841095, 0.848273 (mean 0.842753, SD 0.004906). Ensemble AP is not the arithmetic mean of individual AP values; it is computed from averaged probabilities.

Model 4 trails Model 2 in 11 of 16 basin-level AP point estimates. Gains occur in Batticaloa, Gal Oya, Malwathu Oya, Colombo metro and Nilwala; these are descriptive subgroup comparisons without multiplicity-adjusted intervals. By test year, Model 4 vs Model 2 AP is 2021: 0.7558 vs 0.7578; 2022: 0.5696 vs 0.5549; 2023: 0.8881 vs 0.8924; 2024: 0.7884 vs 0.8256. The largest yearly deficit is in 2024.

## Relation to Models 1–3

| Historical run | Reported 24h AP |
|---|---:|
| Model 1, M5_bce (5 seeds) | 0.7386 |
| Model 2, N5_bce (5 seeds; documented historical result) | 0.8355 |
| Model 3, P3 (5 seeds) | 0.7527 |

Model 4’s 0.8428 is numerically above those historical rows, but preprocessing, sample boundaries, calibration/selection splits, auxiliary objectives and seed counts differ. It is not a controlled superiority claim. The newly retrained Model 2 control at 0.8554 is the relevant neural comparator. Historical sources: [RESULTS.md](RESULTS.md) and [models/README.md](../models/README.md).

## Research decision

Retain Model 2 as the leading DNN reference and LightGBM as the 24h accuracy benchmark. Retain Model 4 as an efficiency/consistency ablation, not the best-performing upgrade. The next justified development work is calibration and onset performance, using chronological development folds with more flood events. Do not select a new method by repeatedly optimising the already exposed 2021–2024 test scores.

The dataset remains a reanalysis discharge-exceedance proxy with inherited interpolated/backfilled weather/soil. Strong AP here does not establish inundation prediction or real-time availability.

Machine-readable evidence: [metrics](../runs/model4_20260910/metrics.json), [paired intervals](../runs/model4_20260910/paired_comparisons.json), [verification audit](../runs/model4_20260910/comparison_audit.json). The original archive was not modified.
