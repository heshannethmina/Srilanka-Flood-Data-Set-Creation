# Model 4 v2: candidate upgrade and dataset audit

The completed Model 4 scored 24h AP 0.8428, versus the matched Model 2 control
0.8554 and LightGBM 0.8606. See [the verified comparison](MODEL4_RESULTS.md).
This revision is an unmeasured candidate; better accuracy is not guaranteed.

## What changes

The original current-day piecewise numerical embeddings and four historical
summaries remain. A second branch processes every day in the 14-day history
using the existing Model 2 periodic embeddings, three temporal transformer
layers, feature attention and static FiLM. A gate fuses the sequence with the
current/summary state. Separate wet/dry hazards, cumulative horizon consistency,
four shared-weight ensemble members, BCE and AMP overflow recovery remain.

The hypothesis is that ordered daily histories recover information lost by
summary statistics. This is informed by the stronger matched Model 2 results,
not a claim of a novel attention method. Numerical representations follow
[Gorishniy et al.](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html).

Three equally seeded models are retrained: v2 (`model4` in result tables),
the original summary encoder (`model4_summary`), and `model2_control`. Each
uses three independent seeds. LightGBM and persistence/percentile references
are included. The old one-seed current-only ablation is replaced by the
three-seed summary control. All see identical inputs, dates, targets, losses,
batch size, optimizer schedule and calibration policy. Architecture-specific
parameter counts are saved. The larger encoder uses batch size 512 by default.

## Run on Kaggle

Push these repository changes, import `notebooks/tfstgnn_kaggle.ipynb`, enable
GPU T4 x2 and Internet, attach
`uom230429e/sri-lanka-flood-tabular-graph-2003-2025`, and Run All. The same
clone-and-run procedure is retained. The notebook now calls
`models/kaggle_run.py --stage model4_v2`.

Optionally attach `uom230429e/flood-data-set` for the image audit. **This
experiment trains on tabular data, not image pixels or image-derived labels.**
It does not yet train a new CNN or claim multimodal improvement.

Outputs go to `/kaggle/working/runs/model4_v2`, separately from the old run.
Download `runs.zip`. Resume only v2 artifacts made with matching code/config/data;
v1 checkpoints cannot initialize/resume this different architecture. The
original architecture is still available through `--stage model4`.

The parquet reader selects required columns and converts numeric inputs to
float32. The loader slices windows per batch instead of duplicating the full
dataset into all overlapping windows. Image auditing reads the small index CSV,
not the large pixel archive; PNG/NPY arrays are never loaded in this experiment.

## What the image audit records

`image_audit.json` lists actual frame counts, dates, median/90th-percentile/
maximum acquisition gaps by site, coverage by training/evaluation period,
duplicate site/date keys and image-label disagreements with the tabular task.
Multiple attached indices are reported separately, not silently combined.

Alignment uses the latest *past* acquisition, not nearest-date matching. For
date-only indices a one-day availability lag is assumed; this is conservative
for acquisition time, but does not verify data-product publication latency.
Images older than 14 days are unavailable in this audit's coverage calculation.
Acquisition age is measured in actual calendar days, even on a sparse panel.
The old event-selected index must use `actual_date`, never `target_date`.

If a site has images on Jan 1, Jan 5 and Jan 10, the audit measures gaps of
4 and 5 days. It does not create five independent image observations by
repeating the Jan 1 frame. Future pixel experiments should retain acquisition
age, presence/quality masks and orbit information; a CNN should process unique
frames and reuse their representation when available. The existing image
labels must not be counted as an independent supervised flood dataset.

## Evaluation and limitations

Train 2003–2017, checkpoint selection 2018–2019, calibration 2020, test
2021–2024. All future target windows crossing boundaries are excluded. The
calibration and threshold policy is intentionally unchanged to isolate the
architecture change. Raw AP and raw Brier are saved alongside calibrated
metrics. The earlier calibration deterioration is unresolved; 2020 had only
50 primary positives. Calibration methods should next be compared on
chronological development folds, not chosen using the exposed test scores.

The test years have already influenced this candidate's design. The rerun is
exploratory, even with paired block-bootstrap intervals; confirmatory claims
need a new independently locked evaluation. A larger model, a larger parquet,
or more pixel bytes does not establish stronger independent evidence.

Local synthetic checks cover sequence order, gradients, monotone horizons,
parquet projection, irregular acquisition gaps, no-future-image alignment,
label consistency auditing, full workflow output, resume and CLI defaults.
They do not establish Kaggle GPU speed or accuracy on the real datasets.
