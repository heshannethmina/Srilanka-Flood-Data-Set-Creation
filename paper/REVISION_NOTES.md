# Manuscript revision — 8 September 2026

The manuscript is now a standard IEEE conference-format DNN application study,
titled **Terrain-Conditioned Deep Neural Networks for Next-Day High-Flow
Prediction in Sri Lanka**. The previous source, PDF, figure script and figures
are preserved under `archive/original-2026-09-08/`.

## Editorial and structural changes

- Replaced the rhetorical evaluation-critique framing with research questions
  about numerical representation, attention, static conditioning, optimisation
  and radar fusion.
- Rewrote the abstract, introduction, related work, method, results, discussion
  and conclusion in restrained academic language.
- Kept conventional baselines as historical context; they no longer dominate
  the contribution or appear as matched competitors in the DNN figures.
- Replaced unsupported disaster statistics and dismissive comparisons with
  prior Sri Lankan studies with a focused, respectful literature discussion.
- Added practical issue-time input availability, independent label validation,
  missed-event evidence, ensemble cost and an explicit deployment research path.
- Retained editable author fields because author and affiliation details have
  not been supplied. No journal/conference has been selected.

## Technical corrections verified against the implementation

| Previous issue | Correction and evidence |
|---|---|
| Main model described as if graph structure participates | `models/model2/model.py` is graph-free; no spatial message passing occurs. Graph comparisons are historical and confounded. |
| Sequential temporal-to-feature-attention drawing | The feature branch consumes shared embeddings in parallel; the two representations merge by addition and LayerNorm. |
| Ensemble averages logits | `floodlib/engine.py` averages probabilities, converts to logits, then calibrates. |
| FiLM begins as an exact identity | The affine part is neutral, but the block still applies LayerNorm. |
| SAR gate starts completely closed | Bias −3 gives sigmoid ≈0.0474. Absence masking, rather than this bias, makes an unavailable image contribute zero. |
| Pure focal-loss ablation | `focal_conf` changes alpha weighting, the focusing factor and label-confidence weighting together. |
| Pretrained CNN claimed as established | `docs/RESULTS.md` says pretraining was not completed. The optional pretrained path is not evidence that the recorded run used it. |
| 33 channels include eight discharge channels | Schema has nine discharge channels. Static dimensions are two continuous values plus seven indicators. |
| Supervision described as observed gauges | Labels are derived from reanalysis discharge; node locations are not a 51-gauge observation dataset. |
| 98th percentile treated as a return-period threshold | A daily percentile does not establish an annual flood return period or an official warning stage. |
| 409,350 valid rows / 1,469 episodes attributed to 2003–2024 | Validation counts extend into 2025. Catalogue filtering gives 1,464 episodes with onset through 2024; exact model sample counts remain unavailable. |
| Gin holdout stated as three nodes | The supplied inventory has four Gin nodes. |
| Training-only preprocessing claimed for every split | Dynamic normalisation uses the temporal train mask; static continuous inputs use all inventory nodes. Strict spatial transfer needs a separate audit. |
| Reliability diagram requested as equal-count bins | Current ECE/reliability code uses 15 equal-width bins. No curve can be recovered from aggregate ECE. |
| Validation seed range treated as test significance band | These refer to different datasets and metrics. Removed the comparison; no confidence intervals were invented. |
| Random split proves loss of forecasting capability | Held-out-row masking changes alarm opportunities and apparent onset in the event routine. The reported difference is a diagnostic association, not an isolated causal effect. |
| Six-day lead inferred from a next-day classifier | Seven-day pre-onset matching is a retrospective association metric, not proof of a calibrated multi-day forecast. |
| Test touched once / fully leakage-controlled claims | Removed. Architecture development uses the same test record; future targets can cross split and truncation boundaries. |
| SAR is conclusively useless | N6-gated improves over the focal ensemble but differs from the strongest BCE reference in objective and other settings. Evidence is mixed, without a clean radar control. |
| Best neural score presented as direct win/loss against trees | Older references predate the truncation policy. A matched rerun is required. |

## Figures

1. Actual study coordinates, inventory zones and directed connectivity, with a
   Natural Earth country outline.
2. Annual proxy-episode counts from the supplied catalogue, explicitly filtered
   to 2003–2024 and marked by temporal split.
3. Corrected full-width architecture showing parallel attention branches and
   optional radar fusion before the prediction MLP.
4. DNN ablation dot plot with exact recorded AP values and objective markers.
5. Parameter-count versus AP comparison, including ensemble interpretation and
   a clearly labelled logarithmic parameter axis.
6. Selected DNN detection/FAR operating points, without joining unrelated
   configurations into an implied operating curve.

The old seed-noise graph, causal leakage diagram and empty reliability box are
removed from the active paper. No raw observations or predictions were invented.

## Work still needed for scientific submission

These tasks require experimental artifacts or new experiments; editing cannot
complete them. The manuscript now discloses the limitations.

1. Recover the exact model-2 panel and run manifests. Recompute eligible
   train/validation/test sample and episode counts after truncation and after
   removing target horizons that cross boundaries.
2. Retain complete prediction arrays, and recompute AP with grouped tied scores
   for all compared models. Check that ranking changes do not alter conclusions.
3. Re-run baselines and neural comparisons on identical dates and masks. Keep
   the encoder, seeds, objective, calibration and panel fixed for graph ablations.
4. Report horizon-aligned event detection with full pre-onset coverage, frozen
   validation thresholds, realised test FAR and a false-warning-episode metric.
5. Produce reliability curves, PR curves and paired uncertainty intervals from
   predictions. Preserve time/storm dependence rather than resampling node-days
   as independent observations.
6. Complete a matched BCE radar control with documented pretraining and common
   coverage. Do not infer the result of an unrun STG-Former experiment.
7. Obtain independent gauge/inundation data for correct, missed and false-alarm
   event hydrographs; replay the actual input publication times.
8. Add author details and adapt the page count/reference style to the chosen
   venue. Fill the experimental evidence gaps before treating this as camera-ready.

No model code or historical result ledger was altered. New training was not
performed, and no performance improvement is claimed from this editorial work.
