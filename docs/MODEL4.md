# Model 4: Hydro-TEM

Use [the Kaggle notebook](../notebooks/tfstgnn_kaggle.ipynb), following the same
procedure as Models 1–3. Push the updated repository, attach
`uom230429e/sri-lanka-flood-tabular-graph-2003-2025`, enable GPU T4 x2 and Internet,
then Save & Run All. The notebook clones the repository, validates the input,
and calls the existing runner with `--stage model4`. No imagery dataset or
notebook code edits are needed.

The equivalent two commands in a fresh Kaggle session are:

```python
!git clone -q https://github.com/heshannethmina/Srilanka-Flood-Data-Set-Creation /kaggle/working/repo
!python -u /kaggle/working/repo/models/kaggle_run.py --stage model4
```

The notebook refreshes its disposable clone when rerun, leaving saved outputs
in `/kaggle/working/runs/model4`. Download `runs.zip`, as in earlier experiments.
The stage includes its own matched baselines and is run separately from the
older `--stage all` schedule. Defaults are 80 epochs, three seeds and a batch
of 1024 node windows; legacy stage defaults remain unchanged. `--epochs`,
`--seeds`, `--batch-size` and `--time-budget-hours` are supported.

[`notebooks/model4_kaggle.py`](../notebooks/model4_kaggle.py) contains the same
short launcher if a single code cell is preferred. The local packaging helper
also follows this procedure when pasted into Kaggle; it no longer raises the
`__file__` guard or downloads and executes a notebook as JSON.

## Why this candidate

Recorded Model 2/N5 BCE AP is 0.8355, compared with LightGBM 0.8496. Model 3/P3
is 0.7527; adding graph propagation hurt the strong numerical encoder. The
existing SAR branch did not improve the headline result. The next justified
experiment is stronger tabular modelling, rather than more graph/CNN capacity.

Hydro-TEM is a custom temporal neural ensemble. Its numerical embedding follows
the piecewise-linear approach from [Gorishniy et al. (2022)](https://arxiv.org/abs/2203.05556).
Its shared-weight ensemble is inspired by [TabM (2025)](https://arxiv.org/abs/2410.24210).
It is not the authors' TabM implementation, a proven new scientific contribution,
or a demonstrated improvement before training.

- Train-only quantile knots embed 34 numeric channels: the existing 33 features
  plus log discharge relative to each node's training Q98 threshold.
- Missingness indicators, current flood state, annual sine/cosine, and static terrain accompany the
  current-day branch. No node-ID embedding is used.
- A learned gate adds yesterday, trailing four-day mean, fourteen-day mean,
  and fourteen-day change. These summaries encode observed history, with no
  future weather forecasts assumed available.
- Three residual MLP layers use shared matrices and member-specific input/output
  modulation. Four members receive separate BCE supervision; three independently
  seeded networks are averaged equally. No seed is selected using test scores.
- Separate wet/dry first-day hazards model flood persistence and onset.
  Conditional day-two/day-three hazards produce cumulative probabilities, enforcing
  `p24 <= p48 <= p72`. Onset is exactly `p24 * (1 - current_flood_state)`.
- Unweighted BCE, AdamW, warmup/cosine learning rate, gradient clipping at 2,
  mixed precision on CUDA, and early stopping replace focal/confidence weighting.

## Evaluation contract

Training is 2003–2017; checkpoint selection is 2018–2019; calibration is 2020;
test is 2021–2024. Origins with any of their three label days in another period
are purged. Missing days are reindexed before target shifts, and all three
future observations must exist. Training medians, normalisation and quantile
knots are saved. The node Q98 definition is checked against archived thresholds
when `thr_high` is present. Rebuilt-target disagreements are recorded in the audit.

The notebook retrains Model 2's existing architecture using the same new input
contract, split, BCE head weights, optimiser and three seeds. This is an
architecture control, not a reproduction of N5: regression auxiliaries are not
trained, model selection changes, and the input count increases. Its parameter
count therefore differs from the historical 711k network. Do not compare new
and historical headline scores as an isolated architecture contrast.

Matched LightGBM has current features, terrain and causal discharge/weather
lags; its tree count is selected on 2018–2019. Persistent state and percentile
rules are included. A current-day-only Model 4 ablation uses one seed and is
exploratory; the multi-seed control and tree comparisons are the main contrasts.

Average precision uses scikit-learn's tie-aware definition. A positive-slope
logit mapping fitted on 2020 calibrates all three flood horizons jointly, retaining
their ordering. Thresholds maximise calibration F1 or recall subject to a
calibration FAR of at most 0.231. If no alarm threshold meets that FAR, a
no-alarm threshold is recorded explicitly. Test FAR is measured, not forced.

Outputs include all four heads, raw/calibrated scores, seed results, onset AP
restricted to currently dry nodes, by-basin/year AP, and event detection using
the true full-panel episode start and a fully observed warning window. The
24-hour and onset heads use a one-day warning window; the 72-hour head uses
three days. These event metrics should not be directly compared with the old
seven-day diagnostic.

Paired AP bootstrap intervals resample 28-day time blocks with all nodes kept
together, preserving contemporaneous spatial dependence. They condition on fitted
models; a single block length and three seeds do not exhaust uncertainty.

## Reproducibility and limits

The notebook executes Model 4 from the cloned repository through the shared
runner. The commit, exact model sources and environment are saved alongside
the outputs. Regenerate launcher cells after edits with
`python scripts/build_model4_notebook.py`.
Source/config/data fingerprints reject incompatible resumed runs. Epoch
checkpoints retain model, optimiser, scaler and RNG state; each completed seed
also saves prediction arrays. Training can resume from attached extracted
`runs/model4` output; the previous `model4_runs` layout is accepted too.
Attach only your own checkpoint artifacts.

Mixed-precision gradient overflow is handled before clipping: the loss scaler
skips the invalid update, reduces its scale, and retries the same batch. Three
unsuccessful attempts switch that seed to full precision. Non-finite full-precision
gradients still fail explicitly. Retries, loss scale and precision mode are saved
in epoch histories; the mode is restored with checkpoints. The Model 2 control
computes its sigmoid in float32 before probability-space BCE.

The interrupted pre-fix run from commit `34d5b3d` can resume automatically from
its last saved epoch. This exception accepts only its exact old workflow hash,
unchanged dataset/configuration/other model sources, and unfinished outputs
without completed prediction caches. The previous manifest and available source
snapshot are preserved, and the migration is recorded in the new manifest.
Other incompatible runs still fail the fingerprint check. Keep the existing
`runs/model4` directory in the same Kaggle session, or attach the extracted
previous output in a new session, then rerun the usual notebook.

The 7.5-hour cooperative training budget reserves approximately 30 minutes
before an eight-hour total for baseline fitting, evaluation and packaging; it
cannot guarantee a finish on every GPU/session. Partial work is explicitly
labelled and no final comparison is emitted until the neural schedule completes.
Failures raise normally, with `failure.txt` and the available output ZIP saved
by the runner's `finally` block.

The inherited processed dataset uses interpolation/backfill in weather/soil
features. Those values cannot be restored to their original missingness from
this parquet. Daily reanalysis also does not establish real-time availability.
Purging labels fixes cross-period targets, not these upstream limitations.
The target is a discharge-quantile exceedance proxy, not measured inundation;
Q98 is not automatically a two-year return-period threshold. Since earlier
experiments already exposed 2021–2024 test scores, this is an exploratory
follow-up benchmark. Confirmatory claims need a new locked holdout, ideally
independent observed flood labels and an explicit data-availability audit.

Local tests use synthetic data for alignment, future-data isolation, probability
constraints, gradients, tie-aware AP, checkpoint reload, notebook packaging, and
an end-to-end CPU experiment. They establish implementation behaviour, not
Kaggle GPU runtime or predictive performance on the full dataset.
