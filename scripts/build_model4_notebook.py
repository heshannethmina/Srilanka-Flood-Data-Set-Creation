"""Package the exact local sources into the upload-and-run Kaggle notebook.

No clone, mutable remote branch, or unpublished commit is needed at runtime.
Run this script after changing any bundled module.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ['models/model4/__init__.py', 'models/model4/workflow.py',
         'models/model2/__init__.py', 'models/model2/config.py',
         'models/model2/modules.py', 'models/model2/model.py',
         'models/floodlib/__init__.py', 'models/floodlib/schema.py',
         'models/floodlib/traincfg.py', 'models/floodlib/blocks.py']


def cell(kind, source):
    obj = dict(cell_type=kind, metadata={}, id=hashlib.sha256(source.encode()).hexdigest()[:12],
               source=source.splitlines(keepends=True))
    if kind == 'code':
        obj.update(execution_count=None, outputs=[])
    return obj


intro = '''# Model 4 — Hydro-TEM: temporal embedded ensemble

**Run all without editing code.** This notebook embeds the complete implementation,
finds the attached tabular dataset, and downloads the public dataset automatically
if it is absent. Use a Kaggle **GPU** accelerator; enable **Internet** if the dataset
or a dependency is missing. SAR images are not required. No GitHub clone is used.

The previous results favour numerical embeddings and BCE: Model 2 AP 0.8355;
Model 3's graph AP 0.7527; LightGBM AP 0.8496. These are historical context,
not scores for this new experiment. **Model 4 has not yet been measured.**

Model 4 combines train-quantile piecewise-linear numerical embeddings, a strong
current-day branch, gated lag/mean/change features over 14 days, terrain and
seasonality, and four jointly trained neural ensemble members per seed. Separate
dry/wet hazards handle onset versus persistence; cumulative hazards enforce
P(24h) ≤ P(48h) ≤ P(72h). Three independent seeds are averaged.

The implementation is inspired by [numerical embeddings](https://arxiv.org/abs/2203.05556)
and [efficient tabular ensembling](https://arxiv.org/abs/2410.24210), and is a custom
research architecture rather than an official TabM reproduction.
'''

protocol = '''## Predeclared experiment

| Purpose | Dates |
|---|---|
| Training and fitted preprocessing | 2003–2017 |
| Checkpoint / early-stopping selection (next-day AP) | 2018–2019 |
| Probability calibration and decision thresholds | 2020 |
| Final evaluation | 2021–2024 |

Origins whose three-day targets cross any boundary are removed. Future targets
are rebuilt within nodes with a full three-day observation requirement. Historical
context from the preceding period remains allowed. No random split, test-tuned
threshold, test-selected seed, focal loss, or class oversampling is used.

**Comparators:** Model 2 architecture retrained with the same new inputs, split,
classification loss and seed count; LightGBM; discharge-percentile rule; persistence;
and a one-seed current-day-only Model 4 ablation. The control is not a rerun of
the old N5 configuration: preprocessing, loss auxiliaries and selection differ.
Compare the freshly generated rows. The ablation is exploratory because it has
one seed. Per-seed scores and paired 28-day block-bootstrap intervals are saved.

**Interpretation:** these labels are GloFAS discharge-Q98 exceedances, not surveyed
flood inundation. The processed dataset already contains interpolated/backfilled
weather/soil and reanalysis data. This notebook cannot undo that inherited
availability leakage. Earlier models have already exposed the test years, so a
positive result here still requires a new locked future/event holdout before a
confirmatory or operational claim. A Q98 percentile is not itself a fitted
two-year return level. No best-in-research or state-of-the-art result is promised.
'''

setup = '''import importlib.util
import os
from pathlib import Path
import subprocess
import sys

assert Path('/kaggle/working').exists(), 'Run this notebook on Kaggle.'
required = {'torch':'torch', 'numpy':'numpy', 'pandas':'pandas',
            'sklearn':'scikit-learn', 'scipy':'scipy', 'pyarrow':'pyarrow',
            'lightgbm':'lightgbm', 'matplotlib':'matplotlib', 'joblib':'joblib'}
missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', *missing], check=True)
import torch
assert torch.cuda.is_available(), 'Enable a GPU accelerator in Kaggle Settings, then Run All.'
print('GPU:', torch.cuda.get_device_name(0), '| PyTorch:', torch.__version__)

hits = sorted(Path('/kaggle/input').rglob('flood_dataset.parquet'))
hits = [p for p in hits if (p.parent/'nodes.csv').exists()]
if not hits:
    if importlib.util.find_spec('kagglehub') is None:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', 'kagglehub'], check=True)
    import kagglehub
    downloaded = Path(kagglehub.dataset_download('uom230429e/sri-lanka-flood-tabular-graph-2003-2025'))
    hits = [p for p in downloaded.rglob('flood_dataset.parquet') if (p.parent/'nodes.csv').exists()]
if not hits:
    raise FileNotFoundError('The tabular input must contain flood_dataset.parquet and nodes.csv.')
if len(hits)>1:
    import hashlib
    fingerprints = {hashlib.sha256(p.read_bytes()).hexdigest() for p in hits}
    if len(fingerprints)>1:
        raise RuntimeError('Multiple different panel versions are attached; retain one intended tabular dataset.')
DATA_ROOT = hits[0].parent
print('Dataset:', DATA_ROOT)
'''

run = '''import shutil
import traceback
from model4.workflow import Config, run

OUT = Path('/kaggle/working/model4_runs')
OUT.mkdir(exist_ok=True)
# To resume a previous session, attach its extracted model4_runs output folder.
# Only checkpoint files generated by your own previous run should be attached.
if not (OUT/'manifest.json').exists():
    candidates = list(Path('/kaggle/input').rglob('model4_runs/manifest.json'))
    if len(candidates)==1:
        shutil.copytree(candidates[0].parent, OUT, dirs_exist_ok=True)
        print('Restored previous run:', candidates[0].parent)
    elif len(candidates)>1:
        raise RuntimeError('Multiple resumable outputs found; attach one previous model4_runs folder.')

CFG = Config()  # Complete default experiment; no user edits needed.
try:
    run(DATA_ROOT, OUT, CFG)
except Exception:
    (OUT/'failure.txt').write_text(traceback.format_exc())
    raise
finally:
    # Always preserve finished seeds and epoch checkpoints, including on error.
    shutil.copytree(CODE_ROOT, OUT/'source', dirs_exist_ok=True)
    with (OUT/'environment.txt').open('w') as environment:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=environment, check=False)
    archive = shutil.make_archive('/kaggle/working/model4_results', 'zip', OUT.parent, OUT.name)
    print('Download:', archive)
'''

display = '''import json
import pandas as pd
from IPython.display import display, Image, FileLink

status = json.loads((OUT/'status.json').read_text())
display(status)
if status['complete']:
    results = pd.read_csv(OUT/'summary.csv')
    display(results[['model','head','pr_auc','brier','ece','pod','far','csi']])
    display(Image(filename=str(OUT/'evaluation.png')))
    display(json.loads((OUT/'paired_comparisons.json').read_text()))
else:
    print('Partial run. Checkpoints are saved; attach extracted output to a new session and Run All to resume.')
display(FileLink('/kaggle/working/model4_results.zip'))
'''

finish = '''## Outputs and next decisions

Save the notebook version to retain `/kaggle/working/model4_results.zip`.
It includes model/optimizer/RNG checkpoints, exact source and preprocessing,
dataset hashes, training histories, all four heads' raw/calibrated predictions,
per-seed results, validation-fitted thresholds, basin/year diagnostics, PR and
reliability plots, and paired confidence intervals against the new controls.

The 7.5-hour training budget leaves time for final evaluation and packaging.
If training is incomplete, `status.json` names pending work and **no final test
comparison is issued**. Attach the extracted `model4_runs` directory from the
previous output to resume. Code/config/data hashes prevent mixing experiments.
Budget is cooperative, not a guarantee against platform termination.

Read the fresh Model 4 vs Model 2 and LightGBM intervals before claiming a gain.
Intervals condition on trained models and use 28-day time blocks with all nodes
together; they do not establish independence of long events. Thresholds target
FAR ≤ 0.231 on calibration data; achieved test FAR may differ. Onset AP is
reported both over all samples and only currently dry samples. Event lead
windows are one day for 24h/onset heads and three days for the 72h head.
'''


def build():
    sources = {p: (ROOT/p).read_text(encoding='utf-8') for p in FILES}
    bundle = ('# Exact source snapshot: self-contained, including the Model 2 control.\n'
              'import hashlib, json, sys\nfrom pathlib import Path\n'
              "CODE_ROOT = Path('/kaggle/working/model4_code')\n"
              'SOURCES = ' + repr(sources) + '\n'
              'for relative, source in SOURCES.items():\n'
              '    path = CODE_ROOT/relative\n'
              '    path.parent.mkdir(parents=True, exist_ok=True)\n'
              "    path.write_text(source, encoding='utf-8')\n"
              "sys.path.insert(0, str(CODE_ROOT/'models'))\n"
              "print('Embedded modules:', len(SOURCES))\n")
    notebook = dict(cells=[cell('markdown',intro),cell('markdown',protocol),cell('code',setup),
                          cell('code',bundle),cell('code',run),cell('code',display),cell('markdown',finish)],
                    metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
                                  language_info=dict(name='python',version='3.11'),
                                  kaggle=dict(accelerator='gpu',isInternetEnabled=True,isGpuEnabled=True,language='python',sourceType='notebook')),
                    nbformat=4,nbformat_minor=5)
    path=ROOT/'notebooks/tfstgnn_kaggle.ipynb'
    path.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
    print(path)


if __name__=='__main__':
    build()
