"""Maintain the thin Kaggle launcher; models live in the cloned repository.

Local: regenerate the notebook and equivalent single-cell Python launcher.
Kaggle: pasting this script runs the same clone-and-stage procedure directly.
"""
import hashlib
import json
from pathlib import Path

INTRO = '''# Model 4 — Hydro-TEM: Kaggle session

This follows the same procedure as Models 1–3: clone the repository, check the
attached dataset and GPU, then run `models/kaggle_run.py --stage model4`.
Push the updated repository before running so the clone includes the new stage.

**Kaggle settings**

1. Add Input: `uom230429e/sri-lanka-flood-tabular-graph-2003-2025`.
2. Accelerator: GPU T4 x2. The experiment uses one GPU.
3. Internet: On, for the repository clone.

Then **Save Version → Save & Run All**. No code edits or imagery dataset needed.

The stage trains Model 4, a matched three-seed Model 2 control, a one-seed
current-day ablation, LightGBM and simple baselines. Outputs and resumable
checkpoints go to `/kaggle/working/runs/model4`; download `runs.zip` afterward.
Model 4's performance is unmeasured until you complete this experiment.
'''

CLONE = '''# 1. Get the code, following the existing Kaggle notebook procedure.
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = 'https://github.com/heshannethmina/Srilanka-Flood-Data-Set-Creation'
DEST = Path('/kaggle/working/repo')
assert Path('/kaggle/working').is_dir(), 'Run this on Kaggle.'
# This is the notebook's disposable clone; results live separately in runs/.
assert DEST.resolve() == Path('/kaggle/working/repo')
if DEST.exists():
    shutil.rmtree(DEST)
subprocess.run(['git', 'clone', '-q', REPO, str(DEST)], check=True)
head = subprocess.check_output(['git', '-C', str(DEST), 'rev-parse', 'HEAD'], text=True).strip()
print('Repository commit:', head)
runner = DEST/'models/kaggle_run.py'
assert (DEST/'models/model4/workflow.py').exists() and 'def stage_model4(' in runner.read_text(), (
    'This GitHub checkout has no Model 4 runner stage. Push models/kaggle_run.py '
    'and models/model4, then rerun this cell.'
)
'''

CHECK = '''# 2. Validate the same attached tabular input and GPU as earlier runs.
import torch

assert torch.cuda.is_available(), 'Enable GPU T4 x2 in Kaggle settings.'
print('GPU:', torch.cuda.get_device_name(0), '| PyTorch:', torch.__version__)
hits = [p for p in Path('/kaggle/input').rglob('flood_dataset.parquet')
        if (p.parent/'nodes.csv').exists()]
assert hits, 'Add Input: uom230429e/sri-lanka-flood-tabular-graph-2003-2025'
print('Tabular dataset:', min(hits, key=lambda p: len(str(p))).parent)
'''

TRAIN = '''# 3. Run Model 4 through the established model runner.
# The stage includes its own matched baselines; no separate baselines stage.
subprocess.run([sys.executable, '-u', str(DEST/'models/kaggle_run.py'),
                '--stage', 'model4', '--time-budget-hours', '8'], check=True)
'''

DISPLAY = '''# 4. Inspect completion and download results, as in earlier notebooks.
import pandas as pd
from IPython.display import display, Image, FileLink

OUT = Path('/kaggle/working/runs/model4')
status = json.loads((OUT/'status.json').read_text())
display(status)
if status['complete']:
    display(pd.read_csv(OUT/'summary.csv'))
    display(Image(filename=str(OUT/'evaluation.png')))
    display(json.loads((OUT/'paired_comparisons.json').read_text()))
else:
    print('Partial run: checkpoints are saved. Attach the prior output and rerun to resume.')
display(FileLink('/kaggle/working/runs.zip'))
'''

PROTOCOL = '''## Evaluation and saved output

Train: 2003–2017. Checkpoint selection: 2018–2019. Calibration and thresholds:
2020. Test: 2021–2024. Three-day target windows crossing these boundaries are
removed. The Model 2 control is retrained with the new protocol and inputs;
historical Model 1–3 scores are context, not a matched comparison.

Model 4 uses numerical embeddings, gated observed-history summaries, and neural
ensembling with BCE. Its cumulative hazards enforce `p24 <= p48 <= p72`, and
onset is zero for nodes currently flooding. See `docs/MODEL4.md` in the repository
for the architecture, sources and full evaluation contract.

The labels are reanalysis discharge-Q98 exceedances, not measured inundation.
Processed weather/soil already contains interpolation/backfill that this model
cannot undo. Prior experiments have exposed the test years: confirmatory claims
still need a new locked holdout and an audit of operational data availability.

The eight-hour budget assigns 7.5 hours to neural training and leaves time for
baselines and packaging. Runtime depends on the GPU. Incomplete runs are labelled
partial rather than reported as final comparisons. Attach the previous extracted
`runs/model4` output to resume; the older `model4_runs` layout is also accepted.
Code, config and data hashes prevent mixing incompatible checkpoints.

Download `runs.zip` from the notebook output. It includes weights, optimiser/RNG
checkpoints, preprocessing, source, environment, commit, raw/calibrated predictions,
per-seed results, calibration-fitted thresholds, basin/year diagnostics, plots and
paired 28-day bootstrap intervals. Test results do not guarantee improvement.
'''

CODE_CELLS = [CLONE, CHECK, TRAIN, DISPLAY]


def cell(kind, source):
    result = dict(cell_type=kind, metadata={}, source=source.splitlines(keepends=True),
                  id=hashlib.sha256(source.encode()).hexdigest()[:12])
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def build(root):
    notebook = dict(cells=[cell('markdown', INTRO), *[cell('code', s) for s in CODE_CELLS],
                          cell('markdown', PROTOCOL)],
                    metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                                  language_info=dict(name='python', version='3.11'),
                                  kaggle=dict(accelerator='gpu', isInternetEnabled=True, isGpuEnabled=True,
                                              language='python', sourceType='notebook')),
                    nbformat=4, nbformat_minor=5)
    path = root/'notebooks/tfstgnn_kaggle.ipynb'
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+'\n', encoding='utf-8')
    print(path)
    script = root/'notebooks/model4_kaggle.py'
    script.write_text('# Model 4: paste this whole file into one Kaggle cell.\n'
                      '# Same clone + kaggle_run.py procedure as Models 1–3.\n\n'
                      + '\n\n'.join(CODE_CELLS), encoding='utf-8')
    print(script)


if __name__ == '__main__':
    if '__file__' in globals():
        build(Path(__file__).resolve().parents[1])
    else:
        # Compatibility for users opening this file instead of the launcher.
        scope = {'__name__': '__main__'}
        for i, source in enumerate(CODE_CELLS):
            exec(compile(source, f'model4_launcher_cell_{i}', 'exec'), scope)
