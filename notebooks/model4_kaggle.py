# Model 4: paste this whole file into one Kaggle cell.
# Same clone + kaggle_run.py procedure as Models 1–3.

# 1. Get the code, following the existing Kaggle notebook procedure.
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


# 2. Validate the same attached tabular input and GPU as earlier runs.
import torch

assert torch.cuda.is_available(), 'Enable GPU T4 x2 in Kaggle settings.'
print('GPU:', torch.cuda.get_device_name(0), '| PyTorch:', torch.__version__)
hits = [p for p in Path('/kaggle/input').rglob('flood_dataset.parquet')
        if (p.parent/'nodes.csv').exists()]
assert hits, 'Add Input: uom230429e/sri-lanka-flood-tabular-graph-2003-2025'
print('Tabular dataset:', min(hits, key=lambda p: len(str(p))).parent)


# 3. Run Model 4 through the established model runner.
# The stage includes its own matched baselines; no separate baselines stage.
subprocess.run([sys.executable, '-u', str(DEST/'models/kaggle_run.py'),
                '--stage', 'model4', '--time-budget-hours', '8'], check=True)


# 4. Inspect completion and download results, as in earlier notebooks.
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
