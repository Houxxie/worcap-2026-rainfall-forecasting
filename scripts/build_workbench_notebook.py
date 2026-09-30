"""Package the common user entry point without changing archived experiment code."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from build_notebooks import code, markdown
from build_spatial_notebooks import build
from research.workbench.runner import sources
from research.workflow.config import BLOCKS, DEFAULT_EVIDENCE
from research.workflow.runner import REPORT_FILES


def main():
    names=list(sources())
    evidence=['signature.json','global.csv']+[f'{b}/{n}' for b in BLOCKS for n in ('complete.json',)+REPORT_FILES]
    names += [(DEFAULT_EVIDENCE/n).relative_to(ROOT).as_posix() for n in evidence]
    build('research/workbench/rainfall_workbench.ipynb','research/workbench/README.md',names,[
        markdown('## 1. Choose the task\nStart with review to read an existing result without retraining. Set train only when you intend to run all seven blocks.'),
        code('''from research.workflow.runner import write_json
TASK = "review"  # review, evaluate, train, prepare
EXPERIMENT = "sst_unet_blend_v1"  # evaluate: hybrid_unet_v1, fixed_blend_v1, sst_unet_blend_v1
MANUAL_INPUTS = {}  # Example: {"results": "/kaggle/input/my-reports/sst_unet_blend_reports.zip"}
TARGET_MONTH = "2026-10"  # Used only by prepare; the registry enforces its dates/deadlines
MAP_MONTH = None  # Optional retrospective map month, e.g. "2022-12"; requires full maps
DEVICE = "cuda"  # Used only by train
SETTINGS = dict(schema="rainfall_workbench_v1", name="rainfall-"+TASK, task=TASK,
    inputs=MANUAL_INPUTS, search_roots=["/kaggle/input", "/kaggle/working"],
    output_root=str(BASE/"rainfall_runs"))
if TASK in {"evaluate", "train"}:
    SETTINGS["experiment"] = "hybrid_unet_v1" if TASK == "train" else EXPERIMENT
if TASK == "train": SETTINGS["device"] = DEVICE
if TASK == "prepare": SETTINGS["target"] = TARGET_MONTH
if MAP_MONTH is not None: SETTINGS["map_month"] = MAP_MONTH
CONFIG_FILE = RUN_ROOT/"settings.json"
write_json(CONFIG_FILE, SETTINGS)
print(json.dumps(SETTINGS, indent=2))
'''),
        markdown('## 2. Check inputs before execution\nThe table explains all missing or ambiguous inputs. This cell does not train or contact data providers.'),
        code('''import pandas as pd
from IPython.display import display
from research.workbench.config import load
from research.workbench.runner import check
CHECK = check(load(CONFIG_FILE))
display(pd.DataFrame(CHECK["checks"]))
print(CHECK["note"])
if not CHECK["ready"]:
    print("\\n".join(CHECK["errors"]))
assert CHECK["ready"], "Correct the input paths above, then rerun configuration and checks."
if TASK == "prepare":
    display(pd.DataFrame(CHECK["audit"]["sources"]))
    print(CHECK["audit"]["next_action"])
'''),
        markdown('## 3. Execute or reuse completed work\nA matching verified result is reused. Failures preserve their outputs; use the recovery cell below rather than immediately training again.'),
        code('''from research.workbench.runner import run, verify
RUN = run(CONFIG_FILE)
print(verify(RUN))
'''),
        markdown('## 4. Read and download\nSave the complete output as well as the small report ZIP if you need maps/models later.'),
        code('''from IPython.display import HTML, FileLink
from research.workbench.runner import export
display(HTML((RUN/"report.html").read_text(encoding="utf-8")))
display(FileLink(str((RUN/"report.html").relative_to(BASE))))
ZIP = RUN.parent/(RUN.name+"-reports.zip")
if not ZIP.exists(): ZIP = export(RUN)
display(FileLink(str(ZIP.relative_to(BASE))))
'''),
        markdown('## 5. Inspect previous or interrupted runs (optional)\nThis cell only inspects. Recovery may supply new settings that reuse completed work without training.'),
        code('''from research.workbench.runner import status, recover
display(pd.DataFrame(status(BASE/"rainfall_runs")))
RECOVER_RUN = None  # Set to the preserved run directory shown in an error
if RECOVER_RUN:
    print(json.dumps(recover(RECOVER_RUN), indent=2))
''')])
    p=ROOT/'research/workbench/rainfall_workbench.ipynb'
    nb=json.loads(p.read_text(encoding='utf-8'))
    reset='''# A previous notebook can leave a different research package in memory.
import importlib
for name in list(sys.modules):
    if name == "research" or name.startswith("research."):
        del sys.modules[name]
importlib.invalidate_caches()
from research.common.inputs import ROOT as ACTIVE_CODE_ROOT
assert ACTIVE_CODE_ROOT.resolve() == CODE_ROOT.resolve(), "Unexpected code bundle in memory."
'''
    nb['cells'][1]['source'] += reset.splitlines(keepends=True)
    p.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':main()
