"""Package a saved-map-only blend comparison for Kaggle CPU execution."""
from pathlib import Path
from build_notebooks import markdown, code
from build_spatial_notebooks import build

ROOT = Path(__file__).resolve().parents[1]

RESET_IMPORTS = '''# A reused Kaggle kernel may still hold research modules from another notebook.
import importlib, sys
from pathlib import Path
CODE_ROOT = (Path('/kaggle/working') if Path('/kaggle/working').is_dir()
             else Path.cwd()/'outputs')/'sst_unet_fixed_blend'/'code'
for relative in [
    'research/sst_extension/evidence/completed_20260930/frozen.json',
    'research/temporal_extension/evidence/kaggle_354026629/frozen.json',
]:
    assert (CODE_ROOT/relative).is_file(), 'Run the source-bundle cell first: ' + relative
for name in list(sys.modules):
    if name == 'research' or name.startswith('research.'):
        del sys.modules[name]
sys.path.insert(0, str(CODE_ROOT))
importlib.invalidate_caches()
from research.common.inputs import ROOT as ACTIVE_CODE_ROOT
assert ACTIVE_CODE_ROOT.resolve() == CODE_ROOT.resolve(), 'Unexpected source directory.'
print('Active comparison code:', ACTIVE_CODE_ROOT)
'''


def main():
    names = ['research/common/inputs.py', 'research/diagnostics/analyze_errors.py', 'research/fixed_blend/evaluate.py',
             'research/sst_unet_blend/protocol.json']
    for folder in ['research/sst_extension/evidence/completed_20260930',
                   'research/temporal_extension/evidence/kaggle_354026629']:
        names += [folder + '/frozen.json', folder + '/evaluation/global.csv']
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT / 'research/sst_unet_blend').glob('*.py')]
    build('research/sst_unet_blend/sst_unet_fixed_blend.ipynb', 'research/sst_unet_blend/README.md', names, [
        markdown('## 1. Synthetic checks\nNo neural or tree model is fitted, including in these checks.'),
        code(RESET_IMPORTS + '''import unittest
suite = unittest.defaultTestLoader.loadTestsFromName("research.sst_unet_blend.test_blend")
assert unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful(), "Synthetic checks failed."
'''),
        markdown('## 2. Locate the exact saved maps\nAttach the original observations, the complete SST output and the complete U-Net extension output. Reports ZIPs are insufficient. Keep `None` for automatic discovery, or supply complete file paths.'),
        code('''from research.sst_unet_blend.experiment import protocol, locate_files, execute, export_reports
OBSERVATIONS = None   # Full path to treino_tp.nc, if needed
SST_PREDICTIONS = None   # Full path to the SST models/predictions.nc, if needed
UNET_PREDICTIONS = None  # Full path to the U-Net extension predictions.nc, if needed
INPUTS = locate_files(protocol(), dict(observations=OBSERVATIONS,
    sst_predictions=SST_PREDICTIONS, unet_predictions=UNET_PREDICTIONS))
print("All required maps are present. No fitting will be performed.")
'''),
        markdown('## 3. Evaluate the fixed 75/25 candidate\nOne previously consulted block, 2021–2022. Primary comparison: the existing hybrid/U-Net blend. Secondary comparison: the SST hybrid alone.'),
        code('''import pandas as pd
from IPython.display import display, HTML, FileLink
OUTPUT = RUN_ROOT / "results"
RESULT = execute(OUTPUT, **INPUTS)
assert RESULT["complete_comparison"] and RESULT["archived_scores_reproduced"]
for name in ["global_metrics", "years", "regions"]:
    print(name)
    display(pd.read_csv(OUTPUT / (name + ".csv")))
display(HTML((OUTPUT / "report.html").read_text(encoding="utf-8")))
REPORT = export_reports(OUTPUT)
display(FileLink(str(REPORT.relative_to(BASE))))
print("Complete comparison:", RESULT["complete_comparison"])
print("Models fitted:", RESULT["model_fits"])
print("Download sst_unet_blend_reports.zip and preserve the complete output.")
''')])


if __name__ == '__main__':
    main()
