"""Package the fixed-budget, user-run residual U-Net pilot."""
from pathlib import Path
from build_notebooks import markdown, code
from build_spatial_notebooks import build

ROOT = Path(__file__).resolve().parents[1]


def main():
    names = ['research/residual_unet/protocol.json', 'research/spatial_unet/protocol.json',
        'research/lagged_sources/library.py', 'research/lagged_sources/ocean_indices.csv',
        'research/lagged_sources/official_hashes.json', 'research/lagged_sources/evidence/metricas_blocos.csv']
    for folder in ['common', 'spatial_unet', 'residual_unet']:
        names += [p.relative_to(ROOT).as_posix() for p in (ROOT / 'research' / folder).glob('*.py')]
    build('research/residual_unet/residual_unet_pilot.ipynb', 'research/residual_unet/README.md', names, [
        markdown('## 1. Small synthetic checks\nThese checks use tiny synthetic grids. They validate the implementation, not forecast skill.'),
        code('''import unittest
suite = unittest.defaultTestLoader.loadTestsFromNames([
    "research.residual_unet.test_residual", "research.spatial_unet.test_spatial"
])
assert unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful(), "Synthetic checks failed."
'''),
        markdown('## 2. Run the bounded pilot\nSame three Inputs, GPU enabled. There are six hybrid fits and only two eight-epoch neural fits. No search or seven-block run is launched.'),
        code('''from research.residual_unet.experiment import execute, export_reports
OFFICIAL = None
SEAS5 = None
CFSV2 = None
OUTPUT = RUN_ROOT/"results"
PREVIOUS_OUTPUT = None
if PREVIOUS_OUTPUT is not None:
    previous = Path(PREVIOUS_OUTPUT).resolve()
    assert (previous/"signature.json").is_file(), "Attach the full saved output, not the reports ZIP."
    if not OUTPUT.exists():
        shutil.copytree(previous, OUTPUT)
    else:
        assert (OUTPUT/"signature.json").read_bytes() == (previous/"signature.json").read_bytes(), "Different existing run."
RESULT = execute(OUTPUT, official=OFFICIAL, seas5=SEAS5, cfsv2=CFSV2, device="cuda")
assert RESULT["pilot_complete"] and RESULT["blocks"] == ["C"]
'''),
        markdown('## 3. Results and download\nCompletion means `pilot_complete: true`. Seven-block completion remains false by design. Preserve the full saved output for later diagnostics.'),
        code('''import pandas as pd
from IPython.display import display, FileLink, HTML
for name in ["global", "years", "regions"]:
    print(name)
    display(pd.read_csv(OUTPUT/(name+".csv")))
display(HTML((OUTPUT/"report.html").read_text(encoding="utf-8")))
REPORT = export_reports(OUTPUT)
display(FileLink(str(REPORT.relative_to(BASE))))
print("Pilot complete:", RESULT["pilot_complete"])
print("Complete seven blocks:", RESULT["complete_seven_blocks"], "— expected: this is a one-block pilot.")
print("Download residual_unet_pilot_reports.zip and save the full notebook output.")
''')])


if __name__ == '__main__':
    main()
