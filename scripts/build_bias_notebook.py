"""Package the single bias-calibration candidate for user-run Kaggle evaluation."""
from pathlib import Path
from build_notebooks import markdown, code
from build_spatial_notebooks import build

ROOT=Path(__file__).resolve().parents[1]


def main():
    names=['research/bias_calibration/protocol.json','research/spatial_unet/protocol.json',
        'research/lagged_sources/library.py','research/lagged_sources/ocean_indices.csv',
        'research/lagged_sources/official_hashes.json','research/lagged_sources/evidence/metricas_blocos.csv']
    for folder in ['common','spatial_unet','bias_calibration']:
        names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'research'/folder).glob('*.py')]
    build('research/bias_calibration/unet_bias_calibration.ipynb','research/bias_calibration/README.md',names,[
        markdown('## 1. Synthetic checks and existing inputs\nEnable a GPU and attach the same three data sources. These tests are small; the next cell performs the real seven-block experiment.'),
        code('''import unittest
from research.bias_calibration.experiment import check, execute, export_reports
OFFICIAL = None
SEAS5 = None
CFSV2 = None
DEVICE = "cuda"
suite = unittest.defaultTestLoader.loadTestsFromNames([
    "research.bias_calibration.test_calibration", "research.spatial_unet.test_spatial"
])
assert unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful(), "Synthetic checks failed."
PATHS, INPUT_AUDIT = check(OFFICIAL, SEAS5, CFSV2, DEVICE)
print({k: str(v) for k,v in PATHS.items()})
'''),
        markdown('## 2. Run all seven blocks\nCompleted blocks are verified and reused. Leave `PREVIOUS_OUTPUT=None` for a new run. After a session reset, attach the full saved output and provide its directory containing `signature.json`. Reports alone cannot resume fitting.'),
        code('''OUTPUT = RUN_ROOT/"results"
PREVIOUS_OUTPUT = None
if PREVIOUS_OUTPUT is not None:
    source = Path(PREVIOUS_OUTPUT).resolve()
    assert (source/"signature.json").is_file(), "Attach the full saved bias-calibration output."
    if not OUTPUT.exists():
        shutil.copytree(source, OUTPUT)
    else:
        assert (OUTPUT/"signature.json").read_bytes() == (source/"signature.json").read_bytes(), "Different existing run."
RESULT = execute(OUTPUT, **PATHS, device=DEVICE)
assert RESULT["complete_seven_blocks"], "Only a partial comparison is available."
'''),
        markdown('## 3. Review and download\nSend the report ZIP for analysis. Save the full notebook output to preserve prediction maps and fitted models. No result automatically changes the operational forecast.'),
        code('''import pandas as pd
from IPython.display import display, FileLink, HTML
for name in ["global", "blocks", "years", "offsets"]:
    print(name)
    display(pd.read_csv(OUTPUT/(name+".csv")))
display(HTML((OUTPUT/"report.html").read_text(encoding="utf-8")))
REPORT = export_reports(OUTPUT)
display(FileLink(str(REPORT.relative_to(BASE))))
print("Completed seven blocks:", RESULT["complete_seven_blocks"])
print("Download bias_calibration_reports.zip and save the full notebook output.")
''')])


if __name__=='__main__': main()
