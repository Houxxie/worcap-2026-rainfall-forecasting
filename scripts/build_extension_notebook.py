"""Package the locked 2021–2022 extension as a standalone Kaggle notebook."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_notebooks import markdown, code
from build_spatial_notebooks import build


def main():
    names = ['research/spatial_unet/protocol.json', 'research/temporal_extension/protocol.json',
             'research/lagged_sources/library.py', 'research/lagged_sources/official_hashes.json',
             'research/lagged_sources/ocean_indices.csv', 'competition/metadata/NOAA/indices_noaa.csv']
    for folder in ['common', 'spatial_unet', 'temporal_extension']:
        names += [p.relative_to(ROOT).as_posix() for p in (ROOT / 'research' / folder).glob('*.py')]
    build('research/temporal_extension/temporal_extension.ipynb', 'research/temporal_extension/README.md', names, [
        markdown('## 1. Inputs and preflight\nAttach the official dataset and both prepared seasonal datasets. No archived fold models or SST are required. This cell does not train or evaluate the 2021–2022 targets.'),
        code('''import unittest
from research.temporal_extension.experiment import check, freeze, evaluate, verify_frozen
OFFICIAL = None
SEAS5 = None
CFSV2 = None
DEVICE = "cuda"
checks = unittest.defaultTestLoader.loadTestsFromNames([
    "research.temporal_extension.test_extension", "research.spatial_unet.test_spatial"
])
assert unittest.TextTestRunner(verbosity=2).run(checks).wasSuccessful(), "Synthetic checks failed."
PATHS, INPUT_AUDIT = check(OFFICIAL, SEAS5, CFSV2, DEVICE)
print({name: str(path) for name, path in PATHS.items()})
'''),
        markdown('## 2. Fit and freeze, or restore a complete frozen run\nLeave `PREVIOUS_RUN=None` for the first run. To continue after a session reset, attach its saved output and enter the directory containing `frozen.json`. This verifies and restores its package without refitting. Never substitute the final operational model trained through 2022.'),
        code('''from datetime import datetime, timezone
PREVIOUS_RUN = None  # Example: "/kaggle/input/.../temporal_extension/runs/20260930T..."
if PREVIOUS_RUN is None:
    RUN = RUN_ROOT/"runs"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    freeze(RUN, **PATHS, device=DEVICE)
else:
    source = Path(PREVIOUS_RUN).resolve()
    verify_frozen(source)
    RUN = RUN_ROOT/"restored"/source.name
    if RUN.exists():
        verify_frozen(RUN)
        assert (RUN/"frozen.json").read_bytes() == (source/"frozen.json").read_bytes(), "Different restored run."
    else:
        shutil.copytree(source, RUN)
    verify_frozen(RUN)
print("Frozen historical prediction package:", RUN)
'''),
        markdown('## 3. Open the target rainfall and evaluate once\nThe four forecast fields are already frozen. This cell verifies their package before decoding January 2021–December 2022 observations. It does not select weights, epochs or a winner for operational use.'),
        code('''from research.common.inputs import sha256
if (RUN/"evaluation"/"complete.json").is_file():
    verify_frozen(RUN)
    completed = json.loads((RUN/"evaluation"/"complete.json").read_text())
    assert completed["frozen_record_sha256"] == sha256(RUN/"frozen.json")
    for name, expected in completed["files"].items():
        artifact = (RUN/"evaluation"/name).resolve()
        assert artifact.is_relative_to((RUN/"evaluation").resolve()), "Invalid evaluation path."
        assert sha256(artifact) == expected, "Changed evaluation artifact: " + name
    RESULT = json.loads((RUN/"evaluation"/"summary.json").read_text())
    print("Reusing the previously completed evaluation; no new scoring.")
else:
    RESULT = evaluate(RUN, PATHS["official"]/"treino_tp.nc")
print(json.dumps(RESULT, indent=2))
'''),
        markdown('## 4. Read and export the results\nDownload the small report ZIP for review. Save the complete notebook output to retain frozen models and prediction maps; the report ZIP alone cannot restore a run.'),
        code('''import zipfile
import pandas as pd
from IPython.display import display, FileLink, HTML
display(pd.read_csv(RUN/"evaluation"/"global.csv"))
display(HTML((RUN/"evaluation"/"report.html").read_text(encoding="utf-8")))
REPORT_ZIP = RUN_ROOT/"temporal_extension_reports.zip"
with zipfile.ZipFile(REPORT_ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in RUN.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".csv", ".png", ".html"} and "code" not in path.relative_to(RUN).parts:
            archive.write(path, path.relative_to(RUN))
display(FileLink(str(REPORT_ZIP.relative_to(BASE))))
display(FileLink(str((RUN/"evaluation"/"report.html").relative_to(BASE))))
print("Save Version with all outputs. The small ZIP excludes model weights and rainfall maps.")
''')])


if __name__ == '__main__':
    main()
