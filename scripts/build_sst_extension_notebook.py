"""Build the standalone, CPU-only SST temporal-extension notebook."""
from pathlib import Path
from build_notebooks import markdown, code
from build_spatial_notebooks import build

ROOT = Path(__file__).resolve().parents[1]


def main():
    names = ['research/common/inputs.py', 'research/common/presentation.py', 'research/common/baseline.py', 'research/common/synthetic_fixture.py',
             'research/lagged_sources/library.py', 'research/lagged_sources/ocean_indices.csv',
             'research/lagged_sources/official_hashes.json', 'competition/metadata/NOAA/indices_noaa.csv',
             'research/temporal_extension/evidence/kaggle_354026629/evaluation/global.csv',
             'research/sst_extension/protocol.json']
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT / 'research/sst_extension').glob('*.py')]
    build('research/sst_extension/sst_temporal_extension.ipynb', 'research/sst_extension/README.md', names, [
        markdown('## 1. Small synthetic checks\nThese validate the implementation on tiny grids, not forecast skill.'),
        code('''import unittest
suite = unittest.defaultTestLoader.loadTestsFromName("research.sst_extension.test_sst")
assert unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful(), "Synthetic checks failed."
'''),
        markdown('## 2. Verify inputs and freeze both forecasts\nCPU is sufficient. Four fixed tree fits; no neural training or parameter search. Keep each manual path as `None` for automatic discovery, or set the directory containing the corresponding files.'),
        code('''from research.sst_extension.experiment import freeze, score, export_reports
OFFICIAL = None
SEAS5 = None
CFSV2 = None
SST = None  # Directory containing ersstv5_4graus_198201_202411.nc
OUTPUT = RUN_ROOT / "results"
freeze(OUTPUT, official=OFFICIAL, seas5=SEAS5, cfsv2=CFSV2, sst=SST)
'''),
        markdown('## 3. Open evaluation rainfall and compare\nThe saved predictions and source hashes are verified first. These years were previously consulted; this is not an independent holdout.'),
        code('''import pandas as pd
from research.common.presentation import display_frame
from IPython.display import display, HTML, Image, FileLink
RESULT = score(OUTPUT, official=OFFICIAL)
for name in ["global", "years", "regions"]:
    print(name)
    display(display_frame(pd.read_csv(OUTPUT / "evaluation" / (name + ".csv"))))
display(HTML((OUTPUT / "evaluation/report.html").read_text(encoding="utf-8")))
display(Image(filename=str(OUTPUT / "evaluation/monthly_comparison.png")))
REPORT = export_reports(OUTPUT)
display(FileLink(str(REPORT.relative_to(BASE))))
print("Complete extension:", RESULT["complete_extension"])
print("Baseline reproduced:", RESULT["baseline_reproduced"])
print("Download sst_temporal_extension_reports.zip and save the complete notebook output.")
assert RESULT["baseline_reproduced"], "The reference differs: inspect the reports before interpreting any SST gain."
''')])


if __name__ == '__main__':
    main()
