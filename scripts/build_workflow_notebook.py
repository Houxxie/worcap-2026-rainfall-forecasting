"""Package the common workflow and existing adapters for a standalone Kaggle notebook."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_notebooks import markdown, code
from build_spatial_notebooks import build
from research.workflow.runner import code_files, REPORT_FILES
from research.workflow.config import BLOCKS, DEFAULT_EVIDENCE


def main():
    names = [p.relative_to(ROOT).as_posix() for p in code_files()]
    evidence = ['signature.json', 'global.csv'] + [f'{b}/{n}' for b in BLOCKS for n in ('complete.json',) + REPORT_FILES]
    names += [(DEFAULT_EVIDENCE / p).relative_to(ROOT).as_posix() for p in evidence]
    settings = '''from research.workflow.config import discover_saved, WorkflowError
from research.workflow.runner import write_json
MODE = "saved"  # "train" reruns the complete fixed seven-block comparison.
EXPERIMENT = "hybrid_unet_v1"  # "fixed_blend_v1" evaluates only 75% reference model + 25% U-Net, in saved mode.
MANUAL_INPUTS = None
# Saved example:
# MANUAL_INPUTS = {"observations": "/kaggle/input/.../treino_tp.nc",
#                  "predictions": "/kaggle/input/.../spatial_unet/results",
#                  "evidence": "/kaggle/input/.../spatial_unet/results"}
if MODE == "saved":
    INPUTS = MANUAL_INPUTS if MANUAL_INPUTS is not None else discover_saved()
elif MODE == "train":
    def unique_parent(filename):
        found = sorted(Path('/kaggle/input').rglob(filename))
        if len(found) != 1:
            raise WorkflowError(f"Found {len(found)} {filename} files. Set MANUAL_INPUTS to the intended directories.")
        return str(found[0].parent)
    INPUTS = MANUAL_INPUTS if MANUAL_INPUTS is not None else {
        "official": unique_parent("treino_tp.nc"),
        "seas5": unique_parent("seas5_51_manifesto.json"),
        "cfsv2": unique_parent("cfsv2_manifesto.json"),
    }
else:
    raise WorkflowError("MODE must be saved or train.")
CONFIG = {"schema": "rainfall_workflow_v1", "name": "hybrid-unet-comparison",
          "mode": MODE, "experiment": EXPERIMENT, "output_root": str(BASE/'experiments'), "inputs": INPUTS}
if MODE == "train":
    CONFIG["device"] = "cuda"
CONFIG_FILE = RUN_ROOT/'workflow_config.json'
write_json(CONFIG_FILE, CONFIG)
print(json.dumps(CONFIG, indent=2))
'''
    build('research/workflow/experiment_workflow.ipynb', 'research/workflow/README.md', names, [
        markdown('## 1. Configure paths and mode\nSaved mode evaluates the previous seven-block output and uses no GPU. Keep the default unless you intend to refit all models. The lookup requires the forecast maps, not only the report archive.'), code(settings),
        markdown('## 2. Check before execution\nThese tests use small synthetic fixtures. Preflight verifies the configured real files without fitting a model. Missing data, changed hashes or an incompatible training environment stop here.'),
        code("import unittest\nfrom research.workflow.__main__ import main\nsuite = unittest.defaultTestLoader.loadTestsFromNames(['research.workflow.test_workflow', 'research.diagnostics.test_diagnostics', 'research.fixed_blend.test_blend'])\nassert unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful(), 'Checks failed.'\nassert main(['check', '--config', str(CONFIG_FILE)]) == 0, 'Preflight stopped; read the message above.'\n"),
        markdown('## 3. Execute the configured comparison\nEach execution creates a new run directory and preserves its configuration, code, evidence, logs and outputs. No result changes the current forecasting model or issues a forecast.'),
        code("from research.workflow.runner import run, verify_run\nRUN = run(CONFIG_FILE)\nprint(verify_run(RUN))\n"),
        markdown('## 4. Read and preserve the report\nThe HTML embeds its figures and can be downloaded on its own. Save the complete notebook output version for the audit files and, in training mode, the fitted models and predictions.'),
        code("from IPython.display import display, HTML, FileLink\ndisplay(FileLink(str((RUN/'report.html').relative_to(BASE))))\ndisplay(HTML((RUN/'report.html').read_text(encoding='utf-8')))\nprint('Complete run:', RUN)\nprint('Save Version with outputs to preserve this run.')\n")])


if __name__ == '__main__':
    main()
