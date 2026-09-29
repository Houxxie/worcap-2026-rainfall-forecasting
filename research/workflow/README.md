# One workflow for development comparisons

Use one configuration and one command to produce a verified comparison, tables, figures and a portable HTML report. The first supported experiment is the existing **fixed hybrid versus compact U-Net**, on all seven chronological blocks from 2007–2020. Architecture, training windows, input lags, weights and epoch selection remain unchanged.

There are two explicit modes:

| Mode | Inputs | Work performed |
|---|---|---|
| `saved` — default | Official observed rainfall and the seven saved prediction maps | Check provenance, reproduce errors, generate diagnostics and a report; no training |
| `train` | Official training files, prepared SEAS5 and CFSv2 | Run the existing seven-block experiment, then the same checks and report |

These are previously consulted **development years**. Neither mode claims an independent holdout, promotes a model or issues a prospective forecast. Operational data collection and its incomplete-source gate remain separate.

## Easiest route: Kaggle

Import [experiment_workflow.ipynb](experiment_workflow.ipynb). Its source bundle is self-contained: no private GitHub login, API key or new source download is required for saved-mode evaluation.

1. Attach the official competition data and the **saved output** of the seven-block U-Net notebook. Verify that the output contains `spatial_unet/results/H1/predictions.nc` through `C/predictions.nc`, not just the notebook source or report ZIP.
2. Leave `MODE = "saved"`. The configuration cell finds one official rainfall file and one complete seven-block output. If duplicates are attached, set `MANUAL_INPUTS` to the intended paths; it will not silently choose a copy.
3. Run the check cell, then the execution cell. No GPU is needed in saved mode.
4. Open or download `report.html`. Save the notebook version **with outputs** to retain the run package.

For training, set `MODE = "train"`, attach the three prepared input datasets and enable a GPU. This runs all seven blocks and needs the [fixed training dependencies](../spatial_unet/requirements.txt). It can be substantially more expensive than saved-mode analysis. The notebook does not install packages, download data or start training during its check cell.

## Local use

Run commands from the repository root using Python 3.12 or later. For saved-mode analysis in a separate environment:

```bash
python -m pip install -r research/workflow/requirements_evaluation.txt
python -m unittest research.workflow.test_workflow research.diagnostics.test_diagnostics -v
```

Copy [saved.example.json](saved.example.json) to your own configuration file and edit its paths. Keep `evidence: null` only when using the original archived maps: it selects the bundled seven-block evidence. For another complete run of the same protocol, set both `predictions` and `evidence` to that run's results folder. Do not mix its maps with the original run's hashes.

Paths may contain spaces. Relative paths resolve from the **configuration file's directory**, not the shell's working directory. `output_root` must be separate from the input datasets and source folders.

```bash
python -m research.workflow check --config "/path/to/my_config.json"
python -m research.workflow run --config "/path/to/my_config.json"
python -m research.workflow verify --run "/path/to/outputs/experiments/RUN_ID"
python -m research.workflow list --root "/path/to/outputs/experiments"
```

`check` validates paths, source protocol, hashes and dependencies without creating a run. `run` repeats preflight to catch intervening changes. `verify` checks the complete output package, including added or missing files. `list` displays run status and recorded scores; it does not choose a winner or assert that listed runs have just been reverified.

The [training example](train.example.json) uses explicit official, SEAS5 and CFSv2 directories and a `device` setting. Use the training environment rather than installing evaluation pins over it. CUDA availability and the existing exact baseline package versions are checked before fitting. A changed architecture or scientific protocol needs a separate reviewed adapter; unknown configuration keys are rejected.

## What each run contains

Each run gets a UTC timestamp and a fingerprint derived from configuration-relevant settings, code, input hashes and environment. Paths and output-directory names are not scientific identity. Repeated evaluations create separate directories, with the same fingerprint when the recorded scientific and software inputs match.

```text
RUN_ID/
  config.json              Resolved user settings
  identity.json            Protocol, source/code hashes, environment and summary
  state.json               Running, completed, failed or interrupted
  execution.log            Progress from this run
  code/                    Source snapshot captured before execution
  source_evidence/         Signature, block completion records and histories
  diagnostics/             Pooled metrics, error maps and four figures
  comparison/              Block/year comparisons against the hybrid
  report.html              Portable report with embedded figures
  REPORT.md                Short result and report entry point
  manifest.json            Completed output inventory and hashes
  training/                Models and forecasts, only in train mode
```

The output snapshot preserves configuration and code but does not vendor Python packages or copy the large source datasets in saved mode. Reproduction still requires the same input snapshots. The HTML can be read by itself; full verification requires the entire run directory. Hashes check byte integrity and consistency, not independent authenticity or real historical publication dates.

Existing runs are never overwritten. A failed or interrupted run keeps its partial output and error stage, and `verify` refuses it. Correct the cause and launch a new run. If fitting completed but report generation failed, saved mode can evaluate the complete `training/` output, avoiding another fit. Incomplete training blocks are not silently resumed or treated as a seven-block comparison.

## Checks and interpretation

The workflow reuses the [archived-map diagnostic](../diagnostics/README.md): exact dates, units and grid; prediction and observation hashes; 2,016 monthly error records; pooled global metrics; intensity partitions; and original epoch selection. It adds strict configuration checks, a code/input fingerprint, source snapshots, lifecycle status, an output manifest and an HTML report. Evaluation dependencies are recorded separately from the environment that originally trained the models.

Regional, seasonal and intensity differences remain descriptive. Future observed rainfall cannot select an operational model. No blend is fitted from the report, and a small development-score improvement is not automatically an accepted model.

This implements the evaluation consolidation and brings forward part of the usability work: one configuration, early error messages, an explicit saved-versus-training choice, reproducible outputs and tests. New architectures, source migration, model tuning and a general-purpose experiment platform are outside this change.

## Validation of this integration

The CLI completed saved-mode evaluation on all 168 real months and reproduced the archived RMSEs: hybrid **1.7533987173382521**, U-Net **1.764725739441376**. The completed run verified all 115 output artifacts. Thirteen configuration, integrity and diagnostic tests passed. The standalone notebook bootstrap was executed in an isolated local directory and its preflight verified all 73 configured input artifacts; notebook syntax and bundled-source consistency were also checked.

The HTML report was inspected in a browser. The full Kaggle notebook and a new GPU training run were not executed for this integration. Train mode delegates to the unchanged experiment implementation; the existing seven-block training result remains the measured evidence. See the [validation record](evidence/validation.json).
