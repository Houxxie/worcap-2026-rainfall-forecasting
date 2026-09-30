# Rainfall workbench

To explore the project with no input files or Python installation, use the [offline demo](../../demo/README.md). For a fresh full run, start with [reproduction](../../docs/REPRODUCTION.md); the paths below are for the shared experiment interface.

Use one notebook or command to check inputs, evaluate saved forecasts, train the existing experiment, and inspect monthly readiness. The model definitions, source lags, training windows and fixed weights are unchanged.

## Choose the task

| Task | What it does | Required input | Compute |
|---|---|---|---|
| `review` | Verify an existing result and build an easier-to-read report | Completed workflow folder, SST extension reports, or SST/U-Net reports ZIP | CPU; no recalculation of forecasts |
| `evaluate` | Score saved maps with the existing comparison protocol | Observations and complete prediction outputs | CPU; no model fitting |
| `train` | Run the existing seven-block hybrid/U-Net protocol, then evaluate | Official data, prepared SEAS5 and CFSv2 | GPU recommended; potentially long |
| `prepare` | Check the latest registry, source arrivals, frozen model and issue deadline | Complete registry with `eventos/` and `objetos/` | CPU; read-only, no provider request |

**Start with `review` when the results already exist.** The small report ZIP is sufficient for review, but not for saved-map evaluation. `prepare` does not acquire data, fit the final model or emit a forecast; those actions remain in the [existing operational workflow](../prospective/BASELINE_RUN.md). The [operational record](../prospective/OPERATIONAL_STATUS.md) documents October's issued forecast and the separate, still-unresolved compatibility of the successor CCSR service.

## Kaggle

Import [rainfall_workbench.ipynb](rainfall_workbench.ipynb). No GitHub credentials are needed. Evaluation and report modes do not download data or install packages.

1. In the configuration cell, choose `TASK`. The default is `review`.
2. Attach the relevant data as Input. Keep `MANUAL_INPUTS = {}` for discovery, or enter the named file/folder paths. The check table shows missing or ambiguous inputs and candidate paths together.
3. Run the checks. The notebook stops before expensive work if required inputs are missing, changed or incompatible.
4. Run the execution cell, then open the report or download its ZIP.

Examples of manual inputs:

```python
# Review the reports you have already downloaded:
MANUAL_INPUTS = {"results": "/kaggle/input/my-reports/sst_unet_blend_reports.zip"}

# Evaluate the fixed SST/U-Net blend from full saved maps:
MANUAL_INPUTS = {
    "observations": "/kaggle/input/competition/treino_tp.nc",
    "sst_predictions": "/kaggle/input/sst-output/results/models/predictions.nc",
    "unet_predictions": "/kaggle/input/unet-output/runs/EXECUTION/predictions.nc",
}

# Check the existing operational registry:
MANUAL_INPUTS = {"registry": "/kaggle/input/registry/registry"}
```

Use one matching example at a time. The two prediction files may share a filename; the SST comparison verifies their different full hashes. For the original seven-block evaluation, use `observations`, `predictions` (the parent of H1 through C), and optionally `evidence` (matching run metadata). Duplicate copies require an explicit choice. A Notebook Input containing only HTML and `.ipynb` files has no prediction maps.

The notebook clears previously imported `research` modules from memory before using its own code bundle. It does not delete stored files. Different source bytes at an existing bundle destination still stop the bootstrap; use a separate `RUN_ROOT` or a fresh notebook session rather than overwriting a recorded run.

## Supported experiments

- `hybrid_unet_v1`: existing hybrid/U-Net comparison on 2007–2020; saved evaluation or unchanged seven-block training.
- `fixed_blend_v1`: existing 75% hybrid / 25% U-Net comparison on 2007–2020; saved evaluation only.
- `sst_unet_blend_v1`: fixed 75% SST hybrid / 25% U-Net comparison on 2021–2022; saved evaluation only.

Review also accepts completed SST-extension reports from 2021–2022. Other experiment layouts need a reviewed adapter; the workbench does not guess their scientific protocol. All these historical periods were already consulted in development.

## Local commands

Use Python 3.12 or later. For reports and evaluation, install [requirements.txt](requirements.txt) in an evaluation environment. Training retains the separate fixed [training environment](../spatial_unet/requirements.txt).

Copy and edit one settings file: [review](review.example.json), [evaluate](evaluate.example.json), [train](train.example.json), or [prepare](prepare.example.json). Paths resolve from the settings file, not the shell working directory. Set `output_root` to a separate output folder.

```bash
python -m research.workbench check --config /path/to/settings.json
python -m research.workbench run --config /path/to/settings.json
python -m research.workbench status --root /path/to/experiments
python -m research.workbench verify --run /path/to/experiments/RUN
python -m research.workbench export --run /path/to/experiments/RUN
python -m research.workbench recover --run /path/to/experiments/RUN
```

`check` performs discovery, provenance and environment checks without training or creating a run. A blocked operational forecast can still have a successfully completed readiness report. Read its readiness status separately from the report's completion status.

## Files, reuse and recovery

```text
experiments/
  NAME-TIMESTAMP-FINGERPRINT/
    settings.json       User choices and paths
    identity.json       Verified inputs, code, environment and fingerprint
    state.json          Current stage, completion or preserved failure
    execution.log       Execution messages
    code/               Exact source snapshot
    payload.json        Existing or newly completed calculation to reuse
    metrics.csv         Standardized comparison, when applicable
    years.csv           Yearly results, when applicable
    readiness.json      Source and deadline status, for preparation
    summary.json        Report context and limitations
    report.html         Portable report with embedded figures
    report_bundle.json  Inventory of the small report package
    completion.json     Full output inventory after successful completion
```

Repeated runs with identical checked inputs, code and environment reuse a **verified complete output**. Readiness always gets a fresh timestamp and is not reused as a current status. Changed artifacts block reuse. A lock prevents a second simultaneous job with the same fingerprint; after a hard crash, check that the process has ended before manually removing its stale lock.

Failures preserve partial outputs. `recover` is read-only and provides settings for a new report-only continuation when the calculation already completed. If all seven training blocks finished before reporting failed, it proposes saved-map evaluation instead of retraining. It never treats incomplete model files as a completed experiment or automatically restarts training.

The report uses shared-scale monthly maps when a verified full prediction file is present; otherwise it shows the available diagnostic figures and states the report-only limitation. `map_month` can select one available month using `YYYY-MM`. The figures are retrospective predictions, not operational forecasts. Readiness reports show missing sources rather than fabricating a forecast map.

Download the small ZIP for sharing and retain the **complete output** for later scientific work. The ZIP omits model weights and prediction arrays. Report-only review checks included hashes; it does not independently rescore observations or authenticate first-publication times.

No task in this interface promotes a model, changes a source policy, or publishes a repository. Future skill must still be assessed through properly issued prospective forecasts.
