# Reproduce an experiment

Start with the [offline demo](../demo/README.md) if you only want to inspect results. Full reproduction needs the files in the [data guide](DATA.md); a manifest is not a substitute for its NetCDF files.

| Path | Purpose | Output |
|---|---|---|
| Competition model | Reproduce the submitted 2023–2024 CSV | `submission_hibrida.csv`, with the recorded SHA-256 |
| Research comparison | Refit the reference model with longer input lags and U-Net on seven historical blocks | Prediction maps, metrics, execution record and report |
| Current forecasting model | Fit once on 1993–2022 and forecast future months with checked arrivals | See the [monthly guide](MONTHLY_FORECAST.md) |

The operational final fit must not be used to score earlier historical validation years.

## Competition: Kaggle (recommended)

1. Download [reproduce_from_github.ipynb](../competition/reproduce_from_github.ipynb) with GitHub's **Download raw file** button and import it into a new Kaggle notebook.
2. Attach the official WorCAP competition dataset with **Add Input**. Keep a CPU session with sufficient RAM; no GPU or CDS key is needed.
3. Enable Internet and run the cells in order. The notebook fetches the pinned public code release, creates an isolated Python 3.12.13 environment with uv, installs the reference packages, downloads about 17 MB of prepared seasonal data, and checks every input hash.
4. The final model fits and generates the CSV. Save a version **with outputs** and retain the CSV and audit records.

The default run does not retrain seven historical validation blocks. Displaying their archived tables is not a new validation run. The setup supports a newer Kaggle notebook interpreter by running the model in its separate Python 3.12 environment. Interpreter and package downloads are additional to the data download. The full grids require substantial RAM; the original environment was Kaggle/Linux. Runtime depends on input loading and CPU.

The [explanatory competition notebook](../competition/hybrid_forecast.ipynb) remains available for reading the model step by step and for offline execution with all inputs attached. Its manual directory settings still work.

## Competition: local Python

Use Python 3.12 and create an isolated environment:

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on Linux/macOS. From the repository root:

```bash
python -m pip install -r competition/requirements.txt
python scripts/download_data.py
python scripts/prepare_official.py --archive "/path/to/competition-download.zip"
python scripts/reproduce_competition.py --check-only
python scripts/reproduce_competition.py
```

The official ZIP must be your own [Kaggle download](DATA.md#prepare-a-new-checkout). The default layout is `data/official`, `data/seas5`, `data/cfsv2`. Edit [competition.inputs.json](../configs/competition.inputs.json) for another layout and pass `--config path/to/config.json`. Relative input paths resolve against the configuration file's parent directory. You no longer need to edit paths in the model code.

The launcher checks exact historical bytes and four core package versions, then fits the final model in a subprocess. Each run creates a timestamped directory under `outputs/reproduction/worcap_banca_hibrida/`. Use `--output path/to/runs` to change that root. Add `--validation` only when intentionally refitting the seven historical blocks too; it takes substantially longer.

Successful exact reproduction generates `submission_hibrida.csv` with 1,885,464 rows and SHA-256:

```text
e98e8954debb8f7c764665a31804ec5512a1c2578c1ed4a24e50c51dedfb429b
```

The run prints its output directory. Original feature order, intermediate CSV rounding and input hashes are intentional. Do not relax a hash check to make a changed snapshot pass.

## Research: seven-block comparison

This fixed experiment compares the reference model with longer input lags with direct U-Net on 2007–2020. It does not issue a real-time forecast or produce a competition CSV. Archived pooled RMSE: **1.753399** for the reference model and **1.764726** for U-Net, in mm/day. Neural execution can vary across hardware; preserve and compare the actual environment.

For Kaggle, import [rainfall_workbench.ipynb](../research/workbench/rainfall_workbench.ipynb), attach official files and seasonal development partitions, choose `TASK = "train"`, `EXPERIMENT = "hybrid_unet_v1"`, and enable a GPU. Use the fixed training environment. The notebook includes the source bundle; see [workbench instructions](../research/workbench/README.md).

For local execution, use a **separate environment** with the fixed [training requirements](../research/spatial_unet/requirements.txt) and compatible CUDA/PyTorch setup. A full run can take hours. From the repository root:

```bash
python -m pip install -r research/spatial_unet/requirements.txt
python scripts/check_inputs.py --config configs/research.inputs.json
python -m research.workbench check --config configs/research.train.json
python -m research.workbench run --config configs/research.train.json
```

For custom input locations, update both [research.inputs.json](../configs/research.inputs.json) and [research.train.json](../configs/research.train.json). The latter selects the experiment and destination, not a new parameter search. Its default output is `outputs/experiments/`.

Expect a unique run directory with `report.html`, metrics, configuration and completion records. Retain the **complete run**, including model files and prediction arrays, for rescoring. The small report ZIP is for viewing. A complete experiment must contain H1, H2, H3, H4, A, B and C; a pilot does not satisfy that condition.

Use workbench `evaluate` for saved maps or `review` for existing reports, without retraining. SST and residual-network experiments have [separate notebooks](../research/README.md).

## Common problems

| Symptom | What to check |
|---|---|
| Missing manifest or NetCDF | Attach actual output files; notebook source and HTML are not the dataset |
| Several matches | Specify the intended folder |
| Hash mismatch | Locate the matching archived version |
| Different package version | Use the reference environment in a new environment/session |
| A report but no prediction maps | Saved-map evaluation needs the full output, not only the reports ZIP |

## Checks without training

```bash
python scripts/verify_repository.py
python -m unittest discover -s tests -v
```

The repository inventory and data-download tests use the standard library; presentation tests also need the model environment (including pandas). These checks are not a new full-grid fit. Scientific tests have separate dependencies, for example `python research/prospective/test_registry.py` in the research environment.
