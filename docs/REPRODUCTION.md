# Reproduce an experiment

Start with the [offline demo](../demo/README.md) if you only want to inspect results. Full reproduction needs the files in the [data guide](DATA.md); a manifest is not a substitute for its NetCDF files.

| Path | Purpose | Output |
|---|---|---|
| Competition model | Reproduce the submitted 2023–2024 CSV | `submission_hibrida.csv`, with the recorded SHA-256 |
| Research comparison | Refit the reference model with longer input lags and U-Net on seven historical blocks | Prediction maps, metrics, execution record and report |
| Current forecasting model | Fit once on 1993–2022 and forecast future months with checked arrivals | See the [monthly guide](MONTHLY_FORECAST.md) |

The operational final fit must not be used to score earlier historical validation years.

## Competition: Kaggle

1. Download and import [competition notebook](../competition/hybrid_forecast.ipynb).
2. Attach the official competition data and prepared SEAS5/CFSv2 datasets. Each seasonal source needs its manifest and all three `.nc` partitions. Expand Input and confirm the actual files are present, not only `notebook.ipynb` and `results.html`.
3. Use Python 3.12 and the [reference environment](../competition/requirements.txt). The notebook checks core versions rather than silently upgrading packages. No GPU, CDS key or Internet is required with inputs attached.
4. Leave the three `BANCA_PASTA_*` values as `None` for discovery of a single matching input, or set their directories explicitly.
5. Leave `REEXECUTAR_VALIDACAO = False` for the final fit and archived historical tables. Set it to `True` only to retrain the seven validation blocks too. Run cells in order.
6. Save a version **with outputs** and retain the CSV and audit records. Displaying archived tables is not a new validation run.

The full grids need substantial RAM. Runtime depends on loading, CPU and whether validation is enabled.

## Competition: local Python

Create an isolated environment:

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on Linux/macOS, then:

```bash
python -m pip install -r competition/requirements.txt
python scripts/check_inputs.py --config configs/competition.inputs.json
```

The default layout is `data/official`, `data/seas5`, `data/cfsv2`. Edit [competition.inputs.json](../configs/competition.inputs.json) for another layout. The read-only checker verifies exact historical bytes and four core package versions, not available RAM or the entire runtime.

Set `BANCA_PASTA_DADOS`, `BANCA_PASTA_SEAS5` and `BANCA_PASTA_CFSV2` at the top of [competition script](../competition/hybrid_forecast.py) to the same **absolute directories printed by the check**. The historical script does not read this JSON automatically. Then run:

```bash
python competition/hybrid_forecast.py
```

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

These use the standard library to check packaging and the getting-started paths, not a new full-grid fit. Scientific tests have separate dependencies, for example `python research/prospective/test_registry.py` in the research environment.
