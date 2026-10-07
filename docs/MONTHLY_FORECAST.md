# Run a monthly forecast

The reference is **two LightGBM models + local ridge**, weights 0.375 / 0.375 / 0.25, fitted once on **January 1993–December 2022**. Atmospheric inputs are T−4, ocean indices T−3, seasonal initializations T−1. U-Net and global SST/PCA remain separate experiments.

The [fixed plan](../research/prospective/plan.json) covers October 2026–September 2027. It does not support arbitrary later dates or replacing the model inside that series.

## New registry or existing run?

- **New user:** obtain [historical inputs](DATA.md), fit below and start your own registry with actual current timestamps. This cannot reproduce our earlier issue time.
- **Existing run:** restore the complete registry (`eventos/` and `objetos/`) and continue it. A report ZIP or summary JSON is insufficient. Once a model is registered, skip fitting.

October was already issued. Inspect it in the [saved demo](../demo/README.md). Do not replace it, change receipt times or label a later reconstruction as the original forecast.

## 1. Prepare the environment and inputs

Use Python 3.12 in an isolated environment. After activating it, from the repository root:

```bash
python -m pip install -r research/prospective/requirements_model.txt
python scripts/check_inputs.py --config configs/operational.inputs.json
```

Default folders: `data/official`, `data/seas5`, `data/cfsv2`. Seasonal development and final-fit partitions are needed; 2023–2024 test partitions are not used. Change paths in [input settings](../configs/operational.inputs.json) and the fitting command together if needed.

The exact historical SEAS5/CFSv2 inputs are available with `python scripts/download_data.py`; official files still come from Kaggle. Pretrained operational packages are not distributed; see [availability](DATA.md#availability). CPU fitting is supported; no GPU is required. GRIB decoding needs ecCodes. Preserve the package versions recorded at fitting for inference.

## 2. Fit once (new registry only)

```bash
python research/prospective/fit_baseline.py --registry outputs/prospective --output outputs/prospective_model_v1 --official data/official --seas5 data/seas5 --cfsv2 data/cfsv2
```

This initializes the fixed plan, fits and registers the model. It stops if a model is already registered or the output package directory exists. It does not issue a forecast.

## 3. Collect sources

The examples use **November 2026**. Set every target consistently and run before the target month's deadline, within the plan. November sources may not all be available yet; a missing source must remain a blocked status.

Configure CDS credentials outside Git through the normal `cdsapi` configuration and accept the relevant dataset terms. On Kaggle, enable the `CDS_API_KEY` secret; the adapter receives it as `CDSAPI_KEY` with `CDSAPI_URL=https://cds.climate.copernicus.eu/api`. Never print the secret or save it in settings.

```bash
python research/prospective/collect_sources.py --registro outputs/prospective --alvo 2026-11
python research/prospective/collect_cds.py --registry outputs/prospective --target 2026-11
python research/prospective/prepare_cfsv2_recipe.py --registry outputs/prospective --target 2026-11
```

The public collector records NOAA indices and an optional SST arrival; SST is not a model input. The CFSv2 command uses the audited original IRI recipe and checks member completeness. The separate CCSR successor has not been approved as equivalent. A failed request must not trigger silent source substitution.

## 4. Check readiness, then issue

Edit the target and registry path in [monthly.prepare.json](../configs/monthly.prepare.json):

```bash
python -m research.workbench check --config configs/monthly.prepare.json
python -m research.workbench run --config configs/monthly.prepare.json
```

Install [report dependencies](../research/workbench/requirements.txt) in a separate reporting environment if needed, preserving the model environment. A completed readiness **report** can still say issuance is blocked. Read the source/deadline status in `report.html` or `readiness.json`; configuration success alone is not forecast readiness.

When all required sources and the model pass, before the deadline, use the model environment:

```bash
python research/prospective/issue_forecast.py --registry outputs/prospective --target 2026-11 --output outputs/forecast_2026_11.nc
```

Expected output: a NetCDF with forecast and training climatology and a successful `previsao_congelada` event. A file without that event is not an issued forecast. Source versions are checked again during issuance; an already issued month or missed deadline is rejected.

## 5. Preserve and evaluate

Back up the complete registry, model and forecast, including Kaggle outputs. The local hash chain is not an independent timestamp or external signature.

The first verification uses final ERA5 received from T+4 onward: February 2027 for October 2026. This evaluates an ERA5 target, not independent rain gauges. Preserve the forecast while researching new candidates separately.

For Kaggle continuation, import [baseline_inference.ipynb](../research/prospective/baseline_inference.ipynb) with the complete existing registry. The [detailed workflow](../research/prospective/BASELINE_RUN.md) and [operational record](../research/prospective/OPERATIONAL_STATUS.md) document adapter checks and October's issuance.
