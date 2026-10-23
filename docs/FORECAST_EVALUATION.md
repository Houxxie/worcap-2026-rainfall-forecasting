# Evaluate an issued monthly forecast

This workflow compares the **forecast and training climatology already frozen in the registry** with final ERA5 rainfall. It does not refit the model or produce a new forecast.

October 2026 becomes eligible on **1 February 2027 at 00:00 UTC** under the fixed T+4 protocol. That is the earliest permitted check, not a guarantee that final ERA5 will be available. November becomes eligible on 1 March 2027. The command also requires final-product metadata in the downloaded bytes.

## Prepare once

Use Python 3.12 and a **separate virtual environment** for evaluation. Keep the frozen model's inference environment unchanged. From the repository root, after activating the new environment:

```bash
python -m pip install -r research/prospective/requirements_verification.txt
```

Restore the **complete latest registry** with its `eventos/` and `objetos/` folders. A report ZIP, summary or forecast PNG is insufficient. If you extract a registry backup, point `--registry` to its inner `registry/` directory. The commands below use `outputs/prospective` as an example; replace it with your restored registry path.

Make a fresh backup before adding evaluation events:

```bash
python research/prospective/backup_registry.py --registry outputs/prospective --output outputs/before-evaluation.zip
```

Keep the resulting ZIP and its verification sidecar outside the temporary Kaggle session. Use a new name if a backup already exists.

## 1. Check the month without downloading

```bash
python research/prospective/verify_forecast.py check --registry outputs/prospective --target 2026-10
```

This verifies the event chain, all referenced objects, the frozen forecast's grid and issue time, and the evaluation date. It prints `ready`, `eligible_from_utc`, `forecast_id` and any blockers. Exit code **2** means blocked; before February this is expected for October. It creates no registry or report, and makes no provider request. There is no CLI option to override the date.

## 2. Retrieve and evaluate final ERA5

After the check passes, configure CDS credentials through the standard `cdsapi` configuration outside Git and accept the dataset terms. Then run:

```bash
python research/prospective/verify_forecast.py evaluate --registry outputs/prospective --target 2026-10 --download --output outputs/evaluation-2026-10
```

The request is for **the target month's total precipitation**, not the lagged atmospheric inputs used for prediction. It retrieves one GRIB field from [ERA5 monthly averaged single-level data](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means). Requests only start after the eligibility check. Downloads may queue at CDS.

The decoder requires `class=ea`, `expver=1`, `stream=moda`, precipitation parameter 228, the correct month and a complete **301 × 261** grid. It rejects ERA5T, duplicate fields, missing cells, wrong coordinates and negative rainfall. No interpolation, filling or clipping is performed. For this specific monthly product, metres are converted to **mm/day by multiplying by 1000**, following the [ECMWF documentation](https://confluence.ecmwf.int/pages/viewpage.action?pageId=216495524).

If you already downloaded the **original GRIB**, use this alternative:

```bash
python research/prospective/verify_forecast.py evaluate --registry outputs/prospective --target 2026-10 --grib /path/to/final-era5-october.grib --output outputs/evaluation-2026-10
```

A local import records the time it is received by this workflow. It does not claim to know when you originally downloaded it. The same eligibility and header checks apply. Use the original unarchived GRIB; a converted NetCDF may have lost the headers needed to distinguish ERA5 from ERA5T.

The registry retains the original GRIB, the decoded and normalized observations, conversion evidence, package versions, exact verification code and references to the existing forecast. Observation verification is separate from the model-input receipts. After success, back up the updated registry again using a new archive name.

## 3. Read or regenerate the report

Open `report.html` in the output folder. Its companion files are:

| File | Contents |
|---|---|
| `metrics.csv` | Monthly RMSE, MAE, bias and area-weighted RMSE for forecast and climatology; primary results and labelled revisions |
| `primary_metrics.csv` | Pooled primary metrics for the full domain and three latitude bands |
| `months.csv` | All 12 planned months, including months never issued or not yet evaluated |
| `summary.json` | Coverage, pooled results and the exact registry snapshot |
| `verification_events.json` | Verification events, hashes, source receipt information and conversion evidence |
| `complete.json` | Hashes and sizes of the exported report files; written last |

Global RMSE is **sqrt(total squared error / total evaluated grid cells)**, not an average of monthly RMSEs. Bias is forecast minus ERA5. Latitude bands are south of 35°S, 35°S to 15°S, and north of 15°S. Boundary rows belong to the northern band. Area-weighted RMSE uses cosine-of-latitude weights. The frozen training climatology is the comparison for each month.

Reports can be generated at any time, including before observations become eligible:

```bash
python research/prospective/verify_forecast.py report --registry outputs/prospective --output outputs/evaluation-overview
```

This reads recorded results without downloading, scoring or appending events. It also recovers a report if export failed after the verification event was written. Use a new output folder; an existing report is never overwritten. A missing `complete.json` indicates an incomplete export, not necessarily a failed registry update.

## Revisions and repeat runs

- Importing the same GRIB bytes again reuses the recorded verification and exports a new report.
- A different observation for an already evaluated month is rejected unless you explicitly add `--revision` to `evaluate`.
- A revision is appended with its own event and observation hashes. **The first evaluation remains the primary result** in pooled comparisons.
- `--revision` is not accepted before a primary evaluation exists.
- Do not remove a registry lock automatically after an interruption. First confirm no other process is writing and inspect the registry/backup.

This measures agreement with **ERA5 reanalysis**, not independent rain-gauge observations. One evaluated month cannot establish general skill in future years. The local registry records actual workflow receipt times and hashes; it is not an independent timestamp authority.

The new workflow is covered by [automatic synthetic checks](CHECKS.md). Those checks do not evaluate October or substitute synthetic data in the operational registry. The live CDS download and real evaluation can only be completed when the required final observations are available.
