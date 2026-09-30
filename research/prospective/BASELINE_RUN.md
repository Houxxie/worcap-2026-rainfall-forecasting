# Fit and issue the lagged baseline

For first-time setup, start with the [monthly forecast guide](../../docs/MONTHLY_FORECAST.md). It separates a new registry from continuation of an existing run. This page retains the detailed record of October's workflow.

The workflow covers final fitting, model registration, source normalization, inference and freezing. The final fit and authenticated ERA5/SEAS5 acquisition completed on 29 September 2026. **October's forecast was frozen on 30 September at 18:18 UTC**, after the original IRI recipe supplied the complete CFSv2 ensemble. The [operational record](OPERATIONAL_STATUS.md) identifies the frozen package, source-equivalence evidence and issued forecast.

The separate `baseline_inference.ipynb` bundles this code for Kaggle. Attach the official training data, the two prepared seasonal datasets, and the **complete existing registry** with its `eventos` and `objetos` directories. It must resume that chain rather than create a replacement with invented earlier receipt dates.

The first cell installs missing NetCDF/GRIB/CDS readers before importing xarray. Model-library versions are checked against the fixed training environment; they are not automatically upgraded. In a session where xarray was already imported before a reader was installed, restart from the saved registry instead of treating a cached missing reader as a missing source.

## Training

The package is fitted once on January 1993–December 2022 with atmospheric lag 4, index lag 3, the existing 29/31-feature tree models and local ridge. The archived NOAA table supplies only training origins; current inference reads the individually recorded PSL responses. Input hashes, source snapshot, features, settings and code are recorded. The 2023–2024 test partitions are not loaded. A competition model or historical fold cannot substitute for this fit.

```bash
python research/prospective/fit_baseline.py --registry outputs/prospective --output outputs/prospective_model_v1 --official /path/to/official --seas5 /path/to/seas5 --cfsv2 /path/to/cfsv2
```

An existing model event blocks refitting. A package directory is never silently overwritten. The registered files match the original plan exactly; the plan and earlier events are not rewritten.

## Sources

Public NOAA/CFSv2 collection continues through the existing adapters. ERA5 and SEAS5 use an existing CDS credential, supplied through the normal `cdsapi` environment or local `.cdsapirc`, never through committed code. On Kaggle, the operational run uses an enabled `CDS_API_KEY` secret, passed through the environment variable `CDSAPI_KEY` with `CDSAPI_URL=https://cds.climate.copernicus.eu/api`; do not print it. The secret label and environment-variable name are different. The CDS dataset terms must already have been accepted by the account owner.

```bash
python research/prospective/collect_cds.py --registry outputs/prospective --target 2026-10
```

The CDS adapter requests monthly **GRIB**, so the ERA5 `expver=1` evidence remains available. It requires the full quarter-degree atmospheric grid, specified units and all variables, plus all 51 SEAS5 members for system 51, forecast month 2. The existing 0.01 mm/day negative-residual tolerance is retained and each correction is audited. Unknown metadata, missing members, negative values beyond tolerance or an incomplete grid block normalization.

ERA5 documentation explains why a NetCDF containing only ERA5 or only ERA5T cannot reliably distinguish the two: [ECMWF data documentation](https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5+data+documentation). A delay alone is not treated as proof of a final vintage.

To register a GRIB already downloaded, preserving the actual *current* import time:

```bash
python research/prospective/prepare_cds.py --registry outputs/prospective --target 2026-10 --source seas5 --file /path/to/source.grib
```

Original bytes, normalized bytes and adapter source are linked in the registry. Local import cannot establish an earlier download time. The authenticated run validated June 2026 ERA5 final monthly fields and the 51-member September 2026 SEAS5 forecast for October; future acquisitions must pass the same checks.

## Inference

```bash
python research/prospective/issue_forecast.py --registry outputs/prospective --target 2026-10 --output outputs/forecast_2026_10.nc
```

Inference materializes the exact model and source objects selected by the registry, reconstructs nonnegative rainfall, and freezes both forecast and training climatology. A source-version change during inference invalidates freezing. An existing monthly prediction or a passed deadline blocks emission. A file produced before a failed freeze is only an unregistered artifact; it is not an issued forecast.

The October example is already completed in the latest registry. Do not rerun it to replace the prediction. Its local inference used the exact frozen NumPy 2.0.2, pandas 2.3.3, xarray 2025.12.0 and LightGBM 4.6.0 versions; the actual Windows/Python 3.12.7 environment is retained in the evidence. The synthetic fitted-package replay passed. GRIB checks passed separately in the acquisition environment; inference reuses the normalized registered inputs.

The U-Net experiment has a separate development protocol. It neither changes this baseline plan nor registers a neural prediction in the baseline's name.
