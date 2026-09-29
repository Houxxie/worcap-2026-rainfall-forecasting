# Fit and issue the lagged baseline

The executable workflow now covers final fitting, model registration, source normalization, inference and freezing. **Implementation is not an issued forecast:** a real run must still fit the package and acquire every required source before its deadline. Missing inputs block the run.

The separate `baseline_inference.ipynb` bundles this code for Kaggle. Attach the official training data, the two prepared seasonal datasets, and the **complete existing registry** with its `eventos` and `objetos` directories. It must resume that chain rather than create a replacement with invented earlier receipt dates.

## Training

The package is fitted once on January 1993–December 2022 with atmospheric lag 4, index lag 3, the existing 29/31-feature tree models and local ridge. The archived NOAA table supplies only training origins; current inference reads the individually recorded PSL responses. Input hashes, source snapshot, features, settings and code are recorded. The 2023–2024 test partitions are not loaded. A competition model or historical fold cannot substitute for this fit.

```bash
python research/prospective/fit_baseline.py --registry outputs/prospective --output outputs/prospective_model_v1 --official /path/to/official --seas5 /path/to/seas5 --cfsv2 /path/to/cfsv2
```

An existing model event blocks refitting. A package directory is never silently overwritten. The registered files match the original plan exactly; the plan and earlier events are not rewritten.

## Sources

Public NOAA/CFSv2 collection continues through the existing adapters. ERA5 and SEAS5 use an existing CDS credential, supplied through the normal `cdsapi` environment or local `.cdsapirc`, never through committed code. On Kaggle, an enabled `CDS_KEY` secret can be passed through `CDSAPI_KEY` with `CDSAPI_URL=https://cds.climate.copernicus.eu/api`; do not print it. The CDS dataset terms must already have been accepted by the account owner.

```bash
python research/prospective/collect_cds.py --registry outputs/prospective --target 2026-10
```

The CDS adapter requests monthly **GRIB**, so the ERA5 `expver=1` evidence remains available. It requires the full quarter-degree atmospheric grid, specified units and all variables, plus all 51 SEAS5 members for system 51, forecast month 2. The existing 0.01 mm/day negative-residual tolerance is retained and each correction is audited. Unknown metadata, missing members, negative values beyond tolerance or an incomplete grid block normalization.

ERA5 documentation explains why a NetCDF containing only ERA5 or only ERA5T cannot reliably distinguish the two: [ECMWF data documentation](https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5+data+documentation). A delay alone is not treated as proof of a final vintage.

To register a GRIB already downloaded, preserving the actual *current* import time:

```bash
python research/prospective/prepare_cds.py --registry outputs/prospective --target 2026-10 --source seas5 --file /path/to/source.grib
```

Original bytes, normalized bytes and adapter source are linked in the registry. Local import cannot establish an earlier download time. Live authenticated CDS retrieval remains to be verified in the user's configured environment; isolated tests are not a substitute for that check.

## Inference

```bash
python research/prospective/issue_forecast.py --registry outputs/prospective --target 2026-10 --output outputs/forecast_2026_10.nc
```

Inference materializes the exact model and source objects selected by the registry, reconstructs nonnegative rainfall, and freezes both forecast and training climatology. A source-version change during inference invalidates freezing. An existing monthly prediction or a passed deadline blocks emission. A file produced before a failed freeze is only an unregistered artifact; it is not an issued forecast.

The U-Net experiment has a separate development protocol. It neither changes this baseline plan nor registers a neural prediction in the baseline's name.
