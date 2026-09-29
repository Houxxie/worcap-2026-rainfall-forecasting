# Data sources

| Source | Product and role | Link |
|---|---|---|
| WorCAP 2026 | Rainfall, nine atmospheric fields, test features and sample IDs | [Competition data](https://www.kaggle.com/competitions/previsao-climatica-de-precipitacao-sobre-a-america-do-sul/data) |
| ECMWF / C3S | SEAS5 system 51, monthly precipitation, initialization T−1, forecast month 2 | [CDS](https://cds.climate.copernicus.eu/datasets/seasonal-monthly-single-levels) |
| NOAA PSL | Monthly Niño 1+2, Niño 3.4, TNA and TSA indices | [Monthly series](https://psl.noaa.gov/data/timeseries/month/) |
| NCEP / IRI | CFSv2 `PENTAD_SAMPLES_FULL`, lead 1.5 | [IRI Data Library](https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.NCEP-CFSv2/.HINDCAST/.PENTAD_SAMPLES_FULL/.prec/) |
| NOAA NCEI | ERSSTv5, used only by the SST research candidate | [ERSSTv5 archive](https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/netcdf/) |

## Required files

Official inputs: `treino_tp.nc`, `treino_tp_alvo.nc`, `teste_features.nc`, `sample_submission.csv` and nine `treino_<variable>.nc` atmospheric files. Names and SHA-256 values are in [reference.json](../competition/reference.json). Access depends on Kaggle and the organizers; these files are not redistributed here.

Each seasonal source needs its manifest and three NetCDF files:

| Source | Manifest | NetCDF prefix |
|---|---|---|
| SEAS5 | `seas5_51_manifesto.json` | `seas5_51_` |
| CFSv2 | `cfsv2_manifesto.json` | `cfsv2_` |

Append `desenvolvimento.nc`, `somente_ajuste_final.nc` and `teste.nc` to each prefix. These compatibility names denote development through 2020, final-fit-only 2021–2022 and forecast targets 2023–2024. Training uses the documented common seasonal coverage.

Manifest copies: [SEAS5](../competition/metadata/SEAS5/seas5_51_manifesto.json), [CFSv2](../competition/metadata/CFSv2/cfsv2_manifesto.json). The [NOAA table](../competition/metadata/NOAA/indices_noaa.csv) is also embedded in the code. The SST research snapshot is `ersstv5_4graus_198201_202411.nc`; its hash is in the [protocol](../research/lagged_sources/protocol.json).

## Preparation and limitations

SEAS5 rates in m/s are converted to mm/day by multiplying by 86,400,000. Negative residuals up to 0.01 mm/day in magnitude are zeroed per member and audited; larger negatives stop preparation. Ensemble membership is checked before averaging.

CFSv2 normally has 24 or 28 expected members. The historical August 2019 missing-member exception was individually verified. It does not authorize partial future ensembles. Seasonal fields are bilinearly interpolated without extrapolation; that does not increase their effective physical resolution.

Raw data, models and large prediction arrays stay outside Git. There is currently no public project download for the exact prepared NetCDF snapshots. Provider links locate the products, but revised downloads may differ. Exact reproduction requires the snapshots matching the hashes.

Historical acquisition scripts remain in `pesquisa-v0.1.0`. Newly acquired CFSv2 is audited by the active prospective adapter. Source terms remain separate from the code license. See [temporal validity](TEMPORAL_VALIDITY.md).
