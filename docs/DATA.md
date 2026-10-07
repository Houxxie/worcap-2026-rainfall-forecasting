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

Raw data, models and large prediction arrays stay outside Git. The [versioned data release](https://github.com/Houxxie/worcap-2026-rainfall-forecasting/releases/tag/data-snapshots-v1) provides the exact prepared SEAS5/CFSv2 snapshots, plus optional research SST. The [catalog](../configs/data_snapshots.json) records archive and member hashes. Provider downloads may have been revised; use the recorded snapshots for exact reproduction. See [distribution and attribution](DATA_RIGHTS.md) for the separate data conditions.

Historical acquisition scripts remain in `pesquisa-v0.1.0`. Newly acquired CFSv2 is audited by the active prospective adapter. Source terms remain separate from the code license. See [temporal validity](TEMPORAL_VALIDITY.md).

## Availability

| Artifact | Included in the repository? | How to obtain or use it |
|---|---|---|
| Competition and research code | Yes | Download ZIP or clone |
| Lightweight demo, archived metrics and selected map images | Yes | Open `demo/index.html` after downloading; no credentials |
| Official competition NetCDFs and sample CSV | No | Use the competition data link above; access is controlled by Kaggle/organizers |
| Exact historical SEAS5/CFSv2 prepared NetCDFs | Release assets, not Git objects | Run `python scripts/download_data.py` (about 17 MB, no credentials) |
| Historical ERSSTv5 research snapshot | Optional release asset | Add `--include-sst` (about 3.5 MB); not used by the competition model |
| Archived NOAA indices | Yes | Included in the code and metadata; current inference acquires fresh recorded responses |
| Trained operational model and full registry | No | Fit with the matching historical inputs, or restore your own complete saved registry |
| Complete historical forecast maps and experiment model weights | No | Retain the full output of your experiment; the small reports ZIP is insufficient for rescoring |
| Current operational sources | No | Collect using the monthly workflow; ERA5/SEAS5 need CDS access |

The first-time execution path is therefore **download → inspect demo → obtain data → check inputs → run**. Exact numerical reproduction is not promised for newly downloaded provider revisions.

For the default local examples, arrange your files as follows (these folders are ignored by Git):

```text
data/
  official/   treino_*.nc, teste_features.nc, sample_submission.csv
  seas5/      seas5_51_manifesto.json, seas5_51_*.nc
  cfsv2/      cfsv2_manifesto.json, cfsv2_*.nc
```

Run `python scripts/check_inputs.py --config configs/competition.inputs.json` from the repository root. It reports every missing/changed file and core package. Research and operational-fit profiles need fewer partitions; use their separate configs.

For provenance, the archived preparation sources are [SEAS5 acquisition](https://github.com/Houxxie/worcap-2026-rainfall-forecasting/blob/pesquisa-v0.1.0/entregas/etapa7A_SEAS5/01_SEAS5_baixar_e_auditar.py) and [CFSv2 acquisition](https://github.com/Houxxie/worcap-2026-rainfall-forecasting/blob/pesquisa-v0.1.0/entregas/etapa8A_CFSv2_dados/01_CFSv2_baixar_e_auditar.py). They are historical sources, not a guarantee that today's endpoints return the same bytes. Review provider compatibility before using them for a new dataset; keep changed data under a new experiment identity.

## Prepare a new checkout

From the repository root, Python's standard library is enough to download the prepared sources:

```bash
python scripts/download_data.py
```

For the SST research experiment, use `python scripts/download_data.py --include-sst`. This does not fit PCA or download new monthly observations. Existing matching files are reused; changed files are preserved and reported as an error.

Download the official dataset with **Download All** on the competition's Data page, then:

```bash
python scripts/prepare_official.py --archive "/path/to/previsao-climatica-de-precipitacao-sobre-a-america-do-sul.zip"
```

This verifies and extracts the 13 original files to `data/official`. It requires space for the ZIP, roughly 2 GB of extracted files and temporary staging. It never uploads your files. If access to the competition data is unavailable, the demo and climate snapshots remain usable, but full competition reproduction cannot proceed. The project does not substitute a different rainfall dataset silently.

On Kaggle, attach the official competition dataset with **Add Input** instead of extracting another copy. The [setup notebook](../competition/reproduce_from_github.ipynb) downloads the seasonal snapshots and finds that input directory.

For an offline machine, download the ZIP assets from the release on another machine, transfer them, and use `python scripts/download_data.py --archive-dir /path/to/archives`. No API keys are needed for these historical snapshots. Current operational acquisition is a separate workflow with its own arrival checks and credentials.
