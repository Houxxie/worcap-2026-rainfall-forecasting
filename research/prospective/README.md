# Prospective source and forecast registry

This workflow records received data and prepares monthly forecast records for October 2026–September 2027. **Collection has begun; no real forecast has been issued.** The [fitting and inference workflow](BASELINE_RUN.md) now implements the model package and ERA5/SEAS5 adapters. Real fitting and complete audited inputs are still required before emission.

The [notebook](prospective_registry.ipynb) collects public sources, audits inputs, restores records and reports readiness. It does not train or substitute a model when inputs are unavailable.

## Plan and checks

Atmosphere T−4; ocean indices T−3; seasonal initialization T−1. SST T−2 is archived for research only. Final training is fixed to January 1993–December 2022 with the research lags. Old fold models or competition models with different lags cannot substitute for that package.

Checks cover actual UTC receipt time, hashes, required months, units, grids, ensemble completeness, parent files, model-package completeness, deadline, duplicate emission and forecast validity. Unknown first-publication time remains unknown. Missing sources block emission. Forecast and training climatology are frozen together; later ERA5 revisions remain separate verification records.

The local hash chain is not an independent timestamp or tamper-proof store. Reports retain months without forecasts instead of omitting them.

## Run

Use Python 3.12 and [requirements.txt](requirements.txt):

```bash
python research/prospective/test_registry.py
python research/prospective/collect_sources.py --alvo 2026-10 --cfsv2
python research/prospective/prepare_cfsv2.py --alvo 2026-10
```

`--alvo` means target month; `--atualizar` on the preparation command requests a fresh source version. Compatibility keys remain stable. Commands run once; there is no automatic schedule.

Save outputs. Attach the complete previous registry in Kaggle to restore its chain and objects. Restarting without it creates another chain and cannot prove earlier acquisitions.

On 28 September 2026, September CFSv2 lacked members 21–24 and was blocked without a partial mean. No real forecast was frozen. This record does not assert the source's current state. Sources are listed in [DATA.md](../../docs/DATA.md).
