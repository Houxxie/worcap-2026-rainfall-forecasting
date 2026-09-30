# Prospective source and forecast registry

This workflow records received data and monthly forecasts for October 2026–September 2027. **The first forecast, for October 2026, was frozen on 30 September at 18:18 UTC**, before the 1 October deadline. The fixed hybrid uses eight validated source snapshots. An explicit evaluation of the original IRI recipe recovered the four CFSv2 members missing from its named product, with exact agreement on the available members. See the [operational record](OPERATIONAL_STATUS.md) and [fitting and inference workflow](BASELINE_RUN.md).

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

On 28–30 September 2026, the named CFSv2 product lacked members 21–24. The same provider's native source and explicit catalog recipe returned all 24 on 30 September. The additional route keeps the original member/date/unit/grid contract and requires exact overlap with the named product:

```bash
python research/prospective/prepare_cfsv2_recipe.py --registry outputs/prospective --target 2026-11
```

This command downloads and audits once; it does not issue a forecast or select another provider. Restore the latest **44-event registry** before continuing. October already has a frozen prediction and cannot be issued again. A future month's completeness must be checked anew.

`cfsv2_compatibility.py` separately compares original IRI and successor CCSR fields for diagnostics. CCSR did not reproduce the original values on the four checked months and is not an approved substitute. Sources are listed in [DATA.md](../../docs/DATA.md).
