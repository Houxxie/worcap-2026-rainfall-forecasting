# Time alignment and availability

A file describing T−1 may be published after the forecast deadline. A model initialization date is not the time an aggregated download became available.

The competition model aligns atmospheric and ocean inputs to T−1 and seasonal forecasts to initializations before the target. Final training ends in December 2022, without updating the fit using 2023–2024 rainfall. This checks alignment, not the historical availability of every consolidated product and hindcast version.

## Research information calendar

| Input for T | Month used |
|---|---|
| Consolidated ERA5 atmosphere | T−4 |
| NOAA ocean indices | T−3 |
| Experimental ERSSTv5 input | T−2 |
| Seasonal initialization / nominal issue | T−1 |
| Latest block training rainfall | First target month minus four months |

These are conservative experimental assumptions, not publication receipts. The project has no verified historical arrival calendar for the CFSv2/IRI aggregate.

The prospective registry records actual UTC receipt time, content hashes, versions and parent files. Unknown publication times remain unknown. Missing or incomplete required inputs block emission. No real forecast had been frozen at the recorded state of 28 September 2026.

The local hash chain is not an independent timestamp or tamper-proof store. Future skill requires forecasts frozen before their target months and verified afterwards. ERA5 is a reanalysis target, not independent rain-gauge truth. Already-consulted development years and the 2025 evaluation are not new untouched holdouts.
