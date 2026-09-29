# Research record

The SST candidate appends eight ERSSTv5 PCs to both LightGBMs while retaining ridge and blend weights. Ocean mask, monthly climatology, centering and full-SVD PCA are fitted only on training data, with square-root cosine-latitude weighting.

## Retrospective 2025 evaluation

With January 1993–December 2022 training and consolidated T−1 inputs, archived global RMSE was **1.684157**, versus **1.667615** with SST. It evaluated an additional year at that stage. The year is now consulted and cannot remain an untouched holdout for later tuning. Consolidated inputs do not prove historical real-time availability.

## Lagged-source comparison

Over seven two-year blocks in 2007–2020, RMSE was **1.753399**, versus **1.752762** with SST. The candidate improved 5/7 blocks and 6/14 years but worsened MAE and bias. The baseline was retained. The [active comparison](../research/lagged_sources/README.md) preserves its evidence tables; earlier scripts and reports remain in `pesquisa-v0.1.0`.

## Prospective collection

On 28 September 2026, September CFSv2 lacked expected members 21–24 and was rejected. ERA5/SEAS5 acquisition and final model integration were incomplete. No real forecast had been issued. This is a dated record, not a live provider-status claim.

These studies have different periods and information calendars. Their scores are not interchangeable benchmarks.
