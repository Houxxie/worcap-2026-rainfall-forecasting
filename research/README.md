# Research after the hackathon

The aim is to evaluate future monthly forecasts with inputs that actually arrived before issue time. The competition hybrid supplies the starting architecture; later changes are evaluated separately.

## Current baseline

Two LightGBMs and local ridge, fixed weights **0.375 / 0.375 / 0.25**, up to 30 training years, atmosphere T−4, indices T−3 and seasonal initializations T−1. Additional SST principal components are not part of the baseline.

| Archived 2007–2020 experiment | RMSE | MAE | Bias |
|---|---:|---:|---:|
| Hybrid with lagged sources | 1.753399 | 1.047213 | +0.004916 |
| Same hybrid + 8 SST PCs | 1.752762 | 1.048663 | +0.017217 |

Units: mm/day. SST reduced RMSE by 0.0363% but worsened MAE and bias; it remains experimental.

- [Paired experiment](lagged_sources/README.md)
- [Prospective registry](prospective/README.md)
- [Final baseline fitting and inference](prospective/BASELINE_RUN.md)
- [Spatial U-Net experiment](spatial_unet/README.md)
- [First U-Net pilot: results and execution record](spatial_unet/PILOT_RESULTS.md)
- [Complete U-Net comparison: seven blocks and execution evidence](spatial_unet/RESULTS.md)
- [Error maps, rainfall intensity, seasonality and training diagnostics](diagnostics/README.md)
- [Unified evaluation workflow and standalone Kaggle notebook](workflow/README.md)
- [One fixed 75% hybrid / 25% U-Net combination](fixed_blend/README.md)
- [Prepared additional chronological evaluation: 2021–2022](temporal_extension/README.md)
- [Earlier research results](../docs/RESEARCH_HISTORY.md)

The operational hybrid was fitted and frozen on 29 September 2026. October issuance is still blocked by incomplete CFSv2 inputs; see the [operational status](prospective/OPERATIONAL_STATUS.md). No real prospective forecast has been emitted. The U-Net completed all seven development blocks: RMSE was 1.764726 versus 1.753399 for the hybrid, with no improved block. A subsequent single fixed 75% hybrid / 25% U-Net blend reached 1.750628, improving six blocks but worsening global absolute bias. It remains a research candidate; the hybrid remains the operational reference. Both diagnostics used the archived maps without retraining.

Freeze hypotheses and configurations before new evaluation. Compare identical dates, grid points and information cutoffs, inspect temporal/regional behavior, and record decisions even without improvement. Small gains on consulted years do not establish future skill.
