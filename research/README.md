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
- [Earlier research results](../docs/RESEARCH_HISTORY.md)

The baseline has been evaluated historically. Final fitting, source adapters and inference are implemented; a real prospective run still needs its fitted package and complete inputs. No real forecast has been emitted by the implementation or synthetic checks. The U-Net workflow is a separate development experiment, without a claimed improvement.

Freeze hypotheses and configurations before new evaluation. Compare identical dates, grid points and information cutoffs, inspect temporal/regional behavior, and record decisions even without improvement. Small gains on consulted years do not establish future skill.
