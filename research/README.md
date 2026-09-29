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
- [Earlier research results](../docs/RESEARCH_HISTORY.md)

The baseline has been evaluated historically. Its final prospective inference package still needs fitting and integration with complete inputs. No real forecast has been emitted.

Freeze hypotheses and configurations before new evaluation. Compare identical dates, grid points and information cutoffs, inspect temporal/regional behavior, and record decisions even without improvement. Small gains on consulted years do not establish future skill.
