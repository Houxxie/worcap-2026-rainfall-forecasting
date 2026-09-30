# Residual U-Net pilot: completed, not adopted

The C-block pilot evaluated January 2019–December 2020. It completed in **692.7 seconds** with eight completed stages, six hybrid fits and two eight-epoch neural fits. `pilot_complete` is true; `complete_seven_blocks` remains false because only one block was planned.

| Model | RMSE | MAE | Signed bias |
|---|---:|---:|---:|
| Fixed hybrid | 1.740802 | 1.030739 | +0.162351 |
| Matched direct U-Net | 1.752793 | 1.042738 | +0.201477 |
| Fixed 75% hybrid / 25% direct U-Net | 1.738480 | 1.030046 | +0.172132 |
| Hybrid corrected by residual U-Net | 1.755200 | 1.043702 | +0.224886 |

Units: mm/day. The residual candidate increased RMSE by **0.014399**, worsened both evaluated years and improved only 9 of 24 monthly RMSE values. Its mean applied correction was positive while the hybrid already had a positive aggregate bias. Training loss decreased, but only final checkpoints were retained: this does not establish which validation epoch would have been best.

The pilot used the same 31 inputs and matched training settings for the two neural targets. Residual targets came from five chronological hybrid fits, each with its own earlier 30-year reference and four-month label gap. The final hybrid remains unchanged. These are consulted development years, not evidence of generalization to future years. The result does not rule out every residual architecture, but it does not support expanding this configuration to seven blocks.

The supplied `results (1).zip` has SHA-256 `1f65aae29e508941f1174f5ce3b40cf3827aa4e226d36a7dfede4f515366b202`. Review verified 69 artifact hashes, 19 source files, shared scalers, cross-fit chronology and prediction identities; pooled metrics were recomputed from saved monthly error sums. Raw competition observations were not included, so independent recomputation from raw truth was not performed. The full ZIP and maps are retained locally; the small evidence subset is under `evidence/completed_20260930/`.

The run's `progress.json` was stale and still reported running. Completed stage markers, `run_state.json`, predictions and metric checks establish completion. This reporting defect does not change the saved scores.

Decision: keep this configuration as a completed negative experiment and proceed to the fixed SST temporal extension. No operational promotion or new neural fit follows this pilot.
