# Residual U-Net: a bounded pilot

Can a U-Net improve a rainfall forecast by learning the **remaining error of the hybrid**, instead of predicting the full rainfall anomaly independently?

```text
Training target = observed rainfall - chronological hybrid forecast
Prediction = maximum(0, final hybrid forecast + U-Net correction)
```

The output layer starts at zero, so the residual network initially reproduces the hybrid. Both positive and negative corrections remain learnable. The existing 31 source-feature maps are the neural inputs; the hybrid is the reconstruction base, not an additional input channel. The network therefore learns a correction from source features, rather than directly receiving the hybrid map as a 32nd feature.

## A smaller first test

- **One block only:** C, January 2019–December 2020, already consulted development data.
- **Two neural fits:** one direct-anomaly control and one residual candidate, each with **eight fixed epochs** on the same **120 monthly training maps** (October 2008–September 2018).
- Identical architecture, 31 input channels, preprocessing, seed, minibatch order, optimizer and compute budget. No checkpoint selection, weight search or new external source.
- Six hybrid fits: five chronological fits to produce training residuals and one unchanged final reference fit.
- No automatic extension to seven blocks, no 2021–2022 evaluation and no operational promotion.

This explicitly reduces neural training history and fixes the epoch count to bound computation. The direct control is newly trained under the same budget; its scores must **not** be presented as reproducing the earlier full-history U-Net. Eight epochs are a pilot budget, not a claim that either network has converged.

The choice of C keeps the complete 30-year climatologies required by the existing hybrid. Generating ten years of earlier hybrid forecasts for H1 would require observations before the available rainfall archive. This pilot rejects missing history rather than shortening, backfilling or silently changing those climatologies. C is the latest original development block, not an untouched holdout or a substitute for broad evaluation.

## Preventing training-error leakage

Split the 120 neural training months into five consecutive 24-month groups. Each group receives hybrid forecasts from a separate fit whose last rainfall label is **four months before the group's first target**. All hybrid coefficients and references use only that fit's permitted history. Later groups can use observations from earlier groups, as in chronological validation.

The network learns `observation - out-of-fit prediction`. It never learns the residual of a hybrid that already fitted that target observation. Earlier hybrids have shorter common seasonal histories than the final hybrid, although all retain 30-year climatologies; transfer of their error patterns is a hypothesis, not guaranteed.

Both networks share input maps and scaling constructed from permitted outer-training data. As in the existing U-Net, the training rainfall climatology feature excludes the entire target map. Input preprocessing can use the outer training window; it cannot use outer evaluation rainfall. All candidate maps are saved and hashed before scoring. The loader holds the historical archive in memory, so this is code-enforced date separation, not a claim of physically inaccessible observations or independent timestamps.

## Comparisons

The report contains the unchanged hybrid, the new direct U-Net control, its fixed 75% hybrid / 25% direct-U-Net combination, and the residual candidate. The residual candidate applies its learned correction at weight one; no correction strength is searched.

Compare residual versus hybrid first, then residual versus the matched direct control and its fixed blend. Inspect RMSE, MAE, absolute bias, the two years separately and latitude bands. The full grid includes ocean. A positive result would justify a subsequent, separately specified evaluation; one block cannot establish future performance. A poor result would reject this pilot configuration, not every residual architecture.

## Run on Kaggle

1. Import **`residual_unet_pilot.ipynb`** into a new private notebook.
2. Attach the same three Inputs: official competition data, `worcap-seas5-dados`, and `worcap-cfsv2-dados`. The development NetCDFs and manifests are required. No earlier model output is needed.
3. Enable a **GPU**. Use the same baseline environment: NumPy 2.0.2, pandas 2.3.3, xarray 2025.12.0, LightGBM 4.6.0. PyTorch and NetCDF readers must be available. The notebook does not install packages or download sources.
4. Use **Save Version → Save & Run All**. Keep the pilot budget unchanged for this comparison.
5. Download **`residual_unet_pilot_reports.zip`** and preserve the full output with weights and prediction maps.

The successful endpoint is **`pilot_complete: true`** and **`status: complete`**. **`complete_seven_blocks: false` is expected**, because this notebook intentionally runs only C.

Based on the earlier T4 timings, an initial planning estimate is **15–30 minutes**, including chronological hybrid fits and input preparation. This is not a measured runtime for this new notebook or a guarantee. Stage timings and each of the eight neural epochs are printed. The code fails if no GPU is available instead of silently using a slow CPU.

Completed stages are reused only after signature and artifact-hash checks. An interrupted stage is preserved and causes an explicit stop; select a fresh output directory for a clean retry. After a session reset, attach the full saved output and set `PREVIOUS_OUTPUT` to the results directory containing `signature.json`. A reports-only ZIP cannot resume training.

## Implementation checks

Fourteen synthetic tests passed locally and from the notebook's extracted source bundle in an isolated directory. They cover chronological cutoffs, rejection of insufficient history, identical inputs for the direct and residual targets, exact hybrid reconstruction with a zero correction, nonnegative rainfall, future-target isolation, artifact integrity, and a tiny-grid end-to-end fit. The [validation record](evidence/validation.json) documents their scope. They do not estimate real forecast quality. Real Kaggle pilot results are pending.
