# Training-only U-Net bias calibration

This is one new **development experiment**, not a replacement for the current forecasting model. It compares the reference model, the original fixed 75% reference model + 25% U-Net blend, and the same blend after a simple correction of the neural contribution. The unchanged component models and climatology are also reported.

## What changes

Only code and newly fitted model artifacts change. Use the same official competition dataset and the same audited SEAS5 and CFSv2 datasets. There are no downloads, new sources, SST features, credentials or Internet requirements. The neural architecture, features, original epoch selection and blend weights stay fixed.

For each of the seven original two-year blocks (2007–2020):

1. Refit the original reference model and U-Net using the original procedure. The reference model must reproduce its archived metrics before proceeding. These regenerated components are shared by both blend candidates.
2. Reserve the last 24 months of that outer training window for calibration. An auxiliary U-Net is trained only on earlier months, respecting the four-month label gap.
3. Select the auxiliary network's epoch count using a further, earlier chronological split. **The 24 calibration months do not choose those epochs.** Refit the auxiliary network from the original seed before predicting the calibration months.
4. Compute the mean prediction-minus-observation error across each complete monthly map, then average those 24 monthly errors. Apply a fixed 50% shrinkage toward zero, equivalent to 24 prior zero-bias months. This regularization strength is a prespecified experimental choice, not an optimized value.
5. Subtract that signed scalar from the original outer U-Net forecasts, clipping rainfall at zero, and combine with the unchanged reference model using 75/25 weights. No observation from the outer block enters the scalar.

```text
offset = mean(calibration prediction - calibration observation) / 2
corrected U-Net = maximum(0, original outer U-Net - offset)
candidate = 0.75 * original reference model + 0.25 * corrected U-Net
```

The correction is global and constant throughout each outer block. It can be positive or negative; no month, pixel or region selects its own adjustment. Bias from an earlier model with a shorter training history may not transfer to the final refitted model. This is the hypothesis being tested. Clipping also means the final mean shift need not equal the scalar exactly.

The original outer network retains its original inner selection, including training-window months later used by the auxiliary calibration procedure. This is allowed training-data reuse. The auxiliary forecasts used to estimate the offset are generated with **neither fitting nor epoch selection on their target months**. No outer target selects a correction, epoch, weight or winner.

## Run the notebook

1. Import [unet_bias_calibration.ipynb](unet_bias_calibration.ipynb) into a **new private Kaggle notebook**.
2. Attach the same three Inputs: the official precipitation competition dataset, `worcap-seas5-dados`, and `worcap-cfsv2-dados`. Manifest files and development NetCDF partitions are required. Previous prediction ZIPs are unnecessary.
3. Enable a GPU (the previous T4 configuration is suitable). Keep the same environment: NumPy 2.0.2, pandas 2.3.3, xarray 2025.12.0, LightGBM 4.6.0, with PyTorch and NetCDF readers. No package is automatically installed or upgraded.
4. Use **Save Version → Save & Run All** so the output is preserved. Leave the seven-block list fixed. The notebook prints each fit and epoch and writes a partial report after each completed block.
5. Download **`bias_calibration_reports.zip`** when all seven blocks finish. The final summary must say `complete_seven_blocks: true`. Keep the full notebook output as well: it contains weights, calibration predictions and candidate maps omitted from the small ZIP.

This is more work than the single 2021–2022 extension: **four neural fits per block, 28 neural fits in total**, plus the original reference model fits. The two extra fits per block isolate calibration from epoch selection. Actual duration depends on early stopping and the GPU; no short runtime is promised.

## Resume and safeguards

In the same session, rerunning the execution cell uses completed blocks only after checking their signatures and every saved artifact. An interrupted block is retried in a new attempt folder; its incomplete attempt is preserved. Inputs, code and environment must match. If they changed, choose a fresh output directory.

After a session reset, attach the **full saved notebook output**, not just the small reports ZIP, and set `PREVIOUS_OUTPUT` to its directory containing `signature.json`. The notebook copies it to working storage and verifies completed blocks before resuming. Do not point this variable at another experiment.

The underlying development loader reads the historical rainfall archive. Selection and calibration explicitly use only permitted dates. The unchanged reference runner records its already-known outer scores before calibration; those scores are never calibration inputs. This workflow does **not** claim that all outer observations remained physically unopened. Candidate hashes provide integrity records, not independent timestamps.

## How to read the result

Primary comparison: corrected blend versus the original blend on pooled full-grid RMSE. Also compare against the reference model, and inspect MAE, absolute bias, annual and block stability, area-weighted RMSE and latitude bands. The grid includes ocean. A lower overall RMSE with worse bias or inconsistent years is a tradeoff, not automatic acceptance.

The 2007–2020 years are development data already inspected. The 2021–2022 findings motivated this hypothesis; those years are neither used to estimate the offsets nor rescored here. This comparison cannot establish future skill. Keep the operational model unchanged until a separate prospective evaluation supports a change.

Seventeen synthetic checks passed from the notebook's extracted source bundle in an isolated directory. They cover chronology, future-observation isolation, target-free prediction equivalence, clipping, alignment, nested fitting, saved-block integrity, and resuming after an interrupted block. The [validation record](evidence/validation.json) records their scope. They verify implementation; they do not measure forecast quality. No real calibration run has been performed during notebook preparation.
