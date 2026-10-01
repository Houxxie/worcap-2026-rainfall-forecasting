# Global SST: an additional chronological comparison

This notebook tests whether **eight global SST principal components** add information to the fixed rainfall reference model. It evaluates January 2021–December 2022, with training capped at September 2020. It does not fit a neural network.

## Why this experiment

The existing seven-block 2007–2020 comparison already used SST T−2 and training-only PCA. SST reduced pooled RMSE from **1.753399 to 1.752762 mm/day**, but increased MAE from 1.047213 to 1.048663 and signed bias from +0.004916 to +0.017217. There is little justification for repeating that run unchanged.

This extension keeps its SST method fixed and checks an additional period. **2021–2022 has already been consulted in the reference model/U-Net extension and earlier final fits. It is not an untouched holdout.** Results cannot establish future forecast skill.

## Data and temporal boundaries

Attach four Kaggle Inputs:

1. The original competition training files, including `treino_tp.nc` and the nine atmospheric files.
2. The prepared SEAS5 dataset, containing `seas5_51_manifesto.json` and its development and final-fit NetCDF partitions.
3. The prepared CFSv2 dataset, containing `cfsv2_manifesto.json` and the same two partitions.
4. The original SST snapshot: `ersstv5_4graus_198201_202411.nc`. SHA-256 must be `908034f6418833302ea432f25850ad2982d1114b747b7b298e26a350eaf2891d`.

The fourth file is the prepared NOAA ERSSTv5 4° subset, 60°S–60°N, rather than a new download of a possibly revised product. The source is [NOAA PSL ERSSTv5](https://psl.noaa.gov/data/gridded/data.noaa.ersst.v5.html). The archived NOAA index table is bundled and checked against the earlier snapshot. No API keys or Internet access are needed. CPU is sufficient; leave the accelerator disabled.

| Quantity | Rule |
|---|---|
| Rainfall training targets | October 1990–September 2020, exactly 360 months |
| Atmospheric fields | Target month T−4 |
| Ocean indices | T−3 |
| SST | T−2 |
| Seasonal initialization | T−1 |
| PCA fitting origins | August 1990–July 2020 |
| SST origins for evaluation | November 2020–October 2022 |

The ocean mask, monthly SST means, centering and PCA components are fitted only on training origins. Anomalies receive sqrt(cos(latitude)) weights; no pixelwise variance standardization is added. Missing selected ocean cells stop execution. Later rainfall labels are decoded only after both prediction files have been frozen.

These monthly lags are an availability assumption for retrospective evaluation. The archived products do **not** establish which historical revision was available on each issue date. This notebook neither emits an operational forecast nor changes the live source-receipt registry.

## Paired comparison

Both treatments use exactly the same sampled pixels, rainfall labels, base features and climatologies. Eight PC scores are appended only to the two LightGBM components: 29/31 predictors become 37/39. The local ridge is fitted once and shared. Each tree model keeps 300 trees, 31 leaves, learning rate 0.05 and seed 42; weights remain 0.375 / 0.375 / 0.25 and residual scales 0.9 / 0.875.

Budget: **one block, four LightGBM fits, one shared ridge, one PCA, zero neural fits**. Progress reports each tree component's fit and prediction time. Runtime has not been measured for this new notebook; input loading, hash checks and inference also take time.

The reconstructed reference model must match its archived 2021–2022 RMSE **1.7967836119** and the other recorded global metrics within 10⁻⁶. A mismatch is reported and blocks interpreting the run as a clean SST comparison.

## Running and reading the output

Import this notebook into a **new Kaggle notebook**, attach the four Inputs and select **Run All**. Use the same reference versions: NumPy 2.0.2, pandas 2.3.3, xarray 2025.12.0, LightGBM 4.6.0, scikit-learn 1.6.1 and SciPy 1.16.3. Versions and source hashes are recorded. Package versions are not upgraded automatically.

The first checks use synthetic fields: tiny grids for model fits and constant full-grid maps for output validation. Their printed scores are artificial. The next cell performs the real paired fit, and the last opens evaluation rainfall. Read `baseline_reproduced`, pooled RMSE, MAE, signed bias, both years and monthly/regional differences together. No weight search, PCA-size search or automatic adoption follows the score.

Download **`sst_temporal_extension_reports.zip`** and preserve the full Kaggle output. The small ZIP contains reports, source snapshots and hashes; model weights, PCA state and prediction maps remain in the full output. `complete_extension: true` describes this one 24-month extension; `complete_seven_blocks` is false by design.

An unchanged completed fit can be verified and reused. An incomplete run is preserved and requires a new `OUTPUT` directory; it is never silently overwritten.
