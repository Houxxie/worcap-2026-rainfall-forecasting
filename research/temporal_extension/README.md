# Additional chronological evaluation: 2021–2022

This notebook evaluates the unchanged lagged hybrid, compact U-Net and fixed **75% hybrid + 25% U-Net** combination on one additional two-year block. It tests whether the development improvement persists in another period, without searching weights or changing architecture.

**Status: prepared for Kaggle; the real 2021–2022 experiment has not been run.** The initial local checks use synthetic data and do not measure forecast skill.

## What this evaluation can establish

The records reviewed for this step contain development scores for 2007–2020, competition feedback for 2023–2024, and a separate retrospective 2025 evaluation. The 2021–2022 observations were already used by earlier final training runs. No separate 2021–2022 comparison was found in those reviewed records; absence of a recorded score does not prove the period was never inspected.

For that reason this is an **additional retrospective chronological evaluation, not an untouched independent holdout**. Freezing predictions before scoring is an execution safeguard, not proof that historical targets were unknown to the project. A convincing future-skill assessment still requires prospectively frozen predictions with verified source availability.

The existing workflow deliberately supports its seven original blocks only. This extension has a separate locked adapter and notebook so that adding a period cannot silently reinterpret the original run or bypass its baseline-reproduction checks.

## Fixed experimental design

| Item | Setting |
|---|---|
| Evaluation | January 2021–December 2022, 24 months |
| Outer training and rainfall reference | October 1990–September 2020, 360 months |
| Inner epoch-selection validation | October 2018–September 2020 |
| Inner training/reference cutoff | June 2018; four-month label lag |
| Atmospheric inputs | T−4 |
| Ocean indices | T−3 |
| Seasonal initialization | T−1, from the existing audited SEAS5/CFSv2 snapshots |
| Network | Original channels 16/32/64, seed 42, AdamW, maximum 40 epochs, patience 6 |
| Final neural fitting | Fresh initialization; epoch count selected only inside the outer training window |
| Hybrid | Same two LightGBMs and local ridge, refitted on the stated outer window |
| Combination | Exactly 75% hybrid + 25% U-Net; no new clipping or recalibration |
| Primary comparison | Pooled full-grid RMSE of the blend against the refitted hybrid |

All models remain fixed throughout the 24 evaluation months. Source fields update for each target month according to the lags. The final operational model trained through December 2022 is **not** reused for this earlier block. The original [U-Net protocol](../spatial_unet/protocol.json) is hash-pinned by the [extension protocol](protocol.json).

## Run in Kaggle

1. Import [temporal_extension.ipynb](temporal_extension.ipynb) into a **new private notebook**.
2. Attach the official competition dataset, `worcap-seas5-dados`, and `worcap-cfsv2-dados`. Each seasonal input needs its manifest and both `desenvolvimento` and `somente_ajuste_final` NetCDF files. The `teste` partitions and SST are not used. The seven old model outputs are unnecessary because this experiment refits the models with a new cutoff.
3. Enable a GPU. Use the original baseline environment: NumPy 2.0.2, pandas 2.3.3, xarray 2025.12.0 and LightGBM 4.6.0. PyTorch and the NetCDF readers must also be present. Preflight stops on incompatible baseline versions or unavailable CUDA; it does not replace packages automatically. Actual PyTorch/GPU versions are recorded.
4. Run cells in order. The check cell runs small synthetic tests and checks the input hashes. The fitting cell trains the hybrid, selects neural epochs internally, refits the network and freezes predictions. The evaluation cell then verifies that package before decoding the target rainfall.
5. Read `report.html`, download `temporal_extension_reports.zip`, and save the notebook version **with all outputs**. The small ZIP contains reports and verification metadata; the full saved output also retains models and prediction maps.

No Internet, CDS key, new downloads or GitHub login are needed when the prepared datasets are attached. This step performs one new outer block, with one inner neural fit and one final neural fit. It can take longer than evaluation of saved maps. The model prints each epoch's timing and progress.

If auto-discovery finds duplicate inputs, set the three directory variables explicitly. Existing runs are not overwritten. A failed fitting run is preserved and needs a new output directory. If predictions froze successfully but the session ended before evaluation, use the restore cell to point to the complete saved run; do not refit it just to obtain the report. An interrupted evaluation leaves `opened.json` and must not be represented as unopened data.

## Local entry points

Use the [existing model dependencies](../spatial_unet/requirements.txt); avoid installing evaluation-only pins over a training environment.

```bash
python -m unittest research.temporal_extension.test_extension research.spatial_unet.test_spatial -v
python -m research.temporal_extension.experiment check --official /data/official --seas5 /data/seas5 --cfsv2 /data/cfsv2 --device cuda
python -m research.temporal_extension.experiment freeze --output /outputs/new_run --official /data/official --seas5 /data/seas5 --cfsv2 /data/cfsv2 --device cuda
python -m research.temporal_extension.experiment evaluate --output /outputs/new_run --observations /data/official/treino_tp.nc
```

The fit path decodes only the training rainfall window. Forecast inference uses an adapter with no target-rainfall access. Source files are still byte-hashed in full for provenance, including the shared rainfall archive; hashing does not decode those future values into model inputs. Scaling and climatologies are fitted inside the appropriate training window. Existing monthly-map inference is checked against the target-free adapter on synthetic data with nonzero network predictions.

## Artifacts and interpretation

`signature.json`, `plan.json`, `split.json`, `source_calendar.csv` and `code/` record the experiment before fitting. The model folders preserve parameters, scalers, references and histories. `predictions.nc` contains all four forecast fields; `frozen.json` records their verified package hashes before evaluation. The local record is not an independent timestamp or tamper-proof ledger.

The separate `evaluation/` directory contains pooled global, annual, calendar-month and latitude-band metrics, monthly error sums, the opening record, a plot and a portable HTML report. Lower RMSE alone does not hide worsened MAE or absolute bias. Two years provide limited temporal diversity, the supplied grid includes ocean, and consolidated source files do not establish historical publication vintages.

No result automatically replaces the operational hybrid, changes the ongoing source gate or issues a live forecast. Record the result once, including a negative result, before planning another hypothesis.

## Preparation checks

Fourteen synthetic checks passed from the notebook's extracted source bundle in an isolated directory. They cover the original spatial model and this extension, including target-free inference equivalence, training-window selection, rejection of modified frozen predictions, pooled-error metrics and complete report generation. See the [validation record](evidence/validation.json). Real-data preflight, GPU fitting and the 2021–2022 scores remain pending in Kaggle.
