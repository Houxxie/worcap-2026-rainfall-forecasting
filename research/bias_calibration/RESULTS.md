# Completed global bias-calibration experiment

All seven development blocks completed. The uploaded reports ZIP has SHA-256 `b8977b6c2670b1923df209be151d169950d6a68f5edcff64923b2bdf9a115e7d`. Both completion records report `status: complete` and `complete_seven_blocks: true`.

| Full-grid 2007–2020 metric | Original 75/25 blend | Bias-corrected blend |
|---|---:|---:|
| RMSE (mm/day) | 1.750628042478 | 1.750622525262 |
| MAE (mm/day) | 1.046127917861 | 1.046806894829 |
| Signed mean error (mm/day) | +0.008991573526 | +0.014264277901 |
| Area-weighted RMSE (mm/day) | 1.829104138688 | 1.829110442596 |

The RMSE reduction is only 0.000005517216 mm/day (about 0.000315%). MAE, absolute global bias and area-weighted RMSE worsen. RMSE improves in three of seven blocks and six of fourteen years; blocks A, B and C all worsen. The standalone corrected U-Net also has worse pooled RMSE than its uncorrected version.

**Decision:** do not adopt this global correction. Retain the uncorrected blend as a research candidate and keep the current forecasting model unchanged. This result concerns the specified transfer of an earlier auxiliary model's global bias; it does not reject every neural calibration or residual-learning method. The next separate hypothesis is [learning chronological reference model residuals](../residual_unet/README.md), starting with a bounded pilot.

The selected [report tables and verification record](evidence/completed_20260930/) retain the observed outcome. Locally, 154 included fold artifacts matched their recorded hashes, all 20 source files matched the notebook code (allowing line-ending normalization), all seven reference model checks reproduced, the calibration dates and offsets were checked, and pooled metrics were recomputed from 168 monthly full-grid records per model. The monthly aggregate was checked against the union of the seven fold tables.

The small ZIP omits 182 referenced model/data artifacts. Their hashes are recorded in the full run, but those absent files were not independently verified here. Raw observations and prediction maps were not recomputed from this ZIP. The years were previously consulted development data, not an independent holdout. No 2021–2022 rescore, live forecast or model promotion occurred.
