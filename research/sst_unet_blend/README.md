# Global SST and U-Net: one fixed combination

This notebook evaluates one candidate using **saved prediction maps** for January 2021–December 2022:

```text
Candidate = 0.75 × hybrid with 8 SST PCs + 0.25 × existing U-Net
Reference = 0.75 × original hybrid       + 0.25 × the same U-Net
```

It tests whether the SST contribution persists in the existing blend. Comparing the candidate with the SST hybrid alone also shows whether the U-Net adds anything after SST is present. There are **zero model fits**, no search over weights and no source-provider downloads. Use CPU, with the accelerator disabled. Runtime is mostly file checks, NetCDF I/O and scoring, not training.

These years have already been evaluated in both source experiments. This is a retrospective development comparison, **not an independent holdout**. Source models were fitted through September 2020 with the existing lag policy; historical publication vintages remain unverified. No operational model is replaced automatically.

## Inputs: observations and two saved prediction files

Only these three files are needed. You do not need to attach raw SST, SEAS5 or CFSv2 for this test.

| Input | Where to find it | SHA-256 prefix |
|---|---|---|
| `treino_tp.nc` | Original competition input; use the unshifted rainfall file | `012bebbe0e38` |
| SST `predictions.nc` | Complete output of the SST extension: `sst_temporal_extension/results/models/predictions.nc` | `8888ce1301a3` |
| U-Net `predictions.nc` | Complete output of the earlier hybrid/U-Net extension, Kaggle version **354026629** | `5e5341220a58` |

The U-Net run is [available here](https://www.kaggle.com/code/houxie/worcap-hybrid-and-u-net-temporal-extension/output?scriptVersionId=354026629). Use its complete output, not `temporal_extension_reports.zip`. Likewise, `sst_temporal_extension_reports.zip` does **not** contain the SST maps.

Import this notebook into a new Kaggle notebook. Add the competition data and the two saved notebook outputs through **Add Input → Notebook**, selecting the appropriate saved versions. Alternatively, download each `predictions.nc` and upload them in separate folders within a private dataset. Both files share the same basename; keep them in different folders. If renamed, set their complete file paths in the configuration cell.

Automatic discovery searches `/kaggle/input` and `/kaggle/working` and checks the **full hashes**, not just filenames. Identical copies are harmless; a missing or different run stops the notebook before calculation. A sidebar entry containing only notebook HTML/code is not sufficient.

## Checks and computation

1. Verify both source map hashes and the original observation hash against the archived receipts.
2. Require all 24 months, the same 301 × 261 grid, correct units and finite nonnegative rainfall. Reject reordered or mismatched coordinates.
3. Require identical hybrid and climatology maps across both experiments. Reconstruct the archived reference blend exactly.
4. Form the candidate in float64 from the already reconstructed rainfall maps, with no additional clipping or correction. Save it and check an exact readback before opening evaluation labels.
5. Reproduce the archived hybrid, SST-hybrid, U-Net, climatology and reference-blend scores. Then compare the new candidate globally, annually, monthly and by latitude band.
6. Check both equivalent fixed-blend MSE identities using the paired error products. Error correlation is descriptive; no optimized weight is computed.

The report includes RMSE, MAE, signed/absolute bias differences and area-weighted RMSE. A lower aggregate RMSE alone does not imply that every year, month or region improved. Preserve the full output for later diagnostics.

## Run and return

Select **Run All**. The synthetic tests in the first section check implementation behavior and do not measure forecast skill. The next section locates the three inputs; the final section evaluates the real saved maps.

Completion is reported as `complete_comparison: true`. `complete_seven_blocks: false` is expected because this notebook evaluates one 24-month extension. Download **`sst_unet_blend_reports.zip`**. It contains the small reports, code snapshot and checks; full candidate maps remain in the saved output.

If a completed run is found with the same inputs/code/environment, the notebook verifies and reuses it. An interrupted output is preserved; select a fresh output directory before retrying.
