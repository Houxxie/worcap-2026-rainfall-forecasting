# Lagged sources and global SST

The [notebook](lagged_sources.ipynb) compares the baseline with eight SST PCs appended to both LightGBMs. Dates, sampled points, targets, ridge and blend weights are identical between treatments.

Atmosphere T−4, indices T−3, SST T−2, seasonal initialization T−1. Each training window ends four months before its first target, contains at most 360 months and begins no earlier than March 1982. Seven 24-month blocks cover 2007–2020; models stay fixed within each block while input fields update monthly.

Each LightGBM uses 300 trees, 31 leaves, learning rate 0.05, seed 42 and 5,000 points per training month. Residual scales are 0.9 and 0.875. Ridge uses penalty 0.1 on mean squared error. Blend weights are 0.375, 0.375 and 0.25.

PCA uses eight components, full SVD, monthly anomalies and square-root cosine-latitude weighting, without per-pixel variance scaling. All transformations fit within training. Rainfall climatology and residual labels exclude each training observation from its own mean.

## Run

In Kaggle, attach official data, seasonal snapshots and `ersstv5_4graus_198201_202411.nc`. Use the [reference environment](requirements.txt). Run the notebook in order: materialize code, run synthetic temporal checks, train paired models and export reports. Inputs already attached require no token or downloads.

Output: `/kaggle/working/rainfall_lagged_sources`. Resume requires identical code, input and environment hashes. The English code starts separately from historical runs.

```bash
python research/lagged_sources/test_temporal_contract.py
```

## Evidence

Archived baseline RMSE: **1.753399**; SST: **1.752762**. MAE and bias worsened, so the baseline remains. Evaluation includes 78,561 cells × 168 months = 13,198,248 errors per model. RMSE is `sqrt(sum(SSE)/sum(n))`, not the mean of block RMSEs.

Original tables under `evidence/` retain compatibility fields: `modelo` = model, `mes_alvo` = target month, `ano` = year, `bloco` = block, `regiao` = region, `vies` = bias, `soma_erro` = sum of errors, `soma_erro_absoluto` = sum of absolute errors. `controle_defasado` is the baseline; `sst_defasada` is the candidate.

These development years were already consulted. The experiment tests sensitivity to an assumed information calendar, not verified historical operational vintages or future skill. See [temporal validity](../../docs/TEMPORAL_VALIDITY.md).
