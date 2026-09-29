# Spatial rainfall experiment

A compact U-Net predicts a monthly rainfall-anomaly map over South America. This is the first spatial neural-network experiment in this project. **No measured improvement is claimed yet.**

The question is whether neighboring atmospheric and seasonal-forecast patterns help beyond the existing combination of two LightGBMs and local ridge regression. No new data source or SST component is added in this experiment.

## What changes

The network receives 31 channels on the original 301 × 261 grid: nine atmospheric fields and their monthly anomalies, latitude, longitude, month sine/cosine, rainfall climatology, four ocean-index anomalies, and the raw/anomaly SEAS5 and CFSv2 forecasts. Global indices are broadcast across the map. The output is one continuous anomaly, reconstructed as `max(0, training climatology + anomaly)` in mm/day.

Two encoder stages, a bottleneck and two decoder stages use 16, 32 and 64 channels. Skip connections retain spatial detail. Group normalization supports small batches; bilinear upsampling targets each skip map's exact dimensions, including odd grid sizes. No geographic flips, random spatial split, pretrained vision weights, SST, or tuned blending weights are used.

The architecture adapts the encoder/decoder idea from [Ronneberger et al. (2015)](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) to continuous regression. It is not the original biomedical segmentation model.

## Fair comparison and its limits

Both methods receive the same source information, target months, grid and outer training cutoff. Atmosphere uses T−4, ocean indices T−3, and seasonal initialization T−1. Training rainfall ends at T−4 before the first forecast in a block. At most 360 monthly training maps are available.

The U-Net trains on complete maps; the fixed tree baseline retains its original 5,000 sampled pixels per month. This is a comparison of modeling approaches, **not an identical-computation or identical-pixel-sample ablation**. Spatial pixels are not independent monthly climate realizations.

All climatologies and normalization are fitted within training. The rainfall climatology supplied for a training example excludes that example's entire target map, including neighboring pixels. Validation and forecast maps use the complete training climatology.

## Chronological selection

1. Each outer block spans two years: 2007–2008 through 2019–2020.
2. The last 24 available training target months form an inner validation period. Inner fitting ends four months before its first target. Its reference history is the outer reference truncated at that earlier cutoff, never extended into older unavailable inputs.
3. Up to 40 epochs are run with AdamW, learning rate 0.001, weight decay 0.0001, batch size 2 and patience 6. Epoch count is chosen by inner reconstructed RMSE, with earliest-epoch tie breaking.
4. A fresh model and preprocessing are fitted on the complete outer training window for that epoch count. Outer errors never choose the epoch count.
5. The unchanged hybrid is refitted and must reproduce its archived block RMSE, MAE and bias within 1e-6 before the comparison is accepted.

These historical years were already consulted during project development. They are **not an untouched holdout**, and successful historical results alone cannot establish performance in future years. The existing prospective hybrid plan remains unchanged; a future neural forecast series would need its own frozen protocol.

## Run in Kaggle

Import `spatial_unet.ipynb` into a separate notebook. Enable a GPU and attach the official training files, prepared SEAS5 and prepared CFSv2 datasets. The same inputs used by the lagged-source experiment are sufficient; SST is not required. Input hashes are checked before training, and test partitions for 2023–2024 are never opened.

The notebook contains all source modules and metadata. No private GitHub authentication is needed. Automatic input discovery requires a single copy of each source; explicit directory settings are available if duplicates exist.

Run the synthetic tests and input preflight first. The default is **H1 only**, to establish runtime, memory use and successful control reproduction. It still uses the full prespecified H1 protocol. Set `BLOCKS` to all seven identifiers for the complete comparison; completed blocks can be reused only with matching input, code and environment hashes. A one-block result is labeled as a pilot and does not imply overall improvement.

Temporary feature maps use approximately 4 GiB per training stage and are removed after that stage. Save the resulting output version: it contains weights, normalization, reference maps, training curves, selected epochs, source calendars, forecasts and errors by month, year, block and latitude band. No forecast is automatically promoted or submitted.

## Local commands

Python 3.12, the pinned baseline environment and PyTorch are required. The implementation was tested with PyTorch 2.8.0 on CPU; use a compatible GPU build for Kaggle. GPU kernels and package versions can change numerical results; the full environment is recorded. See [PyTorch's reproducibility notes](https://docs.pytorch.org/docs/2.8/notes/randomness.html).

```bash
python research/spatial_unet/test_spatial.py
python research/spatial_unet/run_experiment.py --output outputs/spatial_unet_v1 --blocks H1 --official /path/to/official --seas5 /path/to/seas5 --cfsv2 /path/to/cfsv2
```

Primary metric: pooled RMSE, calculated from error sums rather than averaged block RMSE. Secondary metrics: MAE, signed bias, cosine-latitude weighted RMSE, annual results and three latitude bands. The comparison includes the hybrid, U-Net and climatology, without searching for an ensemble weight.

Sources and constraints: [data documentation](../../docs/DATA.md), [temporal validity](../../docs/TEMPORAL_VALIDITY.md), and the machine-readable [protocol](protocol.json).
