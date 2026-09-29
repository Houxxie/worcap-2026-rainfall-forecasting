# WorCAP 2026 · Rainfall Forecasting

Monthly rainfall forecasting over South America, combining seasonal climate forecasts with gradient boosting and local ridge regression.

I started this project during the **WorCAP 2026 Hackathon**, where it finished in the **Top 10**, with an official final score of **1.80114**. This repository presents the hybrid model submitted for the competition and the research I continued afterwards.

![Model overview: atmospheric and ocean inputs feed two LightGBM components and a local ridge model, combined into a monthly rainfall forecast.](assets/model_overview.png)

*Architecture of the competition model. The diagram shows the calculation, not a forecast map.*

## What it does

The model predicts **monthly mean daily precipitation, in mm/day**, on a 0.25° grid covering South America. It learns rainfall anomalies relative to a local monthly climatology and combines three components:

| Component | Inputs | Share of the final forecast |
|---|---|---:|
| LightGBM · seasonal | Atmosphere, ocean indices, location, seasonality and SEAS5 | 37.5% |
| LightGBM · multisystem | The same inputs, plus CFSv2 | 37.5% |
| Local ridge regression | SEAS5 and CFSv2 forecast anomalies at each grid point | 25% |

Training uses up to 30 years of history. Climatologies and transformations are fitted within each training window. The [explained notebook](competition/hybrid_forecast.ipynb) covers the data, anomaly construction, validation, fitting and final CSV export.

## Technologies

Python · NumPy · pandas · xarray · LightGBM · scikit-learn · Matplotlib · NetCDF/GRIB · Kaggle

## Run the competition model

1. Open [hybrid_forecast.ipynb](competition/hybrid_forecast.ipynb) in Kaggle.
2. Attach the official competition data and the prepared SEAS5 and CFSv2 snapshots listed in the [data guide](docs/DATA.md).
3. Use the [reference environment](competition/requirements.txt) and run the cells in order.
4. The notebook fits the components and writes `submission_hibrida.csv`, checking its IDs, grid coverage and expected SHA-256.

The same implementation is available as a [Python script](competition/hybrid_forecast.py). Raw datasets and fitted models are not stored in Git. Exact reproduction requires the recorded snapshots; downloading a revised dataset today may not reproduce the same bytes. See [reproduction instructions](docs/REPRODUCTION.md).

## The process

I began with a climatology-based model, then added seasonal forecasts and their anomalies to represent departures from normal rainfall. The final hybrid combines nonlinear tree models with a simpler local regression. I evaluated it with seven chronological validation blocks spanning 2007–2020, rather than randomly mixing months.

After the competition, the focus shifted from the leaderboard to future forecasts. That raised a separate question: **was each input actually available when the forecast would have been issued?** The research version tests longer source lags and records when new files arrive.

The [research folder](research/README.md) contains the current baseline, a paired SST/PCA experiment and the prospective data registry. This is still a research project; the final prospective model package and complete input pipeline are not yet integrated.

## What I learned

- **Time-aware validation.** Chronological splits are only part of the problem: observation month, forecast initialization and publication time are different dates.
- **Climatologies and anomalies.** Seasonal reference values make rainfall patterns easier to model. Excluding a training observation from its own climatology avoids leaking it into the residual target.
- **Working with climate data.** Combining NetCDF and GRIB products requires checking units, calendars, coordinates, ensemble members and interpolation boundaries.
- **Combining different models.** A local linear model can contribute useful information alongside gradient boosting. Its value has to be checked in the combined prediction.
- **Evaluating beyond one score.** RMSE, MAE, bias and results by year or region reveal trade-offs that an aggregate ranking can hide.
- **Reproducibility.** Fixed configurations, input hashes and explicit data snapshots matter. Even CSV rounding order can affect exact reproduction.

## Repository guide

| Location | Contents |
|---|---|
| [`competition/`](competition/README.md) | Explained hybrid notebook, script and reference metadata |
| [`research/`](research/README.md) | Current lagged-source comparison and prospective registry |
| [`docs/`](docs/REPRODUCTION.md) | Reproduction, sources, temporal limitations and research history |
| [`scripts/`](scripts/README.md) | Repository checks, notebook builds and diagram generation |
| [`CHANGELOG.md`](CHANGELOG.md) | Changes to the project |

The original competition delivery is preserved under the `competicao-2026` Git tag. The current branch contains the English presentation of the hybrid and the active research code. [Version notes](docs/VERSIONS.md) explain the distinction.

## License and acknowledgements

Code is available under the [MIT license](LICENSE). Data providers retain their own terms; see [data sources](docs/DATA.md) and [NOTICE.md](NOTICE.md).

Developed by [Houxxie](https://github.com/Houxxie), with AI assistance in implementation, debugging and documentation. Thanks to the WorCAP organizers and the ECMWF, NOAA and IRI teams for the challenge and data products.
