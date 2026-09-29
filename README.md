🌎 # WorCAP 2026 · Rainfall Forecasting

Monthly rainfall forecasting over South America, combining seasonal climate forecasts, gradient boosting, and local ridge regression.

This project started during the **WorCAP 2026 Hackathon**, where it finished in the **Top 10**. The hybrid model documented here scored **1.78758 in the 2024 evaluation**.

After the competition, I decided to keep working on it. This repository brings together the submitted model, an explanation of how it works, and the experiments I’m developing now.

![Overview of the rainfall forecasting model](assets/model_overview.png)

📝 ## About the project

The goal is to predict monthly rainfall patterns across South America using atmospheric data, ocean indices, and seasonal climate forecasts.

The model predicts **monthly mean daily precipitation, in mm/day**, on a 0.25° grid. A monthly climatology provides the reference, and the models estimate how much rainfall might differ from those usual conditions.

For the competition, I combined three components:

| Component | Main inputs | Final weight |
|---|---|---:|
| LightGBM · seasonal | Atmospheric variables, ocean indices, location, seasonality, and SEAS5 | 37.5% |
| LightGBM · multisystem | The same inputs, with CFSv2 added | 37.5% |
| Local ridge regression | SEAS5 and CFSv2 forecast anomalies at each grid point | 25% |

Training uses up to 30 years of data, with climatologies and transformations fitted within each training window.

The [competition notebook](competition/hybrid_forecast.ipynb) explains the full process, from loading the data to generating the final CSV.

💾 ## Technologies

- Python
- NumPy
- pandas
- xarray
- LightGBM
- scikit-learn
- Matplotlib
- NetCDF/GRIB
- Kaggle

🛠️ ## How I built it

The starting point was a model based on rainfall climatology. From there, I gradually added atmospheric and ocean information. Seasonal forecasts provided another way to represent the conditions expected for each target month.

Forecast anomalies became an important part of the model. These describe how much a prediction differs from the usual conditions for a particular location and time of year.

For the final hybrid, I combined two LightGBM components with a local ridge regression. Predictions were evaluated across seven chronological validation blocks covering 2007–2020, checking both the overall results and the differences between years.

After the competition, my focus shifted toward forecasts for future months. That meant looking more closely at when each data source would actually be available. I’m now testing longer input lags and recording when new files arrive.

The [research folder](research/README.md) contains this work: the current baseline, an experiment with global SST and PCA, and a registry of source arrivals. The complete pipeline for issuing future forecasts is still being integrated.

💡 ## What I learned

### Working with climate data

I became much more comfortable working with NetCDF and GRIB files. Combining different sources taught me to check units, coordinates, calendars, and ensemble members before using the data. It also made me pay closer attention to interpolation and what each dataset’s resolution actually represents.

### Thinking about time in a forecasting problem

One of the biggest lessons for me was separating the month a value describes from the date it becomes available. This changed how I approach validation: I now check both the training cutoff and whether the inputs could have been available when the forecast was made.

### Understanding climatologies and anomalies

Working with monthly climatologies helped me understand how to represent the seasonal rainfall cycle. From there, I learned how anomalies capture departures from that cycle, and why a training observation needs to be excluded from its own rainfall reference when constructing the residual target.

### Combining different models

I explored how a simpler local regression could contribute alongside gradient boosting. Evaluating the blend helped me understand how components can complement each other and why their individual scores only tell part of the story.

### Looking beyond a single metric

Over time, I started paying more attention to MAE, bias, and results by year and region. Looking at these together helped me spot trade-offs that were easy to miss when I focused mainly on the overall RMSE.

### Making my work reproducible

Reproducing the submission taught me to keep track of configurations, package versions, input files, and hashes. It also showed me how something as small as the order of CSV rounding can change the final file.

📝 ## Running the competition model

The notebook is prepared to run in Kaggle:

1. Import [hybrid_forecast.ipynb](competition/hybrid_forecast.ipynb).
2. Attach the official competition data and the prepared SEAS5 and CFSv2 snapshots listed in the [data guide](docs/DATA.md).
3. Use the [reference environment](competition/requirements.txt) and run the cells in order.
4. The notebook fits the model components and generates `submission_hibrida.csv`, checking the IDs, coverage, and expected SHA-256.

The same implementation is also available as a [Python script](competition/hybrid_forecast.py).

Raw datasets and fitted models are kept outside Git. Exact reproduction requires the recorded data snapshots, since providers may revise their files over time. The [reproduction guide](docs/REPRODUCTION.md) covers these requirements.

## Finding your way around

| Folder | Contents |
|---|---|
| [`competition/`](competition/README.md) | The explained hybrid notebook, executable script, and reference metadata |
| [`research/`](research/README.md) | Current experiments and the prospective source registry |
| [`docs/`](docs/REPRODUCTION.md) | Data sources, reproduction instructions, and temporal limitations |
| [`scripts/`](scripts/README.md) | Repository checks, notebook builds, and diagram generation |

The original competition delivery is preserved under the `competicao-2026` Git tag. The current branch contains the English presentation and my ongoing research. Changes are recorded in the [changelog](CHANGELOG.md).

## License and acknowledgements

The code is shared under the [MIT license](LICENSE). Datasets remain subject to their providers’ terms, documented in the [data guide](docs/DATA.md) and [NOTICE.md](NOTICE.md).

Thanks to the WorCAP organizers for the challenge, and to ECMWF, NOAA, and IRI for the data products used in this work.
