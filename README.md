# WorCAP 2026 · Rainfall Forecasting

I started this project during the **WorCAP 2026 Hackathon**, working on monthly rainfall forecasts for South America. My solution finished in the **Top 10**, with an official final score of **1.78758**.

I wanted to keep working on it after the competition, so I put together this repository to share the model I submitted, explain how I built it, and document the experiments I’m working on now.

![Overview of the rainfall forecasting model](assets/model_overview.png)

## About the project

My goal is to predict monthly rainfall patterns across South America using atmospheric data, ocean indices, and seasonal climate forecasts.

The model predicts **monthly mean daily precipitation, in mm/day**, on a 0.25° grid. I use a monthly climatology as a reference, then train the models to estimate how much rainfall might differ from that reference.

For the competition, I combined three components:

| Component | Main inputs | Final weight |
|---|---|---:|
| LightGBM · seasonal | Atmospheric variables, ocean indices, location, seasonality, and SEAS5 | 37.5% |
| LightGBM · multisystem | The same inputs, with CFSv2 added | 37.5% |
| Local ridge regression | SEAS5 and CFSv2 forecast anomalies at each grid point | 25% |

I used up to 30 years of training data and fitted the climatologies and transformations within each training window.

I explain the full process in the [competition notebook](competition/hybrid_forecast.ipynb), from loading the data to generating the final CSV.

## Technologies I used

Python · NumPy · pandas · xarray · LightGBM · scikit-learn · Matplotlib · NetCDF/GRIB · Kaggle

## How I built it

I started with a model based on rainfall climatology and gradually added atmospheric and ocean information. Adding seasonal forecasts gave me another way to represent the conditions expected for each target month.

I then worked with forecast anomalies: how much a prediction differed from the usual conditions for that location and time of year. This became an important part of the model.

For the final hybrid, I combined two LightGBM components with a local ridge regression. I evaluated the predictions across seven chronological validation blocks covering 2007–2020, checking the overall results and the differences between years.

After the competition, I started looking more closely at when each data source would actually be available. I’m now testing longer input lags and recording when new files arrive, with the goal of evaluating forecasts for future months.

I keep that work in the [research folder](research/README.md). It includes the current baseline, an experiment with global SST and PCA, and the source-arrival registry. I’m still integrating the complete pipeline for issuing future forecasts.

## What I learned

### Working with climate data

I got much more comfortable working with NetCDF and GRIB files. I learned to check units, coordinates, calendars, and ensemble members before combining different sources. I also learned to pay attention to interpolation and what the resolution of each dataset actually represents.

### Thinking about time in a forecasting problem

I learned to separate the month a value describes from the date it becomes available. This changed how I look at validation: I now check both the training cutoff and whether the inputs could have been available when a forecast was made.

### Understanding climatologies and anomalies

I learned how monthly climatologies help represent the seasonal rainfall cycle, and how anomalies make departures from that cycle easier to model. I also learned why a training observation needs to be excluded from its own rainfall reference when constructing the residual target.

### Combining different models

I explored how a simpler local regression could contribute alongside gradient boosting. This helped me understand why I need to evaluate the combined predictions, rather than judge each component only by its individual score.

### Looking beyond a single metric

I started paying more attention to MAE, bias, and results by year and region. Looking at these together helped me see trade-offs that were easy to miss when I focused mainly on the overall RMSE.

### Making my work reproducible

I learned to keep track of configurations, package versions, input files, and hashes. While reproducing the submission, I also saw how something as small as the order of CSV rounding can change the final file.

## Running the competition model

I prepared the notebook to run in Kaggle:

1. Import [hybrid_forecast.ipynb](competition/hybrid_forecast.ipynb).
2. Attach the official competition data and the prepared SEAS5 and CFSv2 snapshots listed in the [data guide](docs/DATA.md).
3. Use the [reference environment](competition/requirements.txt) and run the cells in order.
4. The notebook fits the model components and generates `submission_hibrida.csv`, checking the IDs, coverage, and expected SHA-256.

I also included the same implementation as a [Python script](competition/hybrid_forecast.py).

I keep raw datasets and fitted models outside Git. Exact reproduction requires the recorded data snapshots, since providers may revise their files over time. I documented the requirements in the [reproduction guide](docs/REPRODUCTION.md).

## Finding your way around

| Folder | What I keep there |
|---|---|
| [`competition/`](competition/README.md) | The explained hybrid notebook, executable script, and reference metadata |
| [`research/`](research/README.md) | Current experiments and the prospective source registry |
| [`docs/`](docs/REPRODUCTION.md) | Data sources, reproduction instructions, and temporal limitations |
| [`scripts/`](scripts/README.md) | Repository checks, notebook builds, and diagram generation |

I preserved the original competition delivery under the `competicao-2026` Git tag. The current branch contains the English presentation and my ongoing research. Changes are recorded in the [changelog](CHANGELOG.md).

## License and acknowledgements

I’m sharing the code under the [MIT license](LICENSE). The datasets remain subject to their providers’ terms, which I reference in the [data guide](docs/DATA.md) and [NOTICE.md](NOTICE.md).

I developed this project with AI assistance in implementation, debugging, and documentation.

Thanks to the WorCAP organizers for the challenge, and to ECMWF, NOAA, and IRI for the data products used in this work.
