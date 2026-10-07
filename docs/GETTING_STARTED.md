# Start here

There are two model paths: the **competition model** and the **current forecasting model**. Both combine two LightGBMs and local ridge. The operational version uses longer input lags and a separately fitted, frozen model package. U-Net and SST/PCA remain research candidates.

| What you want to do | Start here | What you need |
|---|---|---|
| Explore saved results | [Offline demo](../demo/README.md) | A browser; everything shown is included |
| Reproduce the competition or a research experiment | [Reproduction guide](REPRODUCTION.md) | Python, official files and audited seasonal snapshots |
| Fit the operational baseline and issue future forecasts | [Monthly forecast guide](MONTHLY_FORECAST.md) | Historical fitting inputs, current sources, CDS access and a persistent registry |

## Get the code

Clone the repository, or use **Code → Download ZIP** and extract it. A notebook export by itself does not include the data files produced by its execution.

```bash
git clone https://github.com/Houxxie/worcap-2026-rainfall-forecasting.git
cd worcap-2026-rainfall-forecasting
```

Run terminal commands from this directory unless a guide says otherwise. On Windows, `py -3.12` can replace `python` when selecting the interpreter.

## Try something before installing anything

Open `demo/index.html` from the extracted repository in your browser. It contains two views: the competition results and the current model's historical evaluation plus the frozen October forecast. No Python, accounts, model weights or Internet connection are needed to view it.

The demo displays saved evidence. It does not run inference or establish future forecast skill. Historical metrics describe earlier validation fits, not the final model trained on 1993–2022.

## Before a full run

Read the [data availability table](DATA.md#availability). Code, small reports and metadata are included. Exact SEAS5/CFSv2 snapshots are available in a public data release; `python scripts/download_data.py` installs and verifies them. Official competition files still come from Kaggle, and fitted model packages are produced by your run. See the [reproduction guide](REPRODUCTION.md) for a notebook that handles setup and the final fit.

The [input checker](../scripts/check_inputs.py) lists missing or mismatched historical files and core packages together, before expensive work. Copy or edit the appropriate file in `configs/`. Paths in those JSON files resolve relative to the JSON file, not your terminal directory. These settings do not change model parameters.
