# Explore the results

Download the repository ZIP, extract it, and open **`demo/index.html`** in a browser (I made it to be really simple). You can also download [index.html](index.html) with GitHub's **Download raw file** button; all images and values are embedded in that one file. GitHub's source view itself does not execute the viewer.

No Python, credentials, Internet connection or model fitting is needed to view it.

- **Competition model:** archived 8G validation metrics, yearly charts and the first monthly prediction from its verified submitted CSV.
- **Current forecasting model:** the lagged hybrid's historical evaluation and the October 2026 forecast, explicitly marked **not yet evaluated**.

Switch the model tab and the RMSE/MAE/bias selector. Tables contain the exact saved values. October's anomaly map is a departure from climatology, not forecast error.

The competition map is a submitted prediction, not an observation comparison: full 8G validation maps are not bundled. The historical metrics do not evaluate that particular map. The operational tab's historical metrics come from earlier fold fits, not from scoring 2007–2020 with the final 1993–2022 fit. Different source-lag protocols should not be treated as a controlled paired comparison.

## Rebuild or verify (optional)

Python 3.12, standard library only. From the repository root:

```bash
python demo/explore.py
python demo/explore.py --check
```

Or generate a copy with `python demo/explore.py --output outputs/demo/index.html`. The renderer verifies the hashes in [sources.json](sources.json). It reads only the listed repository assets, CSVs and metadata; it does not download sources, train models or rescore rainfall.

[explore.py](explore.py) fills [template.html](template.html). The resulting HTML is intentionally versioned as the ready-to-open demo. Map provenance is in [competition figure metadata](../assets/competition_forecast_example.json) and [October figure metadata](../assets/forecast_october_2026.json). Regenerating map pixels requires their original inputs and scientific plotting dependencies; viewing the bundled maps does not.

Continue with [reproduction](../docs/REPRODUCTION.md) or [monthly forecasts](../docs/MONTHLY_FORECAST.md).
