# Reproduction

## Competition hybrid

Use Python 3.12 and [competition/requirements.txt](../competition/requirements.txt). The notebook checks NumPy 2.0.2, pandas 2.3.3, xarray 2025.12.0 and LightGBM 4.6.0. It does not silently replace packages.

In Kaggle, import [hybrid_forecast.ipynb](../competition/hybrid_forecast.ipynb), attach the official files and prepared seasonal snapshots from [DATA.md](DATA.md), then run all cells. `REEXECUTAR_VALIDACAO = False` performs only the final fit and displays archived historical results. Set it to `True` to rerun those blocks too.

For local execution, configure the three `BANCA_PASTA_*` directories at the top of [hybrid_forecast.py](../competition/hybrid_forecast.py), install the requirements in an isolated environment, then run:

```bash
python competition/hybrid_forecast.py
```

The script needs substantial memory for gridded data. Successful reproduction means the generated `submission_hibrida.csv` matches its expected SHA-256, not just that training completed. Original filenames, feature ordering and CSV rounding order are intentional.

## Research

The [lagged-source notebook](../research/lagged_sources/lagged_sources.ipynb) embeds its modules and fixed protocol. Attach official data, SEAS5, CFSv2 and the recorded ERSSTv5 snapshot. It compares baseline and candidate, without producing a competition submission.

The [prospective notebook](../research/prospective/prospective_registry.ipynb) collects sources and checks readiness. It needs Internet for public downloads. It is not yet a complete forecasting service; see its [guide](../research/prospective/README.md).

## Checks without training

```bash
python scripts/verify_repository.py
python research/lagged_sources/test_temporal_contract.py
python research/prospective/test_registry.py
```

The first uses the standard library to check hashes, syntax, notebook/script consistency and documentation links. The other two require research dependencies and use synthetic data in temporary folders. These checks do not claim a fresh real-data training run.
