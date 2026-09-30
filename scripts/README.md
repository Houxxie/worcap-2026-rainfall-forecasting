# Maintenance scripts

- `verify_repository.py`: check inventory, hashes, links, syntax and notebook/script consistency without training.
- `build_notebooks.py`: rebuild the self-contained research notebooks from their guides and modules.
- `build_spatial_notebooks.py`: build the separate U-Net experiment and final-baseline inference notebooks.
- `build_workflow_notebook.py`: build the unified saved-evaluation/training notebook and its checked source bundle.
- `build_extension_notebook.py`: build the fixed 2021–2022 temporal-extension notebook, with separate prediction freezing and evaluation.
- `render_overview.py`: regenerate the model architecture image with Matplotlib.
- `render_october_forecast.py`: render the frozen October forecast, climatology and anomaly after checking the forecast and Natural Earth coastline hashes in `assets/forecast_october_2026.json`.
- `build_workbench_notebook.py`: build the shared notebook for result review, saved-map evaluation, existing training and read-only operational readiness.
- `build_residual_notebook.py`, `build_sst_extension_notebook.py`, `build_sst_unet_blend_notebook.py`: package the separate, fixed research experiments.

Historical sources remain in Git tags. A scientific change needs a new protocol and output directory; do not overwrite historical hashes to disguise it.

To reproduce the October figure, use the forecast NetCDF from the complete operational backup and the coastline file identified by URL and SHA-256 in [the figure metadata](../assets/forecast_october_2026.json). The coastline is only a visual reference; it is not a model input. Use a new output path:

```bash
python scripts/render_october_forecast.py --forecast /path/to/forecast_2026_10.nc --coastline /path/to/ne_110m_coastline.geojson --output /path/to/october_preview.png
```

After reviewing and staging intentional source changes, run `python scripts/verify_repository.py --refresh-inventory` to refresh current repository hashes, then stage `repository_manifest.json`. This updates the packaging inventory, not source-data hashes or frozen protocols. Rebuild research notebooks after editing their bundled modules or guides.
