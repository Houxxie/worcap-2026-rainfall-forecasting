# Maintenance scripts

- `verify_repository.py`: check inventory, hashes, links, syntax and notebook/script consistency without training.
- `build_notebooks.py`: rebuild the self-contained research notebooks from their guides and modules.
- `build_spatial_notebooks.py`: build the separate U-Net experiment and final-baseline inference notebooks.
- `build_workflow_notebook.py`: build the unified saved-evaluation/training notebook and its checked source bundle.
- `render_overview.py`: regenerate the model architecture image with Matplotlib.

Historical sources remain in Git tags. A scientific change needs a new protocol and output directory; do not overwrite historical hashes to disguise it.

After reviewing and staging intentional source changes, run `python scripts/verify_repository.py --refresh-inventory` to refresh current repository hashes, then stage `repository_manifest.json`. This updates the packaging inventory, not source-data hashes or frozen protocols. Rebuild research notebooks after editing their bundled modules or guides.
