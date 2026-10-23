# Automatic repository checks

The [Repository checks workflow](../.github/workflows/checks.yml) runs on pushes, pull requests and manual dispatch. See the [latest runs on GitHub](https://github.com/Houxxie/worcap-2026-rainfall-forecasting/actions/workflows/checks.yml).

It uses Python 3.12 on a CPU runner, read-only repository permissions and pinned action revisions. It needs no project secrets, Kaggle access or model/data downloads. Dependency installation uses PyPI. The workflow does not fit models, issue forecasts, alter the registry or publish a release.

## What it checks

- The tracked file inventory, hashes, local documentation links, Python syntax and clean notebook cells.
- Agreement between the competition script and notebook, bundled research sources, and the exact registered operational inference code.
- Reproducibility of the offline demo from the saved evidence.
- Input discovery, archive integrity, source compatibility, missing-member rejection and report recovery.
- Forecast deadlines, frozen artifacts, observation units/grids and complete registry backups.
- A synthetic forecast lifecycle using real GRIB encoding/decoding: restore a registry, reject premature/nonfinal observations, evaluate a full grid, export reports, reject tampering and preserve the primary evaluation after an explicit revision.

These are software and data-contract checks. A green run does **not** establish forecast accuracy, check whether NOAA has published a new month, or train all research experiments. Branch protection is a separate GitHub setting; this workflow does not change it.

## Run the same checks locally

Use a separate Python 3.12 environment, then run from the repository root:

```bash
python -m pip install -r requirements-checks.txt
python scripts/run_checks.py
```

The exact bounded suite is listed in [run_checks.py](../scripts/run_checks.py). Test fixtures live only in temporary directories. No full historical training run or GPU test is included.

## After editing files

Review your changes and stage only the intended files. If you changed a module embedded in a standalone notebook, rebuild that notebook with its [maintenance script](../scripts/README.md). Then refresh the packaging inventory:

```bash
python scripts/verify_repository.py --refresh-inventory
git add repository_manifest.json
python scripts/run_checks.py
```

The inventory reflects tracked/staged files, so stage new files before refreshing it. CI verifies the committed inventory; it never refreshes it to hide a mismatch. The inventory is a consistency check, not an independent signature or a substitute for reviewing the diff. Data hashes, registered scientific protocols and frozen inference identity remain separate and must not be changed to make a check pass.

Push the reviewed commit and confirm the workflow is green before closing a stable version. A future release should also identify the preserved model, required external data, operational status and remaining evaluation limits.
