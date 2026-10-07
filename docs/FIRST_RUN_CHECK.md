# First-run verification

On **7 October 2026**, the competition path was tested from a new public checkout and a new Kaggle notebook, using the published setup instructions. The final model was fitted again, and its output matched the submitted CSV **byte for byte**.

| Check | Result |
|---|---|
| Download public code without GitHub credentials | Passed |
| Download SEAS5, CFSv2 and optional SST release assets without credentials | Passed; archive and member hashes matched |
| Import the retained official Kaggle ZIP locally | All 13 files matched their recorded hashes |
| Kaggle inputs | Official competition attached; seasonal files fetched from the public release |
| Preflight | 21 files, including manifests, and four core package versions passed |
| Model execution | Two LightGBM components and local ridge fitted on 1993–2022; exit code 0 |
| Output | 1,885,464 rows; 57,689,422 bytes; IDs, order and coverage checked |
| Final CSV SHA-256 | `e98e8954debb8f7c764665a31804ec5512a1c2578c1ed4a24e50c51dedfb429b` |

The [machine-readable record](../competition/results/reproduction_20261007.json) identifies the tested code revision, data release and runtime. The reproduction entry point is pinned to `reproduction-v1`; the independent `data-snapshots-v1` tag identifies the climate assets.

## Problems caught by the test

The new Kaggle editor was running Python 3.13.15, while the recorded numerical environment uses Python 3.12. Installing the older NumPy directly in the new kernel attempted a source build. The setup now creates a separate **Python 3.12.13** environment and installs binary packages there. The editor can keep its own interpreter.

A fresh clone also caught an existing inventory mismatch caused by one file's line endings. The current repository inventory was corrected and verified against a fresh checkout. The historical data and competition submission hash were not changed.

## What this verifies

This test verifies data access, setup and final competition reproduction in the recorded Kaggle/Linux environment. The seven historical validation blocks were **not** retrained. SST download was tested locally; SST is not an input to this competition model. No operational model was refitted and no new monthly forecast was issued.

The official competition files remain subject to Kaggle/organizer access. An account without access cannot complete this full run; the demo and public climate snapshots still work independently. Matching a historical submission is also different from demonstrating accuracy on future months.

To follow the tested path, use the [reproduction guide](REPRODUCTION.md).
