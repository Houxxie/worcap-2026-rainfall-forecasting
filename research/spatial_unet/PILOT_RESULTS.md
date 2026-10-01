# First spatial pilot

Historical pilot record. The subsequent unchanged seven-block run is documented in [the complete comparison](RESULTS.md).

The prespecified H1 run completed on Kaggle on 29 September 2026 (UTC). It evaluates January 2007 through December 2008. This is a development result on previously consulted years, not an independent holdout or a forecast of future skill.

| Model | RMSE | MAE | Bias |
|---|---:|---:|---:|
| Monthly climatology | 1.842602 | 1.076899 | -0.092341 |
| Fixed reference model | **1.741364** | 1.036688 | -0.033539 |
| Compact U-Net | 1.743602 | **1.036606** | -0.069229 |

Units are mm/day; each row covers the same 1,885,464 grid points and months. The U-Net RMSE is 0.002238 higher (about 0.129%). Its small MAE reduction does not offset the worse primary metric or larger absolute bias. It is **not adopted as the new baseline**.

## What was verified

- The reference model reproduced its archived H1 RMSE, MAE and bias within the prespecified 1e-6 tolerance before neural training started.
- Inner chronological validation selected five epochs. A fresh U-Net was then fitted for those five epochs on all 295 allowed outer training months, March 1982 through September 2006.
- The U-Net has 122,945 parameters. It used the same source information as the reference model, with no new SST inputs. As documented in the protocol, its complete monthly training maps differ from the tree models' sampled pixels.
- The eight spatial synthetic tests passed in Kaggle. Local isolated checks across spatial training, inference, registry and temporal logic totalled 35 passing tests.
- After export, the run signature and all 11 block artifacts included in the report archive matched the completed-run hashes. Global, block, annual and regional scores were independently recomputed from the exported monthly error sums.

## Reproduction record

The private [Kaggle notebook](https://www.kaggle.com/code/houxie/worcap-spatial-u-net-research) stores version 1, **H1 pilot - chronological U-Net and reference model**, with its outputs. Source modules correspond to Git commit `1ae8c11a0fe397e383f06b97e450233facff8aa4`; the later report-link fix does not change training.

The actual training environment was Python 3.12.13, NumPy 2.0.2, pandas 2.3.3, xarray 2025.12.0, LightGBM 4.6.0 and **PyTorch 2.10.0+cu128 on one Tesla T4**. The separate CPU test environment used PyTorch 2.8.0. Matching code alone does not guarantee bitwise reproduction across these environments.

Full-precision [scores](evidence/h1/global.csv), the [run signature](evidence/h1/signature.json), [source calendar](evidence/h1/H1/source_calendar.csv), [inner training history](evidence/h1/H1/inner/training_history.json) and [selection record](evidence/h1/H1/inner/training.json) are retained here. The evidence directory contains the JSON/CSV report export, not model weights or forecast maps. The completed-run inventory also names those larger artifacts, which are retained in Kaggle output rather than Git.

## Next comparison

At the pilot checkpoint, the next step was to run the remaining six blocks with the same protocol before changing architecture or choosing an ensemble weight. Only H1 had been executed. The pilot established feasibility and baseline reproduction; it neither established an overall improvement nor ruled one out. No neural model was promoted and no prospective forecast was issued. The follow-up is now available in [the complete comparison](RESULTS.md).
