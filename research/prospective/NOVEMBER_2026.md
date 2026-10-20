# November preparation and CFSv2 continuity

Checked on **20 October 2026**. November has **not** been issued. October's forecast, the registered model and the fixed plan are unchanged.

## Source readiness

| Source | Required month | Result of this check |
| --- | --- | --- |
| CFSv2 | October initialization → November target | Complete: 24 members through the original IRI recipe; exact equality on all 20 complete named-product members |
| Niño 1+2, Niño 3.4, TNA, TSA | August | The four files received on 20 October still end in July; their bytes are unchanged since 9 October |
| ERA5 surface | July | Acquired, independently audited and registered: final ERA5, all three variables, exact 301 × 261 grid |
| ERA5 at 850 hPa | July | Authenticated request accepted; still waiting for the CDS result |
| SEAS5 | October initialization → November target | Acquired separately while ERA5 waits: all 51 members, target month, units and grid independently audited and registered |

September ERSSTv5 was also received, but is optional archival data, not a reference-model input. A successful download does not establish that a file includes its required month. The readiness report now shows both the required month and the latest valid month in recorded receipts.

The deadline is **before 1 November, 00:00 UTC / 31 October, 21:00 Brasília**. Missing sources block issuance. July indices cannot replace August indices, and an incomplete seasonal ensemble cannot supply its mean.

- [Current preparation, SEAS5 audit and latest NOAA receipts](evidence/november_seas5_2026_10_20.json)
- [Earlier surface acquisition and model integrity check](evidence/november_preparation_2026_10_20.json)
- [Earlier preparation check on 9 October](evidence/november_preparation_2026_10_09.json)
- [Original IRI recipe versus successor comparison](evidence/cfsv2_compatibility_2026_10_09.json)
- [Current complete registry backup verification](evidence/registry_backup_seas5_2026_10_20.json)

The previous Kaggle draft session was off, with no preserved November outputs proving its outcome. The surface data were acquired again and saved with their original GRIB, normalized field and metadata in private checkpoint version 360548741. A separate acquisition-only **Save & Run All** execution, version 360550645, is running so completed files survive the interactive session. The earlier notebook is retained in version history and a local backup. The saved run does not fit a model, start another registry or issue a forecast.

SEAS5 finished in a separate interactive acquisition and was preserved in private version 360568015. Its transport ZIP passed a second local audit before registration. All four NOAA files were checked again at 21:47 UTC (18:47 Brasília); they still contain no August values and have identical bytes to the earlier check.

There are currently **3 of 8 required sources ready in the canonical registry**. Pressure-level ERA5 is the only remaining CDS input. When the saved run finishes, download its transport ZIP and independently audit that source before appending it to the same registry. Check that its surface and SEAS5 files match the existing receipts rather than appending duplicate inputs. A running notebook is not evidence that those files are ready.

## Successor comparison

For nominal October 2026, CCSR returned **4 complete members out of 24 expected**. The original IRI recipe returned all 24. On the identical four-member subset, the difference between ensemble means has RMSE **0.621811 mm/day**; the maximum individual member/grid-point difference is **12.981549 mm/day**. These compare source fields, not forecast scores against observed rainfall.

Units, nominal origin, November target bounds and one-degree coordinates passed the diagnostic checks. Completeness and numerical equivalence did not. Member IDs alone do not establish matching initialization histories. No successor values were registered as operational inputs.

The original recipe resolves November's CFSv2 acquisition **while IRI remains accessible**. It still depends on IRI; migration away from that service is not complete. The [shutdown notice](https://iri.columbia.edu/resources/data-library/sunset/) remains relevant. The provider's [regridding comparison](https://github.com/iridl/sigrid/blob/19d023912628a38a72435a335b68ba6f209bb178/server/test-regridding/compare_ingrid.py) documents processing differences; this does not establish equivalence for our model.

Continuing the existing series through another source requires a complete member/date contract and reproduction of the original processing, checked over historical overlaps and the requested month. If equivalent inputs cannot be obtained, the new processing requires a separate model/validation series with its own climatologies. This workflow has no fitted rescaling or automatic source substitution.

## Continue from the current registry

The verified continuation has **66 events**, retaining the exact earlier 60-event history. Tip: `8571b9bcf5c3190f223d187ccc5e724d791c93a9ded1d776571d74306efd56d6`.

The local `registry_2026_11_seas5_20261020.zip` contains the complete chain and all 46 referenced objects, including the frozen model and October prediction. Its verification sidecar identifies its hash. Keep this backup; it is not a public Git asset. Earlier backups remain preserved but omit the latest receipts and SEAS5 data.

All 11 registered model files passed hash checks. A presentation edit had changed only an exception message in `infer_baseline.py`, causing the existing exact-code guard to reject it. The file now matches the registered inference code byte for byte. Repository verification protects this identity. Weights and computation did not change.

Finish the saved acquisition of July pressure-level ERA5, then continue this complete registry. Recheck NOAA for August and run readiness again. Issue November only when all eight required sources pass and the deadline has not passed. No recurring monitoring schedule was created.

To acquire new diagnostic evidence, use a new output directory in a separate acquisition environment:

```bash
python -m pip install -r research/prospective/requirements_acquisition.txt
python research/prospective/cfsv2_compatibility.py --output outputs/cfsv2-comparison-new --origins 2026-10 --engine pydap --include-recipe --download
```

Pydap was used after the local NetCDF4 OPeNDAP reader failed. This changes the HTTPS reader, not the provider or data contract; TLS verification stays enabled. Download/contract failures produce inconclusive reports, never migration approval. Raw IRI responses, decoded successor subsets, hashes and receipts are preserved separately.

After any successful continuation, make a verified backup outside the registry directory:

```bash
python research/prospective/backup_registry.py --registry outputs/prospective --output outputs/registry-next.zip
```

The command rejects empty or altered histories and existing destination files. Local receipt times and hashes are not independent timestamps.
