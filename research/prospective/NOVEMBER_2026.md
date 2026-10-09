# November preparation and CFSv2 continuity

Checked on **9 October 2026 UTC (8 October in Brasília)**. November has **not** been issued. October's forecast, the registered model and the fixed plan are unchanged.

## Source readiness

| Source | Required month | Result of this check |
| --- | --- | --- |
| CFSv2 | October initialization → November target | Complete: 24 members through the original IRI recipe; exact equality on all 20 complete named-product members |
| Niño 1+2, Niño 3.4, TNA, TSA | August | The four newly received NOAA files still end in July; August remains missing |
| ERA5 surface and 850 hPa | July | Not acquired in this run: the private Kaggle editor failed to load; no local CDS credential was available |
| SEAS5 | October initialization → November target | Same authenticated-access blocker; provider availability was not established |

September ERSSTv5 was also received, but is optional archival data, not a reference-model input. A successful download does not establish that a file includes its required month. The readiness report now shows both the required month and the latest valid month in recorded receipts.

The deadline is **before 1 November, 00:00 UTC / 31 October, 21:00 Brasília**. Missing sources block issuance. July indices cannot replace August indices, and an incomplete seasonal ensemble cannot supply its mean.

- [Preparation, source receipts and model integrity](evidence/november_preparation_2026_10_09.json)
- [Original IRI recipe versus successor comparison](evidence/cfsv2_compatibility_2026_10_09.json)
- [Complete registry backup verification](evidence/registry_backup_2026_10_09.json)

## Successor comparison

For nominal October 2026, CCSR returned **4 complete members out of 24 expected**. The original IRI recipe returned all 24. On the identical four-member subset, the difference between ensemble means has RMSE **0.621811 mm/day**; the maximum individual member/grid-point difference is **12.981549 mm/day**. These compare source fields, not forecast scores against observed rainfall.

Units, nominal origin, November target bounds and one-degree coordinates passed the diagnostic checks. Completeness and numerical equivalence did not. Member IDs alone do not establish matching initialization histories. No successor values were registered as operational inputs.

The original recipe resolves November's CFSv2 acquisition **while IRI remains accessible**. It still depends on IRI; migration away from that service is not complete. The [shutdown notice](https://iri.columbia.edu/resources/data-library/sunset/) remains relevant. The provider's [regridding comparison](https://github.com/iridl/sigrid/blob/19d023912628a38a72435a335b68ba6f209bb178/server/test-regridding/compare_ingrid.py) documents processing differences; this does not establish equivalence for our model.

Continuing the existing series through another source requires a complete member/date contract and reproduction of the original processing, checked over historical overlaps and the requested month. If equivalent inputs cannot be obtained, the new processing requires a separate model/validation series with its own climatologies. This workflow has no fitted rescaling or automatic source substitution.

## Continue from the current registry

The verified continuation has **54 events**, retaining the exact original 44-event prefix. Tip: `78f3f0eae1b0c236cb91d362a0ab91b89d8da89300017d2bd2523b8766985b16`.

The local `registry_2026_11_prepared_20261009.zip` contains the complete chain and all 42 referenced objects, including the frozen model and October prediction. Its verification sidecar identifies its hash. Keep this backup; it is not a public Git asset. An older October-only backup omits the November receipts.

All 11 registered model files passed hash checks. A presentation edit had changed only an exception message in `infer_baseline.py`, causing the existing exact-code guard to reject it. The file now matches the registered inference code byte for byte. Repository verification protects this identity. Weights and computation did not change.

After access to the existing private Kaggle notebook is restored, continue the complete registry and collect July ERA5/October SEAS5 with the existing CDS adapter. Recheck NOAA for August, then run readiness. Issue November only when all eight required sources pass and the deadline has not passed. No monitoring schedule was created.

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
