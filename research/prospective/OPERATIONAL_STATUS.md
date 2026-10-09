# October 2026 operational forecast

For the latest continuation, see [November preparation](NOVEMBER_2026.md): **54 events**, November not yet issued. The October issuance evidence below remains unchanged.

**Issued and frozen: 30 September 2026, 18:18:00 UTC / 15:18:00 Brasília.** October's forecast passed the complete-source checks before the 1 October, 00:00 UTC deadline. The registered prediction contains 78,561 points on the 301 × 261 quarter-degree grid, with the training climatology stored beside it. Units are monthly mean precipitation in mm/day. This is a forecast; its future skill has not yet been measured.

On 29 September, the fixed reference model with longer input lags was fitted on January 1993–December 2022 and registered with all 11 required package files. That package and its protocol remain unchanged. On 30 September, evaluating the original IRI product recipe explicitly returned all 24 expected CFSv2 members. Every available named-product value was reproduced exactly. The complete ensemble from that single expression was used, without joining datasets or filling missing values.

| Component | Recorded state |
| --- | --- |
| Final reference model | Fitted once; two LightGBMs and local ridge; 52,248,030 bytes in the registered package |
| ERA5 surface fields | June 2026; GRIB final-vintage metadata, units and quarter-degree grid validated |
| ERA5 at 850 hPa | June 2026; six variables validated |
| Ocean indices | July 2026 values available in recorded NOAA PSL responses |
| SEAS5 | September initialization, October target; members 0–50 validated |
| CFSv2 | September nominal initialization, October target; all 24 expected members validated through the original IRI recipe |
| Forecast | Frozen at 18:18 UTC on 30 September; all eight required source snapshots identified |

## Source resolution and frozen evidence

The named IRI endpoint was still incomplete on 30 September. Native IRI fields contained members 21–24, and the provider's [published product recipe](https://github.com/iridl/dlentries/blob/7b8bcdd8be1c2beec01e441bcc09e31d5854958a/entries/Models/NMME/NCEP-CFSv2/FORECAST/PENTAD_SAMPLES/MONTHLY/index.tex) supplied the full ensemble after applying its original `regridAverage` operation and unit conversion. The internal cause of the named-product failure has not been confirmed by the provider.

| Nominal initialization | Expected | Complete named-product members | Complete explicit-recipe members | Largest difference on overlap |
| --- | ---: | ---: | ---: | ---: |
| September 2024 | 24 | 24 | 24 | 0 |
| November 2025 | 28 | 28 | 28 | 0 |
| August 2026 | 24 | 20 | 24 | 0 |
| September 2026 | 24 | 20 | 24 | 0 |

These checks establish equality for the inspected fields. Future acquisitions must pass the same checks; the result does not certify every provider vintage. The new preparation command records all three HTTP responses, the unchanged member/date/mean adapter, exact-overlap proof and derived mean in the existing registry.

The successor CCSR product differs numerically in all four comparisons and remains diagnostic only. Its software [documents a regridding difference](https://github.com/iridl/sigrid/blob/19d023912628a38a72435a335b68ba6f209bb178/server/test-regridding/compare_ingrid.py), which supports investigating preprocessing as a cause but does not by itself explain every field difference. The [announced IRI shutdown](https://iri.columbia.edu/resources/data-library/sunset/) makes a separately validated migration necessary for continued operation.

- [Source-resolution evidence](evidence/cfsv2_resolution_2026_09_30.json)
- [Issued forecast and environment audit](evidence/issued_forecast_2026_10.json)
- [Exact forecast registry event](evidence/forecast_event_2026_10.json)
- [Portable-backup verification](evidence/forecast_backup_2026_10.json)

The October registry snapshot contains **44 events** and has tip `712c60325428670c50ed1c059af09be6e58c192ac7a3ef25ff8cea6e35961a7e`. The forecast is 957,631 bytes, SHA-256 `7ec16a8e96e942877aa25a0d817dc5ab3709e8672bae765781356fcce4bf45f7`. Its original full backup is 64,828,247 bytes; all 144 inventoried archive entries and every referenced registry object were checked.

For new work, resume from the latest complete snapshot identified in [November preparation](NOVEMBER_2026.md), preserving `eventos` and `objetos` together. October must not be regenerated or replaced. The first verification against final ERA5 is eligible from **1 February 2027**, under the frozen T+4 verification protocol. Receipt records use the local UTC clock and hashes, not an independent timestamp. No automatic monitoring job is enabled.

## Initial preparation — historical record

The following records describe the earlier blocked state. Their archived ZIPs remain unchanged; they are not the latest continuation point.

The original protocol event and all 13 events present before this run were retained. The completed preparation contains 26 events, including the frozen model, source receipts and failed source audits. Raw bytes and normalized bytes are retained separately. Receipt times indicate when the files were acquired; they do not establish the first time a provider published them.

The [machine-readable audit](evidence/operational_2026_10.json) includes model-file hashes, training settings, input hashes and the selected source-event IDs.

The downloaded 102,870,856-byte ZIP matched SHA-256 `eb8ca693f053e1ec81d71e70e411798b43704e90cdfe7ab49570bafbb2e02990`. [Local verification](evidence/operational_2026_10_verification.json) checked every referenced object and extended the original local chain from 13 to 26 events without changing earlier files. The Kaggle compute session was stopped after both copies were verified.

- Protocol event: `b61d3ae3dac9e65da02c1b6d5aa3b66f9d918bae94db51a9f55c3728cf490e37`.
- Model event: `8b7fae77efa361b7ec09d4212f20f429a2ba30a01ebdba36ec44f0b9733e6d82`, registered at `2026-09-29T19:42:08.785802+00:00`.
- Initial 26-event registry tip: `6a42acfd497209f61ae80d9f5c4d46638b02806132b9df652fd6428bab1d3a97`.

### Original continuation instructions — superseded by the issued forecast above

Restore the complete latest registry, including `eventos` and `objetos`. Do not start from the earlier 13-event snapshot or retrain the model. Reacquire CFSv2 with `prepare_cfsv2.py --atualizar`, keeping the new receipt and audit. If it passes and the deadline has not passed, use `issue_forecast.py` to infer from the exact registered source objects and freeze the prediction with its training climatology. Existing audited ERA5, SEAS5 and index objects can be reused.

If the source remains incomplete at the deadline, record October as a month without an issued forecast. A later successful download cannot authorize a backdated October prediction. No automatic monitoring job is enabled.

The [private Kaggle run, version 1](https://www.kaggle.com/code/houxie/worcap-operational-rainfall-forecast?scriptVersionId=353975624) retains the complete model and initial 26-event registry, including `operational_october_2026.zip`. Its embedded 13-event starting snapshot documents the first execution. A continuation must restore the latest completed registry, including the follow-up events below. Git contains code and audit metadata, not source data, credentials or model binaries. This preparation does not measure future rainfall skill. The local hash chain is not an independent timestamp or tamper-proof storage.

## Follow-up acquisition and provider migration — 29 September

At 20:11 UTC on 29 September, a fresh acquisition from the original IRI endpoint again returned complete fields for members 1–20 and no finite values for expected members 21–24. Separate requests for each missing member confirmed 0 of 5,304 finite grid points. The recorded initialization table assigns all four to 3 September 2026.

IRI's [migration announcement](https://iri.columbia.edu/resources/data-library/sunset/) describes the transition to forecast.ccsr and warns of changes during beta testing. This notice does not establish the cause of the missing fields. Its successor [CFSv2 pentad forecast dataset](https://forecast.ccsr.columbia.edu/data/NMME/NOAA-NCEP/CFSv2/pentad_samples/forecast/pr) supplied all 24 expected member fields in a diagnostic acquisition at 20:16 UTC. The explicit target bounds are 1 October–1 November 2026; the new lead coordinate is 1, whereas the original IRI coordinate is 1.5.

The two services are **not established as interchangeable**. Comparing matching member IDs 1–20 on the same one-degree coordinates gives an RMSE difference of 0.626789 mm/day between their 20-member means, and a maximum individual-member grid-point difference of 18.998323 mm/day. These are differences between source fields, not prediction errors against observed rainfall. The cause—such as preprocessing or member mapping—has not been determined. The decoded successor subset is retained only as a diagnostic artifact, explicitly distinguished from raw transport bytes; it is not a validated forecast input.

The [recheck report](evidence/cfsv2_recheck_2026_09_29.json) records the comparison, receipt times and unchanged model/protocol IDs. [Snapshot verification](evidence/cfsv2_recheck_2026_09_29_verification.json) checked all referenced object hashes and the new 35-event ZIP. This ZIP is 54,232,856 bytes, SHA-256 `4ffb55e4d99dd57fc3f9db2a392321176cb0912c5c27ad864aef89453fa38044`. It contains the registry rather than the duplicated model/code directory layout of the original Kaggle ZIP.

The 35-event registry was the continuation point on 29 September; the **44-event registry above supersedes it**. No source-policy change, retraining or forecast emission had occurred at that time. Successor-service availability alone does not satisfy the frozen input contract.
