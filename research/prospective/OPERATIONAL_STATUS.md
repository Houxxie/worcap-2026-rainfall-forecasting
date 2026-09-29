# October 2026 operational preparation

On 29 September 2026, the fixed lagged hybrid was fitted on 360 target months, January 1993–December 2022, and registered with all 11 required package files. **The forecast is blocked, not issued.** The acquired September CFSv2 fields do not contain complete members 21–24. No partial ensemble, month substitution, imputation or model fallback was used.

The [follow-up acquisition](#follow-up-acquisition-and-provider-migration) below extends the local registry to 35 events. The saved Kaggle version and original ZIP retain the initial 26-event preparation.

| Component | Recorded state |
| --- | --- |
| Final hybrid | Fitted once; two LightGBMs and local ridge; 52,248,030 bytes in the registered package |
| ERA5 surface fields | June 2026; GRIB final-vintage metadata, units and quarter-degree grid validated |
| ERA5 at 850 hPa | June 2026; six variables validated |
| Ocean indices | July 2026 values available in recorded NOAA PSL responses |
| SEAS5 | September initialization, October target; members 0–50 validated |
| CFSv2 | September nominal initialization; members 21–24 missing or partial in the received fields |
| Forecast | None; emission requires all sources before 1 October 2026, 00:00 UTC |

The original protocol event and all 13 events present before this run were retained. The completed preparation contains 26 events, including the frozen model, source receipts and failed source audits. Raw bytes and normalized bytes are retained separately. Receipt times indicate when the files were acquired; they do not establish the first time a provider published them.

The [machine-readable audit](evidence/operational_2026_10.json) includes model-file hashes, training settings, input hashes and the selected source-event IDs.

The downloaded 102,870,856-byte ZIP matched SHA-256 `eb8ca693f053e1ec81d71e70e411798b43704e90cdfe7ab49570bafbb2e02990`. [Local verification](evidence/operational_2026_10_verification.json) checked every referenced object and extended the original local chain from 13 to 26 events without changing earlier files. The Kaggle compute session was stopped after both copies were verified.

- Protocol event: `b61d3ae3dac9e65da02c1b6d5aa3b66f9d918bae94db51a9f55c3728cf490e37`.
- Model event: `8b7fae77efa361b7ec09d4212f20f429a2ba30a01ebdba36ec44f0b9733e6d82`, registered at `2026-09-29T19:42:08.785802+00:00`.
- Registry tip: `6a42acfd497209f61ae80d9f5c4d46638b02806132b9df652fd6428bab1d3a97`.

## Continue from these artifacts

Restore the complete latest registry, including `eventos` and `objetos`. Do not start from the earlier 13-event snapshot or retrain the model. Reacquire CFSv2 with `prepare_cfsv2.py --atualizar`, keeping the new receipt and audit. If it passes and the deadline has not passed, use `issue_forecast.py` to infer from the exact registered source objects and freeze the prediction with its training climatology. Existing audited ERA5, SEAS5 and index objects can be reused.

If the source remains incomplete at the deadline, record October as a month without an issued forecast. A later successful download cannot authorize a backdated October prediction. No automatic monitoring job is enabled.

The [private Kaggle run, version 1](https://www.kaggle.com/code/houxie/worcap-operational-rainfall-forecast?scriptVersionId=353975624) retains the complete model and initial 26-event registry, including `operational_october_2026.zip`. Its embedded 13-event starting snapshot documents the first execution. A continuation must restore the latest completed registry, including the follow-up events below. Git contains code and audit metadata, not source data, credentials or model binaries. This preparation does not measure future rainfall skill. The local hash chain is not an independent timestamp or tamper-proof storage.

## Follow-up acquisition and provider migration

At 20:11 UTC on 29 September, a fresh acquisition from the original IRI endpoint again returned complete fields for members 1–20 and no finite values for expected members 21–24. Separate requests for each missing member confirmed 0 of 5,304 finite grid points. The recorded initialization table assigns all four to 3 September 2026.

IRI's [migration announcement](https://iri.columbia.edu/resources/data-library/sunset/) describes the transition to forecast.ccsr and warns of changes during beta testing. This notice does not establish the cause of the missing fields. Its successor [CFSv2 pentad forecast dataset](https://forecast.ccsr.columbia.edu/data/NMME/NOAA-NCEP/CFSv2/pentad_samples/forecast/pr) supplied all 24 expected member fields in a diagnostic acquisition at 20:16 UTC. The explicit target bounds are 1 October–1 November 2026; the new lead coordinate is 1, whereas the original IRI coordinate is 1.5.

The two services are **not established as interchangeable**. Comparing matching member IDs 1–20 on the same one-degree coordinates gives an RMSE difference of 0.626789 mm/day between their 20-member means, and a maximum individual-member grid-point difference of 18.998323 mm/day. These are differences between source fields, not prediction errors against observed rainfall. The cause—such as preprocessing or member mapping—has not been determined. The decoded successor subset is retained only as a diagnostic artifact, explicitly distinguished from raw transport bytes; it is not a validated forecast input.

The [recheck report](evidence/cfsv2_recheck_2026_09_29.json) records the comparison, receipt times and unchanged model/protocol IDs. [Snapshot verification](evidence/cfsv2_recheck_2026_09_29_verification.json) checked all referenced object hashes and the new 35-event ZIP. This ZIP is 54,232,856 bytes, SHA-256 `4ffb55e4d99dd57fc3f9db2a392321176cb0912c5c27ad864aef89453fa38044`. It contains the registry rather than the duplicated model/code directory layout of the original Kaggle ZIP.

Continue from the **35-event registry**, not from either earlier snapshot. No source-policy change, retraining or forecast emission occurred. Further work must resolve source equivalence before using the successor service; availability alone does not satisfy the frozen input contract. No automatic monitoring job is enabled.
