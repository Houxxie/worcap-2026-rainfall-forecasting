# October 2026 operational preparation

On 29 September 2026, the fixed lagged hybrid was fitted on 360 target months, January 1993–December 2022, and registered with all 11 required package files. **The forecast is blocked, not issued.** The acquired September CFSv2 fields do not contain complete members 21–24. No partial ensemble, month substitution, imputation or model fallback was used.

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

The [private Kaggle run, version 1](https://www.kaggle.com/code/houxie/worcap-operational-rainfall-forecast?scriptVersionId=353975624) retains the complete model and registry, including `operational_october_2026.zip`. Its embedded 13-event starting snapshot documents this first execution; a continuation must restore the completed output with 26 events, as described above. Git contains code and audit metadata, not source data, credentials or model binaries. This preparation does not measure future rainfall skill. The local hash chain is not an independent timestamp or tamper-proof storage.
