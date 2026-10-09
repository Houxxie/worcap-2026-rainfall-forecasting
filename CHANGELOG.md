# Changelog

## November preparation and source continuity — 2026-10-09 UTC

- Acquire and audit all 24 CFSv2 members for November through the original IRI recipe. Preserve the October forecast/model and extend the verified registry from 44 to 54 events.
- Record that current NOAA index receipts end in July; August and authenticated ERA5/SEAS5 acquisition remain pending. November is not issued.
- Recheck CCSR with an explicit Pydap transport: four complete members, differing from IRI. Add original-recipe comparison and inconclusive failure reports; do not authorize migration.
- Show latest recorded valid months in readiness reports and add a verified full-registry backup command.
- Restore the exact frozen inference script after a message-only edit changed its hash. Protect its recorded identity in repository verification; model calculations and parameters remain unchanged.

## Reproducible data access — 2026-10-07

- Publish hash-verified historical SEAS5, CFSv2 and optional ERSSTv5 snapshots as versioned release assets, with source attribution and separate data-use conditions. Official competition files remain on Kaggle.
- Add standard-library download and official-ZIP preparation commands, with archive/member integrity checks and protection against overwriting changed inputs.
- Add a configuration-based competition launcher and a setup notebook for a clean Kaggle session. Preserve model parameters, historical evidence and the required final CSV hash.
- Document the concrete first-run steps and the distinction between these historical snapshots and operational source acquisition.
- Verify anonymous downloads and a clean Kaggle CPU run using an isolated Python 3.12.13 environment. The final 1,885,464-row CSV matches the recorded SHA-256 byte for byte; seven historical blocks were not retrained.

## Clear model names — 2026-10-01

- Use competition model, current forecasting model and reference model in documentation, notebook explanations and generated reports. Remove unexplained development version labels from the presentation.
- Refresh the offline viewer, diagram, map caption and preview animation with those names.
- Preserve historical tags, source identifiers, numerical evidence and CSV reproduction checks. Display tables translate model names on copies without changing saved results.

## Getting started and offline demo — 2026-09-30

- Add three entry paths: explore saved results, reproduce a competition/research run, and issue monthly forecasts. Preserve the personal README and distinguish the two model versions.
- Include a self-contained offline viewer with checked archived metrics, yearly charts, a verified competition CSV example and October's frozen map. October remains unscored; historical metrics refer to earlier validation fits.
- Document exact input availability, missing prepared-data/model downloads, new versus existing registries, credentials, expected outputs and realistic compute requirements.
- Add read-only historical input checks and portable config examples. Preserve all scientific model implementations, protocols, archived numerical results and issued forecasts.

## First prospective monthly forecast — 2026-09-30

- Resolve incomplete CFSv2 acquisition by explicitly evaluating the original IRI catalog recipe. All 24 expected September members are present; every overlapping value matches the named product exactly. Four-month checks include complete 24- and 28-member ensembles.
- Keep successor CCSR fields as diagnostics: differences on all four checked months prevent treating that service as interchangeable. No source substitution, ensemble filling, retraining or weight change.
- Freeze October 2026 rainfall and training climatology at 18:18 UTC, before the deadline, using the existing model and eight validated sources. Verify the full 44-event registry and a portable backup containing all referenced objects.
- Add 11 source-compatibility tests; verify 14 registry checks and seven inference/acquisition checks across their respective environments. Preserve actual inference versions, source IDs, receipt times and forecast hashes. Future rainfall skill remains unmeasured.

## Rainfall workbench and usability — 2026-09-30

- Add one CLI and standalone Kaggle notebook for saved evaluation, the existing seven-block training protocol, read-only monthly preparation, and completed-result review.
- Show missing/ambiguous inputs together; verify complete outputs before reuse; preserve failed runs and propose report-only or saved-map recovery instead of silently restarting fitting.
- Provide a common portable report with model comparisons, annual scores, saved maps/diagnostics, source records, readiness blockers and explicit report-only limitations. No scientific configuration or operational source policy changes.
- Verify 18 workbench/legacy tests, 10 additional isolated notebook checks, stale-kernel recovery, real saved-report execution, historical map rendering and the existing 35-event registry. No new model fit, provider download or forecast emission.

## Completed saved-map SST/U-Net blend — 2026-09-30

- Complete the fixed 75/25 comparison on 2021–2022 without fitting models. Candidate RMSE is 1.792304 versus 1.795289 for the original blend.
- Record the small 0.0184% RMSE gain over the SST model alone, alongside worse MAE, higher absolute bias and deterioration in 2022. Keep the operational reference unchanged.
- Verify included hashes, source snapshots, aggregate metrics and paired squared-error identities. Actual prediction maps are absent from the small report ZIP; local raw-observation rescoring was not performed.
- Fix notebook initialization for a reused kernel that still holds modules from the preceding SST experiment; all six synthetic checks pass with the stale-import scenario reproduced.

## Saved-map SST/U-Net blend — 2026-09-30

- Prepare one fixed 75% SST model / 25% existing U-Net comparison on 2021–2022. Compare primarily against the existing reference model/U-Net blend and secondarily against the SST model alone.
- Require the exact archived prediction hashes, identical reference model/climatology maps, reproduced source metrics and exact fixed-blend MSE identities. No fitting, new source acquisition or weight search.
- Add a standalone CPU notebook, saved-output verification, small report export and synthetic integration checks. Real comparison remains pending the two complete prediction files; the supplied report ZIPs do not contain them.

## Completed SST temporal extension — 2026-09-30

- Record full 2021–2022 completion and exact reproduction of the archived reference model: RMSE 1.796784 versus 1.792633 with eight SST PCs, a 0.2310% decrease.
- Record lower RMSE and MAE in both years, 12 improved months out of 24, and higher aggregate positive bias. Preserve the unchanged operational reference.
- Verify included artifact/source hashes and recalculate reported metrics from monthly error sums. The small report ZIP omits models and maps; independent raw-observation scoring was not performed.
- Pool the existing 2007–2020 and new 2021–2022 error sums descriptively: SST improves RMSE slightly but worsens MAE and absolute bias overall. All periods remain consulted development evidence.

## Fixed SST temporal extension — 2026-09-30

- Record the completed residual U-Net pilot: RMSE worsened from 1.740802 to 1.755200 on 2019–2020; retain the reference model and avoid an automatic seven-block expansion.
- Reuse the existing SST/PCA definition on the additional 2021–2022 block, with the same paired training samples, lags, weights and fixed tree settings.
- Prepare a separate CPU notebook: four LightGBM fits, one shared ridge, one PCA, no neural fit or search. Verify the archived baseline, freeze predictions before opening evaluation labels, and export small reports.
- The earlier seven-block SST result is already available; this extension does not repeat it. Both the old years and 2021–2022 were consulted previously. Real extension results remain pending.

## Bounded residual U-Net pilot — 2026-09-30

- Record the completed seven-block bias correction: essentially unchanged RMSE, worse MAE and global absolute bias; do not adopt it.
- Prepare a separate C-block pilot with the same 31 features and two matched eight-epoch neural fits on 120 months: direct rainfall anomaly versus chronological reference model error.
- Generate residual targets using five earlier reference model fits with complete 30-year climatologies and the existing four-month label gap. Keep the final reference model unchanged.
- Save source/input signatures, stage timings, predictions and paired reports. No new data, weight search, automatic seven-block run or operational promotion. Real pilot results are pending.

## Prepared neural bias-calibration experiment — 2026-09-30

- Add a standalone Kaggle notebook using the same three inputs and the original seven development blocks, with no new data or weight search.
- Estimate one signed offset from 24 earlier chronological forecasts, using an additional nested epoch-selection split and fixed 50% shrinkage. Keep the original component models and 75/25 blend weights unchanged.
- Add target-isolation, numerical and saved-artifact checks, resumable completed blocks, preserved failed attempts and partial reports. Keep 2021–2022 out of this run.
- This is a prepared experiment for user execution, not a measured improvement or operational promotion.

## Completed chronological extension — 2026-09-29

- Record successful Kaggle version `354026629`, using the unchanged training cutoff, architecture and fixed 75/25 blend.
- Record 2021–2022 pooled RMSE of 1.796784 for the reference model and 1.795289 for the blend: a 0.0832% decrease, with worse MAE, absolute bias and 2022 RMSE.
- Preserve the small reports and audit records; verify their hashes and recompute pooled metrics from monthly sums. Model weights and prediction maps remain in the saved Kaggle output.
- Retain the current forecasting model and document a training-only bias-correction hypothesis for a separate experiment; no new fit or weight search is performed.

## Temporal-extension input coverage fix — 2026-09-29

- Preserve failed Kaggle version `354007635`: its development-only ocean index table ended before the first 2021 forecast origin, so no outer evaluation was produced.
- Load the existing full NOAA archive, verify its exact overlap with the development table, and reject missing training or forecast origins before fitting.
- Add calendar-coverage regression checks and rebuild the standalone notebook. Architecture, weights and evaluation protocol remain fixed.

## Prepared chronological extension — 2026-09-29

- Add a separate locked Kaggle notebook for the unchanged reference model, U-Net and 75/25 blend on January 2021–December 2022, with fitting capped at September 2020.
- Keep fitting/freezing and evaluation in separate calls; neural inference can run without target rainfall, and the frozen artifact package is checked before scoring.
- Document that earlier final models used these observations, so the extension is retrospective rather than an untouched holdout. Real training and evaluation remain pending.

## Fixed reference model / U-Net blend — 2026-09-29

- Evaluate one prespecified 75% reference model / 25% U-Net combination in the common workflow, using the original seven-block maps without retraining.
- Preserve candidate maps, verify their exact readback and the MSE identities, and compare global, temporal, regional and intensity errors.
- Record a development RMSE reduction from 1.753399 to 1.750628, with improvements in six of seven blocks and a worse global absolute bias.
- Keep the blend experimental and the current forecasting model unchanged; document the complete outcome and reproduction path.

## Research evaluation workflow — 2026-09-29

- Add one configured entry point for evaluating saved reference model/U-Net maps or running their existing seven-block comparison.
- Add a self-contained Kaggle notebook, early input checks, distinct run directories, source/configuration snapshots, failure records and output verification.
- Generate a portable HTML report with embedded diagnostic figures and paired block/year metrics.
- Validate saved-mode integration on the real 2007–2020 predictions, reproducing archived scores without refitting or changing the operational model.

## English presentation — 2026-09-28

- Rename the repository to `worcap-2026-rainfall-forecasting`.
- Present the reference model as the sole competition solution on the current branch.
- Add English guides, explained notebooks, runtime messages and a model diagram.
- Add “What I learned”; report the Top 10 finish and official final score.
- Organize active code under `competition/` and `research/`; preserve original deliveries and earlier experiments in historical tags.
- Keep numeric settings, feature ordering and the reference model CSV identity check.

This is a presentation and packaging update, not a new model or evaluation.

## Initial research snapshot — 2026-09-28

`pesquisa-v0.1.0`: first archive of the lagged-source baseline, SST experiments and prospective registry. No real prospective forecast had been issued.

## Competition archive — 2026-09-28

`competicao-2026`: import of the original delivery. Git timestamps reflect the actual import date, not simulated competition-period commits.
