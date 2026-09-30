# Changelog

## Completed chronological extension — 2026-09-29

- Record successful Kaggle version `354026629`, using the unchanged training cutoff, architecture and fixed 75/25 blend.
- Record 2021–2022 pooled RMSE of 1.796784 for the hybrid and 1.795289 for the blend: a 0.0832% decrease, with worse MAE, absolute bias and 2022 RMSE.
- Preserve the small reports and audit records; verify their hashes and recompute pooled metrics from monthly sums. Model weights and prediction maps remain in the saved Kaggle output.
- Retain the operational hybrid and document a training-only bias-correction hypothesis for a separate experiment; no new fit or weight search is performed.

## Temporal-extension input coverage fix — 2026-09-29

- Preserve failed Kaggle version `354007635`: its development-only ocean index table ended before the first 2021 forecast origin, so no outer evaluation was produced.
- Load the existing full NOAA archive, verify its exact overlap with the development table, and reject missing training or forecast origins before fitting.
- Add calendar-coverage regression checks and rebuild the standalone notebook. Architecture, weights and evaluation protocol remain fixed.

## Prepared chronological extension — 2026-09-29

- Add a separate locked Kaggle notebook for the unchanged hybrid, U-Net and 75/25 blend on January 2021–December 2022, with fitting capped at September 2020.
- Keep fitting/freezing and evaluation in separate calls; neural inference can run without target rainfall, and the frozen artifact package is checked before scoring.
- Document that earlier final models used these observations, so the extension is retrospective rather than an untouched holdout. Real training and evaluation remain pending.

## Fixed hybrid / U-Net blend — 2026-09-29

- Evaluate one prespecified 75% hybrid / 25% U-Net combination in the common workflow, using the original seven-block maps without retraining.
- Preserve candidate maps, verify their exact readback and the MSE identities, and compare global, temporal, regional and intensity errors.
- Record a development RMSE reduction from 1.753399 to 1.750628, with improvements in six of seven blocks and a worse global absolute bias.
- Keep the blend experimental and the operational hybrid unchanged; document the complete outcome and reproduction path.

## Research evaluation workflow — 2026-09-29

- Add one configured entry point for evaluating saved hybrid/U-Net maps or running their existing seven-block comparison.
- Add a self-contained Kaggle notebook, early input checks, distinct run directories, source/configuration snapshots, failure records and output verification.
- Generate a portable HTML report with embedded diagnostic figures and paired block/year metrics.
- Validate saved-mode integration on the real 2007–2020 predictions, reproducing archived scores without refitting or changing the operational model.

## English presentation — 2026-09-28

- Rename the repository to `worcap-2026-rainfall-forecasting`.
- Present the hybrid as the sole competition solution on the current branch.
- Add English guides, explained notebooks, runtime messages and a model diagram.
- Add “What I learned”; report the Top 10 finish and official final score.
- Organize active code under `competition/` and `research/`; preserve original deliveries and earlier experiments in historical tags.
- Keep numeric settings, feature ordering and the hybrid CSV identity check.

This is a presentation and packaging update, not a new model or evaluation.

## Initial research snapshot — 2026-09-28

`pesquisa-v0.1.0`: first archive of the lagged-source baseline, SST experiments and prospective registry. No real prospective forecast had been issued.

## Competition archive — 2026-09-28

`competicao-2026`: import of the original delivery. Git timestamps reflect the actual import date, not simulated competition-period commits.
