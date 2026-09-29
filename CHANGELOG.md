# Changelog

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
