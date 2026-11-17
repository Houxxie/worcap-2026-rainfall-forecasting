# Versions and provenance

| Git reference | Purpose |
|---|---|
| `competicao-2026` | Original competition delivery, preserved without rewriting |
| `pesquisa-v0.1.0` | First research archive, including earlier experiments |
| `operational-2026-10` | First frozen monthly forecast, source-compatibility tests and completed research reports |
| `v1.0.0` | Final reviewed version: competition model, frozen forecasting model, research reports, automatic checks and the evaluation workflow |
| `main` | English presentation; no active development after `v1.0.0` |

Original competition reproduction script SHA-256:

```text
547bd7ac54c8873b5d6e237627f4b6dba94851259548a455f8574e2ecfff8616
```

The English notebook and script are derivatives. Presentation has changed; numeric settings, feature ordering and final CSV identity checks are retained. The intermediate tree average remains internal because the competition model depends on its rounding order.

Historical tags preserve original Portuguese documents and commit messages. The current branch presents only the competition solution. Earlier experiments remain inspectable without mixing their results with the active reference.

Original input filenames, feature identifiers and machine-readable contracts remain where compatibility requires them. Frozen JSON protocols and original evidence tables retain their bytes, including their original field names and descriptions. Reader-facing text calls the submitted solution the competition model and the operational version the current forecasting model. Existing filenames and archived model keys remain stable so saved runs can still be read. Instructions and runtime explanations are in English; see the [field glossary](../research/lagged_sources/README.md).

`v1.0.0` is stable in the software sense: the code, guides and checks were reviewed together and are not expected to change. It is not a statement about forecast quality. October 2026 is the only issued forecast and has not been evaluated; November 2026 was not issued. The 2007–2022 results are retrospective and those years were consulted during development. The fitted model package, source data and operational registry stay outside Git.

Repository checks and synthetic tests verify packaging and selected contracts. They do not claim that all real-data training was repeated for this update. No new forecast skill is claimed.

The October operational record identifies the actual forecast bytes and emission time. The `operational-2026-10` tag identifies the code and documentation snapshot published afterwards; it is not an independent timestamp of the original forecast emission. Source data and fitted-model files remain in the separately verified backup.
