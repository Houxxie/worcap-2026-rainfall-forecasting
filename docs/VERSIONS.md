# Versions and provenance

| Git reference | Purpose |
|---|---|
| `competicao-2026` | Original competition delivery, preserved without rewriting |
| `pesquisa-v0.1.0` | First research archive, including earlier experiments |
| `operational-2026-10` | First frozen monthly forecast, source-compatibility tests and completed research reports |
| `main` | English hybrid presentation and active research |

Original competition reproduction script SHA-256:

```text
547bd7ac54c8873b5d6e237627f4b6dba94851259548a455f8574e2ecfff8616
```

The English notebook and script are derivatives. Presentation has changed; numeric settings, feature ordering and final CSV identity checks are retained. The intermediate tree average remains internal because the hybrid depends on its rounding order.

Historical tags preserve original Portuguese documents and commit messages. The current branch presents only the hybrid competition solution. Earlier experiments remain inspectable without mixing their results with the active reference.

Original input filenames, feature identifiers and machine-readable contracts remain where compatibility requires them. Frozen JSON protocols and original evidence tables retain their bytes, including their original field names and descriptions. Instructions and runtime explanations are in English; see the [field glossary](../research/lagged_sources/README.md).

Repository checks and synthetic tests verify packaging and selected contracts. They do not claim that all real-data training was repeated for this update. No new forecast skill is claimed.

The October operational record identifies the actual forecast bytes and emission time. The `operational-2026-10` tag identifies the code and documentation snapshot published afterwards; it is not an independent timestamp of the original forecast emission. Source data and fitted-model files remain in the separately verified backup.
