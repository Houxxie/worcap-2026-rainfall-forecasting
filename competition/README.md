# Competition hybrid

Two LightGBM residual models plus local ridge regression, developed for the WorCAP 2026 Hackathon. **Top 10 · official final score: 1.78758.**

- [Explained notebook](hybrid_forecast.ipynb)
- [Executable script](hybrid_forecast.py)
- [Reference environment](requirements.txt)
- [Input hashes and audited results](reference.json)
- [Required data snapshots](../docs/DATA.md)
- [Explore saved results without installing anything](../demo/README.md)
- [Step-by-step Kaggle and local reproduction](../docs/REPRODUCTION.md)

Final training: January 1993–December 2022. Predictions: January 2023–December 2024 on the 301 × 261 official grid. Fixed component weights: 0.375, 0.375 and 0.25. Expected final CSV SHA-256:

```text
e98e8954debb8f7c764665a31804ec5512a1c2578c1ed4a24e50c51dedfb429b
```

The default run fits the final components; optional validation repeats seven development blocks. Intermediate component files preserve the original rounding order. They are not separate competition solutions presented here.

The notebook and script are English presentation derivatives. Submitted originals remain under `competicao-2026`. Historical dataset names and feature identifiers remain compatible. See [version notes](../docs/VERSIONS.md).

Previous-month input alignment does not prove consolidated products were published before issue time. Read the [temporal limitations](../docs/TEMPORAL_VALIDITY.md) before interpreting this as a real-time forecast.
