"""A small standalone report for the additional two-year comparison."""
from html import escape
from pathlib import Path
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(output, result, plan):
    output = Path(output)
    table = pd.read_csv(output / 'global.csv')
    table = table[['modelo', 'rmse', 'mae', 'vies', 'rmse_area', 'n']].rename(columns={
        'modelo': 'Model', 'rmse': 'RMSE', 'mae': 'MAE', 'vies': 'Bias', 'rmse_area': 'Area-weighted RMSE', 'n': 'Samples'})
    months = pd.read_csv(output / 'monthly_metrics.csv')
    months = months[months.regiao == 'dominio_inteiro'].copy()
    months['rmse'] = np.sqrt(months.sse / months.n)
    colors = dict(hybrid='#087e8b', unet='#db654a', blend='#6246a5', climatology='#9299a1')
    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    for model in ['hybrid', 'unet', 'blend']:
        rows = months[months.modelo == model].sort_values('mes_alvo')
        ax.plot(pd.to_datetime(rows.mes_alvo), rows.rmse, label=model.title(), color=colors[model])
    ax.set(xlabel='Target month', ylabel='Full-grid RMSE (mm/day)', title='Additional chronological block: 2021–2022')
    ax.legend(frameon=False)
    fig.savefig(output / 'monthly_comparison.png', dpi=150)
    plt.close(fig)
    encoded = base64.b64encode((output / 'monthly_comparison.png').read_bytes()).decode('ascii')
    delta = result['delta_rmse']
    parts = ['<h1>Temporal extension: 2021–2022</h1>',
             f'<p>Fixed 75% hybrid + 25% U-Net. Blend minus hybrid RMSE: <strong>{delta:+.6f} mm/day</strong>.</p>',
             '<p>Retrospective extension, not an independent holdout. These years were used in earlier final training. This run refitted every component only on October 1990–September 2020, with epochs chosen inside that training window. Predictions were saved before evaluation labels were decoded.</p>',
             table.to_html(index=False, float_format=lambda value: f'{value:.6f}', border=0),
             f'<img alt="Monthly comparison" src="data:image/png;base64,{encoded}">']
    for name in ['years', 'regions', 'calendar']:
        frame = pd.read_csv(output / (name + '.csv'))
        cols = [c for c in ['modelo', 'ano', 'mes', 'regiao', 'rmse', 'mae', 'vies', 'rmse_area', 'n'] if c in frame]
        frame = frame[cols].rename(columns={'modelo': 'Model', 'ano': 'Year', 'mes': 'Month', 'regiao': 'Region', 'vies': 'Bias'})
        parts += [f'<details><summary>{escape(name.title())}</summary>' + frame.to_html(index=False, float_format=lambda value: f'{value:.6f}', border=0) + '</details>']
    parts += ['<p>Full supplied grid, including ocean. The unweighted pooled RMSE is primary; area-weighted RMSE is secondary. Two years provide limited temporal diversity. No alternative blend weights were tried. No operational forecast was issued and no model was automatically promoted.</p>',
              '<p>Historical inputs are consolidated source snapshots, not reconstructed publication vintages. Saving a prediction before scoring a historical target is an execution safeguard, not proof that the target was previously unknown to the project.</p>']
    html = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rainfall temporal extension</title><style>body{font:16px/1.55 system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;color:#172b36}img{max-width:100%}table{display:block;overflow:auto;border-collapse:collapse}th,td{padding:8px 12px;border-bottom:1px solid #ddd;text-align:right}details{margin:20px 0}summary{cursor:pointer;font-weight:600}</style><body>' + '\n'.join(parts) + '</body></html>'
    (output / 'report.html').write_text(html, encoding='utf-8', newline='\n')
