"""Figures and a self-contained report for the prespecified fixed blend."""
from html import escape
from pathlib import Path
import base64
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from research.diagnostics.analyze_errors import BLOCKS, BIN_NAMES, figure_setup
from research.workflow.report import table_html

COLORS = dict(hybrid='#087e8b', unet='#db654a', blend='#6246a5', climatology='#9299a1')


def plot_results(tables, maps, output):
    figure_setup()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, name, key, order in [(axes[0], 'blocks', 'block', BLOCKS),
                                (axes[1], 'years', 'year', range(2007, 2021))]:
        frame = tables[name + '_comparison'].set_index(key).loc[list(order)]
        ax.bar(range(len(frame)), frame.delta_rmse, color=np.where(frame.delta_rmse < 0, COLORS['blend'], COLORS['unet']))
        ax.axhline(0, color='#555', linewidth=.8)
        ax.set_xticks(range(len(frame)), frame.index, rotation=45 if name == 'years' else 0)
        ax.set(xlabel=key.title(), ylabel='Blend minus hybrid RMSE (mm/day)')
    fig.suptitle('Fixed 75% hybrid + 25% U-Net | negative differences favor the blend')
    fig.savefig(output / 'temporal_comparison.png', dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for model in ['hybrid', 'unet', 'blend']:
        month = tables['calendar'].query('model == @model').sort_values('calendar_month')
        intensity = tables['intensity'].query('model == @model').set_index('intensity').loc[list(BIN_NAMES)]
        axes[0].plot(month.calendar_month, month.rmse, marker='o', label=model.title(), color=COLORS[model])
        axes[1].plot(range(5), intensity.rmse, marker='o', label=model.title(), color=COLORS[model])
    axes[0].set(xlabel='Target calendar month', ylabel='RMSE (mm/day)', xticks=range(1, 13))
    axes[0].legend(frameon=False)
    axes[1].set_xticks(range(5), ['0–<1', '1–<3', '3–<5', '5–<10', '≥10'])
    axes[1].set(xlabel='Observed monthly mean rainfall (mm/day)', ylabel='Conditional RMSE (mm/day)')
    fig.suptitle('Development diagnostics | observed intensity cannot select a forecast')
    fig.savefig(output / 'calendar_intensity.png', dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    difference = maps.blend_rmse - maps.hybrid_rmse
    bound = max(float(np.abs(difference).max()), 1e-12)
    artist = ax.pcolormesh(maps.lon, maps.lat, difference, cmap='RdBu_r', vmin=-bound, vmax=bound, shading='auto')
    ax.set(xlabel='Longitude (°)', ylabel='Latitude (°)', xlim=(-90, -25), ylim=(-60, 15), aspect='equal',
           title='Blend − hybrid RMSE | all 168 months\nBlue favors the blend; full grid includes ocean')
    fig.colorbar(artist, ax=ax, label='Difference in RMSE (mm/day)')
    fig.savefig(output / 'spatial_comparison.png', dpi=150); plt.close(fig)


def render(run, record):
    run = Path(run)
    folder = run / 'fixed_blend'
    summary = json.loads((folder / 'summary.json').read_text())
    scores = pd.read_csv(folder / 'global.csv')
    scores['delta_rmse_vs_hybrid'] = scores.rmse - summary['hybrid_rmse']
    scores = scores[['model', 'rmse', 'mae', 'bias', 'delta_rmse_vs_hybrid', 'n']]
    scores['model'] = scores.model.replace({'blend': 'Fixed 75/25 blend'})
    delta = summary['delta_rmse_vs_hybrid']
    outcome = ('The fixed blend reduces pooled development RMSE.' if delta < 0 else
               'The fixed blend increases pooled development RMSE.' if delta > 0 else 'Pooled RMSE is tied.')
    parts = ['<h1>One fixed blend: hybrid + U-Net</h1>',
             f'<p class="lead">{outcome} Difference: <strong>{delta:+.6f} mm/day</strong>.</p>',
             '<p>75% hybrid + 25% U-Net · January 2007–December 2020 · Seven chronological blocks</p>',
             '<p class="note">One weight fixed before this calculation, after inspecting the original model diagnostics. No weight search, retraining, new source or automatic promotion. These years are development data, not an independent holdout.</p>',
             '<h2>Overall comparison</h2><p>Metrics pool errors across the full supplied grid, including ocean. Bias is prediction minus observation; a smaller absolute bias is closer to zero.</p>',
             table_html(scores),
             '<details><summary>Run identity</summary><dl>' + ''.join(
                 f'<dt>{escape(k)}</dt><dd>{escape(str(v))}</dd>' for k, v in {
                     'Run': record['run_id'], 'Started UTC': record['started_at'], 'Fingerprint': record['fingerprint'],
                     'Source signature': record['source_signature_sha256'], 'Formula': '0.75 × hybrid + 0.25 × U-Net',
                 }.items()) + '</dl></details>']
    for name, title, text in [
        ('temporal_comparison.png', 'Temporal consistency',
         f"RMSE improves in {summary['blocks']['improved']}/7 blocks and {summary['years']['improved']}/14 years. A global mean can hide losses in specific periods."),
        ('calendar_intensity.png', 'Season and rainfall intensity',
         'Intensity groups use observed monthly means, not daily extremes. Outcomes are unknown at issue time; these groups are descriptive and cannot choose weights.'),
        ('spatial_comparison.png', 'Location',
         f"RMSE improves in {summary['blend_lower_rmse_grid_fraction']:.2%} of grid cells. This is a cell count, not a geographic-area or land-only fraction."),
    ]:
        encoded = base64.b64encode((folder / name).read_bytes()).decode('ascii')
        parts += [f'<h2>{title}</h2><p>{escape(text)}</p><img alt="{title}" src="data:image/png;base64,{encoded}">']
    for name, key, title in [('blocks', 'block', 'Blocks'), ('years', 'year', 'Years'),
                              ('calendar', 'calendar_month', 'Calendar months'), ('region', 'region', 'Latitude bands'),
                              ('intensity', 'intensity', 'Observed intensity')]:
        frame = pd.read_csv(folder / (name + '_comparison.csv'))
        cols = [key, 'n', 'hybrid_rmse', 'blend_rmse', 'delta_rmse', 'delta_mae', 'hybrid_bias', 'blend_bias', 'delta_absolute_bias']
        parts += [f'<details><summary>{title}: paired numbers</summary>' + table_html(frame[cols]) + '</details>']
    products = json.loads((folder / 'complementarity.json').read_text())
    rho = products['centered_error_correlation']
    correlation = f'{rho:.6f}' if rho is not None else 'undefined (zero error variance)'
    parts += ['<h2>Complementarity check</h2>',
              f'<p>Pooled error correlation: {correlation}. The exact MSE identity was checked against the blended maps. No optimal weight was estimated. The correlation is descriptive; it mixes dates and locations and is not an independent-sample significance test.</p>',
              '<h2>What was verified</h2><p>Original input hashes, dates, coordinates and units; original model scores; the fixed blend written and read back in float64; complete intensity partitions; the MSE identity; and the workflow artifact inventory. Archived rainfall maps were already nonnegative, so no additional clipping was applied.</p>',
              '<p>The workflow keeps the original diagnostics under diagnostics/ and the new candidate results under fixed_blend/. Code, input signatures and configuration are recorded. The operational hybrid and source-availability gate are unchanged.</p>']
    css = 'body{font:16px/1.55 system-ui,sans-serif;color:#172b36;max-width:1100px;margin:40px auto;padding:0 24px;background:#f8fafb}h1,h2{line-height:1.2}h2{margin-top:38px}.lead{font-size:20px}.note{border-left:4px solid #6246a5;padding:12px 16px;background:#f0edf7}details{margin:18px 0}summary{cursor:pointer;font-weight:650}dd{overflow-wrap:anywhere;margin-bottom:12px}table{border-collapse:collapse;width:100%;display:block;overflow:auto;font-size:14px;background:white}th,td{padding:9px 12px;border-bottom:1px solid #dae2e6;text-align:right}th:first-child,td:first-child{text-align:left}img{max-width:100%;height:auto}'
    (run / 'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fixed rainfall blend</title><style>' + css + '</style><body>' + '\n'.join(parts) + '</body></html>', encoding='utf-8', newline='\n')
    (run / 'REPORT.md').write_text('# Fixed 75/25 blend\n\n' + outcome + f' Difference: {delta:+.6f} mm/day.\n\n'
                                  + 'Open report.html for tables and embedded figures. One fixed weight; no fitting or promotion. Previously consulted development years.\n', encoding='utf-8', newline='\n')
