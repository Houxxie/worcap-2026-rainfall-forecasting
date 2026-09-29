"""Generate a portable HTML report from verified diagnostic tables."""
from html import escape
from pathlib import Path
import base64
import json
import pandas as pd


def table_html(frame):
    frame = frame.copy()
    if 'model' in frame:
        frame['model'] = frame.model.replace({'hybrid': 'Hybrid', 'unet': 'U-Net', 'climatology': 'Climatology'})
    if 'n' in frame:
        frame['n'] = frame.n.map(lambda n: f'{int(n):,}')
    labels = {'model': 'Model', 'n': 'Samples', 'rmse': 'RMSE', 'mae': 'MAE', 'bias': 'Bias',
              'delta_rmse_vs_hybrid': 'Δ RMSE vs hybrid', 'calendar_month': 'Month', 'hybrid_rmse': 'Hybrid RMSE',
              'unet_rmse': 'U-Net RMSE', 'delta_rmse': 'Δ RMSE', 'region': 'Latitude band',
              'intensity': 'Observed mm/day', 'block': 'Block', 'year': 'Year', 'selected_epochs': 'Selected epoch',
              'inner_epochs_run': 'Epochs run', 'first_inner_rmse': 'First inner RMSE',
              'best_inner_rmse': 'Best inner RMSE', 'last_inner_rmse': 'Last inner RMSE'}
    return frame.rename(columns=labels).to_html(index=False, float_format=lambda x: f'{x:.6f}', border=0)


def render(run, record):
    run = Path(run)
    report = run / 'diagnostics'
    summary = json.loads((report / 'summary.json').read_text())
    scores = pd.read_csv(report / 'global.csv')
    baseline = scores.set_index('model').loc['hybrid', 'rmse']
    scores['delta_rmse_vs_hybrid'] = scores.rmse - baseline
    scores = scores[['model', 'rmse', 'mae', 'bias', 'delta_rmse_vs_hybrid', 'n']]
    signed = summary['unet_rmse'] - summary['hybrid_rmse']
    outcome = ('The hybrid has lower pooled RMSE.' if signed > 0 else
               'The U-Net has lower pooled RMSE on these development years.' if signed < 0 else 'Pooled RMSE is tied.')
    metadata = {
        'Run': record['run_id'], 'Mode': record['mode'], 'Comparison': 'Fixed hybrid / compact U-Net / monthly climatology',
        'Period': 'January 2007–December 2020 · seven chronological blocks',
        'Started (UTC)': record['started_at'], 'Run fingerprint': record['fingerprint'],
        'Source signature': record['source_signature_sha256'], 'Scope': 'Development years already consulted; not an independent holdout',
    }
    parts = [f'<h1>Rainfall experiment report</h1><p class="lead">{escape(outcome)} U-Net − hybrid RMSE: <strong>{signed:+.6f} mm/day</strong>.</p>',
             '<p class="note">No model is automatically promoted. No prospective forecast is issued. This report does not establish future skill.</p>',
             '<p>January 2007–December 2020 · Seven chronological blocks · Full 0.25° grid</p>',
             '<details><summary>Run details and provenance</summary><dl>' + ''.join(f'<dt>{escape(k)}</dt><dd>{escape(str(v))}</dd>' for k, v in metadata.items()) + '</dl></details>',
             '<h2>Overall comparison</h2><p>Monthly mean daily precipitation, in mm/day. RMSE pools squared errors and counts; it does not average block RMSEs. The full supplied grid includes ocean cells.</p>',
             table_html(scores)]
    sections = [
        ('Calendar month', 'calendar_month.png', 'calendar_comparison.csv', ['calendar_month', 'hybrid_rmse', 'unet_rmse', 'delta_rmse'],
         'Each month pools fourteen years. Positive differences favor the hybrid.'),
        ('Location', 'spatial_errors.png', 'region_comparison.csv', ['region', 'hybrid_rmse', 'unet_rmse', 'delta_rmse'],
         f"U-Net RMSE is lower in {summary['unet_lower_rmse_grid_fraction']:.2%} of grid cells. This is not an area-weighted or land-only percentage."),
        ('Observed rainfall intensity', 'rainfall_intensity.png', 'intensity_comparison.csv', ['intensity', 'n', 'hybrid_rmse', 'unet_rmse', 'delta_rmse'],
         'Bins describe the observed monthly mean, not daily extremes. The future observation is unavailable at issue time: these bins cannot select an operational model. Conditional bias is affected by regression to the mean.'),
        ('Original training histories', 'training_curves.png', 'training.csv', ['block', 'selected_epochs', 'inner_epochs_run', 'first_inner_rmse', 'best_inner_rmse', 'last_inner_rmse'],
         'Epoch selection uses inner validation only. Training losses are online anomaly losses; validation uses fixed end-of-epoch weights and clipped rainfall. Their gap is not an exact generalization-gap estimate.'),
    ]
    for title, image, table, columns, explanation in sections:
        encoded = base64.b64encode((report / image).read_bytes()).decode('ascii')
        parts += [f'<h2>{escape(title)}</h2><p>{escape(explanation)}</p>',
                  f'<img alt="{escape(title)}" src="data:image/png;base64,{encoded}">',
                  '<details><summary>View numerical results</summary>' + table_html(pd.read_csv(report / table)[columns]) + '</details>']
    for name, title in [('blocks', 'Two-year blocks'), ('years', 'Individual years')]:
        table = pd.read_csv(run / 'comparison' / (name + '.csv'))
        parts += [f'<details><summary>{title}</summary>', table_html(table), '</details>']
    parts += ['<h2>Reproduction and limits</h2><p>config.json records resolved paths. identity.json records code, data and environment fingerprints. source_evidence preserves the source run metadata. manifest.json records the output hashes. Use the workflow verify command to check the saved package.</p>',
              f'<p>{summary["monthly_region_scores_reproduced"]:,} monthly model × region error records were reproduced from the maps. Dates, grid, units, prediction hashes and selected epochs were checked.</p>',
              '<p>Historical source publication vintages are not established by this check. The tree models sample training grid points; the U-Net trains on complete maps. Spatial samples are correlated. Observed improvements are descriptive, without a claim of statistical significance.</p>']
    html = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rainfall experiment report</title><style>body{font:16px/1.55 system-ui,sans-serif;color:#172b36;max-width:1100px;margin:40px auto;padding:0 24px;background:#f8fafb}h1,h2{line-height:1.2}h2{margin-top:44px}.lead{font-size:20px}.note{border-left:4px solid #087e8b;padding:12px 16px;background:#e9f4f5}details{margin:18px 0}summary{cursor:pointer;font-weight:650;margin-bottom:12px}dt{font-weight:650}dd{margin:0 0 12px;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;font-size:14px;display:block;overflow:auto;background:white}th,td{padding:9px 12px;border-bottom:1px solid #dae2e6;text-align:right}th:first-child,td:first-child{text-align:left}img{max-width:100%;height:auto;background:white;margin:12px 0}</style><body>' + '\n'.join(parts) + '</body></html>'
    (run / 'report.html').write_text(html, encoding='utf-8', newline='\n')
    (run / 'REPORT.md').write_text(
        '# Rainfall experiment report\n\n' + outcome + f' U-Net minus hybrid RMSE: {signed:+.6f} mm/day.\n\n'
        + 'Open report.html for the complete tables and embedded figures.\n\n'
        + f'Run: `{record["run_id"]}`. Mode: `{record["mode"]}`.\n\n'
        + 'Development comparison, 2007–2020. No automatic promotion and no prospective forecast.\n', encoding='utf-8', newline='\n')
