"""Portable, readable reports; figures use saved scientific fields only."""
from html import escape
from pathlib import Path
import base64
import json
import numpy as np
import pandas as pd

from research.common.presentation import MODEL_LABELS as LABELS, display_frame


def forecast_map(path, output, month=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import xarray as xr
    with xr.open_dataset(path) as ds:
        dates = pd.DatetimeIndex(ds.time.values)
        require_month = month or dates[-1].strftime('%Y-%m')
        selected = np.flatnonzero(dates.strftime('%Y-%m') == require_month)
        if len(selected) != 1:
            raise ValueError(f'Map month {require_month} is missing or duplicated in {path}.')
        names = [n for n in ['hybrid', 'hybrid_sst', 'blend_sst', 'blend', 'unet', 'climatology'] if n in ds][:3]
        if not names:
            raise ValueError('No supported rainfall fields for a map preview.')
        fields = [ds[n].isel(time=int(selected[0])).transpose('lat', 'lon').load() for n in names]
        if any(f.attrs.get('units') != 'mm/day' or not np.isfinite(f.values).all() for f in fields):
            raise ValueError('Map preview requires finite rainfall in mm/day.')
        fig, axes = plt.subplots(1, len(fields), figsize=(5*len(fields), 5), squeeze=False, constrained_layout=True)
        limit = max(float(f.max()) for f in fields)
        for axis, name, field in zip(axes[0], names, fields):
            im = axis.pcolormesh(field.lon, field.lat, field.values, shading='auto', vmin=0, vmax=max(limit, 1e-6), cmap='YlGnBu')
            axis.set(title=LABELS.get(name, name), xlabel='Longitude (degrees)', ylabel='Latitude (degrees)')
            axis.set_aspect('equal')
        fig.colorbar(im, ax=axes.ravel().tolist(), label='Monthly mean rainfall (mm/day)', shrink=.8)
        fig.suptitle(f'Saved retrospective predictions · {require_month}')
        fig.savefig(output, dpi=130)
        plt.close(fig)
    return require_month


def table(frame):
    frame = display_frame(frame)
    return frame.to_html(index=False, escape=True, border=0, float_format=lambda v: f'{v:.6f}')


def render(folder, info):
    folder = Path(folder)
    is_prepare = info['kind'] == 'prepare'
    cards = [('Task', info['task']), ('Period', info['period']), ('Status', info['status'])]
    body = [f'<p class="eyebrow">WorCAP · Rainfall forecasting</p><h1>{escape(info["title"])}</h1>',
            '<div class="cards">'+''.join(f'<div><span>{escape(k)}</span><strong>{escape(str(v))}</strong></div>' for k,v in cards)+'</div>',
            f'<p class="notice">{escape(info["limitations"])}</p>']
    if is_prepare:
        snapshot = info['registry']
        body += ['<h2>Monthly preparation</h2>', f'<p>{escape(snapshot["next_action"])}</p>',
                 f'<p>Deadline: <strong>{escape(snapshot["deadline_utc"])}</strong>. Checked: {escape(snapshot["checked_at"])}.</p>',
                 '<ul>'+''.join(f'<li>{escape(x)}</li>' for x in snapshot['status']['faltantes'])+'</ul>',
                 '<h2>Source arrivals</h2>', table(pd.DataFrame(snapshot['sources'])[
                     ['source', 'required_month', 'status', 'received_at', 'latest_receipt']]),
                 '<p>Receipt time is the acquisition time. First publication time remains unknown.</p>',
                 '<h2>Forecast map</h2><p>This task checks preparation and does not issue a forecast or invent a map.</p>']
    else:
        scores = pd.read_csv(folder/'metrics.csv')
        primary = info['primary']
        baseline = scores.set_index('model').loc[primary]
        scores['delta_rmse'] = scores.rmse - baseline.rmse
        scores['delta_mae'] = scores.mae - baseline.mae
        scores['delta_abs_bias'] = scores.bias.abs() - abs(baseline.bias)
        body += ['<h2>Comparison with the reference</h2>',
                 f'<p>Reference: <strong>{escape(LABELS.get(primary,primary))}</strong>. Units: mm/day. Negative differences favor the candidate. Bias is prediction minus observation.</p>',
                 table(scores[['model','rmse','mae','bias','delta_rmse','delta_mae','delta_abs_bias']]),
                 '<p>Inspect all three measures: a lower RMSE can coexist with worse MAE or more systematic overprediction. No model is promoted automatically.</p>',
                 '<h2>Individual years</h2>', table(pd.read_csv(folder/'years.csv')[['model','year','rmse','mae','bias']])]
        if not info['figures']:
            body += ['<h2>Maps</h2><p>No map is present in this report input. Attach the full prediction output to include a preview.</p>']
        for item in info['figures']:
            data = base64.b64encode((folder/item['file']).read_bytes()).decode('ascii')
            body += [f'<h2>{escape(item["title"])}</h2><p>{escape(item["caption"])}</p>',
                     f'<img alt="{escape(item["title"])}" src="data:image/png;base64,{data}">']
    body += ['<h2>Sources and reproducibility</h2>',
             '<details><summary>Show the source record and verification scope</summary><pre>'+escape(json.dumps(info['provenance'], indent=2, ensure_ascii=False))+'</pre></details>',
             '<h2>Files to keep</h2><p>Download the small report ZIP for review. Save the complete notebook output to preserve maps and models. A report ZIP cannot replace those files for a later saved-map experiment.</p>',
             '<p>Run ID: <code>'+escape(info['run_id'])+'</code>. Configuration and completion records are stored alongside this page.</p>']
    css = '''body{font:16px/1.6 system-ui,sans-serif;background:#f4f7fa;color:#182b3b;max-width:1150px;margin:40px auto;padding:0 24px}h1{font-size:36px;line-height:1.2;max-width:900px}h2{margin-top:40px;line-height:1.3}.eyebrow{color:#31667b;letter-spacing:.08em;text-transform:uppercase;font-size:13px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:28px 0}.cards>div{padding:18px;background:white;border:1px solid #dae4ec;border-radius:10px}.cards span{display:block;color:#526475;font-size:13px}.cards strong{display:block;font-size:19px;overflow-wrap:anywhere}.notice{border-left:4px solid #d29a22;padding:14px 18px;background:#fff5d9}table{display:block;overflow:auto;width:100%;border-collapse:collapse;background:white;font-size:14px}td,th{text-align:right;padding:10px 12px;border-bottom:1px solid #dce5ed}td:first-child,th:first-child{text-align:left}img{max-width:100%;height:auto;background:white;border-radius:8px}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eaf0f5;padding:18px;font-size:12px}code{overflow-wrap:anywhere}@media(max-width:700px){.cards{grid-template-columns:1fr}body{padding:0 16px}h1{font-size:28px}}'''
    html = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rainfall workbench report</title><style>'+css+'</style></head><body>'+'\n'.join(body)+'</body></html>'
    (folder/'report.html').write_text(html, encoding='utf-8', newline='\n')
