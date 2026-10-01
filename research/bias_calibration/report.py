"""Readable tables for a fixed candidate; no automatic model selection."""
from html import escape
import json
import pandas as pd
from research.common.presentation import display_frame


def render(output, result):
    parts=['<h1>U-Net bias calibration</h1>',
           '<p>Development comparison, 2007–2020. One global offset per fold, estimated on earlier chronological predictions and shrunk by 50%. Fixed reference model/U-Net weights: 75/25. Units: mm/day.</p>',
           '<p>Status: <strong>'+escape(result['status'])+'</strong>. A partial run is not the seven-block result. No operational model is replaced.</p>']
    for name in ['global','blocks','years','regions','calendar']:
        frame=pd.read_csv(output/(name+'.csv'))
        cols=[c for c in ['modelo','bloco','ano','mes','regiao','rmse','mae','vies','rmse_area','n'] if c in frame]
        parts += ['<h2>'+name.title()+'</h2>',display_frame(frame[cols].rename(columns={'modelo':'Model','vies':'Bias'})).to_html(index=False,float_format=lambda x:f'{x:.6f}',border=0)]
    offsets=[]
    for block in result['blocks']:
        record=json.loads((output/block/'complete.json').read_text())
        row=json.loads((output/block/record['attempt']/'calibration/offset.json').read_text())
        offsets.append(dict(block=block,raw_bias=row['raw_mean_bias'],applied_offset=row['offset_mm_day'],calibration_start=row['calibration_start'],calibration_end=row['calibration_end']))
    frame=pd.DataFrame(offsets);frame.to_csv(output/'offsets.csv',index=False)
    parts += ['<h2>Training-only corrections</h2>',frame.to_html(index=False,border=0),
        '<p>The correction is transferred from an auxiliary model fitted on an earlier window to the original refitted outer model. That transfer may help or hurt. Look for agreement across RMSE, MAE, absolute bias, years and regions.</p>',
        '<p>These years and 2021–2022 have already been inspected. No results from 2021–2022 are used to estimate the offsets, and they are not evaluated by this notebook. Historical source snapshots do not establish publication vintages or future skill.</p>']
    html='<!doctype html><html lang="en"><meta charset="utf-8"><title>U-Net bias calibration</title><style>body{font:16px/1.5 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#172b36}table{display:block;overflow:auto;border-collapse:collapse}td,th{padding:8px 12px;border-bottom:1px solid #ddd;text-align:right}</style><body>'+ '\n'.join(parts)+'</body></html>'
    (output/'report.html').write_text(html,encoding='utf-8')
