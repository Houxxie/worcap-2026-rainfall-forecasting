"""Evaluate a single fixed combination from saved maps; no model fitting is imported."""
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import base64
import json
import platform
import shutil
import zipfile
import numpy as np
import pandas as pd
from research.common.presentation import display_frame
import xarray as xr
from research.common.inputs import ROOT, require, sha256, write_json
from research.fixed_blend.evaluate import fixed_blend, error_products, summarize_products

PROTOCOL = Path(__file__).with_name('protocol.json')
SST_EVIDENCE = 'research/sst_extension/evidence/completed_20260930'
UNET_EVIDENCE = 'research/temporal_extension/evidence/kaggle_354026629'
MODELS = ('hybrid', 'hybrid_sst', 'unet', 'blend_reference', 'blend_sst', 'climatology')
SUMS = ['n', 'sse', 'error_sum', 'absolute_error_sum', 'sse_area', 'weight_area']


def protocol():
    plan = json.loads(PROTOCOL.read_text())
    require(plan['id'] == 'sst_unet_fixed_blend_2021_2022_v1'
            and plan['weights'] == [.75, .25] and plan['training_end'] == '2020-09-01', 'Fixed experiment changed.')
    require(plan['evaluation_start'] == '2021-01-01' and plan['evaluation_end'] == '2022-12-01', 'Evaluation dates changed.')
    require(not any(plan[k] for k in ['new_training', 'weight_search', 'conditional_switching', 'automatic_promotion']),
            'This experiment evaluates saved maps at fixed weights only.')
    # The source receipts establish the intended runs; no similarly named experiment is substituted.
    sst = json.loads((ROOT / SST_EVIDENCE / 'frozen.json').read_text())
    unet = json.loads((ROOT / UNET_EVIDENCE / 'frozen.json').read_text())
    require(plan['inputs']['sst_predictions']['sha256'] == sst['files']['models/predictions.nc']
            and plan['inputs']['unet_predictions']['sha256'] == unet['files']['predictions.nc']
            and plan['inputs']['observations']['sha256'] == unet['observation_file_sha256'], 'Pinned source hashes differ.')
    return plan


def locate_files(plan, supplied=None, roots=None):
    """Find exact bytes, allowing renamed files when their explicit paths are supplied."""
    supplied = supplied or {}
    roots = list(map(Path, roots or ['/kaggle/input', '/kaggle/working']))
    found, hashes = {}, {}
    for name, spec in plan['inputs'].items():
        if supplied.get(name):
            candidate = Path(supplied[name]).resolve()
            require(candidate.is_file(), f'Missing {name}: {candidate}')
            candidates = [candidate]
        else:
            candidates = sorted({p.resolve() for root in roots if root.is_dir()
                                 for p in root.rglob(spec['filename']) if p.is_file()})
        matches = []
        for p in candidates:
            if p not in hashes:
                hashes[p] = sha256(p)
            if hashes[p] == spec['sha256']:
                matches.append(p)
        require(matches, f'Missing exact {name} ({spec["filename"]}). Attach the complete saved output, '
                f'not the reports ZIP. Expected SHA-256 {spec["sha256"]}. Use a file path override if renamed.')
        found[name] = matches[0]
        print('Verified:', name, found[name], flush=True)
    return found


def validate_maps(sst, unet, dates, require_full_grid=True):
    require(set(sst.data_vars) == {'hybrid', 'hybrid_sst', 'climatology'}
            and set(unet.data_vars) == {'hybrid', 'unet', 'blend', 'climatology'}, 'Wrong source-model variables.')
    sst, unet = xr.align(sst, unet, join='exact')
    for ds in [sst, unet]:
        require(pd.DatetimeIndex(ds.time.values).equals(dates), 'Wrong, missing or reordered dates.')
        if require_full_grid:
            require(np.array_equal(ds.lat, np.arange(-60, 15.25, .25))
                    and np.array_equal(ds.lon, np.arange(-90, -24.75, .25)), 'Wrong spatial grid.')
        for name in ds:
            field = ds[name]
            require(field.dims == ('time', 'lat', 'lon') and field.attrs.get('units') == 'mm/day', 'Wrong dimensions or units.')
            require(np.isfinite(field.values).all() and (field.values >= 0).all(), 'Nonfinite or negative input rainfall.')
    for name in ['hybrid', 'climatology']:
        require(np.array_equal(sst[name].values, unet[name].values), f'{name} maps differ between source experiments.')
    reconstructed = fixed_blend(unet.hybrid.values, unet.unet.values)
    require(np.array_equal(reconstructed, unet.blend.values), 'Archived reference blend does not reproduce exactly.')
    return sst, unet


def combine(sst, unet, dates, require_full_grid=True):
    sst, unet = validate_maps(sst, unet, dates, require_full_grid)
    candidate = fixed_blend(sst.hybrid_sst.values, unet.unet.values)
    return xr.Dataset({
        'hybrid': sst.hybrid.astype('float64'), 'hybrid_sst': sst.hybrid_sst.astype('float64'),
        'unet': unet.unet.astype('float64'), 'blend_reference': unet.blend.astype('float64'),
        'blend_sst': xr.DataArray(candidate, dims=('time', 'lat', 'lon'), coords=sst.hybrid.coords,
                                attrs=dict(units='mm/day')),
        'climatology': sst.climatology.astype('float64')},
        attrs=dict(scope='Retrospective 2021-2022 development; no operational issuance', weights='0.75,0.25'))


def pooled(rows, keys):
    out = rows.groupby(keys, as_index=False)[SUMS].sum()
    out['rmse'] = np.sqrt(out.sse / out.n)
    out['mae'] = out.absolute_error_sum / out.n
    out['bias'] = out.error_sum / out.n
    out['rmse_area'] = np.sqrt(out.sse_area / out.weight_area)
    return out


def evaluate_maps(predictions, truth):
    truth, predictions = xr.align(truth, predictions, join='exact')
    require(truth.dims == ('time', 'lat', 'lon') and truth.attrs.get('units') == 'mm/day', 'Wrong observation dimensions/units.')
    require(set(predictions.data_vars) == set(MODELS), 'Incomplete candidate/reference set.')
    require(np.isfinite(truth.values).all() and (truth.values >= 0).all(), 'Invalid observed rainfall.')
    lat = truth.lat.values
    masks = dict(full_domain=np.ones(len(lat), bool), south_of_35S=lat < -35,
                 between_35S_and_15S=(lat >= -35) & (lat < -15), north_of_15S=lat >= -15)
    rows, products = [], {'reference': [], 'sst': []}
    for i, t in enumerate(pd.DatetimeIndex(truth.time.values)):
        observed = truth.values[i].astype('float64')
        for label, hybrid in [('reference', 'hybrid'), ('sst', 'hybrid_sst')]:
            products[label].append(error_products(observed, predictions[hybrid].values[i], predictions.unet.values[i]))
        for name in MODELS:
            pred = predictions[name].values[i].astype('float64')
            require(np.isfinite(pred).all() and (pred >= 0).all(), 'Invalid prediction values.')
            error = pred - observed
            for region, mask in masks.items():
                if not mask.any():
                    continue
                e = error[mask]
                area = np.broadcast_to(np.cos(np.deg2rad(lat[mask]))[:, None], e.shape)
                rows.append(dict(model=name, date=str(t.date()), year=t.year, month=t.month, region=region,
                            n=e.size, sse=float(np.square(e).sum()), error_sum=float(e.sum()),
                            absolute_error_sum=float(np.abs(e).sum()), sse_area=float((area * e**2).sum()), weight_area=float(area.sum())))
    detail = pd.DataFrame(rows)
    whole = detail[detail.region == 'full_domain']
    tables = dict(monthly_metrics=detail, global_metrics=pooled(whole, ['model']),
                  years=pooled(whole, ['model', 'year']), months=pooled(whole, ['model', 'date']),
                  regions=pooled(detail, ['model', 'region']), calendar=pooled(whole, ['model', 'month']))
    products = {k: summarize_products(v) for k, v in products.items()}
    g = tables['global_metrics'].set_index('model')
    for key, model in [('reference', 'blend_reference'), ('sst', 'blend_sst')]:
        np.testing.assert_allclose([products[key]['blend_sse_from_products'], products[key]['blend_sse_from_disagreement']],
                                   g.loc[model, 'sse'], rtol=1e-12, atol=1e-7)
    return tables, products


def compare_references(table):
    current = table.set_index('model')
    deviations = {}
    for folder, mapping in [(SST_EVIDENCE, {'hybrid': 'hybrid', 'hybrid_sst': 'hybrid_sst', 'climatology': 'climatology'}),
                            (UNET_EVIDENCE, {'hybrid': 'hybrid', 'unet': 'unet', 'blend': 'blend_reference', 'climatology': 'climatology'})]:
        archived = pd.read_csv(ROOT / folder / 'evaluation/global.csv').set_index('modelo')
        for old, new in mapping.items():
            delta = {k: float(current.loc[new, k] - archived.loc[old, 'vies' if k == 'bias' else k])
                     for k in ['rmse', 'mae', 'bias', 'rmse_area']}
            require(all(abs(v) < 1e-10 for v in delta.values()), 'Archived scores not reproduced: ' + new)
            require(int(current.loc[new, 'n']) == int(archived.loc[old, 'n']), 'Different reference counts.')
            deviations[f'{folder}/{old}'] = delta
    return deviations


def source_files():
    files = ['research/common/inputs.py', 'research/common/presentation.py', 'research/diagnostics/analyze_errors.py', 'research/fixed_blend/evaluate.py',
             'research/sst_unet_blend/protocol.json']
    for folder in [SST_EVIDENCE, UNET_EVIDENCE]:
        files.extend([folder + '/frozen.json', folder + '/evaluation/global.csv'])
    files += [p.relative_to(ROOT).as_posix() for p in Path(__file__).parent.glob('*.py')]
    return sorted(set(files))


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def execute(output, observations=None, sst_predictions=None, unet_predictions=None):
    started = perf_counter()
    plan = protocol()
    inputs = locate_files(plan, dict(observations=observations, sst_predictions=sst_predictions, unet_predictions=unet_predictions))
    output = Path(output).resolve()
    signature = dict(plan=plan, input_hashes={k: sha256(p) for k, p in inputs.items()},
                     code={n: sha256(ROOT / n) for n in source_files()},
                     versions=dict(numpy=np.__version__, pandas=pd.__version__, xarray=xr.__version__, python=platform.python_version()))
    if (output / 'complete.json').exists():
        record = json.loads((output / 'complete.json').read_text())
        require(json.loads((output / 'signature.json').read_text()) == signature, 'Different inputs, code or environment; select a fresh output.')
        verify_inventory(output, record['files'])
        print('Complete saved comparison verified; no recalculation.', flush=True)
        return record['summary']
    require(not output.exists(), 'An incomplete output exists. Preserve it and choose a new output directory.')
    require(not output.is_relative_to(ROOT / 'research') and not output.is_relative_to(ROOT / 'competition'), 'Output must be separate from code.')
    require(all(not p.is_relative_to(output) for p in inputs.values()), 'An input is inside the output directory.')
    output.mkdir(parents=True)
    write_json(output / 'signature.json', signature)
    for name in signature['code']:
        dest = output / 'code' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
        require(sha256(dest) == signature['code'][name], 'Code changed during snapshot.')
    dates = pd.date_range(plan['evaluation_start'], plan['evaluation_end'], freq='MS')
    with xr.open_dataset(inputs['sst_predictions']) as a, xr.open_dataset(inputs['unet_predictions']) as b:
        predictions = combine(a.load(), b.load(), dates)
    predictions.to_netcdf(output / 'predictions.nc', engine='h5netcdf',
                           encoding={n: dict(zlib=True, complevel=4) for n in predictions})
    with xr.open_dataset(output / 'predictions.nc') as saved:
        for n in predictions:
            np.testing.assert_array_equal(saved[n].values, predictions[n].values)
            require(saved[n].dtype == np.dtype('float64'), 'Output precision changed.')
    write_json(output / 'candidate_frozen.json', dict(frozen_at=utcnow(), sha256=sha256(output / 'predictions.nc'),
               evaluation_labels_opened=False, independent_holdout=False, forecast_issued=False))
    print('Source maps verified; candidate written and read back. Opening observations for scoring.', flush=True)
    with xr.open_dataset(inputs['observations']) as ds:
        truth = ds.tp.sel(time=dates).transpose('time', 'lat', 'lon').load()
    tables, products = evaluate_maps(predictions, truth)
    require((tables['global_metrics'].n == 1885464).all(), 'Incomplete full-grid scoring.')
    checks = compare_references(tables['global_metrics'])
    for name, table in tables.items():
        table.to_csv(output / (name + '.csv'), index=False)
    write_json(output / 'reference_checks.json', checks)
    write_json(output / 'error_products.json', products)
    g = tables['global_metrics'].set_index('model')
    comparisons = {}
    for reference in ['blend_reference', 'hybrid_sst', 'hybrid']:
        entry = {metric: float(g.loc['blend_sst', metric] - g.loc[reference, metric]) for metric in ['rmse', 'mae', 'bias', 'rmse_area']}
        entry['absolute_bias'] = float(abs(g.loc['blend_sst', 'bias']) - abs(g.loc[reference, 'bias']))
        entry['rmse_reduction_percent'] = float(100 * (1 - g.loc['blend_sst', 'rmse'] / g.loc[reference, 'rmse']))
        for name, key in [('years', 'year'), ('months', 'date')]:
            paired = tables[name].pivot(index=key, columns='model', values='rmse')
            delta = paired.blend_sst - paired[reference]
            entry[name] = dict(better=int((delta < 0).sum()), worse=int((delta > 0).sum()), tied=int((delta == 0).sum()))
        comparisons[reference] = entry
    summary = dict(status='complete', complete_comparison=True, complete_seven_blocks=False,
                   primary_reference='blend_reference', comparisons=comparisons,
                   archived_scores_reproduced=True, hybrid_maps_identical=True, fixed_blend_identity_verified=True,
                   model_fits=0, weight_search=False, independent_holdout=False,
                   forecast_issued=False, model_promoted=False, seconds=perf_counter() - started)
    write_json(output / 'summary.json', summary)
    render_report(output, tables, summary)
    require(all(sha256(p) == signature['input_hashes'][k] for k, p in inputs.items()), 'An input changed during evaluation.')
    frozen = json.loads((output / 'candidate_frozen.json').read_text())
    require(sha256(output / 'predictions.nc') == frozen['sha256'], 'Candidate changed during evaluation.')
    files = {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob('*'))
             if p.is_file() and '__pycache__' not in p.parts}
    write_json(output / 'complete.json', dict(files=files, summary=summary))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def verify_inventory(output, files):
    output = Path(output).resolve()
    for name, digest in files.items():
        p = (output / name).resolve()
        require(p.is_relative_to(output) and p.is_file() and sha256(p) == digest, 'Changed or missing artifact: ' + name)


def render_report(output, tables, summary):
    import matplotlib.pyplot as plt
    monthly = tables['months'].pivot(index='date', columns='model', values='rmse')
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), layout='constrained')
    for ax, reference, title in zip(axes, ['blend_reference', 'hybrid_sst'],
                                    ['Contribution of SST to the existing reference model/U-Net blend', 'Contribution of U-Net to the SST model']):
        delta = monthly.blend_sst - monthly[reference]
        ax.bar(range(24), delta, color=['#168293' if v < 0 else '#b56343' for v in delta])
        ax.axhline(0, color='#253746', lw=.8)
        ax.set(title=title, ylabel='RMSE change (mm/day)', xticks=range(0, 24, 3),
               xticklabels=[t[:7] for t in monthly.index[::3]])
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(output / 'monthly_comparison.png', dpi=150)
    plt.close(fig)
    encoded = base64.b64encode((output / 'monthly_comparison.png').read_bytes()).decode()
    html = '<!doctype html><html lang="en"><meta charset="utf-8"><title>SST and U-Net fixed blend</title><style>body{font:16px/1.5 system-ui;max-width:1100px;margin:35px auto;padding:0 20px;color:#213744}table{border-collapse:collapse}td,th{padding:7px;border-bottom:1px solid #ddd}img{max-width:100%}</style><body>'
    html += '<h1>Does SST improve the fixed reference model/U-Net blend?</h1><p>Candidate: 75% SST model + 25% unchanged U-Net. Primary reference: 75% original reference model + 25% the same U-Net. No models were fitted and no weights were searched.</p>'
    html += '<p>2021–2022 is a previously consulted development period, not an independent holdout. Historical publication vintages remain unverified; no operational model is promoted.</p>'
    html += display_frame(tables['global_metrics'][['model', 'rmse', 'mae', 'bias', 'rmse_area']]).to_html(index=False, float_format=lambda v: f'{v:.6f}')
    html += f'<img alt="Monthly contribution of SST and U-Net" src="data:image/png;base64,{encoded}">'
    html += '<h2>Annual errors</h2>' + display_frame(tables['years'][['model', 'year', 'rmse', 'mae', 'bias']]).to_html(index=False, float_format=lambda v: f'{v:.6f}')
    html += '<p>All errors use the full supplied grid. Negative changes favor the candidate. Complementarity was checked using both exact fixed-blend MSE identities.</p></body></html>'
    (output / 'report.html').write_text(html, encoding='utf-8')


def export_reports(output):
    output = Path(output).resolve()
    record = json.loads((output / 'complete.json').read_text())
    verify_inventory(output, record['files'])
    target = output.parent / 'sst_unet_blend_reports.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(output.rglob('*')):
            if p.is_file() and p.suffix in {'.json', '.csv', '.py', '.png', '.html'} and '__pycache__' not in p.parts:
                archive.write(p, p.relative_to(output).as_posix())
    return target
