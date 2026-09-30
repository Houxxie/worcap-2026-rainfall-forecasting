"""One fixed-budget residual-learning pilot; never automatically run the seven folds."""
from pathlib import Path
from datetime import datetime, timezone
import gc
import json
import platform
import shutil
import tempfile
import time
import zipfile
import numpy as np
import pandas as pd
import torch
import xarray as xr
from research.common.inputs import ROOT, Inputs, Context, preflight, require, sha256, write_json
from research.common import baseline
from research.spatial_unet.training import MonthlyMaps, fit_network, inference_maps
from .data import crossfit, crossfit_plan, ResidualMaps, ForecastMaps, forecast_field, validate_field

HERE = Path(__file__).parent


def protocol():
    plan = json.loads((HERE / 'protocol.json').read_text())
    source = ROOT / 'research/spatial_unet/protocol.json'
    require(sha256(source) == plan['source_protocol_sha256'], 'Original U-Net protocol changed.')
    require(plan['id'] == 'hybrid_residual_unet_pilot_v1' and plan['block'] == 'C', 'Different pilot.')
    require(plan['epochs'] == 8 and plan['neural_training_months'] == 120 and plan['neural_fits'] == 2,
            'The pilot compute budget changed.')
    require(plan['evaluation_start'] == '2019-01-01' and plan['evaluation_months'] == 24 and
            plan['crossfit_forecast_months'] == 24 and plan['label_lag_months'] == 4 and plan['features'] == 31,
            'The pilot calendar or inputs changed.')
    require(not plan['weight_search'] and plan['direct_blend'] == {'hybrid': .75, 'direct_unet': .25}, 'Unexpected weight search.')
    config = json.loads(source.read_text())
    config['max_epochs'] = plan['epochs']
    return plan, config


def source_files():
    files = [HERE / 'protocol.json', ROOT / 'research/spatial_unet/protocol.json',
        ROOT / 'research/lagged_sources/library.py', ROOT / 'research/lagged_sources/ocean_indices.csv',
        ROOT / 'research/lagged_sources/official_hashes.json', ROOT / 'research/lagged_sources/evidence/metricas_blocos.csv']
    for folder in ['common', 'spatial_unet', 'residual_unet']:
        files += list((ROOT / 'research' / folder).glob('*.py'))
    return sorted(set(files))


class Stages:
    """Reuse completed stages with exact artifacts; stop on incomplete stages."""
    def __init__(self, output, signature):
        self.output, self.signature = Path(output), signature

    def __call__(self, name, folder, action):
        folder = Path(folder)
        done = folder / 'complete.json'
        if done.is_file():
            record = json.loads(done.read_text())
            require(record['signature'] == self.signature, 'Changed cached-stage signature.')
            for relative, digest in record['files'].items():
                path = (folder / relative).resolve()
                require(path.is_relative_to(folder.resolve()) and path.is_file() and sha256(path) == digest,
                        'Changed stage artifact: ' + relative)
            print(name, '| verified saved stage; no refitting.', flush=True)
            return
        require(not folder.exists(), f'Incomplete stage {folder}. Preserve it and use a new output directory; do not delete completed artifacts.')
        folder.mkdir(parents=True)
        start = time.perf_counter()
        write_json(self.output / 'progress.json', dict(stage=name, status='running', started_utc=datetime.now(timezone.utc).isoformat()))
        print(name, '| starting', flush=True)
        action()
        elapsed = time.perf_counter() - start
        write_json(done, dict(signature=self.signature, seconds=elapsed, files={
            p.relative_to(folder).as_posix(): sha256(p) for p in folder.rglob('*') if p.is_file()}))
        print(name, f'| completed in {elapsed / 60:.1f} min.', flush=True)


def run_pilot(data, plan, config, output, device, stages):
    dates = pd.date_range(plan['evaluation_start'], periods=24, freq='MS')
    cutoff = dates[0] - pd.DateOffset(months=4)
    training = data.lib.calendario_pareado(cutoff)
    reference = data.lib.janela_referencia(cutoff)
    neural_dates = training[-plan['neural_training_months']:]
    require(len(neural_dates) == 120 and neural_dates[-1] == cutoff, 'Invalid neural training calendar.')
    crossfit_plan(data, neural_dates)
    write_json(output / 'split.json', dict(outer=dates.strftime('%Y-%m-%d').tolist(),
        hybrid_training=training.strftime('%Y-%m-%d').tolist(), neural_training=neural_dates.strftime('%Y-%m-%d').tolist(),
        reference=reference.strftime('%Y-%m-%d').tolist(), same_neural_examples=True,
        independent_holdout=False, epoch_selection=False, complete_seven_blocks=False))

    def reference_action():
        folder = output / 'reference'
        state = baseline.fit(data, cutoff, folder / 'hybrid')
        prediction = baseline.predict(data, state, dates)
        validate_field(prediction, data, dates)
        prediction.rename('hybrid').to_netcdf(folder / 'predictions.nc', engine='h5netcdf')
    stages('reference_hybrid_1_of_6', output / 'reference', reference_action)
    with xr.open_dataarray(output / 'reference/predictions.nc') as f:
        hybrid = f.load()
    validate_field(hybrid, data, dates)
    historical = crossfit(data, neural_dates, output / 'crossfit', stages)

    # Full outer-training references stay identical in both neural variants. Only
    # the neural examples are capped at 120 months. No outer rainfall enters them.
    context = Context(data, reference, training)
    context.save(output / 'context')
    temporary = output / 'temporary_maps'
    temporary.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary) as temp:
        train = MonthlyMaps(context, neural_dates, Path(temp) / 'shared_training', training=True)
        try:
            residual = ResidualMaps(train, historical, data)
            for name, dataset, base in [('direct_unet', train, None), ('residual_unet', residual, hybrid)]:
                folder = output / name
                def action(name=name, dataset=dataset, base=base, folder=folder):
                    net, epochs = fit_network(dataset, config, folder / 'network', device, epochs=plan['epochs'])
                    require(epochs == 8, 'Different training budget.')
                    forecasts = inference_maps(net, ForecastMaps(context, dates, train.scaling, base), device, config['batch_size'])
                    field = forecast_field(data, dates, forecasts, name)
                    validate_field(field, data, dates)
                    field.to_netcdf(folder / 'predictions.nc', engine='h5netcdf')
                    del net
                    gc.collect()
                    if device.type == 'cuda': torch.cuda.empty_cache()
                stages(name + '_8_epochs', folder, action)
        finally:
            train.close()
    predictions = {'hybrid': hybrid}
    for name in ['direct_unet', 'residual_unet']:
        with xr.open_dataarray(output / name / 'predictions.nc') as f:
            predictions[name] = f.load()
    predictions['direct_blend'] = .75 * hybrid.astype('float64') + .25 * predictions['direct_unet'].astype('float64')
    predictions['direct_blend'].attrs = dict(units='mm/day')
    xr.Dataset(predictions).to_netcdf(output / 'predictions.nc', engine='h5netcdf',
        encoding={name: dict(zlib=True, complevel=4) for name in predictions})
    with xr.open_dataset(output / 'predictions.nc') as saved:
        for name, field in predictions.items():
            np.testing.assert_array_equal(saved[name].values, field.values)
    # Freeze all candidates before opening the outer rainfall for scoring.
    write_json(output / 'candidate_record.json', dict(predictions_sha256=sha256(output / 'predictions.nc'),
        created_utc=datetime.now(timezone.utc).isoformat(), outer_targets_used_for_fitting=False,
        epoch_selection=False, weight_search=False, independent_holdout=False, model_promoted=False))
    rows = data.lib.pesquisa_diagnosticos(data.rain.sel(time=dates), predictions, plan['block'])
    rows.to_csv(output / 'monthly_metrics.csv', index=False)
    domain = rows[rows.regiao == 'dominio_inteiro']
    for name, table, groups in [('global', domain, ['modelo']), ('years', domain, ['modelo', 'ano']),
                                ('calendar', domain, ['modelo', 'mes']), ('regions', rows, ['modelo', 'regiao'])]:
        data.lib.pesquisa_resumir(table, groups).to_csv(output / (name + '.csv'), index=False)
    scores = pd.read_csv(output / 'global.csv').set_index('modelo')
    archived = pd.read_csv(ROOT / 'research/lagged_sources/evidence/metricas_blocos.csv')
    old = archived[(archived.bloco == plan['block']) & (archived.modelo == 'controle_defasado')].iloc[0]
    for key in ['rmse', 'mae', 'vies']:
        require(abs(scores.loc['hybrid', key] - old[key]) < 1e-6, 'Historical hybrid did not reproduce: ' + key)
    write_json(output / 'baseline_check.json', dict(reproduced=True, tolerance=1e-6,
        rmse_archived=float(old.rmse), rmse_current=float(scores.loc['hybrid', 'rmse'])))
    comparisons = {}
    for comparator in ['hybrid', 'direct_unet', 'direct_blend']:
        a, b = scores.loc['residual_unet'], scores.loc[comparator]
        comparisons[comparator] = {k: float(a[k] - b[k]) for k in ['rmse', 'mae', 'vies', 'rmse_area']}
        comparisons[comparator]['delta_absolute_bias'] = float(abs(a.vies) - abs(b.vies))
    result = dict(status='complete', pilot_complete=True, blocks=[plan['block']], complete_seven_blocks=False,
        comparisons=comparisons, independent_holdout=False, model_promoted=False, forecast_issued=False,
        note='One fixed-budget, previously consulted block. The 120-month/eight-epoch direct control is new, not the archived full-training U-Net.')
    write_json(output / 'summary.json', result)
    html = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Residual U-Net pilot</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#162a38}table{border-collapse:collapse}th,td{padding:9px;border:1px solid #ccd5da;text-align:right}h1,h2{color:#126478}</style><h1>Learning the hybrid forecast error</h1><p>Block C: January 2019–December 2020. Same 120 training months and eight epochs for both neural targets. Five chronological hybrid fits generate out-of-fit residuals. Previously consulted development data; no operational promotion.</p>'
    for name in ['global', 'years', 'regions']:
        html += '<h2>' + name.title() + '</h2>' + pd.read_csv(output / (name + '.csv')).to_html(index=False, float_format=lambda x: f'{x:.6f}')
    (output / 'report.html').write_text(html + '</html>', encoding='utf-8')
    return result


def export_reports(output):
    output = Path(output)
    archive = output / 'residual_unet_pilot_reports.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in output.rglob('*'):
            if p.is_file() and p.suffix in {'.json', '.csv', '.html'} and 'code' not in p.relative_to(output).parts:
                z.write(p, p.relative_to(output))
    return archive


def execute(output, official=None, seas5=None, cfsv2=None, device='cuda'):
    plan, config = protocol()
    require(device == 'cuda' and torch.cuda.is_available(), 'Enable a Kaggle GPU. This pilot does not silently fall back to CPU.')
    paths, audit = preflight(official, seas5, cfsv2)
    output = Path(output).resolve()
    require(not output.is_relative_to(ROOT / 'research') and not output.is_relative_to(ROOT / 'competition') and
            all(not output.is_relative_to(p) for p in paths.values()), 'Keep output outside source/input folders.')
    output.mkdir(parents=True, exist_ok=True)
    code = {p.relative_to(ROOT).as_posix(): sha256(p) for p in source_files()}
    signature = dict(plan=plan, config=config, inputs=audit, code=code, python=platform.python_version(),
        torch=torch.__version__, device=device, gpu=torch.cuda.get_device_name(0))
    sigfile = output / 'signature.json'
    if sigfile.exists():
        require(json.loads(sigfile.read_text()) == signature, 'Changed code, inputs or environment; choose a new OUTPUT.')
    else:
        require(not any(output.iterdir()), 'New output must be empty.')
        write_json(sigfile, signature)
        for name in code:
            p = output / 'code' / name
            p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, p)
    start = time.perf_counter()
    write_json(output / 'run_state.json', dict(status='running', pilot_complete=False, complete_seven_blocks=False))
    print('PILOT C only | six hybrid fits (five chronological + one reference) | two neural fits × eight epochs.', flush=True)
    try:
        data = Inputs(**paths, output=output)
        result = run_pilot(data, plan, config, output, torch.device(device), Stages(output, sha256(sigfile)))
        _, ending = preflight(**paths)
        require(ending == audit and all(sha256(ROOT / n) == h for n, h in code.items()), 'Inputs or source changed during execution.')
        write_json(output / 'run_state.json', dict(status='complete', pilot_complete=True, complete_seven_blocks=False,
            input_hashes_rechecked=True, source_hashes_rechecked=True, seconds_this_invocation=time.perf_counter() - start,
            forecast_issued=False, model_promoted=False))
        print(json.dumps(result, indent=2), flush=True)
        return result
    except Exception as error:
        write_json(output / 'run_state.json', dict(status='failed', pilot_complete=False,
            error_type=type(error).__name__, error=str(error)))
        raise
    finally:
        export_reports(output)
