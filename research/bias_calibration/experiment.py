"""Compare one fixed bias correction on the original seven development blocks."""
from pathlib import Path
from datetime import datetime, timezone
import gc
import json
import platform
import shutil
import tempfile
import zipfile
import numpy as np
import pandas as pd
import torch
import xarray as xr
from research.common.inputs import ROOT, Inputs, Context, preflight, require, sha256, write_json
from research.spatial_unet.run_experiment import run_block as original_block
from research.spatial_unet.training import MonthlyMaps, fit_network, inference_maps
from .calibration import splits, ForecastMaps, estimate, correct
from .report import render

HERE = Path(__file__).parent


def protocol():
    plan = json.loads((HERE/'protocol.json').read_text())
    source = ROOT/'research/spatial_unet/protocol.json'
    require(sha256(source) == plan['source_protocol_sha256'], 'Original U-Net configuration changed.')
    require(plan['id'] == 'unet_global_bias_v1' and plan['blocks'] == ['H1','H2','H3','H4','A','B','C'], 'Changed experiment.')
    require(plan['calibration_months'] == plan['prior_zero_bias_months'] == 24 and plan['label_lag_months'] == 4, 'Changed calibration settings.')
    require(plan['weights'] == {'hybrid': .75, 'unet': .25} and not plan['weight_search'] and not plan['hyperparameter_search'], 'Unexpected search.')
    return plan, json.loads(source.read_text())


def source_files():
    paths = [HERE/'protocol.json', ROOT/'research/spatial_unet/protocol.json',
             ROOT/'research/lagged_sources/library.py', ROOT/'research/lagged_sources/official_hashes.json',
             ROOT/'research/lagged_sources/ocean_indices.csv', ROOT/'research/lagged_sources/evidence/metricas_blocos.csv']
    for folder in ['common', 'spatial_unet', 'bias_calibration']:
        paths += list((ROOT/'research'/folder).glob('*.py'))
    return sorted(set(paths))


def check(official=None, seas5=None, cfsv2=None, device='cuda'):
    protocol()
    require(device in ['cuda', 'cpu'], 'Use cuda or cpu.')
    require(device != 'cuda' or torch.cuda.is_available(), 'Enable a Kaggle GPU before fitting.')
    paths, audit = preflight(official, seas5, cfsv2)
    print('Existing competition, SEAS5 and CFSv2 inputs verified. No new download.', flush=True)
    return paths, audit


def fit_calibration(data, training, reference, config, folder, device):
    """Two extra fits: early epoch selection, then an auxiliary calibration predictor."""
    folder.mkdir(parents=True, exist_ok=False)
    split = splits(training, reference)
    write_json(folder/'split.json', {k: v.strftime('%Y-%m-%d').tolist() for k, v in split.items()})
    with tempfile.TemporaryDirectory(prefix='maps_', dir=folder) as temporary:
        temp = Path(temporary)
        context = Context(data, split['selection_reference'], split['selection_fit'])
        context.save(folder/'selection')
        train = MonthlyMaps(context, split['selection_fit'], temp/'selection_train', training=True)
        valid = MonthlyMaps(context, split['selection_validation'], temp/'selection_valid', scaling=train.scaling)
        try:
            net, epochs = fit_network(train, config, folder/'selection', device, validation=valid)
        finally:
            train.close(); valid.close()
        del net, train, valid, context
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()
        context = Context(data, split['reference'], split['fit'])
        context.save(folder/'auxiliary')
        train = MonthlyMaps(context, split['fit'], temp/'auxiliary_train', training=True)
        try:
            net, _ = fit_network(train, config, folder/'auxiliary', device, epochs=epochs)
            scaling = {k: v.copy() for k, v in train.scaling.items()}
        finally:
            train.close()
        # Target rainfall is not a field of this dataset. It is read later for the offset.
        values = inference_maps(net, ForecastMaps(context, split['calibration'], scaling), device, config['batch_size'])
        prediction = xr.DataArray(values, dims=('time','lat','lon'), coords=dict(
            time=split['calibration'], lat=data.rain.lat, lon=data.rain.lon), attrs=dict(units='mm/day'), name='unet')
        prediction.to_netcdf(folder/'predictions.nc', engine='h5netcdf')
        with xr.open_dataarray(folder/'predictions.nc') as saved: xr.testing.assert_identical(saved, prediction)
        record, monthly = estimate(prediction, data.rain, split['calibration'], training[-1], split['fit'][-1], split['selection_validation'][-1])
        record['auxiliary_selected_epochs'] = epochs
        record['prediction_sha256'] = sha256(folder/'predictions.nc')
        write_json(folder/'offset.json', record)
        monthly.to_csv(folder/'monthly_bias.csv', index=False)
        del net, train, context
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()
    return record


def run_block(data, block, config, directory, device):
    directory.mkdir(parents=True, exist_ok=False)
    # The unchanged original runner reproduces the hybrid and fits the reference U-Net.
    # Its scores are recorded, never passed to calibration or used for selection.
    original_block(data, block, config, directory/'reference', device)
    m = data.lib
    cutoff = m.origem_mensal([pd.Timestamp(block['inicio'])], 4)[0]
    training, reference = m.calendario_pareado(cutoff), m.janela_referencia(cutoff)
    offset = fit_calibration(data, training, reference, config, directory/'calibration', device)
    with xr.open_dataset(directory/'reference/predictions.nc') as source:
        predictions = source.load()
    predictions['unet_corrected'] = correct(predictions.unet, offset)
    predictions['blend'] = .75*predictions.hybrid.astype('float64') + .25*predictions.unet.astype('float64')
    predictions['blend_corrected'] = .75*predictions.hybrid.astype('float64') + .25*predictions.unet_corrected
    for v in predictions.data_vars.values():
        v.attrs['units'] = 'mm/day'
        require(np.isfinite(v.values).all() and (v.values >= 0).all(), 'Invalid candidate map.')
    predictions.attrs['bias_protocol'] = 'unet_global_bias_v1'
    predictions.to_netcdf(directory/'predictions.nc', engine='h5netcdf', encoding={k:dict(zlib=True, complevel=4) for k in predictions})
    with xr.open_dataset(directory/'predictions.nc') as saved: xr.testing.assert_identical(saved, predictions)
    write_json(directory/'candidate_record.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        predictions_sha256=sha256(directory/'predictions.nc'), offset_sha256=sha256(directory/'calibration/offset.json'),
        outer_targets_used_for_calibration=False, independent_holdout=False,
        note='Development years already consulted. The original reference runner scores its control before calibration; these scores are not calibration inputs.'))
    dates = pd.DatetimeIndex(predictions.time.values)
    metrics = m.pesquisa_diagnosticos(data.rain.sel(time=dates), {k:predictions[k] for k in predictions}, block['nome'])
    metrics.to_csv(directory/'monthly_metrics.csv', index=False)
    return metrics


def verify_block(parent, signature):
    record = json.loads((parent/'complete.json').read_text())
    require(record['signature'] == signature, 'Changed completed-block signature.')
    attempt = (parent/record['attempt']).resolve()
    require(attempt.is_relative_to(parent.resolve()) and attempt != parent.resolve(), 'Invalid attempt path.')
    for name, digest in record['files'].items():
        path = (attempt/name).resolve()
        require(path.is_relative_to(attempt) and path.is_file() and sha256(path) == digest, 'Changed completed artifact: '+name)
    return pd.read_csv(attempt/'monthly_metrics.csv')


def summarize(data, output, frames, expected):
    metrics = pd.concat(frames, ignore_index=True)
    require(not metrics.duplicated(['modelo','mes_alvo','regiao']).any(), 'Duplicate evaluation rows.')
    metrics.to_csv(output/'monthly_metrics.csv', index=False)
    domain = metrics[metrics.regiao == 'dominio_inteiro']
    for name, rows, groups in [('global',domain,['modelo']),('blocks',domain,['modelo','bloco']),
            ('years',domain,['modelo','ano']),('regions',metrics,['modelo','regiao']),('calendar',domain,['modelo','mes'])]:
        if name == 'calendar':
            rows = rows.copy(); rows['mes'] = pd.to_datetime(rows.mes_alvo).dt.month
        data.lib.pesquisa_resumir(rows, groups).to_csv(output/(name+'.csv'), index=False)
    totals = pd.read_csv(output/'global.csv').set_index('modelo')
    blocks = sorted(domain.bloco.unique().tolist())
    comparisons = {}
    for reference in ['hybrid','blend']:
        comparisons[reference] = {metric: float(totals.loc['blend_corrected',metric]-totals.loc[reference,metric]) for metric in ['rmse','mae','vies']}
        comparisons[reference]['delta_absolute_bias'] = float(abs(totals.loc['blend_corrected','vies'])-abs(totals.loc[reference,'vies']))
    result = dict(status='complete' if set(blocks)==set(expected) else 'partial', blocks=blocks,
        complete_seven_blocks=len(blocks)==7, comparisons= comparisons, model_promoted=False,
        forecast_issued=False, independent_holdout=False, weight_search=False)
    write_json(output/'summary.json', result)
    render(output, result)
    export_reports(output)
    print(json.dumps(result, indent=2), flush=True)
    return result


def export_reports(output):
    path = output/'bias_calibration_reports.zip'
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in output.rglob('*'):
            if p.is_file() and p.suffix in {'.json','.csv','.html','.png'} and 'code' not in p.relative_to(output).parts:
                archive.write(p, p.relative_to(output))
    return path


def execute(output, official=None, seas5=None, cfsv2=None, device='cuda'):
    plan, config = protocol()
    paths, audit = check(official, seas5, cfsv2, device)
    output = Path(output).resolve()
    require(not output.is_relative_to(ROOT/'research') and not output.is_relative_to(ROOT/'competition'), 'Keep outputs separate from source.')
    require(all(not output.is_relative_to(p) for p in paths.values()), 'Output overlaps an input.')
    output.mkdir(parents=True, exist_ok=True)
    source = {p.relative_to(ROOT).as_posix(): sha256(p) for p in source_files()}
    signature = dict(plan=plan, config=config, inputs=audit, code=source, device=device,
        python=platform.python_version(), torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if device=='cuda' else None)
    sigfile = output/'signature.json'
    if sigfile.exists():
        require(json.loads(sigfile.read_text()) == signature, 'Run inputs/code/environment changed. Keep this output and choose a new OUTPUT folder.')
    else:
        require(not any(output.iterdir()), 'New output must be empty.')
        write_json(sigfile, signature)
        for name in source:
            target=output/'code'/name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/name,target)
    data = Inputs(**paths, output=output)
    torch_device = torch.device(device)
    if device=='cpu': torch.set_num_threads(4)
    frames=[]
    write_json(output/'run_state.json', dict(status='running', complete_seven_blocks=False))
    try:
        for block in data.lib.BLOCOS:
            if block['nome'] not in plan['blocks']: continue
            parent=output/block['nome']; parent.mkdir(exist_ok=True)
            if (parent/'complete.json').exists():
                rows=verify_block(parent, sha256(sigfile))
                print(block['nome'], 'verified completed block; no refitting.', flush=True)
            else:
                # Preserve interrupted attempts. A retry gets a separate folder.
                i=1
                while (parent/f'attempt_{i:03d}').exists(): i+=1
                attempt=parent/f'attempt_{i:03d}'
                print(block['nome'], 'starting',attempt.name,'| four neural fits including calibration.',flush=True)
                try:
                    rows=run_block(data, block, config, attempt, torch_device)
                except Exception as error:
                    attempt.mkdir(exist_ok=True)
                    write_json(attempt/'failure.json', dict(error_type=type(error).__name__, error=str(error)))
                    raise
                require(all(sha256(ROOT/name)==digest for name,digest in source.items()), 'Source changed during fitting.')
                write_json(parent/'complete.json', dict(signature=sha256(sigfile), attempt=attempt.name,
                    files={p.relative_to(attempt).as_posix():sha256(p) for p in attempt.rglob('*') if p.is_file()}))
            frames.append(rows)
            result=summarize(data,output,frames,plan['blocks'])
        require(len(frames)==7, 'Incomplete block set.')
        _, ending=preflight(**paths)
        require(ending==audit, 'Inputs changed during run.')
        require(all(sha256(ROOT/name)==digest for name,digest in source.items()), 'Source changed during run.')
        write_json(output/'run_state.json', dict(status='complete', complete_seven_blocks=True,
            input_hashes_rechecked=True, source_hashes_rechecked=True, forecast_issued=False, model_promoted=False))
        return result
    except Exception as error:
        write_json(output/'run_state.json', dict(status='failed', complete_seven_blocks=False,
            error_type=type(error).__name__, error=str(error)))
        raise
    finally:
        export_reports(output)
