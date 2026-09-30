"""Compare a monthly-map U-Net to the unchanged lagged hybrid on development years."""
from pathlib import Path
import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import sys
import tempfile
import numpy as np
import pandas as pd
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.common.inputs import Inputs, Context, require, write_json, sha256
from research.common import baseline
from research.spatial_unet.training import MonthlyMaps, fit_network, inference_maps, split_inner


def signature(data, protocol, device):
    code = {}
    for folder in ['common', 'spatial_unet']:
        for p in sorted((ROOT / 'research' / folder).glob('*.py')):
            code[p.relative_to(ROOT).as_posix()] = sha256(p)
    return dict(inputs=data.audit, protocol=protocol, code=code,
        python=platform.python_version(), torch=torch.__version__, device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())


def run_block(data, block, config, directory, device):
    m = data.lib
    start = pd.Timestamp(block['inicio'])
    cutoff = m.origem_mensal([start], 4)[0]
    dates = pd.date_range(start, periods=24, freq='MS')
    training = m.calendario_pareado(cutoff)
    reference = m.janela_referencia(cutoff)
    inner_train, inner_reference, inner_dates = split_inner(training, reference, config['inner_validation_months'])
    directory.mkdir(parents=True, exist_ok=True)
    m.calendario_emissoes(dates, cutoff).to_csv(directory / 'source_calendar.csv', index=False)
    write_json(directory / 'split.json', dict(train=training.strftime('%Y-%m').tolist(),
        inner_train=inner_train.strftime('%Y-%m').tolist(), inner_reference=inner_reference.strftime('%Y-%m').tolist(),
        inner_validation=inner_dates.strftime('%Y-%m').tolist(), outer=dates.strftime('%Y-%m').tolist(),
        independent_holdout=False, historical_publication_vintages_verified=False))
    # Verify the inexpensive control before spending GPU time on the neural fit.
    state = baseline.fit(data, cutoff, directory / 'hybrid')
    hybrid = baseline.predict(data, state, dates)
    control_metrics = m.pesquisa_diagnosticos(data.rain.sel(time=dates), {'hybrid': hybrid}, block['nome'])
    reference_metrics = pd.read_csv(ROOT / 'research/lagged_sources/evidence/metricas_blocos.csv')
    old = reference_metrics[(reference_metrics.bloco == block['nome']) & (reference_metrics.modelo == 'controle_defasado')].iloc[0]
    now = m.pesquisa_resumir(control_metrics[control_metrics.regiao == 'dominio_inteiro'], ['modelo']).iloc[0]
    for key in ['rmse', 'mae', 'vies']:
        require(abs(float(now[key]) - float(old[key])) < 1e-6, f'Control did not reproduce archived {key}. Stop comparison.')
    write_json(directory / 'baseline_check.json', dict(reproduced=True, tolerance=1e-6,
        rmse_archived=float(old.rmse), rmse_current=float(now.rmse)))
    print(block['nome'], 'fixed hybrid reproduced; starting neural fit.', flush=True)
    baseline_climate = state['climate']
    del state
    gc.collect()
    scratch = directory / 'temporary_maps'
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as temp:
        temp = Path(temp)
        inner = Context(data, inner_reference, inner_train)
        inner.save(directory / 'inner')
        train = MonthlyMaps(inner, inner_train, temp / 'inner_train', training=True)
        valid = MonthlyMaps(inner, inner_dates, temp / 'inner_valid', scaling=train.scaling)
        try:
            net, epochs = fit_network(train, config, directory / 'inner', device, validation=valid)
        finally:
            train.close()
            valid.close()
        del train, valid, inner, net
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # Final preprocessing is refitted after epoch selection, using only outer training.
        context = Context(data, reference, training)
        context.save(directory / 'unet')
        train = MonthlyMaps(context, training, temp / 'outer_train', training=True)
        try:
            net, _ = fit_network(train, config, directory / 'unet', device, epochs=epochs)
            scaling = train.scaling
        finally:
            train.close()
        valid = MonthlyMaps(context, dates, temp / 'outer_eval', scaling=scaling)
        try:
            prediction = inference_maps(net, valid, device, config['batch_size'])
        finally:
            valid.close()
        coords = dict(time=dates, lat=data.rain.lat, lon=data.rain.lon)
        predictions = dict(unet=xr.DataArray(prediction, dims=('time', 'lat', 'lon'), coords=coords, attrs=dict(units='mm/day')))
        del net, train, valid
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    require(np.array_equal(baseline_climate.values, context.climate.values), 'Baseline and U-Net rainfall references differ.')
    predictions['hybrid'] = hybrid
    predictions['climatology'] = xr.DataArray(context.climate.values[dates.month - 1], dims=('time', 'lat', 'lon'), coords=coords, attrs=dict(units='mm/day'))
    metrics = m.pesquisa_diagnosticos(data.rain.sel(time=dates), predictions, block['nome'])
    xr.Dataset(predictions, attrs=dict(training_end=str(cutoff.date()), independent_holdout='false', chosen_epochs=epochs)).to_netcdf(
        directory / 'predictions.nc', engine='h5netcdf', encoding={k: dict(zlib=True, complevel=4) for k in predictions})
    metrics.to_csv(directory / 'monthly_metrics.csv', index=False)
    return metrics


def summarize(data, directory, frames):
    all_metrics = pd.concat(frames, ignore_index=True)
    require(not all_metrics.duplicated(['modelo', 'mes_alvo', 'regiao']).any(), 'Duplicate evaluation rows.')
    all_metrics.to_csv(directory / 'monthly_metrics.csv', index=False)
    domain = all_metrics[all_metrics.regiao == 'dominio_inteiro']
    for name, frame, groups in [('global', domain, ['modelo']), ('blocks', domain, ['modelo', 'bloco']),
                                ('years', domain, ['modelo', 'ano']), ('regions', all_metrics, ['modelo', 'regiao'])]:
        summary = data.lib.pesquisa_resumir(frame, groups)
        summary.to_csv(directory / (name + '.csv'), index=False)
    totals = pd.read_csv(directory / 'global.csv').set_index('modelo')
    complete = len(domain.bloco.unique()) == 7
    result = dict(complete_seven_blocks=complete, blocks=sorted(domain.bloco.unique().tolist()),
        unet_rmse=float(totals.loc['unet', 'rmse']), hybrid_rmse=float(totals.loc['hybrid', 'rmse']),
        delta_rmse=float(totals.loc['unet', 'rmse'] - totals.loc['hybrid', 'rmse']),
        model_promoted=False, forecast_issued=False, independent_holdout=False,
        interpretation='Development comparison on previously consulted years. A pilot is not the seven-block result.')
    write_json(directory / 'summary.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def execute(output, blocks=('H1',), official=None, seas5=None, cfsv2=None, device=None):
    config = json.loads(Path(__file__).with_name('protocol.json').read_text())
    require(config['id'] == 'spatial_unet_v1' and config['lags'] == dict(atmosphere=4, indices=3, seasonal_initialization=1, training_rainfall=4), 'Different temporal protocol.')
    require(config['inner_label_lag'] == 4 and config['max_training_months'] == 360, 'Different training protocol.')
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cpu':
        torch.set_num_threads(4)
        print('CPU selected: full-grid training may be slow. Prefer a Kaggle GPU.', flush=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = Inputs(official, seas5, cfsv2, output=output)
    run_signature = signature(data, config, device)
    signature_file = output / 'signature.json'
    if signature_file.exists():
        require(json.loads(signature_file.read_text()) == run_signature, 'Changed inputs, code or device. Preserve this run and choose another output directory.')
    else:
        write_json(signature_file, run_signature)
    selected = [b for b in data.lib.BLOCOS if b['nome'] in blocks]
    require(len(selected) == len(set(blocks)) and selected, 'Unknown or empty block selection.')
    frames = []
    for block in selected:
        directory = output / block['nome']
        complete = directory / 'complete.json'
        if complete.exists():
            record = json.loads(complete.read_text())
            require(record['signature'] == sha256(signature_file), 'Changed completed-block signature.')
            for name, digest in record['files'].items():
                require(sha256(directory / name) == digest, f'Changed block artifact: {name}')
            metrics = pd.read_csv(directory / 'monthly_metrics.csv')
            print(block['nome'], 'verified cached result', flush=True)
        else:
            require(not directory.exists(), f'Incomplete run in {directory}. Preserve it and use a new output directory.')
            metrics = run_block(data, block, config, directory, device)
            files = {p.relative_to(directory).as_posix(): sha256(p) for p in directory.rglob('*') if p.is_file()}
            write_json(complete, dict(signature=sha256(signature_file), files=files))
        frames.append(metrics)
    return summarize(data, output, frames)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--blocks', nargs='+', default=['H1'])
    parser.add_argument('--official')
    parser.add_argument('--seas5')
    parser.add_argument('--cfsv2')
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    execute(**vars(args))
