"""Train/freeze first, then score an additional historical block in a separate call."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import gc
import json
import platform
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

from research.common.inputs import ROOT, Context, library, preflight, require, sha256, write_json
from research.common import baseline
from research.spatial_unet.training import MonthlyMaps, fit_network, inference_maps, split_inner

PROTOCOL_PATH = Path(__file__).with_name('protocol.json')
SOURCE_PROTOCOL = ROOT / 'research/spatial_unet/protocol.json'
INDEX_SNAPSHOT = ROOT / 'competition/metadata/NOAA/indices_noaa.csv'
DEVELOPMENT_INDICES = ROOT / 'research/lagged_sources/ocean_indices.csv'
INDEX_SNAPSHOT_SHA256 = 'b6f444fbec48681c1e5b46a5c22aa197ad91c7fbbdaff82c3ed3a9c63f7c15a4'
MODELS = ('hybrid', 'unet', 'blend', 'climatology')


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def protocol():
    plan = json.loads(PROTOCOL_PATH.read_text())
    require(plan['id'] == 'temporal_extension_2021_2022_v1', 'Unexpected extension protocol.')
    require(plan['training_start'] == '1990-10-01' and plan['training_end'] == '2020-09-01'
            and plan['evaluation_start'] == '2021-01-01' and plan['evaluation_end'] == '2022-12-01', 'Extension dates changed.')
    require(plan['weights'] == {'hybrid': .75, 'unet': .25} and plan['weight_search'] is False
            and plan['hyperparameter_search'] is False, 'This experiment has one fixed candidate.')
    require(sha256(SOURCE_PROTOCOL) == plan['source_protocol_sha256'], 'Original U-Net protocol changed.')
    return plan, json.loads(SOURCE_PROTOCOL.read_text())


def source_files():
    paths = [SOURCE_PROTOCOL, PROTOCOL_PATH, ROOT / 'research/lagged_sources/library.py',
             ROOT / 'research/lagged_sources/official_hashes.json', DEVELOPMENT_INDICES, INDEX_SNAPSHOT]
    for folder in ['common', 'spatial_unet', 'temporal_extension']:
        paths += list((ROOT / 'research' / folder).glob('*.py'))
    return sorted(set(paths))


def calendar(plan):
    training = pd.date_range(plan['training_start'], plan['training_end'], freq='MS')
    targets = pd.date_range(plan['evaluation_start'], plan['evaluation_end'], freq='MS')
    require(len(training) == 360 and len(targets) == 24 and training[-1] == targets[0] - pd.DateOffset(months=4),
            'Invalid four-month label gap or window length.')
    return training, targets


def require_index_coverage(table, required):
    library().conferir_indices(table)
    missing = pd.DatetimeIndex(required).difference(table.index)
    require(len(missing) == 0, 'Ocean indices do not cover all training/forecast origins. Missing: '
            + ', '.join(missing.strftime('%Y-%m').tolist()[:12]) + '. No fitting or imputation is allowed.')


def index_snapshot(plan):
    """Extend the calendar using the already archived NOAA table, with exact overlap checks."""
    require(sha256(INDEX_SNAPSHOT) == INDEX_SNAPSHOT_SHA256, 'Archived NOAA snapshot changed.')
    full = pd.read_csv(INDEX_SNAPSHOT, parse_dates=['time_origem']).set_index('time_origem')
    development = pd.read_csv(DEVELOPMENT_INDICES, parse_dates=['time_origem']).set_index('time_origem')
    library().conferir_indices(development)
    shared = development.index.intersection(full.index)
    require(len(shared) > 0 and full.columns.equals(development.columns)
            and np.array_equal(full.loc[shared].values, development.loc[shared].values),
            'Archived NOAA values differ from the development snapshot on shared months.')
    training, targets = calendar(plan)
    required = pd.date_range(training[0] - pd.DateOffset(months=3), targets[-1] - pd.DateOffset(months=3), freq='MS')
    require_index_coverage(full, required)
    audit = dict(source=INDEX_SNAPSHOT.relative_to(ROOT).as_posix(), sha256=sha256(INDEX_SNAPSHOT),
                 development_sha256=sha256(DEVELOPMENT_INDICES), shared_months=len(shared), overlap_exact=True,
                 selected_start=str(required[0].date()), selected_end=str(required[-1].date()),
                 selected_months=len(required), new_download=False, historical_publication_vintages_verified=False)
    return full.loc[required].copy(), audit


def read_training_rain(path, training):
    with xr.open_dataset(path) as ds:
        require(ds.tp.attrs.get('units') == 'mm/day', 'Unexpected rainfall units.')
        rain = ds.tp.sel(time=training).transpose('time', 'lat', 'lon').load()
    require(pd.DatetimeIndex(rain.time.values).equals(training), 'Training rainfall calendar differs.')
    require(np.isfinite(rain.values).all() and (rain.values >= 0).all(), 'Invalid training rainfall.')
    return rain


def load_inputs(paths, audit, output, plan):
    training, targets = calendar(plan)
    m = library()
    m.PASTA, m.PASTA_SEAS5_MANUAL, m.PASTA_CFSV2_MANUAL = paths['official'], paths['seas5'], paths['cfsv2']
    m.PASTA_SEAS5, m.MANIFESTO_SEAS5 = m.localizar_seas5()
    m.PASTA_CFSV2, m.MANIFESTO_CFSV2 = m.localizar_cfsv2()
    m.SAIDA = output
    # Only the fitting window is decoded here. No dummy evaluation rainfall is inserted.
    rain = read_training_rain(paths['official'] / 'treino_tp.nc', training)
    require(np.array_equal(rain.lat, np.arange(-60, 15.25, .25)) and np.array_equal(rain.lon, np.arange(-90, -24.75, .25)), 'Different grid.')
    atmosphere, _ = m.carregar_atmosfera(rain, training[0] - pd.DateOffset(months=4), targets[-1] - pd.DateOffset(months=4))
    m.INDICES_OC, indices_audit = index_snapshot(plan)
    write_json(output / 'indices.json', indices_audit)
    phases = ['desenvolvimento', 'somente_ajuste_final']
    seas = xr.concat([m.carregar_seas5(p, rain) for p in phases], dim='time').sel(time=slice(training[0], targets[-1]))
    cfs = xr.concat([m.carregar_cfsv2(p, rain) for p in phases], dim='time').sel(time=slice(training[0], targets[-1]))
    audit = dict(audit, indices=indices_audit, library_sha256=sha256(ROOT / 'research/lagged_sources/library.py'),
                 rainfall_decoded_for_fit=[str(training[0].date()), str(training[-1].date())],
                 evaluation_rainfall_decoded=False, test_partition_read=False, sst_used=False)
    return SimpleNamespace(lib=m, rain=rain, atmosphere=atmosphere, seas=seas, cfs=cfs, audit=audit)


class ForecastMaps(Dataset):
    """Inference adapter that never reads target rainfall; zeros are unused loss placeholders."""
    def __init__(self, context, dates, scaling):
        self.context, self.dates = context, pd.DatetimeIndex(dates)
        self.scaling = {k: np.asarray(v).copy() for k, v in scaling.items()}
        require((self.dates > context.cutoff).all(), 'Forecast targets must follow training.')
        require(set(self.scaling) == {'mean', 'scale'} and self.scaling['mean'].shape == (31,)
                and self.scaling['scale'].shape == (31,), 'Invalid training scaler.')
        require(np.isfinite(self.scaling['mean']).all() and np.isfinite(self.scaling['scale']).all()
                and (self.scaling['scale'] > 0).all(), 'Invalid training scaler values.')

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, index):
        features = self.context.features(self.dates[index], training=False)
        base = features[22].copy()
        x = (features - self.scaling['mean'][:, None, None]) / self.scaling['scale'][:, None, None]
        require(np.isfinite(x).all(), 'Nonfinite forecast features.')
        return torch.from_numpy(x), torch.zeros_like(torch.from_numpy(base)), torch.from_numpy(base)


def check(official=None, seas5=None, cfsv2=None, device='cuda'):
    plan, _ = protocol()
    _, indices = index_snapshot(plan)
    paths, audit = preflight(official, seas5, cfsv2, final=True)
    require(device in {'cpu', 'cuda'}, 'Device must be cpu or cuda.')
    require(device != 'cuda' or torch.cuda.is_available(), 'Enable a Kaggle GPU before fitting, or explicitly select cpu.')
    # These are already prepared seasonal inputs; no provider access is needed.
    print('Ready: training files, seasonal partitions and ocean indices checked through', indices['selected_end'],
          '| exact shared NOAA months:', indices['shared_months'], '| no model fitted.', flush=True)
    return paths, audit


def freeze(output, official=None, seas5=None, cfsv2=None, device='cuda'):
    plan, config = protocol()
    paths, audit = check(official, seas5, cfsv2, device)
    output = Path(output).resolve()
    require(not output.exists(), 'Output already exists. Preserve it and choose a fresh run directory.')
    for path in paths.values():
        require(not output.is_relative_to(path), 'Output must not be inside an input dataset.')
    require(not output.is_relative_to(ROOT / 'research') and not output.is_relative_to(ROOT / 'competition'), 'Output must be separate from source code.')
    output.mkdir(parents=True)
    started = utcnow()
    write_json(output / 'state.json', dict(status='fitting', started_at=started, forecast_issued=False))
    try:
        code = {p.relative_to(ROOT).as_posix(): sha256(p) for p in source_files()}
        for name in code:
            target = output / 'code' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
            require(sha256(target) == code[name], 'Code changed during snapshot.')
        write_json(output / 'plan.json', plan)
        write_json(output / 'signature.json', dict(plan=plan, source_protocol=config, code=code, inputs=audit,
                   python=platform.python_version(), torch=torch.__version__, device=device,
                   gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None, started_at=started))
        data = load_inputs(paths, audit, output, plan)
        training, targets = calendar(plan)
        inner_train, inner_reference, inner_dates = split_inner(training, training, config['inner_validation_months'])
        write_json(output / 'split.json', dict(training=training.strftime('%Y-%m').tolist(),
                   inner_training=inner_train.strftime('%Y-%m').tolist(), inner_reference=inner_reference.strftime('%Y-%m').tolist(),
                   inner_validation=inner_dates.strftime('%Y-%m').tolist(), evaluation=targets.strftime('%Y-%m').tolist(),
                   independent_holdout=False, evaluation_rainfall_decoded=False))
        data.lib.calendario_emissoes(targets, training[-1]).to_csv(output / 'source_calendar.csv', index=False)
        state = baseline.fit(data, training[-1], output / 'hybrid')
        hybrid = baseline.predict(data, state, targets)
        base_climate = state['climate'].copy()
        del state
        gc.collect()
        torch_device = torch.device(device)
        if device == 'cpu':
            torch.set_num_threads(4)
        scratch = output / 'scratch'
        scratch.mkdir()
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            temporary = Path(temporary)
            context = Context(data, inner_reference, inner_train)
            context.save(output / 'inner')
            train = MonthlyMaps(context, inner_train, temporary / 'inner_train', training=True)
            valid = MonthlyMaps(context, inner_dates, temporary / 'inner_valid', scaling=train.scaling)
            try:
                net, epochs = fit_network(train, config, output / 'inner', torch_device, validation=valid)
            finally:
                train.close(); valid.close()
            del net, train, valid, context
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
            context = Context(data, training, training)
            context.save(output / 'unet')
            np.testing.assert_array_equal(base_climate.values, context.climate.values)
            train = MonthlyMaps(context, training, temporary / 'outer_train', training=True)
            try:
                net, _ = fit_network(train, config, output / 'unet', torch_device, epochs=epochs)
                scaling = {k: v.copy() for k, v in train.scaling.items()}
            finally:
                train.close()
            forecast = inference_maps(net, ForecastMaps(context, targets, scaling), torch_device, config['batch_size'])
            del net, train
        coords = dict(time=targets, lat=data.rain.lat, lon=data.rain.lon)
        unet = xr.DataArray(forecast, dims=('time', 'lat', 'lon'), coords=coords, attrs=dict(units='mm/day'))
        blend = .75 * hybrid.astype('float64') + .25 * unet.astype('float64')
        blend.attrs['units'] = 'mm/day'
        climate = xr.DataArray(context.climate.values[targets.month - 1], dims=unet.dims, coords=coords, attrs=dict(units='mm/day'))
        predictions = xr.Dataset(dict(hybrid=hybrid, unet=unet, blend=blend, climatology=climate),
            attrs=dict(protocol=plan['id'], training_end=plan['training_end'], independent_holdout='false', retrospective='true'))
        for field in predictions.data_vars.values():
            require(np.isfinite(field).all() and (field >= 0).all(), 'Invalid prediction map.')
        predictions.to_netcdf(output / 'predictions.nc', engine='h5netcdf', encoding={k: dict(zlib=True, complevel=4) for k in predictions})
        with xr.open_dataset(output / 'predictions.nc') as restored:
            xr.testing.assert_identical(restored, predictions)
        require(all(sha256(ROOT / name) == expected for name, expected in code.items()), 'Source code changed during fitting.')
        _, end_audit = preflight(**paths, final=True)
        require(end_audit == audit, 'Input files changed during fitting.')
        frozen = dict(schema='rainfall_temporal_extension_frozen_v1', frozen_at=utcnow(), started_at=started,
            evaluation_rainfall_decoded=False, independent_holdout=False, model_promoted=False, forecast_issued=False,
            observation_file_sha256=audit['files']['treino_tp.nc'], selected_epochs=epochs,
            files={p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'state.json'})
        write_json(output / 'frozen.json', frozen)
        write_json(output / 'state.json', dict(status='predictions_frozen', frozen_at=frozen['frozen_at'], forecast_issued=False))
        print('Historical predictions frozen:', output, '| selected epochs:', epochs, flush=True)
        return output
    except (Exception, KeyboardInterrupt) as error:
        write_json(output / 'state.json', dict(status='failed', error_type=type(error).__name__, error=str(error), forecast_issued=False))
        raise


def verify_frozen(output):
    output = Path(output).resolve()
    require((output / 'frozen.json').is_file(), 'Freeze predictions before opening evaluation rainfall.')
    frozen = json.loads((output / 'frozen.json').read_text())
    require(frozen['schema'] == 'rainfall_temporal_extension_frozen_v1' and frozen['evaluation_rainfall_decoded'] is False,
            'Invalid freeze record.')
    require({'predictions.nc', 'signature.json', 'plan.json', 'split.json'} <= set(frozen['files']), 'Incomplete frozen package.')
    for name, digest in frozen['files'].items():
        path = (output / name).resolve()
        require(path.is_relative_to(output) and path != output, 'Frozen file path leaves the run.')
        require(path.is_file() and sha256(path) == digest, f'Frozen artifact changed or missing: {name}')
    return frozen


def evaluate(output, observations):
    output, observations = Path(output), Path(observations)
    frozen = verify_frozen(output)
    require(sha256(observations) == frozen['observation_file_sha256'], 'Observation snapshot differs from the training archive.')
    destination = output / 'evaluation'
    require(not destination.exists(), 'This frozen run has already been opened for evaluation. Preserve its report.')
    destination.mkdir()
    write_json(destination / 'opened.json', dict(opened_at=utcnow(), frozen_record_sha256=sha256(output / 'frozen.json'),
               scope='retrospective extension; not an independent holdout', weight_search=False,
               evaluation_code={p.relative_to(ROOT).as_posix(): sha256(p) for p in source_files()},
               python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, xarray=xr.__version__))
    # Observation values are decoded for the first time in this evaluation path.
    plan = json.loads((output / 'plan.json').read_text())
    _, dates = calendar(plan)
    with xr.open_dataset(observations) as source, xr.open_dataset(output / 'predictions.nc') as predictions:
        require(source.tp.attrs.get('units') == 'mm/day', 'Unexpected verification units.')
        obs = source.tp.sel(time=dates).transpose('time', 'lat', 'lon').load()
        predictions.load()
        require(set(predictions.data_vars) == set(MODELS), 'Unexpected candidate models.')
        require(pd.DatetimeIndex(predictions.time.values).equals(dates), 'Wrong prediction calendar.')
        obs, predictions = xr.align(obs, predictions, join='exact')
        require(np.isfinite(obs).all() and (obs >= 0).all(), 'Invalid verification observations.')
        for model in MODELS:
            require(predictions[model].dims == ('time', 'lat', 'lon') and predictions[model].attrs.get('units') == 'mm/day', 'Invalid forecast metadata.')
        m = library()
        monthly = m.pesquisa_diagnosticos(obs, {model: predictions[model] for model in MODELS}, 'D_2021_2022')
    monthly.to_csv(destination / 'monthly_metrics.csv', index=False)
    domain = monthly[monthly.regiao == 'dominio_inteiro']
    for name, table, groups in [('global', domain, ['modelo']), ('years', domain, ['modelo', 'ano']),
                               ('calendar', domain, ['modelo', 'mes']), ('regions', monthly, ['modelo', 'regiao'])]:
        m.pesquisa_resumir(table, groups).to_csv(destination / (name + '.csv'), index=False)
    values = pd.read_csv(destination / 'global.csv').set_index('modelo')
    require((values.n == 24 * 301 * 261).all(), 'Incomplete verification.')
    delta = float(values.loc['blend', 'rmse'] - values.loc['hybrid', 'rmse'])
    result = dict(status='evaluated', protocol=plan['id'], independent_holdout=False,
        hybrid_rmse=float(values.loc['hybrid', 'rmse']), unet_rmse=float(values.loc['unet', 'rmse']),
        blend_rmse=float(values.loc['blend', 'rmse']), delta_rmse=delta,
        delta_mae=float(values.loc['blend', 'mae'] - values.loc['hybrid', 'mae']),
        delta_absolute_bias=float(abs(values.loc['blend', 'vies']) - abs(values.loc['hybrid', 'vies'])),
        model_promoted=False, forecast_issued=False, evaluated_at=utcnow())
    write_json(destination / 'summary.json', result)
    from .report import render
    render(destination, result, plan)
    write_json(destination / 'complete.json', dict(frozen_record_sha256=sha256(output / 'frozen.json'),
        files={p.name: sha256(p) for p in destination.iterdir() if p.is_file()}))
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ['check', 'freeze']:
        p = commands.add_parser(command)
        for name in ['official', 'seas5', 'cfsv2']:
            p.add_argument('--' + name)
        p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
        if command == 'freeze':
            p.add_argument('--output', required=True)
    p = commands.add_parser('evaluate')
    p.add_argument('--output', required=True)
    p.add_argument('--observations', required=True)
    args = vars(parser.parse_args())
    command = args.pop('command')
    globals()[command](**args)
