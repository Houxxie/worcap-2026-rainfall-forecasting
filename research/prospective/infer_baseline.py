"""Inference from frozen model files and exact selected source snapshots only."""
from pathlib import Path
import importlib.metadata
import importlib.util
import json
import numpy as np
import pandas as pd
import xarray as xr


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def psl_value(path, month):
    lines = Path(path).read_text(encoding='utf-8').strip().splitlines()
    first, last = map(int, lines[0].split())
    require(first <= month.year <= last, 'Index source does not cover the requested year.')
    row = lines[1 + month.year - first].split()
    require(len(row) == 13 and int(row[0]) == month.year, 'Malformed NOAA row.')
    value = float(row[month.month])
    require(np.isfinite(value) and abs(value) < 20, 'Unavailable or invalid NOAA value.')
    return value


def predict(package, inputs, target):
    package = Path(package)
    versions = json.loads((package / 'ambiente.json').read_text())
    require(all(importlib.metadata.version(n) == v for n, v in versions.items()), 'Inference environment differs from the fitted baseline.')
    spec = importlib.util.spec_from_file_location('frozen_baseline_library', package / 'biblioteca.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    target = pd.Timestamp(target + '-01')
    source_month = target - pd.DateOffset(months=4)
    with xr.open_dataset(package / 'clima_chuva.nc') as d:
        climate = next(iter(d.data_vars.values())).load()
    fields = {}
    for name in ['era5_sl', 'era5_pl']:
        with xr.open_dataset(inputs[name]) as ds:
            require(ds.attrs.get('schema') == 'worcap_era5_prospective_v1' and ds.attrs.get('expver') == 1,
                    'Require audited final ERA5, never an unaudited or ERA5T input.')
            require(pd.DatetimeIndex(ds.time.values).equals(pd.DatetimeIndex([source_month])), 'Wrong atmospheric source month.')
            for key in ds.data_vars:
                require(key not in fields, 'Duplicate atmospheric field.')
                m.conferir_campo(ds[key], climate, [source_month])
                fields[key] = ds[key].load().astype('float32')
    require(set(fields) == set(m.VARIAVEIS), 'Incomplete atmospheric input.')
    index_month = target - pd.DateOffset(months=3)
    m.INDICES_OC = pd.DataFrame([[psl_value(inputs[n], index_month) for n in m.INDICES_NOMES]], index=[index_month], columns=m.INDICES_NOMES)
    seasonal = []
    for name, variable, schema in [('seas5', 'seas5_tp_media', 'worcap_seas5_prospective_v1'),
                                    ('cfsv2', 'cfsv2_tp_media', 'nimbus_cfsv2_prospectivo_v1')]:
        with xr.open_dataset(inputs[name]) as ds:
            require(ds.attrs.get('schema') == schema and ds[variable].attrs.get('units') == 'mm/day', 'Unaudited seasonal artifact.')
            require(pd.DatetimeIndex(ds.time.values).equals(pd.DatetimeIndex([target])), 'Wrong seasonal target.')
            require(pd.DatetimeIndex(ds.time_origem.values).equals(pd.DatetimeIndex([target - pd.DateOffset(months=1)])), 'Wrong seasonal initialization.')
            a = ds[variable].interp(lat=climate.lat, lon=climate.lon, method='linear').astype('float32').load()
            for coord in ['time_origem', 'inicializacao_mais_recente']:
                if coord in ds:
                    a = a.assign_coords({coord: ('time', ds[coord].values)})
            require(np.isfinite(a.values).all() and (a.values >= 0).all(), 'Invalid seasonal interpolation.')
            seasonal.append(a)
    seas, cfs = seasonal
    with np.load(package / 'clima_atmosfera_indices.npz', allow_pickle=False) as z:
        atmosphere = {n: z[n].copy() for n in z.files}
    with np.load(package / 'ridge.npz', allow_pickle=False) as z:
        ridge = {n: z[n].copy() for n in z.files}
    ridge['corte'] = str(ridge['corte'].item())
    climates = []
    for name in ['seas5', 'cfsv2']:
        with xr.open_dataset(package / ('clima_' + name + '.nc')) as ds:
            climates.append(next(iter(ds.data_vars.values())).load())
    pieces = []
    for name, scale, features in zip(m.NOMES_MODELOS, [.9, .875], [m.FEATURES_SEAS5, m.FEATURES_MULTISSISTEMA]):
        model = m.lgb.Booster(model_file=str(package / (name + '.txt')))
        require(model.feature_name() == features, 'Model feature order differs from the frozen library.')
        anom = m.prever_par(model, name, fields, seas, cfs, atmosphere, climate, *climates, [target])
        pieces.append(m.reconstruir_chuva(climate, anom, scale).values.astype('float64'))
    prediction = .75 * (.5 * pieces[0] + .5 * pieces[1]) + .25 * m.mos_predict(ridge, seas, cfs, [target])
    require(np.isfinite(prediction).all() and (prediction >= 0).all(), 'Invalid hybrid prediction.')
    return xr.Dataset({
        'precipitacao': (('time', 'lat', 'lon'), prediction, dict(units='mm/day')),
        'climatologia': (('time', 'lat', 'lon'), climate.values[[target.month - 1]], dict(units='mm/day'))},
        coords=dict(time=[target], lat=climate.lat, lon=climate.lon))
