"""Fit and predict the fixed lagged hybrid, without changing its architecture."""
from pathlib import Path
import gc
import numpy as np
import xarray as xr
from .inputs import require, sha256, write_json


def fit(data, cutoff, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    m = data.lib
    train = data.rain.sel(time=slice(None, cutoff))
    X, y, ids, dates, climate, atmosphere, sums, counts, seas, cfs = m.amostrar_arvores_multissistema(
        train, data.atmosphere, data.seas, data.cfs, cutoff)
    models = []
    for name, features in [('arvores_seas5', m.FEATURES_SEAS5), ('arvores_multissistema', m.FEATURES_MULTISSISTEMA)]:
        dataset = m.lgb.Dataset(np.ascontiguousarray(X[:, :len(features)]), label=y, feature_name=features)
        model = m.lgb.train(m.PARAMETROS, dataset, num_boost_round=m.ARVORES)
        require(model.feature_name() == features, 'Baseline feature order changed.')
        model.save_model(str(folder / (name + '.txt')))
        models.append(model)
        del dataset
    ridge = m.mos_fit(train, data.seas, data.cfs, cutoff)
    np.savez_compressed(folder / 'ridge.npz', **ridge)
    np.savez_compressed(folder / 'clima_atmosfera_indices.npz', **atmosphere)
    for name, field in [('chuva', climate), ('seas5', seas), ('cfsv2', cfs)]:
        field.to_netcdf(folder / ('clima_' + name + '.nc'))
    audit = dict(training_start=str(dates[0].date()), training_end=str(dates[-1].date()),
        training_months=len(dates), examples=len(y), parameters=m.PARAMETROS,
        tree_count=m.ARVORES, weights=[.375, .375, .25], residual_scales=[.9, .875],
        sample_hash=__import__('hashlib').sha256(ids.tobytes()).hexdigest(), inputs=data.audit)
    del X, y, ids
    gc.collect()
    write_json(folder / 'fit.json', audit)
    return dict(models=models, ridge=ridge, climate=climate, atmosphere=atmosphere, seas=seas, cfs=cfs, audit=audit)


def predict(data, state, dates):
    m = data.lib
    components = []
    for model, name, scale in zip(state['models'], m.NOMES_MODELOS, [.9, .875]):
        anomalies = m.prever_par(model, name, data.atmosphere, data.seas, data.cfs,
            state['atmosphere'], state['climate'], state['seas'], state['cfs'], dates)
        components.append(m.reconstruir_chuva(state['climate'], anomalies, scale).values.astype('float64'))
    ridge = m.mos_predict(state['ridge'], data.seas, data.cfs, dates)
    result = .75 * (.5 * components[0] + .5 * components[1]) + .25 * ridge
    return xr.DataArray(result, dims=('time', 'lat', 'lon'),
        coords=dict(time=dates, lat=data.rain.lat, lon=data.rain.lon), attrs=dict(units='mm/day'))
