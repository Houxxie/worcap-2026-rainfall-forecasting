"""Chronological hybrid forecasts and shared direct/residual map datasets."""
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset
from research.common import baseline
from research.common.inputs import require, write_json


def forecast_field(data, dates, values, name):
    return xr.DataArray(values, dims=('time', 'lat', 'lon'), name=name,
        coords=dict(time=pd.DatetimeIndex(dates), lat=data.rain.lat, lon=data.rain.lon),
        attrs=dict(units='mm/day'))


def validate_field(field, data, dates):
    require(field.dims == ('time', 'lat', 'lon') and field.attrs.get('units') == 'mm/day',
            'Wrong forecast dimensions or units.')
    require(pd.DatetimeIndex(field.time.values).equals(pd.DatetimeIndex(dates)), 'Wrong forecast calendar.')
    require(np.array_equal(field.lat, data.rain.lat) and np.array_equal(field.lon, data.rain.lon),
            'Forecast grid does not match the target grid.')
    require(np.isfinite(field.values).all() and (field.values >= 0).all(), 'Invalid forecast rainfall.')


def crossfit_plan(data, dates, chunk_months=24):
    """Reject unavailable references before fitting any auxiliary hybrid."""
    dates = pd.DatetimeIndex(dates)
    require(len(dates) > 0 and dates.equals(pd.date_range(dates[0], periods=len(dates), freq='MS')),
            'Cross-fit targets must be consecutive monthly dates.')
    require(chunk_months == 24 and len(dates) % chunk_months == 0, 'Use complete two-year cross-fit groups.')
    available = pd.DatetimeIndex(data.rain.time.values)
    groups = []
    for first in range(0, len(dates), chunk_months):
        targets = dates[first:first + chunk_months]
        cutoff = targets[0] - pd.DateOffset(months=4)
        reference = data.lib.janela_referencia(cutoff)
        require(len(reference) == 360 and reference.isin(available).all(),
                'Insufficient past rainfall for the unchanged 30-year hybrid. Do not shorten or fill the reference.')
        training = data.lib.calendario_pareado(cutoff)
        require(training[-1] == cutoff and not training.isin(targets).any(), 'In-sample cross-fit target.')
        require(len(training) >= 24 and len(training) <= 360, 'Invalid hybrid training window.')
        require(data.lib.origem_mensal(reference, 4).isin(
            pd.DatetimeIndex(data.atmosphere[data.lib.VARIAVEIS[0]].time.values)).all(), 'Incomplete past atmosphere.')
        for field in [data.seas, data.cfs]:
            available_forecasts = pd.DatetimeIndex(field.time.values)
            require(training.isin(available_forecasts).all() and targets.isin(available_forecasts).all(),
                    'Incomplete seasonal forecast coverage.')
        groups.append(dict(cutoff=cutoff, targets=targets))
    return groups


def crossfit(data, dates, directory, run_stage):
    """Each target map is forecast by a hybrid fitted strictly before that group."""
    directory = Path(directory)
    groups = crossfit_plan(data, dates)
    fields, records = [], []
    for i, group in enumerate(groups, 1):
        folder = directory / f'group_{i:02d}'
        def action(group=group, folder=folder):
            state = baseline.fit(data, group['cutoff'], folder / 'hybrid')
            require(pd.Timestamp(state['audit']['training_end']) == group['cutoff'], 'Wrong hybrid cutoff.')
            prediction = baseline.predict(data, state, group['targets'])
            validate_field(prediction, data, group['targets'])
            prediction.rename('hybrid').to_netcdf(folder / 'predictions.nc', engine='h5netcdf')
            write_json(folder / 'calendar.json', dict(
                fit_cutoff=str(group['cutoff'].date()), target_months=group['targets'].strftime('%Y-%m-%d').tolist(),
                reference_months=360, label_lag_months=4, in_sample_errors_used=False))
        run_stage(f'crossfit_{i:02d}_of_{len(groups):02d}', folder, action)
        with xr.open_dataarray(folder / 'predictions.nc') as f:
            prediction = f.load()
        validate_field(prediction, data, group['targets'])
        fields.append(prediction)
        records.extend(dict(target_month=str(t.date()), hybrid_cutoff=str(group['cutoff'].date()),
                            group=i, label_gap_months=4) for t in group['targets'])
        gc.collect()
    result = xr.concat(fields, dim='time')
    validate_field(result, data, dates)
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(directory / 'calendar.csv', index=False)
    return result


class ResidualMaps(Dataset):
    """Share the exact input cache/scaler; only replace the target and reconstruction base."""
    def __init__(self, maps, hybrid, data):
        validate_field(hybrid, data, maps.dates)
        self.maps, self.dates, self.scaling = maps, maps.dates, maps.scaling
        self.base = hybrid.values.astype('float32')

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, index):
        x, _, _ = self.maps[index]
        base = self.base[index].copy()
        target = np.asarray(self.maps.y[index], dtype='float32') - base
        return x, torch.from_numpy(target), torch.from_numpy(base)


class ForecastMaps(Dataset):
    """Inference reads source features and a frozen base, never target rainfall values."""
    def __init__(self, context, dates, scaling, hybrid=None):
        self.context, self.dates = context, pd.DatetimeIndex(dates)
        require((self.dates > context.cutoff).all(), 'Forecasts must follow the training cutoff.')
        self.scaling = {k: np.asarray(v).copy() for k, v in scaling.items()}
        require(set(self.scaling) == {'mean', 'scale'} and all(v.shape == (31,) for v in self.scaling.values()), 'Invalid scaler.')
        require(all(np.isfinite(v).all() for v in self.scaling.values()) and (self.scaling['scale'] > 0).all(), 'Invalid scaler values.')
        if hybrid is not None:
            validate_field(hybrid, context.data, self.dates)
        # Preserve the hybrid's float64 precision when adding a float32 correction.
        self.hybrid = None if hybrid is None else hybrid.values.copy()

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, index):
        x = self.context.features(self.dates[index], training=False)
        base = x[22].copy() if self.hybrid is None else self.hybrid[index].copy()
        x = (x - self.scaling['mean'][:, None, None]) / self.scaling['scale'][:, None, None]
        require(np.isfinite(x).all(), 'Invalid inference features.')
        return torch.from_numpy(x), torch.zeros_like(torch.from_numpy(base)), torch.from_numpy(base)
