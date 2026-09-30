"""Chronological splits and one shrunken additive offset, in mm/day."""
import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset
from research.common.inputs import require
from research.spatial_unet.training import split_inner


def splits(training, reference):
    training, reference = pd.DatetimeIndex(training), pd.DatetimeIndex(reference)
    fit, refs, calibration = split_inner(training, reference, 24)
    selection_fit, selection_refs, selection_valid = split_inner(fit, refs, 24)
    require(len(calibration) == 24 and calibration[-1] == training[-1], 'Invalid calibration window.')
    require(fit[-1] == calibration[0] - pd.DateOffset(months=4), 'Calibration label gap changed.')
    require(selection_valid[-1] == fit[-1] and selection_fit[-1] == selection_valid[0] - pd.DateOffset(months=4),
            'Epoch selection overlaps calibration targets.')
    return dict(fit=fit, reference=refs, calibration=calibration,
                selection_fit=selection_fit, selection_reference=selection_refs,
                selection_validation=selection_valid)


class ForecastMaps(Dataset):
    """Generate forecast features without reading any target rainfall."""
    def __init__(self, context, dates, scaling):
        self.context, self.dates = context, pd.DatetimeIndex(dates)
        require((self.dates > context.cutoff).all(), 'Forecast dates must follow training.')
        self.scaling = {k: np.asarray(v).copy() for k, v in scaling.items()}
        require(set(self.scaling) == {'mean', 'scale'} and all(v.shape == (31,) for v in self.scaling.values()), 'Invalid scaler.')
        require(all(np.isfinite(v).all() for v in self.scaling.values()) and (self.scaling['scale'] > 0).all(), 'Invalid scaling values.')

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, index):
        x = self.context.features(self.dates[index], training=False)
        base = x[22].copy()
        x = (x - self.scaling['mean'][:, None, None]) / self.scaling['scale'][:, None, None]
        require(np.isfinite(x).all(), 'Nonfinite forecast features.')
        return torch.from_numpy(x), torch.zeros_like(torch.from_numpy(base)), torch.from_numpy(base)


def estimate(prediction, observations, dates, outer_cutoff, auxiliary_cutoff, selection_end):
    dates = pd.DatetimeIndex(dates)
    outer_cutoff, auxiliary_cutoff, selection_end = map(pd.Timestamp, [outer_cutoff, auxiliary_cutoff, selection_end])
    require(dates.equals(pd.date_range(outer_cutoff - pd.DateOffset(months=23), outer_cutoff, freq='MS')),
            'Calibration must be the last 24 training months, never outer targets.')
    require(auxiliary_cutoff == dates[0] - pd.DateOffset(months=4) and selection_end <= auxiliary_cutoff,
            'Calibration labels entered fitting or epoch selection.')
    require(pd.DatetimeIndex(prediction.time.values).equals(dates), 'Wrong calibration prediction calendar.')
    require(prediction.dims == ('time', 'lat', 'lon'), 'Wrong prediction dimensions.')
    require(prediction.attrs.get('units') == observations.attrs.get('units') == 'mm/day', 'Unexpected units.')
    # Select before decoding/aggregation: no later observation can affect the offset.
    truth = observations.sel(time=dates).transpose('time', 'lat', 'lon')
    prediction, truth = xr.align(prediction, truth, join='exact')
    require(np.isfinite(prediction.values).all() and np.isfinite(truth.values).all()
            and (prediction.values >= 0).all() and (truth.values >= 0).all(), 'Invalid calibration rainfall.')
    monthly = (prediction.values.astype('float64') - truth.values.astype('float64')).mean(axis=(1, 2))
    bias = float(monthly.mean())
    record = dict(schema='unet_global_bias_v1', units='mm/day', calibration_months=24,
                  prior_zero_bias_months=24, shrinkage=.5, raw_mean_bias=bias, offset_mm_day=.5 * bias,
                  calibration_start=str(dates[0].date()), calibration_end=str(dates[-1].date()),
                  outer_cutoff=str(outer_cutoff.date()), auxiliary_cutoff=str(auxiliary_cutoff.date()),
                  selection_end=str(selection_end.date()), outer_targets_used=False)
    return record, pd.DataFrame(dict(target_month=dates.strftime('%Y-%m-%d'), mean_error_mm_day=monthly))


def correct(prediction, record):
    require(record['schema'] == 'unet_global_bias_v1' and record['shrinkage'] == .5 and
            record['calibration_months'] == record['prior_zero_bias_months'] == 24, 'Changed correction rule.')
    require(record['outer_targets_used'] is False and prediction.attrs.get('units') == record['units'] == 'mm/day', 'Invalid correction provenance or units.')
    offset = float(record['offset_mm_day'])
    require(np.isfinite(offset) and np.isclose(offset, .5 * record['raw_mean_bias'], rtol=0, atol=1e-12), 'Invalid offset.')
    require(np.isfinite(prediction.values).all() and (prediction.values >= 0).all(), 'Invalid outer predictions.')
    result = (prediction.astype('float64') - offset).clip(min=0)
    result.attrs = dict(units='mm/day')
    return result
