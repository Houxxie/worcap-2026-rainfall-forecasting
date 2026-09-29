"""Small mathematical and alignment checks; no model fitting or downloads."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from research.diagnostics.analyze_errors import (
    MODELS, stats, pooled, intensity_masks, exact_alignment, verify_file,
)


class DiagnosticChecks(unittest.TestCase):
    def test_rmse_pools_squared_errors_and_sample_counts(self):
        frame = pd.DataFrame([dict(model='x', **stats([0], [4])),
                              dict(model='x', **stats([0, 0, 0], [0, 0, 0]))])
        result = pooled(frame, ['model']).iloc[0]
        self.assertEqual(result.rmse, 2.)
        self.assertEqual(result.mae, 1.)
        self.assertEqual(result.bias, 1.)

    def test_intensity_edges_cover_each_observation_once(self):
        observed = np.array([0., .9, 1., 2.9, 3., 4.9, 5., 9.9, 10., 90.])
        masks = intensity_masks(observed)
        np.testing.assert_array_equal(np.sum(masks, axis=0), np.ones(10))
        self.assertEqual([int(m.sum()) for m in masks], [2] * 5)
        with self.assertRaises(ValueError):
            intensity_masks(np.array([-1.]))

    def test_empty_bin_has_no_invented_zero_rmse(self):
        frame = pd.DataFrame([dict(model='x', **stats([], []))])
        row = pooled(frame, ['model']).iloc[0]
        self.assertEqual(row.n, 0)
        self.assertTrue(np.isnan(row.rmse))

    def test_alignment_rejects_time_shift_and_flipped_grid(self):
        dates = pd.date_range('2007-01-01', periods=2, freq='MS')
        obs = xr.DataArray(np.ones((2, 2, 2)), dims=('time', 'lat', 'lon'),
                           coords=dict(time=dates, lat=[-1., 0.], lon=[-40., -39.]), attrs=dict(units='mm/day'))
        forecast = xr.Dataset({model: obs.copy() for model in MODELS})
        exact_alignment(obs, forecast, dates)
        with self.assertRaises(ValueError):
            exact_alignment(obs, forecast.isel(lat=slice(None, None, -1)), dates)
        with self.assertRaises(ValueError):
            exact_alignment(obs, forecast.assign_coords(time=dates + pd.Timedelta(days=1)), dates)

    def test_changed_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'artifact'
            file.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Hash mismatch'):
                verify_file(file, '0' * 64)


if __name__ == '__main__':
    unittest.main()
