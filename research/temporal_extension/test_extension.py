"""Synthetic checks for withheld targets, unchanged inference and freeze integrity."""
from pathlib import Path
from contextlib import redirect_stdout
import io
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import xarray as xr

from research.common.inputs import Context, write_json, sha256
from research.common.synthetic_fixture import fixture
from research.spatial_unet.network import RainfallUNet, seed_everything
from research.spatial_unet.training import MonthlyMaps, inference_maps, split_inner
from research.temporal_extension.experiment import (
    ForecastMaps, calendar, protocol, read_training_rain, verify_frozen, evaluate,
    index_snapshot, require_index_coverage, DEVELOPMENT_INDICES, INDEX_SNAPSHOT,
)


class ExtensionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_locked_dates_and_inner_label_gap(self):
        plan, config = protocol()
        training, targets = calendar(plan)
        fit, reference, valid = split_inner(training, training, config['inner_validation_months'])
        self.assertEqual(targets[0], pd.Timestamp('2021-01-01'))
        self.assertEqual(training[-1], pd.Timestamp('2020-09-01'))
        self.assertEqual(valid[0], pd.Timestamp('2018-10-01'))
        self.assertEqual(fit[-1], pd.Timestamp('2018-06-01'))
        self.assertEqual(reference[0], pd.Timestamp('1990-10-01'))

    def test_archived_indices_cover_first_and_last_forecast_without_changing_training(self):
        plan, _ = protocol()
        indices, audit = index_snapshot(plan)
        self.assertEqual(len(indices), 387)
        self.assertEqual(indices.index[-1], pd.Timestamp('2022-09-01'))
        self.assertEqual(audit['shared_months'], 526)
        self.assertTrue(audit['overlap_exact'])
        reference = pd.read_csv(DEVELOPMENT_INDICES, parse_dates=['time_origem']).set_index('time_origem')
        training, targets = calendar(plan)
        training_origins = training - pd.DateOffset(months=3)
        pd.testing.assert_frame_equal(indices.loc[training_origins], reference.loc[training_origins])
        archive = pd.read_csv(INDEX_SNAPSHOT, parse_dates=['time_origem']).set_index('time_origem')
        for date in [targets[0], targets[-1]]:
            origin = date - pd.DateOffset(months=3)
            np.testing.assert_array_equal(indices.loc[origin].values, archive.loc[origin].values)

    def test_old_development_table_rejected_before_fitting(self):
        table = pd.read_csv(DEVELOPMENT_INDICES, parse_dates=['time_origem']).set_index('time_origem')
        targets = pd.date_range('2021-01-01', '2022-12-01', freq='MS')
        with self.assertRaisesRegex(ValueError, 'Missing: 2020-10'):
            require_index_coverage(table, targets - pd.DateOffset(months=3))

    def test_training_reader_does_not_include_later_values(self):
        dates = pd.date_range('2020-01-01', periods=12, freq='MS')
        values = np.ones((12, 2, 3)); values[9:] = np.nan
        field = xr.DataArray(values, dims=('time', 'lat', 'lon'), coords=dict(time=dates, lat=[0, 1], lon=[0, 1, 2]), attrs=dict(units='mm/day'))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'rain.nc'
            xr.Dataset(dict(tp=field)).to_netcdf(path, engine='h5netcdf')
            selected = read_training_rain(path, dates[:9])
            self.assertEqual(selected.sizes['time'], 9)
            self.assertTrue(np.isfinite(selected).all())

    def test_label_free_inference_matches_original_adapter(self):
        data = fixture()
        reference = pd.date_range('1976-10-01', '2006-09-01', freq='MS')
        training = reference[reference >= '1982-03-01']
        context = Context(data, reference, training)
        dates = pd.date_range('2007-01-01', periods=2, freq='MS')
        scaling = dict(mean=np.arange(31, dtype='float32'), scale=np.arange(1, 32, dtype='float32'))
        seed_everything()
        net = RainfallUNet(widths=(4, 8, 16))
        # Nonzero outputs ensure the comparison exercises feature normalization.
        with torch.no_grad():
            net.output.weight.fill_(.02)
        with tempfile.TemporaryDirectory() as temporary:
            original = MonthlyMaps(context, dates, Path(temporary) / 'original', scaling=scaling)
            try:
                expected = inference_maps(net, original, torch.device('cpu'))
                data.rain = data.rain.sel(time=reference)
                self.assertFalse(dates.isin(data.rain.time.values).any())
                actual = inference_maps(net, ForecastMaps(context, dates, scaling), torch.device('cpu'))
                np.testing.assert_array_equal(actual, expected)
            finally:
                original.close()

    def test_feature_path_cannot_touch_rainfall(self):
        class ContextWithoutRain:
            cutoff = pd.Timestamp('2020-09-01')
            data = SimpleNamespace()  # No rainfall attribute exists.
            def features(self, date, training=False):
                return np.ones((31, 2, 3), dtype='float32')
        maps = ForecastMaps(ContextWithoutRain(), ['2021-01-01'], dict(mean=np.zeros(31), scale=np.ones(31)))
        x, dummy, base = maps[0]
        self.assertEqual(tuple(x.shape), (31, 2, 3))
        self.assertEqual(float(dummy.sum()), 0.)
        self.assertEqual(float(base.sum()), 6.)

    def test_scoring_requires_unchanged_frozen_predictions_before_opening_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch('research.temporal_extension.experiment.xr.open_dataset', side_effect=AssertionError('Targets opened')):
                with self.assertRaisesRegex(ValueError, 'Freeze predictions'):
                    evaluate(root, root / 'rain.nc')
                for name in ['predictions.nc', 'signature.json', 'plan.json', 'split.json']:
                    (root / name).write_text('fixture')
                record = dict(schema='rainfall_temporal_extension_frozen_v1', evaluation_rainfall_decoded=False,
                              files={p.name: sha256(p) for p in root.iterdir()})
                write_json(root / 'frozen.json', record)
                verify_frozen(root)
                (root / 'predictions.nc').write_text('altered')
                with self.assertRaisesRegex(ValueError, 'changed or missing'):
                    evaluate(root, root / 'rain.nc')

    def test_complete_evaluation_uses_pooled_errors_and_preserves_freeze(self):
        # Alternating errors of 1 and 3 have RMSE sqrt(5), not mean RMSE 2.
        plan, _ = protocol()
        _, dates = calendar(plan)
        coords = dict(time=dates, lat=np.arange(-60, 15.25, .25), lon=np.arange(-90, -24.75, .25))
        observed = xr.DataArray(np.ones((24, 301, 261), dtype='float32'),
                                dims=('time', 'lat', 'lon'), coords=coords, attrs=dict(units='mm/day'))
        error = xr.DataArray(np.tile([1., 3.], 12), dims='time', coords=dict(time=dates))
        predictions = xr.Dataset(dict(hybrid=observed + error, unet=observed + 2,
                                      blend=observed + .75 * error + .5, climatology=observed * 0))
        for field in predictions.data_vars.values():
            field.attrs['units'] = 'mm/day'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            observed.to_dataset(name='tp').to_netcdf(root / 'rain.nc', engine='h5netcdf')
            predictions.to_netcdf(root / 'predictions.nc', engine='h5netcdf')
            write_json(root / 'plan.json', plan)
            for name in ['signature.json', 'split.json']:
                write_json(root / name, dict(synthetic=True))
            write_json(root / 'frozen.json', dict(schema='rainfall_temporal_extension_frozen_v1',
                evaluation_rainfall_decoded=False, observation_file_sha256=sha256(root / 'rain.nc'),
                files={name: sha256(root / name) for name in ['plan.json', 'signature.json', 'split.json', 'predictions.nc']}))
            original_record = (root / 'frozen.json').read_bytes()
            with redirect_stdout(io.StringIO()):
                result = evaluate(root, root / 'rain.nc')
            self.assertAlmostEqual(result['hybrid_rmse'], np.sqrt(5))
            self.assertAlmostEqual(result['blend_rmse'], np.sqrt((1.25**2 + 2.75**2) / 2))
            self.assertFalse(result['independent_holdout'])
            self.assertFalse(result['model_promoted'])
            self.assertEqual(original_record, (root / 'frozen.json').read_bytes())
            report = root / 'evaluation'
            self.assertIn('not an independent holdout', (report / 'report.html').read_text(encoding='utf-8'))
            for name, digest in json.loads((report / 'complete.json').read_text())['files'].items():
                self.assertEqual(sha256(report / name), digest)
            self.assertEqual(len(pd.read_csv(report / 'years.csv')), 8)
            with self.assertRaisesRegex(ValueError, 'already been opened'):
                evaluate(root, root / 'rain.nc')


if __name__ == '__main__':
    unittest.main()
