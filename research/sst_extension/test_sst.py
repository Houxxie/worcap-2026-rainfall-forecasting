"""Small synthetic checks of chronology, paired fitting and frozen artifacts."""
from pathlib import Path
from contextlib import redirect_stdout
import io
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import xarray as xr
from research.common.synthetic_fixture import fixture
from research.common import baseline
from research.common.inputs import sha256, write_json
from .data import calendar, index_snapshot
from .experiment import protocol, fit_pair, metrics, seal, verify_frozen, score, export_reports


def sst_fixture(dates):
    rng = np.random.default_rng(861)
    values = rng.normal(size=(len(dates), 3, 7)).astype('float32')
    values[:, 0, 0] = np.nan
    return xr.DataArray(values, dims=('time', 'lat', 'lon'),
                        coords=dict(time=dates, lat=[-40, 0, 40], lon=np.arange(0, 350, 50)), attrs=dict(units='degC'))


class SSTTests(unittest.TestCase):
    def setUp(self):
        self.data = fixture()
        self.data.audit = dict(synthetic=True)
        self.m = self.data.lib
        self.m.ARVORES, self.m.PONTOS_POR_MES = 2, 4
        self.m.PARAMETROS = dict(self.m.PARAMETROS, num_threads=1, min_data_in_leaf=3, num_leaves=4)
        self.cutoff = pd.Timestamp('2006-09-01')
        self.targets = pd.date_range('2007-01-01', periods=24, freq='MS')
        self.training = self.m.calendario_pareado(self.cutoff)
        self.sst = sst_fixture(pd.DatetimeIndex(self.data.rain.time.values))

    def test_protocol_calendar_uses_fixed_gap_and_30_years(self):
        plan = protocol()
        train, targets = calendar(plan)
        self.assertEqual(train[0], pd.Timestamp('1990-10-01'))
        self.assertEqual(self.m.origem_mensal(train, 2)[-1], pd.Timestamp('2020-07-01'))
        self.assertEqual(self.m.origem_mensal(targets, 2)[-1], pd.Timestamp('2022-10-01'))
        self.assertFalse(plan['independent_holdout'])
        with self.assertRaises(ValueError):
            calendar(dict(plan, training_end='2020-10-01'))

    def test_archived_indices_match_overlap_and_cover_extension_origins(self):
        train, targets = calendar(protocol())
        indices, audit = index_snapshot(self.m, train, targets)
        self.assertEqual(audit['shared_months'], 526)
        self.assertEqual(indices.index[0], pd.Timestamp('1990-07-01'))
        self.assertEqual(indices.index[-1], pd.Timestamp('2022-09-01'))
        self.assertEqual(len(audit['older_unshared_months']), 5)
        self.assertTrue(audit['all_required_months_present'])

    def test_pca_mask_climate_center_components_ignore_post_training_sst(self):
        a = self.m.sst_ajustar_pca(self.sst, self.training)
        changed = self.sst.copy(deep=True)
        changed.loc[dict(time=slice('2006-08-01', None))] = 999.
        changed.loc[dict(time=slice('2006-08-01', None), lat=0, lon=50)] = np.nan
        b = self.m.sst_ajustar_pca(changed, self.training)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        self.assertEqual(a['contagens_mensais'].sum(), len(self.training))

    def test_t_minus_two_scores_ignore_later_sst_and_reject_missing_ocean_cells(self):
        state = self.m.sst_ajustar_pca(self.sst, self.training)
        targets = self.targets[:1]
        before = self.m.sst_transformar(self.sst, state, targets)
        changed = self.sst.copy(deep=True)
        changed.loc[dict(time=slice('2006-12-01', None))] = 300.
        np.testing.assert_array_equal(before, self.m.sst_transformar(changed, state, targets))
        changed.loc[dict(time='2006-11-01', lat=0, lon=50)] = np.nan
        with self.assertRaisesRegex(ValueError, 'Missing SST'):
            self.m.sst_transformar(changed, state, targets)

    def test_paired_fit_reproduces_baseline_without_outer_rainfall(self):
        truth = self.data.rain.sel(time=self.targets).copy()
        truth.attrs['units'] = 'mm/day'
        self.data.rain = self.data.rain.sel(time=slice(None, self.cutoff))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            state = baseline.fit(self.data, self.cutoff, root / 'reference')
            reference = baseline.predict(self.data, state, self.targets)
            with redirect_stdout(io.StringIO()):
                predictions, audit = fit_pair(self.data, self.sst, self.cutoff, self.targets, root / 'pair')
            xr.testing.assert_equal(reference, predictions.hybrid.rename(None))
            self.assertEqual(audit['paired_array_hashes']['ids'], state['audit']['sample_hash'])
            self.assertEqual(len(audit['timings']), 4)
            self.assertFalse(audit['evaluation_labels_read'])
            with xr.open_dataset(root / 'pair/predictions.nc') as saved:
                xr.testing.assert_equal(predictions, saved)
            tables = metrics(predictions, truth, self.m)
            g = tables['global'].set_index('modelo')
            for name in ['hybrid', 'hybrid_sst']:
                error = predictions[name].values - truth.values.astype('float64')
                self.assertAlmostEqual(g.loc[name, 'rmse'], np.sqrt(np.square(error).mean()), places=12)
                self.assertAlmostEqual(g.loc[name, 'mae'], np.abs(error).mean(), places=12)
            self.assertEqual(len(tables['monthly_metrics']), 3 * 24 * 4)
            with self.assertRaises(ValueError):
                metrics(predictions, truth.assign_coords(lon=truth.lon + 1), self.m)

    def test_frozen_artifact_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'models').mkdir()
            (root / 'models/predictions.nc').write_bytes(b'placeholder')
            (root / 'signature.json').write_text('{}')
            seal(root)
            self.assertFalse(verify_frozen(root)['evaluation_labels_read'])
            (root / 'models/predictions.nc').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Changed or missing'):
                verify_frozen(root)

    def test_scoring_and_report_export_from_frozen_full_grid_synthetic_maps(self):
        # Schema-sized constant maps exercise I/O/scoring without a full-grid model fit.
        plan = protocol()
        _, targets = calendar(plan)
        coords = dict(time=targets, lat=np.arange(-60, 15.25, .25), lon=np.arange(-90, -24.75, .25))
        def field(value):
            return xr.DataArray(np.full((24, 301, 261), value, dtype='float32'),
                                dims=('time', 'lat', 'lon'), coords=coords, attrs=dict(units='mm/day'))
        truth = field(2.2)
        predictions = xr.Dataset(dict(hybrid=field(2.), hybrid_sst=field(2.1), climatology=field(1.8)))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output, official = root / 'run', root / 'official'
            (output / 'models').mkdir(parents=True)
            official.mkdir()
            truth.to_dataset(name='tp').to_netcdf(official / 'treino_tp.nc', engine='h5netcdf')
            predictions.to_netcdf(output / 'models/predictions.nc', engine='h5netcdf')
            metrics(predictions, truth, self.m)['global'].to_csv(root / 'reference.csv', index=False)
            write_json(output / 'signature.json', dict(plan=plan, code={'reference.csv': sha256(root / 'reference.csv')},
                       inputs=dict(files={'treino_tp.nc': sha256(official / 'treino_tp.nc')})))
            write_json(output / 'models/fit.json', dict(seconds=0))
            write_json(output / 'plan.json', plan)
            seal(output)
            with patch('research.sst_extension.experiment.ROOT', root), \
                 patch('research.sst_extension.experiment.REFERENCE', 'reference.csv'), \
                 patch('research.sst_extension.experiment.source_files', return_value=['reference.csv']):
                with redirect_stdout(io.StringIO()):
                    result = score(output, official)
                self.assertTrue(result['complete_extension'] and result['baseline_reproduced'])
                self.assertEqual(result['months_better'], 24)
                self.assertTrue(export_reports(output).is_file())
                (output / 'evaluation/global.csv').write_text('tampered')
                with self.assertRaisesRegex(ValueError, 'Changed evaluation artifact'):
                    export_reports(output)


if __name__ == '__main__':
    unittest.main()
