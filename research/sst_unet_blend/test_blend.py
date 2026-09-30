"""Synthetic numerical, alignment and saved-only workflow checks."""
from contextlib import redirect_stdout
from pathlib import Path
import copy
import io
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import xarray as xr
from research.common.inputs import ROOT, sha256
from .experiment import (protocol, locate_files, combine, evaluate_maps, compare_references,
                         execute, export_reports, SST_EVIDENCE, UNET_EVIDENCE)


def maps(full=False):
    dates = pd.date_range('2021-01-01', periods=24, freq='MS')
    coords = dict(time=dates, lat=np.arange(-60, 15.25, .25) if full else [-50., -25., 0.],
                  lon=np.arange(-90, -24.75, .25) if full else [-70., -60.])
    shape = tuple(len(coords[k]) for k in ['time', 'lat', 'lon'])
    def field(value):
        return xr.DataArray(np.full(shape, value, dtype='float64'), dims=('time', 'lat', 'lon'), coords=coords,
                            attrs=dict(units='mm/day'))
    h, hs, u, cl = map(field, [2., 2.1, 2.4, 1.8])
    original = .75 * h + .25 * u
    original.attrs['units'] = 'mm/day'
    return xr.Dataset(dict(hybrid=h, hybrid_sst=hs, climatology=cl)), \
        xr.Dataset(dict(hybrid=h, unet=u, blend=original, climatology=cl)), field(2.2), dates


class BlendTests(unittest.TestCase):
    def test_protocol_pins_two_distinct_completed_runs_and_no_training(self):
        plan = protocol()
        self.assertEqual(plan['weights'], [.75, .25])
        self.assertFalse(plan['new_training'] or plan['weight_search'] or plan['automatic_promotion'])
        self.assertNotEqual(plan['inputs']['sst_predictions']['sha256'], plan['inputs']['unet_predictions']['sha256'])

    def test_discovery_rejects_same_named_wrong_bytes_and_accepts_renamed_explicit_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'a').mkdir(); (root / 'b').mkdir()
            (root / 'a/predictions.nc').write_bytes(b'wrong')
            good = root / 'b/predictions.nc'
            good.write_bytes(b'right')
            plan = dict(inputs=dict(sst_predictions=dict(filename='predictions.nc', sha256=sha256(good))))
            with redirect_stdout(io.StringIO()):
                self.assertEqual(locate_files(plan, roots=[root])['sst_predictions'], good.resolve())
                renamed = root / 'sst_maps.nc'
                good.rename(renamed)
                self.assertEqual(locate_files(plan, dict(sst_predictions=renamed), roots=[root])['sst_predictions'], renamed.resolve())
            with self.assertRaisesRegex(ValueError, 'Missing exact'):
                locate_files(plan, roots=[root])

    def test_fixed_combination_preserves_reference_and_adds_only_sst_change(self):
        sst, unet, truth, dates = maps()
        candidate = combine(sst, unet, dates, require_full_grid=False)
        np.testing.assert_array_equal(candidate.blend_reference, unet.blend)
        np.testing.assert_allclose(candidate.blend_sst - candidate.blend_reference, .75 * (sst.hybrid_sst - sst.hybrid), atol=1e-15)
        np.testing.assert_allclose(candidate.blend_sst, 2.175, rtol=0, atol=1e-15)
        self.assertEqual(candidate.blend_sst.dtype, np.dtype('float64'))
        tables, products = evaluate_maps(candidate, truth)
        g = tables['global_metrics'].set_index('model')
        self.assertAlmostEqual(g.loc['blend_sst', 'rmse'], .025, places=12)
        self.assertAlmostEqual(g.loc['blend_reference', 'rmse'], .1, places=12)
        self.assertAlmostEqual(g.loc['blend_sst', 'bias'], -.025, places=12)
        self.assertAlmostEqual(products['sst']['blend_sse_from_products'], g.loc['blend_sst', 'sse'], places=10)

    def test_misalignment_missing_dates_units_and_changed_reference_are_rejected(self):
        sst, unet, _, dates = maps()
        for invalid in [unet.isel(time=slice(None, -1)), unet.isel(lat=slice(None, None, -1)),
                        unet.assign_coords(lon=unet.lon + .01)]:
            with self.assertRaises(ValueError):
                combine(sst, invalid, dates, require_full_grid=False)
        for variable in ['hybrid', 'climatology', 'blend']:
            invalid = unet.copy(deep=True)
            invalid[variable].values[0, 0, 0] += .01
            with self.assertRaises(ValueError):
                combine(sst, invalid, dates, require_full_grid=False)
        for value in [-1., np.nan, np.inf]:
            invalid = sst.copy(deep=True)
            invalid.hybrid_sst.values[0, 0, 0] = value
            with self.assertRaises(ValueError):
                combine(invalid, unet, dates, require_full_grid=False)
        sst.hybrid_sst.attrs['units'] = 'm'
        with self.assertRaisesRegex(ValueError, 'units'):
            combine(sst, unet, dates, require_full_grid=False)

    def test_recorded_source_scores_are_required_before_interpretation(self):
        old = pd.read_csv(ROOT / UNET_EVIDENCE / 'evaluation/global.csv')
        new = pd.read_csv(ROOT / SST_EVIDENCE / 'evaluation/global.csv').query("modelo == 'hybrid_sst'")
        table = pd.concat([old, new]).rename(columns=dict(modelo='model', vies='bias'))
        table.loc[table.model == 'blend', 'model'] = 'blend_reference'
        self.assertEqual(len(compare_references(table)), 7)
        table.loc[table.model == 'unet', 'rmse'] += .0001
        with self.assertRaisesRegex(ValueError, 'not reproduced'):
            compare_references(table)

    def test_saved_only_end_to_end_export_reuse_and_tamper_detection(self):
        sst, unet, truth, _ = maps(full=True)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = dict(observations=root / 'observations.nc', sst_predictions=root / 'sst.nc', unet_predictions=root / 'unet.nc')
            for key, data in [('observations', truth.to_dataset(name='tp')), ('sst_predictions', sst), ('unet_predictions', unet)]:
                data.to_netcdf(paths[key], engine='h5netcdf')
            plan = copy.deepcopy(protocol())
            for name, p in paths.items():
                plan['inputs'][name]['sha256'] = sha256(p)
            output = root / 'output'
            def check_analytic_reference(table):
                g = table.set_index('model')
                for name, rmse in [('hybrid', .2), ('hybrid_sst', .1), ('unet', .2), ('blend_reference', .1), ('climatology', .4)]:
                    self.assertAlmostEqual(g.loc[name, 'rmse'], rmse, places=12)
                return dict(synthetic=True)
            with patch('research.sst_unet_blend.experiment.protocol', return_value=plan), \
                 patch('research.sst_unet_blend.experiment.compare_references', side_effect=check_analytic_reference), \
                 redirect_stdout(io.StringIO()):
                result = execute(output, **paths)
                self.assertTrue(result['complete_comparison'])
                self.assertEqual(result['model_fits'], 0)
                self.assertEqual(result['comparisons']['blend_reference']['months']['better'], 24)
                self.assertTrue(export_reports(output).is_file())
                with patch('research.sst_unet_blend.experiment.evaluate_maps', side_effect=AssertionError('Unexpected repeated scoring')):
                    self.assertEqual(result, execute(output, **paths))
                (output / 'summary.json').write_text('{}')
                with self.assertRaisesRegex(ValueError, 'Changed or missing'):
                    execute(output, **paths)


if __name__ == '__main__':
    unittest.main()
