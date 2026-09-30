"""Guards against turning an IRI recipe refresh into a source substitution."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import xarray as xr

from research.prospective import prepare_cfsv2_recipe as c


class RecipeTests(unittest.TestCase):
    def arrays(self):
        explicit = np.full((28, 78, 68), np.nan)
        explicit[:24] = np.arange(24)[:, None, None] / 2
        cached = explicit.copy()
        cached[20:] = np.nan
        return cached, explicit, np.arange(28) < 24

    def test_complete_original_recipe_can_recover_wholly_missing_members(self):
        cached, explicit, mask = self.arrays()
        result = c.verify_overlap(cached, explicit, mask)
        self.assertEqual(result['recovered_members'], [21, 22, 23, 24])
        self.assertTrue(result['values_identical_on_overlap'])
        self.assertFalse(result['ensemble_joined_or_imputed'])
        self.assertFalse(result['alternate_provider_used'])

    def test_even_tiny_value_change_fails_exact_compatibility(self):
        cached, explicit, mask = self.arrays()
        explicit[2, 4, 5] += 1e-8
        with self.assertRaisesRegex(ValueError, 'differs from named product'):
            c.verify_overlap(cached, explicit, mask)

    def test_partial_or_still_missing_or_negative_members_are_rejected(self):
        for scenario in ['partial_cached', 'missing_new', 'negative_new', 'inactive_new', 'too_little_overlap']:
            cached, explicit, mask = self.arrays()
            if scenario == 'partial_cached': cached[0, 0, 0] = np.nan
            elif scenario == 'missing_new': explicit[21, 0, 0] = np.nan
            elif scenario == 'negative_new': explicit[21, 0, 0] = -1
            elif scenario == 'inactive_new': explicit[24, 0, 0] = 1
            elif scenario == 'too_little_overlap': cached[19] = np.nan
            with self.subTest(scenario=scenario), self.assertRaises(ValueError):
                c.verify_overlap(cached, explicit, mask)

    def test_wrong_units_or_lead_cannot_be_fixed_by_renaming(self):
        _, values, _ = self.arrays()
        ds = xr.Dataset({'aprod': (('S','L','M','Y','X'), values[None, None],
                         dict(units='mm/day', long_name='Precipitation Rate'))},
                        coords=dict(S=[800.0], L=[1.5], M=np.arange(1,29), Y=np.arange(-61,17), X=np.arange(269,337)))
        ds.S.attrs = dict(calendar='360', units='months since 1960-01-01')
        ds.L.attrs = dict(units='months', pointwidth=1.0)
        ds.Y.attrs['units'] = 'degree_north'; ds.X.attrs['units'] = 'degree_east'
        with tempfile.TemporaryDirectory() as td:
            p, q = Path(td)/'raw.nc', Path(td)/'norm.nc'
            ds.to_netcdf(p)
            actual = c.normalize_recipe(p, q, pd.Timestamp('2026-09-01'))
            np.testing.assert_array_equal(actual, values)
            for field in ['units', 'lead']:
                changed = ds.copy(deep=True)
                if field == 'units': changed.aprod.attrs['units'] = 'mm'
                else: changed = changed.assign_coords(L=[0.5]); changed.L.attrs = ds.L.attrs
                changed.to_netcdf(p)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    c.normalize_recipe(p, q, pd.Timestamp('2026-09-01'))

    def test_hindcast_is_not_routed_to_forecast_recipe(self):
        with self.assertRaises(ValueError):
            c.recipe_url('2010-09-01')
        url = c.recipe_url('2026-09-01')
        self.assertIn('.REALTIME_ENSEMBLE/.FLXF/.surface/.PRATE/', url)
        self.assertIn('regridAverage/nip/', url)
        self.assertNotIn('ccsr', url)


if __name__ == '__main__':
    unittest.main()
