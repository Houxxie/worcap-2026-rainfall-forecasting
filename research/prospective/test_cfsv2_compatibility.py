"""Synthetic source-contract checks; no network or model training."""
import unittest

import numpy as np
import xarray as xr

from research.prospective import cfsv2_compatibility as c


def fixture():
    a = np.ones((1, 28, 1, 78, 68), dtype='float64')
    a[:, 24:] = np.nan
    return xr.Dataset(
        {'pr': (('S', 'M', 'L', 'Y', 'X'), a,
                dict(units='mm/day', standard_name='lwe_precipitation_rate'))},
        coords=dict(S=np.array(['2026-09-01'], dtype='datetime64[ns]'),
                    M=np.arange(1, 29), L=[1], Y=c.legacy.LAT, X=c.legacy.LON_360,
                    target=(('S', 'L'), np.array([['2026-10-01']], dtype='datetime64[ns]')),
                    target_bnds=(('S', 'L', 'nbound'), np.array(
                        [[['2026-10-01', '2026-11-01']]], dtype='datetime64[ns]'))))


class CompatibilityTests(unittest.TestCase):
    def test_matching_overlap_does_not_authorize_missing_legacy_members(self):
        new = c.successor_fields(fixture(), '2026-09-01')
        old = new.copy()
        old[20:] = np.nan
        # Exclude new members from the comparison, even if their rainfall is large.
        new[20:24] = 100
        report = c.compare_fields(old, new, '2026-09-01')
        self.assertEqual(report['ensemble_mean_difference_rmse'], 0)
        self.assertTrue(report['numerical_agreement_on_overlap'])
        self.assertFalse(report['legacy_complete'])
        self.assertTrue(report['successor_complete'])
        self.assertFalse(report['operationally_authorized'])
        self.assertEqual(report['common_complete_members'], list(range(1, 21)))

    def test_changed_values_are_not_accepted_due_to_matching_metadata(self):
        old = c.successor_fields(fixture(), '2026-09-01')
        new = old.copy()
        new[:24, 0, 0] += 10
        report = c.compare_fields(old, new, '2026-09-01')
        self.assertTrue(report['legacy_complete'] and report['successor_complete'])
        self.assertFalse(report['numerical_agreement_on_overlap'])
        self.assertFalse(report['operationally_authorized'])
        self.assertEqual(report['maximum_absolute_member_difference'], 10)

    def test_target_bounds_and_units_are_verified_not_inferred_from_lead(self):
        for change in ['bounds', 'target', 'units', 'origin', 'member', 'grid']:
            ds = fixture()
            if change == 'bounds':
                ds.target_bnds.values[0, 0, 1] = np.datetime64('2026-12-01')
            elif change == 'target':
                ds.target.values[0, 0] = np.datetime64('2026-09-01')
            elif change == 'units':
                ds.pr.attrs['units'] = 'mm'
            elif change == 'origin':
                ds = ds.assign_coords(S=np.array(['2026-08-01'], dtype='datetime64[ns]'))
            elif change == 'member':
                ds = ds.assign_coords(M=np.arange(28))
            elif change == 'grid':
                ds = ds.isel(Y=slice(None, None, -1))
            with self.subTest(change=change), self.assertRaises(ValueError):
                c.successor_fields(ds, '2026-09-01')

    def test_partial_member_is_not_treated_as_a_complete_member(self):
        new = c.successor_fields(fixture(), '2026-09-01')
        old = new.copy()
        new[10, 0, 0] = np.nan
        report = c.compare_fields(old, new, '2026-09-01')
        self.assertFalse(report['successor_complete'])
        self.assertNotIn(11, report['common_complete_members'])
        self.assertEqual(report['successor_finite_pixels'][10], 5303)

    def test_no_overlap_is_inconclusive_and_november_requires_28(self):
        new = c.successor_fields(fixture(), '2026-09-01')
        old = np.full_like(new, np.nan)
        report = c.compare_fields(old, new, '2026-09-01')
        self.assertEqual(report['status'], 'no_complete_overlap')
        self.assertIsNone(report['numerical_agreement_on_overlap'])
        self.assertEqual(int(c.expected_mask('2025-11-01').sum()), 28)
        self.assertFalse(c.compare_fields(new, new, '2025-11-01')['successor_complete'])

    def test_unexpected_members_and_invalid_values_are_rejected(self):
        for value, index in [(1, 24), (-1, 0), (np.inf, 0)]:
            ds = fixture()
            ds.pr.values[0, index, 0, 0, 0] = value
            with self.subTest(value=value, index=index), self.assertRaises(ValueError):
                c.successor_fields(ds, '2026-09-01')


if __name__ == '__main__':
    unittest.main()
