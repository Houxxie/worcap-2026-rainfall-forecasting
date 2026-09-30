"""Synthetic checks for fitted-package replay and receipt adapters."""
from pathlib import Path
import importlib.metadata
import json
import shutil
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from research.common.inputs import LIBRARY, write_json
from research.common.baseline import fit, predict as predict_reference
from research.common.synthetic_fixture import fixture
from infer_baseline import predict, psl_value
from prepare_cds import regular_grid, convert, FIELDS
from collect_cds import request_for


class InferenceTests(unittest.TestCase):
    def test_coordinate_reordering_and_duplicate_rejection(self):
        lat, lon, values = regular_grid([1, 1, 0, 0], [270, 271, 270, 271], [3, 4, 1, 2])
        np.testing.assert_array_equal(lat, [0, 1])
        np.testing.assert_array_equal(lon, [-90, -89])
        np.testing.assert_array_equal(values, [[1, 2], [3, 4]])
        with self.assertRaises(ValueError):
            regular_grid([0, 0, 1, 1], [270, 270, 271, 271], [1, 2, 3, 4])

    def test_requests_respect_lags_and_final_grib_evidence(self):
        for source in ['era5_sl', 'era5_pl', 'seas5']:
            dataset, request = request_for(source, '2026-10')
            self.assertEqual(request['month'], ['09'] if source == 'seas5' else ['06'])
            self.assertEqual(request['data_format'], 'grib')
        self.assertEqual(request_for('era5_pl', '2026-10')[1]['pressure_level'], ['850'])

    def test_psl_rejects_unavailable_month(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'psl.data'
            p.write_text('2026 2026\n2026 1 2 3 4 5 6 7 -99.99 -99.99 -99.99 -99.99 -99.99\n-99.99\n')
            self.assertEqual(psl_value(p, pd.Timestamp('2026-07-01')), 7)
            with self.assertRaises(ValueError):
                psl_value(p, pd.Timestamp('2026-08-01'))

    def test_model_package_replays_direct_baseline(self):
        data = fixture()
        data.audit = {'fixture': True}
        dates = pd.DatetimeIndex(['2007-01-01'])
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            package = temp / 'model'
            state = fit(data, pd.Timestamp('2006-09-01'), package)
            expected = predict_reference(data, state, dates)
            shutil.copyfile(LIBRARY, package / 'biblioteca.py')
            write_json(package / 'ambiente.json', {n: importlib.metadata.version(n) for n in ['numpy','pandas','xarray','lightgbm']})
            paths = {}
            for source, fields in FIELDS.items():
                names = [row[0] for row in fields.values()]
                ds = xr.Dataset({n: data.atmosphere[n].sel(time=[pd.Timestamp('2006-09-01')]) for n in names},
                    attrs=dict(schema='worcap_era5_prospective_v1', expver=1))
                path = temp / (source + '.nc')
                ds.to_netcdf(path)
                paths[source] = path
            for name in data.lib.INDICES_NOMES:
                path = temp / (name + '.data')
                row = data.lib.INDICES_OC.loc['2006-01-01':'2006-12-01', name]
                path.write_text('2006 2006\n2006 ' + ' '.join(str(float(v)) for v in row) + '\n-99.99\n')
                paths[name] = path
            for name, field, schema in [('seas5', data.seas, 'worcap_seas5_prospective_v1'),
                                        ('cfsv2', data.cfs, 'nimbus_cfsv2_prospectivo_v1')]:
                ds = field.sel(time=dates).rename(name + '_tp_media').to_dataset()
                ds.attrs['schema'] = schema
                ds[name + '_tp_media'].attrs['units'] = 'mm/day'
                path = temp / (name + '.nc')
                ds.to_netcdf(path)
                paths[name] = path
            actual = predict(package, paths, '2007-01')
            np.testing.assert_allclose(actual.precipitacao, expected, rtol=0, atol=1e-7)
            np.testing.assert_array_equal(actual.climatologia.values, state['climate'].values[[0]])
            with self.assertRaises(ValueError):
                predict(package, paths, '2007-02')

    def test_real_grib_decoder_final_era5_and_era5t_rejection(self):
        import eccodes as ec
        from data_contracts import LAT, LON
        def write(path, expver):
            with Path(path).open('wb') as f:
                for param in FIELDS['era5_sl']:
                    g = ec.codes_grib_new_from_samples('regular_ll_sfc_grib1')
                    try:
                        for k, v in dict(Ni=len(LON), Nj=len(LAT), latitudeOfFirstGridPointInDegrees=15.,
                            latitudeOfLastGridPointInDegrees=-60., longitudeOfFirstGridPointInDegrees=270.,
                            longitudeOfLastGridPointInDegrees=335., iDirectionIncrementInDegrees=.25,
                            jDirectionIncrementInDegrees=.25, dataDate=20260601, dataTime=0,
                            **{'class':'ea', 'stream':'moda', 'expver':expver, 'paramId':param}).items():
                            ec.codes_set(g, k, v)
                        ec.codes_set_values(g, np.ones(len(LAT) * len(LON)))
                        ec.codes_write(g, f)
                    finally:
                        ec.codes_release(g)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'era5.grib'
            write(path, '0001')
            ds, meta = convert(path, 'era5_sl', '2026-10')
            self.assertEqual(set(ds.data_vars), {'cloud_cover','t2','surface_pressure'})
            self.assertEqual(meta['meses'], ['2026-06'])
            np.testing.assert_array_equal(ds.lat, LAT)
            write(path, '0005')
            with self.assertRaises(ValueError):
                convert(path, 'era5_sl', '2026-10')

    def test_seas5_requires_all_members_and_correct_target(self):
        import eccodes as ec
        def write(path, members):
            with path.open('wb') as f:
                for member in range(members):
                    g = ec.codes_grib_new_from_samples('regular_ll_sfc_grib1')
                    try:
                        settings = dict(localDefinitionNumber=16, stream='msmm', type='fcmean', system=51,
                            origin='ecmf', dataDate=20260901, dataTime=0, forecastMonth=2, verifyingMonth=202610,
                            number=member, numberOfForecastsInEnsemble=51, paramId=172228,
                            Ni=68, Nj=78, latitudeOfFirstGridPointInDegrees=16., latitudeOfLastGridPointInDegrees=-61.,
                            longitudeOfFirstGridPointInDegrees=269., longitudeOfLastGridPointInDegrees=336.,
                            iDirectionIncrementInDegrees=1., jDirectionIncrementInDegrees=1.)
                        for key, value in settings.items():
                            ec.codes_set(g, key, value)
                        ec.codes_set_values(g, np.full(68 * 78, (member + 1) / 86400000.))
                        ec.codes_write(g, f)
                    finally:
                        ec.codes_release(g)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'seas5.grib'
            write(path, 51)
            ds, meta = convert(path, 'seas5', '2026-10')
            np.testing.assert_allclose(ds.seas5_tp_media.values, 26., rtol=1e-6)
            self.assertEqual(len(meta['auditoria']), 51)
            self.assertEqual(meta['meses'], ['2026-09'])
            with self.assertRaises(ValueError):
                convert(path, 'seas5', '2026-11')
            write(path, 50)
            with self.assertRaises(ValueError):
                convert(path, 'seas5', '2026-10')

    def test_pressure_level_contract(self):
        import eccodes as ec
        from data_contracts import LAT, LON
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'pressure.grib'
            with path.open('wb') as f:
                for param in FIELDS['era5_pl']:
                    g = ec.codes_grib_new_from_samples('regular_ll_pl_grib1')
                    try:
                        for key, value in dict(Ni=len(LON), Nj=len(LAT), latitudeOfFirstGridPointInDegrees=15.,
                            latitudeOfLastGridPointInDegrees=-60., longitudeOfFirstGridPointInDegrees=270.,
                            longitudeOfLastGridPointInDegrees=335., iDirectionIncrementInDegrees=.25,
                            jDirectionIncrementInDegrees=.25, dataDate=20260601, dataTime=0,
                            level=850, **{'class':'ea','stream':'moda','expver':'0001','paramId':param}).items():
                            ec.codes_set(g, key, value)
                        ec.codes_set_values(g, np.ones(len(LAT) * len(LON)))
                        ec.codes_write(g, f)
                    finally:
                        ec.codes_release(g)
            ds, meta = convert(path, 'era5_pl', '2026-10')
            self.assertEqual(len(ds.data_vars), 6)
            self.assertEqual(len(meta['auditoria']), 6)


if __name__ == '__main__':
    unittest.main(argv=[__file__])
