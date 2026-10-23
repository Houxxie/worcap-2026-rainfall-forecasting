"""Full-grid synthetic verification, including real GRIB decoding; no provider calls."""
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import json
import tempfile
import unittest
import zipfile

import eccodes as ec
import numpy as np
import xarray as xr

import verify_forecast as v
from backup_registry import backup
from evaluate import avaliar, painel
from registry import Registro, ContratoError, digest, deslocar

ISSUED = datetime(2026, 9, 30, 18, tzinfo=timezone.utc)
ELIGIBLE = datetime(2027, 2, 1, tzinfo=timezone.utc)


def grib_fixture(path, value=.002, **overrides):
    handle = ec.codes_grib_new_from_samples('regular_ll_sfc_grib1')
    try:
        settings = dict(centre='ecmf', setLocalDefinition=1, localDefinitionNumber=1,
            **{'class': 'ea'}, stream='moda', expver='0001', paramId=228, dataDate=20261001, dataTime=0,
            Ni=len(v.LON), Nj=len(v.LAT), latitudeOfFirstGridPointInDegrees=15,
            longitudeOfFirstGridPointInDegrees=270, latitudeOfLastGridPointInDegrees=-60,
            longitudeOfLastGridPointInDegrees=335, iDirectionIncrementInDegrees=.25,
            jDirectionIncrementInDegrees=.25, jScansPositively=0)
        settings.update(overrides)
        for name, val in settings.items():
            ec.codes_set(handle, name, val)
        ec.codes_set_values(handle, np.full(len(v.LAT) * len(v.LON), value))
        with path.open('wb') as f:
            ec.codes_write(handle, f)
    finally:
        ec.codes_release(handle)
    return path


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.clocks = ExitStack()
        self.addCleanup(self.clocks.close)
        self.clock_mocks = [self.clocks.enter_context(patch(name, return_value=ISSUED)) for name in
                            ['registry.agora', 'evaluate.agora', 'verify_forecast.agora', 'backup_registry.agora']]
        self.r = Registro(self.root / 'registry')
        plan = json.loads((v.BASE / 'plan.json').read_text(encoding='utf-8'))
        protocol = self.r.iniciar(plan)
        files = {}
        for name in plan['arquivos_modelo']:
            file = self.root / name
            file.write_bytes(b'synthetic model placeholder; never fitted or used for inference')
            files[name] = file
        self.r.modelo(files, dict(treino_inicio='1993-01', treino_fim='2022-12', protocolo_id=protocol['id']))
        for name, rule in plan['fontes'].items():
            if rule['obrigatoria']:
                self.r.recibo(name, b'synthetic ' + name.encode(),
                    dict(validado=True, meses=[deslocar('2026-10', -rule['lag'])]), {'fixture': True})
        shape = (1, len(v.LAT), len(v.LON))
        # Distinct known errors for the forecast and climatology, over the real grid.
        ds = xr.Dataset({name: (('time', 'lat', 'lon'), np.full(shape, value), {'units': 'mm/day'})
                         for name, value in [('precipitacao', 3.), ('climatologia', 5.)]},
                        coords=dict(time=[np.datetime64('2026-10-01')], lat=v.LAT, lon=v.LON))
        path = self.root / 'forecast.nc'
        ds.to_netcdf(path, engine='h5netcdf')
        ready = self.r.prontidao('2026-10')
        inf = {k: ready[k] for k in ['modelo', 'fontes', 'protocolo']}
        inf['previsao_sha256'] = digest(path.read_bytes())
        self.forecast = self.r.congelar('2026-10', path, inf)
        self.original_events = self.r.eventos(conferir_objetos=True)
        self.grib = grib_fixture(self.root / 'era5.grib')

    def advance(self, value=ELIGIBLE):
        for clock in self.clock_mocks:
            clock.return_value = value

    def test_deadline_blocks_download_and_leaves_registry_and_output_untouched(self):
        self.advance(datetime(2027, 1, 31, 23, 59, 59, tzinfo=timezone.utc))
        self.assertFalse(v.check(self.r.pasta, '2026-10')['ready'])
        with patch.object(v, 'download', side_effect=AssertionError('no network')):
            with self.assertRaisesRegex(ContratoError, r'T\+4'):
                v.evaluate(self.r.pasta, '2026-10', self.root / 'early')
        self.assertFalse((self.root / 'early').exists())
        self.assertEqual(self.r.eventos(), self.original_events)
        self.advance()
        self.assertTrue(v.check(self.r.pasta, '2026-10')['ready'])
        self.assertFalse(v.check(self.r.pasta, '2026-11')['ready'])
        self.assertFalse((self.root / 'absent').exists())
        with self.assertRaisesRegex(ContratoError, 'complete existing registry'):
            v.check(self.root / 'absent', '2026-10')
        self.assertFalse((self.root / 'absent').exists())

    def test_end_to_end_restored_registry_grib_scores_and_portable_report(self):
        backup(self.r.pasta, self.root / 'before.zip')
        with zipfile.ZipFile(self.root / 'before.zip') as archive:
            archive.extractall(self.root / 'restored')  # Only this test's generated archive.
        r = Registro(self.root / 'restored' / 'registry')
        self.advance()
        output = self.root / 'report'
        result = v.evaluate(r.pasta, '2026-10', output, self.grib)
        event = r.obter(result['selected_verification_id'], 'verificacao')
        self.assertTrue(result['primary'])
        self.assertFalse(result['reused'])
        self.assertEqual(event['registrado_em_utc'], ELIGIBLE.isoformat())
        self.assertEqual(r.ler_objeto(event['dados']['original_grib']), self.grib.read_bytes())
        self.assertEqual(r.eventos()[:-1], self.original_events)
        rows = event['dados']['metricas']
        self.assertEqual(len(rows), 8)
        for row in rows:
            expected = 1. if row['modelo'] == 'precipitacao' else 3.
            for metric in ['rmse', 'mae', 'vies', 'rmse_area']:
                self.assertAlmostEqual(row[metric], expected, places=6)
        self.assertEqual(rows[0]['n'], 301 * 261)
        summary = json.loads((output / 'summary.json').read_text())
        self.assertEqual((summary['planned_months'], summary['issued_months'], summary['evaluated_months']), (12, 1, 1))
        self.assertEqual(len(summary['months']), 12)
        self.assertEqual(summary['months'][1]['status'], 'Not issued; deadline passed')
        inventory = json.loads((output / 'complete.json').read_text())['files']
        for name, item in inventory.items():
            self.assertEqual(digest((output / name).read_bytes()), item['sha256'])
        self.assertNotIn('<script src=', (output / 'report.html').read_text())
        backup(r.pasta, self.root / 'after.zip')
        with zipfile.ZipFile(self.root / 'after.zip') as archive:
            archive.extractall(self.root / 'again')
        restored = Registro(self.root / 'again' / 'registry')
        self.assertEqual(restored.eventos(conferir_objetos=True), r.eventos())
        self.assertEqual(self.r.eventos(), self.original_events)

    def test_duplicate_reuse_and_explicit_revision_preserve_primary_metrics(self):
        self.advance()
        first = v.evaluate(self.r.pasta, '2026-10', self.root / 'primary', self.grib)
        primary = painel(self.r)['metricas_primarias']
        second = v.evaluate(self.r.pasta, '2026-10', self.root / 'again', self.grib)
        self.assertTrue(second['reused'])
        self.assertEqual(first['selected_verification_id'], second['selected_verification_id'])
        changed = grib_fixture(self.root / 'revised.grib', .003)
        with self.assertRaisesRegex(ContratoError, 'Different observation'):
            v.evaluate(self.r.pasta, '2026-10', self.root / 'blocked', changed)
        self.assertFalse((self.root / 'blocked').exists())
        revision = v.evaluate(self.r.pasta, '2026-10', self.root / 'revision', changed, revision=True)
        self.assertFalse(revision['primary'])
        self.assertEqual(painel(self.r)['metricas_primarias'], primary)
        self.assertEqual(len(self.r.eventos()), len(self.original_events) + 2)

    def test_report_export_failure_recovers_without_scoring_again(self):
        self.advance()
        with patch.object(v, 'report', side_effect=OSError('synthetic disk failure')):
            with self.assertRaisesRegex(OSError, 'disk failure'):
                v.evaluate(self.r.pasta, '2026-10', self.root / 'interrupted', self.grib)
        events = self.r.eventos()
        with patch.object(v, 'metricas', side_effect=AssertionError('must not rescore')):
            v.report(self.r.pasta, self.root / 'recovered')
        self.assertEqual(self.r.eventos(), events)
        with self.assertRaisesRegex(ContratoError, 'already exists'):
            v.report(self.r.pasta, self.root / 'recovered')
        with self.assertRaisesRegex(ContratoError, 'outside the registry'):
            v.report(self.r.pasta, self.r.pasta / 'report')

    def test_bad_grib_headers_grid_negative_values_and_duplicate_fields_rejected(self):
        self.advance()
        variants = [dict(expver='0005'), dict(stream='mnth'), dict(paramId=167),
                    dict(dataDate=20260901), dict(dataTime=1200),
                    dict(longitudeOfFirstGridPointInDegrees=269), dict(value=-.001)]
        for i, settings in enumerate(variants):
            with self.subTest(settings=settings):
                path = grib_fixture(self.root / f'bad-{i}.grib', **settings)
                with self.assertRaises(ContratoError):
                    v.evaluate(self.r.pasta, '2026-10', self.root / f'bad-report-{i}', path)
        duplicate = self.root / 'duplicate.grib'
        duplicate.write_bytes(self.grib.read_bytes() * 2)
        with self.assertRaisesRegex(ContratoError, 'exactly one'):
            v.evaluate(self.r.pasta, '2026-10', self.root / 'duplicate-report', duplicate)
        self.assertEqual(self.r.eventos(), self.original_events)

    def test_modified_forecast_and_simultaneous_write_are_rejected(self):
        self.advance()
        with self.r.transacao():
            with self.assertRaisesRegex(ContratoError, 'Registry is in use'):
                v.evaluate(self.r.pasta, '2026-10', self.root / 'locked', self.grib)
        ref = self.forecast['dados']['previsao']
        path = self.r.pasta / 'objetos' / (ref['sha256'] + '.bin')
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ContratoError, 'Modified object'):
            v.check(self.r.pasta, '2026-10')

    def test_scores_match_existing_evaluator_and_request_uses_target_not_input_lag(self):
        self.advance()
        folder = self.root / 'decoded'
        folder.mkdir()
        raw, normalized, _ = v.decode_observation(self.grib, '2026-10', folder)
        event = avaliar(self.r, '2026-10', normalized, dict(produto='ERA5 monthly_averaged_reanalysis',
            expver=1, mes_alvo='2026-10', url=v.SOURCE_URL, arquivo_bruto=raw))
        result = v.evaluate(self.r.pasta, '2026-10', self.root / 'same-score', self.grib, revision=True)
        added = self.r.obter(result['selected_verification_id'])
        self.assertEqual(event['dados']['metricas'], added['dados']['metricas'])
        request = v.request_for('2026-10')
        self.assertEqual(request['month'], ['10'])
        self.assertEqual(request['variable'], ['total_precipitation'])
        self.assertEqual(request['product_type'], ['monthly_averaged_reanalysis'])


if __name__ == '__main__':
    unittest.main()
