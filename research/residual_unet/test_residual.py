"""Small synthetic checks for residual targets, chronology, freezing and resume."""
from pathlib import Path
import copy
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
import xarray as xr
from research.common.synthetic_fixture import fixture
from research.common.inputs import Context, write_json
from research.common import baseline
from research.spatial_unet.network import RainfallUNet, restore
from research.spatial_unet.training import MonthlyMaps, inference_maps
from .data import crossfit_plan, ResidualMaps, ForecastMaps, forecast_field, validate_field
from .experiment import protocol, run_pilot, Stages


def extended_fixture():
    data = fixture()
    dates = pd.date_range('1976-06-01', '2021-01-01', freq='MS')
    def extend(field):
        expanded = field.isel(time=np.arange(len(dates)) % field.sizes['time']).assign_coords(time=dates)
        for coordinate in ['time_origem', 'inicializacao_mais_recente']:
            if coordinate in expanded.coords:
                expanded = expanded.assign_coords({coordinate: ('time', data.lib.origem_mensal(dates, 1).values)})
        expanded.attrs = dict(units='mm/day')
        return expanded
    data.rain, data.seas, data.cfs = map(extend, [data.rain, data.seas, data.cfs])
    data.atmosphere = {k: extend(v) for k, v in data.atmosphere.items()}
    indices = data.lib.INDICES_OC
    data.lib.INDICES_OC = pd.DataFrame(indices.to_numpy()[np.arange(len(dates)) % len(indices)], index=dates, columns=indices.columns)
    data.audit = dict(synthetic=True)
    data.lib.ARVORES = 2
    data.lib.PONTOS_POR_MES = 4
    data.lib.PARAMETROS = dict(data.lib.PARAMETROS, num_threads=1, min_data_in_leaf=3, num_leaves=4)
    return data


class ResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.data = extended_fixture()
        self.cutoff = pd.Timestamp('2018-09-01')
        self.training = self.data.lib.calendario_pareado(self.cutoff)
        self.reference = self.data.lib.janela_referencia(self.cutoff)

    def test_five_groups_keep_full_references_and_gap(self):
        groups = crossfit_plan(self.data, self.training[-120:])
        self.assertEqual(len(groups), 5)
        self.assertEqual(groups[0]['cutoff'], pd.Timestamp('2008-06-01'))
        self.assertEqual(groups[-1]['targets'][-1], self.cutoff)
        for group in groups:
            self.assertEqual(group['cutoff'], group['targets'][0] - pd.DateOffset(months=4))
            self.assertLess(group['cutoff'], group['targets'][0])
        with self.assertRaisesRegex(ValueError, 'Insufficient past rainfall'):
            crossfit_plan(self.data, pd.date_range('1996-10-01', periods=120, freq='MS'))
        with self.assertRaises(ValueError):
            crossfit_plan(self.data, self.training[-120:][::-1])

    def test_residual_changes_only_target_and_base(self):
        context = Context(self.data, self.reference, self.training)
        dates = self.training[-2:]
        hybrid = forecast_field(self.data, dates, np.full((2, 9, 13), 2.), 'hybrid')
        with tempfile.TemporaryDirectory() as folder:
            maps = MonthlyMaps(context, dates, Path(folder) / 'maps', training=True)
            try:
                residual = ResidualMaps(maps, hybrid, self.data)
                x, target, base = residual[0]
                torch.testing.assert_close(x, maps[0][0], rtol=0, atol=0)
                np.testing.assert_array_equal(target.numpy(), maps.y[0] - 2.)
                np.testing.assert_array_equal(base.numpy(), np.full((9, 13), 2.))
                self.assertIs(residual.scaling, maps.scaling)
                with self.assertRaises(ValueError):
                    ResidualMaps(maps, hybrid.isel(time=[1, 0]), self.data)
                with self.assertRaises(ValueError):
                    ResidualMaps(maps, hybrid.assign_coords(lat=hybrid.lat + 1), self.data)
            finally:
                maps.close()

    def test_zero_network_exactly_reproduces_hybrid_and_ignores_outer_truth(self):
        context = Context(self.data, self.reference, self.training)
        dates = pd.date_range('2019-01-01', periods=2, freq='MS')
        hybrid = forecast_field(self.data, dates, np.full((2, 9, 13), 2.123456789), 'hybrid')
        scaling = dict(mean=np.zeros(31, 'float32'), scale=np.ones(31, 'float32'))
        dataset = ForecastMaps(context, dates, scaling, hybrid)
        x = dataset[0][0].clone()
        net = RainfallUNet(widths=(4, 8, 16))
        pred = inference_maps(net, dataset, torch.device('cpu'))
        np.testing.assert_array_equal(pred, hybrid.values)
        self.data.rain.loc[dict(time=slice('2018-10-01', None))] = 999.
        torch.testing.assert_close(dataset[0][0], x, rtol=0, atol=0)
        np.testing.assert_array_equal(inference_maps(net, dataset, torch.device('cpu')), pred)
        with torch.no_grad():
            net.output.bias.fill_(-100.)
        self.assertEqual(float(inference_maps(net, dataset, torch.device('cpu')).max()), 0.)

    def test_crossfit_hybrid_does_not_see_its_forecast_targets(self):
        group = crossfit_plan(self.data, self.training[-120:])[0]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            a = baseline.fit(self.data, group['cutoff'], folder / 'a')
            expected = baseline.predict(self.data, a, group['targets'][:2])
            self.data.rain.loc[dict(time=slice(group['cutoff'] + pd.DateOffset(months=1), None))] += 1000.
            b = baseline.fit(self.data, group['cutoff'], folder / 'b')
            actual = baseline.predict(self.data, b, group['targets'][:2])
            xr.testing.assert_equal(expected, actual)
            self.assertEqual(a['audit']['sample_hash'], b['audit']['sample_hash'])

    def test_stages_verify_cached_files_and_preserve_incomplete_work(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            stage = Stages(root, 'example-signature')
            calls = []
            def action():
                calls.append(1)
                (root / 'one/data.txt').write_text('result')
            stage('one', root / 'one', action)
            stage('one', root / 'one', action)
            self.assertEqual(len(calls), 1)
            (root / 'one/data.txt').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Changed stage artifact'):
                stage('one', root / 'one', action)
            (root / 'incomplete').mkdir()
            (root / 'incomplete/keep.txt').write_text('keep')
            with self.assertRaisesRegex(ValueError, 'Incomplete stage'):
                stage('interrupted', root / 'incomplete', action)
            self.assertTrue((root / 'incomplete/keep.txt').is_file())

    def test_small_end_to_end_pilot_with_real_fits_and_checkpoint_reuse(self):
        plan, config = protocol()
        config = copy.deepcopy(config)
        config['architecture']['channels'] = [4, 8, 16]
        dates = pd.date_range('2019-01-01', periods=24, freq='MS')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state = baseline.fit(self.data, self.cutoff, root / 'expected_hybrid')
            hybrid = baseline.predict(self.data, state, dates)
            metrics = self.data.lib.pesquisa_diagnosticos(self.data.rain.sel(time=dates), {'controle_defasado': hybrid}, 'C')
            archived = self.data.lib.pesquisa_resumir(metrics[metrics.regiao == 'dominio_inteiro'], ['modelo', 'bloco'])
            archive = root / 'research/lagged_sources/evidence/metricas_blocos.csv'
            archive.parent.mkdir(parents=True)
            archived.to_csv(archive, index=False)
            output = root / 'result'
            output.mkdir()
            stages = Stages(output, 'synthetic')
            with patch('research.residual_unet.experiment.ROOT', root):
                result = run_pilot(self.data, plan, config, output, torch.device('cpu'), stages)
            self.assertTrue(result['pilot_complete'])
            self.assertFalse(result['complete_seven_blocks'])
            rows = pd.read_csv(output / 'monthly_metrics.csv')
            self.assertEqual(len(rows), 4 * 24 * 4)
            self.assertEqual(set(rows.modelo), {'hybrid', 'direct_unet', 'direct_blend', 'residual_unet'})
            for name in ['direct_unet', 'residual_unet']:
                record = json.loads((output / name / 'network/training.json').read_text())
                self.assertEqual(record['selected_epochs'], 8)
                self.assertEqual(record['train_months'], 120)
                self.assertFalse(record['inner_selection'])
                net, _ = restore(output / name / 'network/network.pt')
                self.assertEqual(net.output.out_channels, 1)
            with (patch('research.residual_unet.experiment.ROOT', root),
                  patch('research.residual_unet.experiment.fit_network', side_effect=AssertionError('Unexpected neural refit')),
                  patch('research.common.baseline.fit', side_effect=AssertionError('Unexpected hybrid refit'))):
                repeated = run_pilot(self.data, plan, config, output, torch.device('cpu'), stages)
            self.assertEqual(repeated, result)


if __name__ == '__main__':
    unittest.main()
