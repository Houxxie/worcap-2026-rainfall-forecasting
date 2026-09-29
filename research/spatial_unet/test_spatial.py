"""Synthetic temporal, spatial and training checks; never a rainfall skill benchmark."""
from pathlib import Path
import copy
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.common.inputs import Context, library
from research.spatial_unet.network import RainfallUNet, seed_everything, restore
from research.spatial_unet.training import MonthlyMaps, split_inner, fit_network, inference_maps


from research.common.synthetic_fixture import fixture


class SpatialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.data = fixture()
        self.reference = pd.date_range('1976-10-01', '2006-09-01', freq='MS')
        self.training = self.reference[self.reference >= '1982-03-01']

    def test_inner_selection_has_label_gap_and_no_new_history(self):
        train, reference, valid = split_inner(self.training, self.reference)
        self.assertEqual(len(valid), 24)
        self.assertEqual(train[-1], valid[0] - pd.DateOffset(months=4))
        self.assertEqual(reference[0], self.reference[0])
        self.assertTrue(reference.max() < valid.min())

    def test_future_observations_do_not_change_context_or_features(self):
        a = Context(self.data, self.reference, self.training)
        expected = a.features('2007-01-01')
        self.data.rain.loc[dict(time=slice('2006-10-01', None))] = 9000
        for field in self.data.atmosphere.values():
            field.loc[dict(time=slice('2006-10-01', None))] = 9000
        self.data.lib.INDICES_OC.loc['2006-11-01':] = 9000
        self.data.seas.loc[dict(time=slice('2007-02-01', None))] = 9000
        self.data.cfs.loc[dict(time=slice('2007-02-01', None))] = 9000
        b = Context(self.data, self.reference, self.training)
        np.testing.assert_array_equal(a.climate, b.climate)
        np.testing.assert_array_equal(expected, b.features('2007-01-01'))

    def test_training_climatology_excludes_entire_target_map(self):
        t = self.training[36]
        a = Context(self.data, self.reference, self.training).features(t, training=True)[22]
        self.data.rain.loc[dict(time=t)] += np.arange(117).reshape(9, 13)
        b = Context(self.data, self.reference, self.training).features(t, training=True)[22]
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)

    def test_context_matches_existing_baseline_transform(self):
        m = self.data.lib
        ctx = Context(self.data, self.reference, self.training)
        climate, _, _ = m.estatisticas_climatologia(self.data.rain, self.reference[-1])
        ca = m.ajustar_clima_atmosfera(self.data.atmosphere, m.origem_mensal(self.reference, 4))
        ca['_clima_indices'] = m.ajustar_clima_indices(m.INDICES_OC, m.origem_mensal(self.reference, 4), self.reference[-1])
        cs = m.ajustar_clima_seas5(self.data.seas, self.reference[-1])
        cc = m.ajustar_clima_previsao(self.data.cfs, self.reference[-1], 'CFSv2')
        t = pd.Timestamp('2007-01-01')
        x = np.column_stack([m.matriz_mes(self.data.atmosphere, ca, climate, t),
            m.valores_seas5(self.data.seas, t), m.valores_anomalia_seas5(self.data.seas, cs, t),
            m.valores_cfsv2(self.data.cfs, t), m.valores_anomalia_cfsv2(self.data.cfs, cc, t)])
        np.testing.assert_array_equal(ctx.features(t).reshape(31, -1).T, x)

    def test_odd_grid_shape_and_initial_climatology(self):
        seed_everything()
        net = RainfallUNet()
        with torch.inference_mode():
            result = net(torch.zeros(1, 31, 301, 261))
        self.assertEqual(tuple(result.shape), (1, 301, 261))
        self.assertEqual(float(result.abs().max()), 0.)

    def test_gradients_and_checkpoint_roundtrip(self):
        seed_everything()
        net = RainfallUNet(widths=(4, 8, 16))
        x = torch.randn(2, 31, 9, 13)
        optimizer = torch.optim.AdamW(net.parameters(), lr=.001)
        loss = ((net(x) - 1) ** 2).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(net.output.weight.grad).all())
        optimizer.step()
        self.assertFalse(torch.equal(net(x), torch.zeros(2, 9, 13)))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'weights.pt'
            torch.save(dict(state_dict=net.state_dict(), widths=[4, 8, 16]), path)
            loaded, _ = restore(path)
            torch.testing.assert_close(net(x), loaded(x), rtol=0, atol=0)

    def test_training_loop_and_scaler_use_only_training_maps(self):
        config = json.loads(Path(__file__).with_name('protocol.json').read_text())
        config.update(max_epochs=2, patience=2)
        config['architecture']['channels'] = [4, 8, 16]
        ctx = Context(self.data, self.reference, self.training)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            train = MonthlyMaps(ctx, self.training[-4:], folder / 'train', training=True)
            valid = MonthlyMaps(ctx, pd.date_range('2007-01-01', periods=2, freq='MS'), folder / 'valid', scaling=train.scaling)
            try:
                np.testing.assert_array_equal(valid.scaling['mean'], train.scaling['mean'])
                model, epochs = fit_network(train, config, folder / 'model', torch.device('cpu'), validation=valid)
                self.assertIn(epochs, [1, 2])
                pred = inference_maps(model, valid, torch.device('cpu'))
                self.assertTrue(np.isfinite(pred).all() and (pred >= 0).all())
                self.assertEqual(pred.shape, (2, 9, 13))
            finally:
                train.close()
                valid.close()

    def test_invalid_month_cannot_be_used_for_training(self):
        ctx = Context(self.data, self.reference, self.training)
        with self.assertRaises(ValueError):
            ctx.features('2007-01-01', training=True)
        with self.assertRaises(ValueError):
            ctx.features('2006-09-01', training=False)


if __name__ == '__main__':
    unittest.main(argv=[__file__])
