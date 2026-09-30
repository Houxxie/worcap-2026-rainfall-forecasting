"""Synthetic checks: chronology, target isolation, arithmetic and saved-run integrity."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
import xarray as xr
from research.common.inputs import ROOT, write_json, sha256
from research.common.synthetic_fixture import fixture
from research.spatial_unet.training import MonthlyMaps, inference_maps
from research.common.inputs import Context
from research.spatial_unet.network import RainfallUNet
from .calibration import splits, ForecastMaps, estimate, correct
from .experiment import protocol, fit_calibration, run_block, verify_block, summarize, execute


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def setUp(self):
        self.data=fixture()
        self.training=pd.date_range('1994-01-01','2003-12-01',freq='MS')
        self.split=splits(self.training,self.training)

    def predictions(self, bias=2.):
        p=self.data.rain.sel(time=self.split['calibration']).astype('float64')+bias
        p.attrs['units']='mm/day';self.data.rain.attrs['units']='mm/day'
        return p

    def estimate(self,p):
        return estimate(p,self.data.rain,self.split['calibration'],self.training[-1],self.split['fit'][-1],self.split['selection_validation'][-1])

    def test_nested_dates_keep_calibration_out_of_fit_and_selection(self):
        s=self.split
        self.assertEqual(s['calibration'][0],pd.Timestamp('2002-01-01'))
        self.assertEqual(s['fit'][-1],pd.Timestamp('2001-09-01'))
        self.assertEqual(s['selection_validation'][-1],s['fit'][-1])
        self.assertEqual(s['selection_fit'][-1],s['selection_validation'][0]-pd.DateOffset(months=4))
        self.assertFalse(s['calibration'].isin(s['fit']).any())
        self.assertFalse(s['calibration'].isin(s['selection_validation']).any())

    def test_half_bias_formula_and_future_observation_isolation(self):
        p=self.predictions()
        a,monthly=self.estimate(p)
        self.assertAlmostEqual(a['offset_mm_day'],1.)
        self.assertEqual(len(monthly),24)
        self.data.rain.loc[dict(time=slice('2004-01-01',None))]=1e8
        b,_=self.estimate(p)
        self.assertEqual(a,b)
        xr.testing.assert_allclose(correct(p,a),p-1.)

    def test_clipping_signed_bias_and_invalid_offset(self):
        p=self.predictions();record,_=self.estimate(p)
        p[:]=.2
        self.assertEqual(float(correct(p,record).max()),0.)
        dry={**record,'raw_mean_bias':-2.,'offset_mm_day':-1.}
        self.assertAlmostEqual(float(correct(p,dry).min()),1.2)
        with self.assertRaisesRegex(ValueError,'Invalid offset'):
            correct(p,{**record,'offset_mm_day':9.})

    def test_wrong_calendar_units_grid_and_selection_leak_rejected(self):
        p=self.predictions()
        with self.assertRaisesRegex(ValueError,'calendar'): self.estimate(p.isel(time=slice(1,None)))
        p.attrs['units']='mm'
        with self.assertRaisesRegex(ValueError,'units'): self.estimate(p)
        p.attrs['units']='mm/day'
        with self.assertRaises(ValueError): self.estimate(p.assign_coords(lat=p.lat+1))
        with self.assertRaisesRegex(ValueError,'entered fitting'):
            estimate(p,self.data.rain,self.split['calibration'],self.training[-1],self.split['fit'][-1],self.split['calibration'][0])

    def test_target_free_adapter_matches_original(self):
        ctx=Context(self.data,self.split['reference'],self.split['fit'])
        net=RainfallUNet(widths=(4,8,16))
        with torch.no_grad(): net.output.bias.fill_(.2)
        scaling=dict(mean=np.zeros(31,dtype='float32'),scale=np.ones(31,dtype='float32'))
        dates=self.split['calibration'][:2]
        with tempfile.TemporaryDirectory() as temp:
            old=MonthlyMaps(ctx,dates,Path(temp)/'old',scaling=scaling)
            try: expected=inference_maps(net,old,torch.device('cpu'))
            finally: old.close()
            adapter=ForecastMaps(ctx,dates,scaling)
            self.data.rain.loc[dict(time=dates)]=np.nan
            actual=inference_maps(net,adapter,torch.device('cpu'))
        np.testing.assert_array_equal(expected,actual)

    def test_real_small_fits_never_select_epochs_on_calibration_rain(self):
        _,config=protocol()
        config.update(max_epochs=1,patience=1,batch_size=16)
        config['architecture']['channels']=[4,8,16]
        self.data.rain.attrs['units']='mm/day'
        with tempfile.TemporaryDirectory() as temp:
            temp=Path(temp)
            a=fit_calibration(self.data,self.training,self.training,config,temp/'a',torch.device('cpu'))
            self.data.rain.loc[dict(time=self.split['calibration'])]+=3
            b=fit_calibration(self.data,self.training,self.training,config,temp/'b',torch.device('cpu'))
            with xr.open_dataarray(temp/'a/predictions.nc') as x, xr.open_dataarray(temp/'b/predictions.nc') as y:
                xr.testing.assert_identical(x,y)
            self.assertEqual(a['auxiliary_selected_epochs'],b['auxiliary_selected_epochs'])
            self.assertAlmostEqual(b['offset_mm_day']-a['offset_mm_day'],-1.5,places=6)

    def test_complete_block_report_and_tamper_rejection(self):
        self.data.rain.attrs['units']='mm/day'
        dates=pd.date_range('2007-01-01',periods=24,freq='MS')
        truth=self.data.rain.sel(time=dates)
        def reference(data,block,config,directory,device):
            directory.mkdir(parents=True)
            xr.Dataset(dict(hybrid=truth+.4,unet=truth+.8,climatology=truth+.6)).to_netcdf(directory/'predictions.nc',engine='h5netcdf')
        def calibration(data,training,reference,config,folder,device):
            folder.mkdir()
            record=dict(schema='unet_global_bias_v1',units='mm/day',shrinkage=.5,calibration_months=24,
                prior_zero_bias_months=24,raw_mean_bias=.8,offset_mm_day=.4,outer_targets_used=False,
                calibration_start='2004-10-01',calibration_end='2006-09-01')
            write_json(folder/'offset.json',record)
            return record
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp);parent=output/'H1';attempt=parent/'attempt_001'
            with patch('research.bias_calibration.experiment.original_block',reference),patch('research.bias_calibration.experiment.fit_calibration',calibration):
                rows=run_block(self.data,dict(nome='H1',inicio='2007-01-01'),{},attempt,torch.device('cpu'))
            write_json(parent/'complete.json',dict(signature='synthetic',attempt=attempt.name,
                files={p.relative_to(attempt).as_posix():sha256(p) for p in attempt.rglob('*') if p.is_file()}))
            loaded=verify_block(parent,'synthetic')
            self.assertEqual(len(rows),len(loaded))
            result=summarize(self.data,output,[rows],['H1','H2'])
            self.assertEqual(result['status'],'partial')
            self.assertAlmostEqual(result['comparisons']['blend']['rmse'],-.1,places=6)
            self.assertTrue((output/'report.html').is_file())
            with xr.open_dataset(attempt/'predictions.nc') as p:
                np.testing.assert_allclose(p.blend,.75*p.hybrid+.25*p.unet)
                np.testing.assert_allclose(p.blend_corrected,.75*p.hybrid+.25*p.unet_corrected)
            (attempt/'monthly_metrics.csv').write_text('changed')
            with self.assertRaisesRegex(ValueError,'Changed completed artifact'): verify_block(parent,'synthetic')

    def test_protocol_pins_original_architecture_and_single_candidate(self):
        plan,config=protocol()
        self.assertEqual(plan['prior_zero_bias_months'],24)
        self.assertEqual(config['architecture']['channels'],[16,32,64])
        self.assertEqual(plan['weights'],dict(hybrid=.75,unet=.25))
        self.assertFalse(plan['evaluation_2021_2022'])

    def test_interrupted_run_resumes_verified_blocks_and_preserves_attempt(self):
        calls=[]
        def fake_block(data, block, config, directory, device):
            directory.mkdir()
            calls.append(block['nome'])
            if block['nome']=='H2' and calls.count('H2')==1:
                raise RuntimeError('synthetic interruption')
            dates=pd.date_range(block['inicio'],periods=24,freq='MS')
            truth=data.rain.isel(time=slice(0,24)).assign_coords(time=dates)
            fields={name:truth+value for name,value in dict(hybrid=.4,unet=.8,blend=.5,unet_corrected=.4,blend_corrected=.4,climatology=.6).items()}
            rows=data.lib.pesquisa_diagnosticos(truth,fields,block['nome'])
            rows.to_csv(directory/'monthly_metrics.csv',index=False)
            (directory/'calibration').mkdir()
            write_json(directory/'calibration/offset.json',dict(raw_mean_bias=.8,offset_mm_day=.4,calibration_start='synthetic',calibration_end='synthetic'))
            return rows
        with tempfile.TemporaryDirectory() as temp:
            temp=Path(temp);paths={name:temp/name for name in ['official','seas5','cfsv2']}
            audit=dict(synthetic=True)
            with patch('research.bias_calibration.experiment.check',return_value=(paths,audit)),patch('research.bias_calibration.experiment.preflight',return_value=(paths,audit)),patch('research.bias_calibration.experiment.Inputs',return_value=self.data),patch('research.bias_calibration.experiment.run_block',fake_block),patch('builtins.print'):
                with self.assertRaisesRegex(RuntimeError,'synthetic interruption'): execute(temp/'run',device='cpu')
                result=execute(temp/'run',device='cpu')
                self.assertTrue(result['complete_seven_blocks'])
                self.assertEqual(calls.count('H1'),1)
                self.assertEqual(calls.count('H2'),2)
                self.assertTrue((temp/'run/H2/attempt_001/failure.json').is_file())
                self.assertEqual(json.loads((temp/'run/H2/complete.json').read_text())['attempt'],'attempt_002')
                execute(temp/'run',device='cpu')
                self.assertEqual(len(calls),8)
                self.assertEqual(json.loads((temp/'run/run_state.json').read_text())['status'],'complete')


if __name__=='__main__': unittest.main()
