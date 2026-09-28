"""Testes de calendário e causalidade; não treinam árvores com dados reais."""
from pathlib import Path
import runpy
import unittest
import json

B = runpy.run_path(str(Path(__file__).resolve().parent/'biblioteca.py'))
import numpy as np
import pandas as pd
import xarray as xr


class ContratoTemporal(unittest.TestCase):
    def test_virada_ano_e_janela(self):
        self.assertEqual(B['origem_mensal'](['2007-01-01'],4)[0],pd.Timestamp('2006-09-01'))
        self.assertEqual(B['origem_mensal'](['2007-01-01'],3)[0],pd.Timestamp('2006-10-01'))
        self.assertEqual(B['origem_mensal'](['2007-01-01'],2)[0],pd.Timestamp('2006-11-01'))
        j=B['janela_referencia']('2006-09-01')
        self.assertEqual(j[0],pd.Timestamp('1976-10-01'))
        self.assertEqual(len(j),360)
        self.assertTrue(all((j.month==m).sum()==30 for m in range(1,13)))
        self.assertEqual(B['calendario_pareado']('2006-09-01')[0],pd.Timestamp('1982-03-01'))

    def test_embargo_chuva(self):
        good=B['calendario_emissoes'](pd.date_range('2007-01-01',periods=24,freq='MS'),'2006-09-01')
        self.assertTrue((good.ultimo_alvo_treino<good.time_alvo).all())
        with self.assertRaises(ValueError):
            B['calendario_emissoes'](['2007-01-01'],'2006-12-01')

    def test_clima_chuva_sem_futuro(self):
        ts=pd.date_range('1976-10-01','2008-12-01',freq='MS')
        a=xr.DataArray(np.arange(len(ts)*4,dtype=np.float32).reshape(-1,2,2),
            dims=('time','lat','lon'),coords=dict(time=ts,lat=[-40.,0.],lon=[-80.,-50.]))
        cl,s,c=B['estatisticas_climatologia'](a,'2006-09-01')
        a.values[ts>pd.Timestamp('2006-09-01')]=1e8
        np.testing.assert_array_equal(cl,B['estatisticas_climatologia'](a,'2006-09-01')[0])
        self.assertTrue((c==30).all())
        # A média LOO deve excluir o próprio alvo em todos os doze meses.
        for m in range(1,13):
            sel=a.sel(time=B['janela_referencia']('2006-09-01'))
            vals=sel.values[pd.DatetimeIndex(sel.time.values).month==m]
            np.testing.assert_allclose((s[m-1]-vals[0])/29,vals[1:].mean(0,dtype=np.float64))

    def test_matriz_usa_meses_reais(self):
        ts=pd.date_range('2006-01-01','2007-03-01',freq='MS')
        clima=xr.DataArray(np.ones((12,2,2),np.float32),dims=('month','lat','lon'),
            coords=dict(month=range(1,13),lat=[-40.,0.],lon=[-80.,-50.]))
        at={n:xr.DataArray(np.broadcast_to((ts.month.to_numpy()+j*100)[:,None,None],(len(ts),2,2)).astype(np.float32).copy(),
            dims=('time','lat','lon'),coords=dict(time=ts,lat=clima.lat,lon=clima.lon))
            for j,n in enumerate(B['VARIAVEIS'])}
        ca={n:np.zeros((12,2,2),np.float32) for n in at}
        ca['_clima_indices']=np.zeros((12,4))
        idx=pd.DataFrame(np.repeat(ts.month.to_numpy()[:,None],4,axis=1),index=ts,columns=B['INDICES_NOMES'])
        # run_path retorna uma cópia do dict; funções conservam o namespace de definição.
        glob=B['matriz_mes'].__globals__
        antigo=glob.get('INDICES_OC')
        glob['INDICES_OC']=idx
        try:
            x=B['matriz_mes'](at,ca,clima,'2007-01-01')
            np.testing.assert_array_equal(x[0,:9],np.arange(9)*100+9)
            np.testing.assert_array_equal(x[0,23:],np.repeat(10.,4))
            for v in at.values():v.values[ts>pd.Timestamp('2006-09-01')]=999999
            idx.loc[idx.index>pd.Timestamp('2006-10-01')]=999999
            np.testing.assert_array_equal(x,B['matriz_mes'](at,ca,clima,'2007-01-01'))
        finally:
            if antigo is None:glob.pop('INDICES_OC',None)
            else:glob['INDICES_OC']=antigo

    def test_pca_lag2_mascara_somente_treino(self):
        ts=pd.date_range('1982-01-01','2008-01-01',freq='MS')
        rng=np.random.default_rng(13)
        a=xr.DataArray(rng.normal(size=(len(ts),4,5)).astype(np.float32),dims=('time','lat','lon'),
            coords=dict(time=ts,lat=[-50.,-20.,20.,50.],lon=np.arange(5)))
        a.values[:,:1,:1]=np.nan
        alvos=pd.date_range('1982-03-01','2006-09-01',freq='MS')
        st=B['sst_ajustar_pca'](a,alvos)
        self.assertEqual(pd.Timestamp(st['origens_treino'][-1]),pd.Timestamp('2006-07-01'))
        before=B['sst_transformar'](a,st,['2007-01-01'])
        a.values[ts>pd.Timestamp('2006-11-01')]=np.nan
        st2=B['sst_ajustar_pca'](a,alvos)
        for k in ['mascara','componentes','centro','clima_mensal']:
            np.testing.assert_array_equal(st[k],st2[k])
        np.testing.assert_array_equal(before,B['sst_transformar'](a,st2,['2007-01-01']))
        # Validação ausente deve parar: não alargar máscara nem imputar usando futuro.
        a.values[ts==pd.Timestamp('2006-11-01'),1,1]=np.nan
        with self.assertRaises(ValueError):B['sst_transformar'](a,st,['2007-01-01'])

    def test_publicacao_desconhecida_bloqueia(self):
        gate=B['exigir_disponibilidade_real']
        with self.assertRaises(ValueError):gate([dict(fonte='CFSv2')],'2026-09-30T23:59:59Z',['CFSv2'])
        r=dict(fonte='SST',disponivel_em='2026-09-05T00:00:00Z',recebido_em='2026-09-06T00:00:00Z',
               versao='snapshot_teste',url='https://example.invalid/teste',sha256='0'*64)
        self.assertTrue(gate([r],'2026-09-30T23:59:59Z',['SST']))
        r['recebido_em']='2026-10-01T00:00:00Z'
        with self.assertRaises(ValueError):gate([r],'2026-09-30T23:59:59Z',['SST'])


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ContratoTemporal))
    if not result.wasSuccessful():raise RuntimeError('Contrato temporal falhou; não treinar.')
    print('Seis testes temporais aprovados; dados sintéticos. Sem alegação de publicação histórica.')
