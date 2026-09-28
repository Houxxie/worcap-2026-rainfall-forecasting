"""Verifica temporalidade real da SST e contratos do novo caminho de inferência."""
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
src = (HERE/'extensao_sst.py').read_text(encoding='utf-8')
funcs = [n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef)]
def exigir(ok,msg):
    if not bool(ok): raise ValueError(msg)
def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
ns = dict(np=np,pd=pd,xr=xr,Path=Path,PCA=PCA,threadpool_limits=threadpool_limits,
          exigir=exigir,sha256=sha256,SST_COMPONENTES=8)
exec(compile(ast.Module(body=funcs,type_ignores=[]),'<extensao>','exec'),ns)
sst,_ = ns['sst_auditar_arquivo'](ROOT/'entregas/etapa8F_M10_SST_PCA/ersstv5_4graus_198201_202411.nc')
alvos = pd.date_range('1982-02-01','2006-12-01',freq='MS')
datas = pd.date_range('2007-01-01',periods=24,freq='MS')
fit = ns['sst_ajustar_pca'](sst,alvos)
ns['sst_auditar_causalidade'](sst,alvos,datas,fit)
scores = ns['sst_transformar'](sst,fit,datas)
assert scores.shape == (24,8)
assert np.datetime64(fit['origens_treino'][-1]) == np.datetime64('2006-11-01')
# Mudar qualquer dado após o treino não altera a PCA (inclui todos os meses de validação).
modified = sst.copy(deep=True)
modified.values[pd.DatetimeIndex(modified.time.values) > pd.Timestamp('2006-11-01')] = 10000
fit2 = ns['sst_ajustar_pca'](modified,alvos)
for k in fit: np.testing.assert_array_equal(fit[k],fit2[k])
# Máscara e centragem se baseiam apenas no treino; janeiro usa os janeiros históricos.
origens=(alvos.to_period('M')-1).to_timestamp()
raw=sst.sel(time=origens).values.reshape(len(origens),-1).astype('float64')[:,fit['mascara']]
for m in range(1,13):
    np.testing.assert_allclose(fit['clima_mensal'][m-1],raw[origens.month==m].mean(axis=0),rtol=0,atol=1e-12)

# A inferência deve manter 29/31 features na mesma ordem e adicionar PCs por mês.
lat=np.array([-45.,-20.,5.]);lon=np.array([-65.,-45.])
clima=xr.DataArray(np.ones((12,3,2),dtype='float32'),dims=('month','lat','lon'),coords=dict(month=range(1,13),lat=lat,lon=lon))
cs=clima.copy();cs.attrs['corte']='2006-12-01'
ns.update(FEATURES_SEAS5=[f'f{i}' for i in range(29)],FEATURES_MULTISSISTEMA=[f'f{i}' for i in range(31)],
    SST_FEATURES=[f'p{i}' for i in range(8)],matriz_mes=lambda a,ca,c,t:np.full((6,27),t.month,dtype='float32'),
    valores_seas5=lambda s,t:np.ones(6,dtype='float32')*28,
    valores_anomalia_seas5=lambda s,c,t:np.ones(6,dtype='float32')*29,
    valores_cfsv2=lambda s,t:np.ones(6,dtype='float32')*30,
    valores_anomalia_cfsv2=lambda s,c,t:np.ones(6,dtype='float32')*31)
class Probe:
    def __init__(self,fs): self.fs=fs;self.calls=[]
    def feature_name(self): return self.fs
    def predict(self,x): self.calls.append(x.copy());return x.sum(axis=1)
for name,fs in [('arvores_seas5',ns['FEATURES_SEAS5']),('arvores_multissistema',ns['FEATURES_MULTISSISTEMA'])]:
    probe=Probe(fs+ns['SST_FEATURES'])
    pred=ns['pesquisa_prever_sst'](probe,name,None,None,None,None,clima,cs,cs,datas,scores)
    assert pred.shape==(24,3,2)
    for i,x in enumerate(probe.calls):
        np.testing.assert_array_equal(x[:,-8:],np.broadcast_to(scores[i],(6,8)))
        assert (x[:,:27]==datas[i].month).all()
        assert (x[:,27]==28).all() and (x[:,28]==29).all()
        if len(fs)==31: assert (x[:,29]==30).all() and (x[:,30]==31).all()
    try: ns['pesquisa_prever_sst'](probe,name,None,None,None,None,clima,cs,cs,['2006-12-01'],scores[:1])
    except ValueError: pass
    else: raise AssertionError('Não rejeitou inferência no treino')

baseline=xr.concat([clima.sel(month=t.month).drop_vars('month') for t in datas],dim=pd.Index(datas,name='time'))
obs=baseline.copy(data=np.zeros(baseline.shape))
dp=ns['pesquisa_diagnosticos'](obs,{'controle':baseline,'sst_8pcs':baseline*2},'H1')
tab=ns['pesquisa_resumir'](dp[dp.regiao=='dominio_inteiro'],['modelo']).set_index('modelo')
np.testing.assert_allclose(tab.loc['controle',['rmse','rmse_area','mae','vies']].to_numpy(dtype=float),1)
np.testing.assert_allclose(tab.loc['sst_8pcs',['rmse','rmse_area','mae','vies']].to_numpy(dtype=float),2)
import tempfile
with tempfile.TemporaryDirectory(prefix='nimbus_netcdf_') as tmpdir:
    original = xr.Dataset(dict(controle=baseline.astype('float64'),sst_8pcs=baseline.astype('float64')*2))
    target = Path(tmpdir)/'previsoes.nc'
    original.to_netcdf(target,engine='h5netcdf',encoding={v:dict(zlib=True,complevel=4) for v in original.data_vars})
    with xr.open_dataset(target,engine='h5netcdf') as loaded:
        xr.testing.assert_identical(original,loaded.load())
report=dict(status='aprovado',extensao_sha256=sha256(HERE/'extensao_sst.py'),
    testes=['SST real: cobertura, unidades e grade','PCA invariável a mudanças fora do treino',
    '12 climatologias calculadas somente no treino','Inferência: ordem das 29/31 features e alinhamento dos oito PCs',
    'Inferência no treino rejeitada','Métricas uniformes e por área verificadas com valores conhecidos',
    'Gravação e releitura NetCDF h5netcdf preservam coordenadas e valores float64'],
    escopo='PCA com SST real; inferência e métricas em grade sintética. Treinamento climático completo depende da execução no Kaggle.',
    variancia_8pcs_H1=float(fit['variancia_explicada'].sum()))
(HERE/'verificacao_local.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(report,indent=2,ensure_ascii=False))
