"""Verificações locais de contrato; não medem desempenho do modelo."""
from pathlib import Path
import hashlib
import importlib.util
import json
import tempfile
import zipfile
import numpy as np
import pandas as pd
import xarray as xr
import nbformat

PASTA=Path(__file__).resolve().parent
ROOT=PASTA.parents[1]
spec=importlib.util.spec_from_file_location('era_teste',PASTA/'adquirir_era5_2025.py')
era=importlib.util.module_from_spec(spec);spec.loader.exec_module(era)
results={}
with tempfile.TemporaryDirectory() as temp:
    path=Path(temp)/'mensal.nc'
    vals=np.full((2,301,261),.003,dtype='float32')
    ds=xr.Dataset({'tp':(('valid_time','latitude','longitude'),vals)},
        coords={'valid_time':pd.to_datetime(['2022-01-01','2022-07-01']),
            'latitude':era.LAT[::-1],'longitude':era.LON+360})
    ds.tp.attrs['units']='m'
    ds.to_netcdf(path)
    v=era.ler_normalizado(path,{'total_precipitation':('tp','tp')})
    assert np.allclose(v.tp,3.) and v.tp.attrs['units']=='mm/day'
    assert np.array_equal(v.lat,era.LAT) and np.array_equal(v.lon,era.LON)
    results['normaliza_grade_e_converte_apenas_1000_sem_dias_mes']=True
    chuva=ds.assign_coords(valid_time=ds.valid_time+np.timedelta64(6,'h'))
    chuva.to_netcdf(Path(temp)/'chuva.nc')
    atmos=ds.rename({'tp':'tcc'}); atmos.tcc.attrs['units']='(0 - 1)'
    atmos.to_netcdf(Path(temp)/'atmos.nc')
    misto=Path(temp)/'mensal_misto.zip'
    with zipfile.ZipFile(misto,'w') as z:
        z.write(Path(temp)/'chuva.nc','chuva.nc'); z.write(Path(temp)/'atmos.nc','atmos.nc')
    v=era.ler_normalizado(misto,{'total_precipitation':('tp','tp'),'total_cloud_cover':('tcc','cloud_cover')})
    assert v.sizes['time']==2 and np.isfinite(v.to_array()).all()
    assert np.allclose(v.tp,3.) and np.allclose(v.cloud_cover,.003)
    results['ZIP_mensal_horarios_00_e_06_sem_duplicacao_ou_media']=True
    ds=ds.assign_coords(longitude=ds.longitude+.125);ds.to_netcdf(path)
    try: era.ler_normalizado(path,{'total_precipitation':('tp','tp')})
    except AssertionError: results['rejeita_grade_deslocada']=True
    else: raise AssertionError('Aceitou grade incorreta')

for n in PASTA.glob('*.py'): compile(n.read_text(encoding='utf-8'),n.name,'exec')
nb=nbformat.read(PASTA/'Nimbus_PA_avaliacao_2025.ipynb',as_version=4)
nbformat.validate(nb)
for i,c in enumerate(nb.cells):
    if c.cell_type=='code':compile(c.source,f'cell{i}','exec')
results['scripts_e_notebook_compilam']=True
protocolo=json.loads((PASTA/'protocolo_congelado.json').read_text(encoding='utf-8'))
assert protocolo['treino_alvos']==['1993-01-01','2022-12-01']
assert protocolo['teste_alvos']==['2025-01-01','2025-12-01']
assert hashlib.sha256((PASTA/'protocolo_congelado.json').read_bytes()).hexdigest()==(PASTA/'protocolo.sha256').read_text().strip()
results['protocolo_2025_fixo_e_sem_treino_na_reserva']=True
for tipo,var,path in [('CFSv2','cfsv2_tp_media','cfsv2_2025.nc'),('SST','sst','ersstv5_4graus_202411_202511.nc')]:
    root=ROOT/'outputs/nimbus_avaliacao_2025'/tipo
    m=json.loads((root/'manifesto_2025.json').read_text(encoding='utf-8'))
    assert m['sha256']==hashlib.sha256((root/path).read_bytes()).hexdigest()
    with xr.open_dataset(root/path) as d:
        expected=pd.date_range('2025-01-01','2025-12-01',freq='MS') if tipo=='CFSv2' else pd.date_range('2024-11-01','2025-11-01',freq='MS')
        assert pd.DatetimeIndex(d.time.values).equals(expected)
        if tipo=='CFSv2':
            assert np.isfinite(d[var]).all() and (d[var]>=0).all()
            assert (pd.to_datetime(d.inicializacao_mais_recente.values)<expected).all()
            assert len(m['pilotos'])==2
        else:
            assert m['comparacao_sobreposicao']['max_abs_degC']==0.
    results[f'{tipo}_real_datas_hash_e_auditoria']=True
indices=pd.read_csv(PASTA/'indices_2025.csv',parse_dates=['time_origem'])
assert pd.DatetimeIndex(indices.time_origem).equals(pd.date_range('2024-12-01','2025-11-01',freq='MS'))
assert np.isfinite(indices.iloc[:,1:].values).all()
results['indices_reais_snapshot_original_12_meses']=True
(PASTA/'verificacao_local.json').write_text(json.dumps(dict(status='PASS',testes=results,
    limite='Aquisição CDS, piloto com dados reais ERA5, treinamento e métricas 2025 ainda não executados.'),indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(results,ensure_ascii=False,indent=2))
