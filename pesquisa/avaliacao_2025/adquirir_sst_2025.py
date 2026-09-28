"""Somente preditores SST de 2025, sem abrir precipitação observada."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import requests
import numpy as np
import pandas as pd
import xarray as xr

BASE = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path.cwd()/'outputs'
SAIDA = BASE/'nimbus_avaliacao_2025'/'SST'
SAIDA.mkdir(parents=True,exist_ok=True)
URL = 'https://psl.noaa.gov/thredds/ncss/grid/Datasets/noaa.ersst.v5/sst.mnmean.nc'
PEDIDO = dict(var='sst',north=60,west=0,east=358,south=-60,horizStride=2,
    time_start='2024-11-01T00:00:00Z',time_end='2025-11-01T00:00:00Z',
    timeStride=1,accept='netcdf4')
path = SAIDA/'ersstv5_4graus_202411_202511.nc'
manifest_path = SAIDA/'manifesto_2025.json'
cached_manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if path.exists() and manifest_path.exists() else None
if cached_manifest:
    assert cached_manifest['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest(), 'SST em cache alterada.'
fallback=[]
if not path.exists():
    print('SST: baixando 13 meses, incluindo uma sobreposição para conferência.',flush=True)
    r=requests.get(URL,params=PEDIDO,timeout=(20,180))
    temp=path.with_suffix('.partial.nc')
    if r.status_code == 200:
        temp.write_bytes(r.content)
    else:
        print('PSL indisponível; usando o produtor NCEI, mesmo ERSSTv5, com conferência de sobreposição.',flush=True)
        pieces=[]
        for t in pd.date_range('2024-11-01','2025-11-01',freq='MS'):
            url=f'https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/netcdf/ersst.v5.{t:%Y%m}.nc'
            raw=SAIDA/f'ncei_{t:%Y%m}.nc'
            if not raw.exists():
                rr=requests.get(url,timeout=(20,60)); rr.raise_for_status()
                raw.write_bytes(rr.content)
            with xr.open_dataset(raw) as source:
                assert source.attrs.get('product_version')=='Version v5'
                assert source.sst.attrs['units']=='degree_C'
                assert pd.DatetimeIndex(source.time.values).to_period('M')[0]==t.to_period('M')
                a=source.sst.isel(lev=0,drop=True).sel(lat=np.arange(60,-61,-4),lon=np.arange(0,360,4)).load()
                a=a.assign_coords(time=[t]); a.attrs['units']='degC'
                pieces.append(a)
            fallback.append(dict(url=url,arquivo=raw.name,sha256=hashlib.sha256(raw.read_bytes()).hexdigest()))
        xr.concat(pieces,dim='time').to_dataset(name='sst').assign_attrs(
            title='NOAA ERSSTv5 — NCEI subset',product_version='Version 5',
            processamento='Seleção exata 4 graus; nível superficial; rótulo mensal no dia 1; degree_C renomeado degC').to_netcdf(temp)
    with xr.open_dataset(temp) as ds:
        if 'sst' not in ds: raise ValueError('Download sem SST.')
    temp.replace(path)
with xr.open_dataset(path) as ds:
    assert ds.sst.dims==('time','lat','lon') and ds.sst.attrs.get('units')=='degC'
    assert 'ERSSTv5' in str(ds.attrs.get('title','')) and ds.attrs.get('product_version')=='Version 5'
    assert pd.DatetimeIndex(ds.time.values).equals(pd.date_range('2024-11-01','2025-11-01',freq='MS'))
    assert np.array_equal(ds.lat,np.arange(60,-61,-4))
    assert np.array_equal(ds.lon,np.arange(0,360,4))
    sst=ds.sst.load()
    finite=sst.values[np.isfinite(sst.values)]
    assert finite.size and not np.isinf(sst.values).any() and finite.min()>=-1.801 and finite.max()<=45
    attrs={k:str(v) for k,v in ds.attrs.items()}
sources=[]
if Path('/kaggle/input').is_dir():
    sources=list(Path('/kaggle/input').rglob('ersstv5_4graus_198201_202411.nc'))
else:
    ref=Path.cwd()/'entregas/etapa8F_M10_SST_PCA/ersstv5_4graus_198201_202411.nc'
    if ref.is_file(): sources=[ref]
overlap=None
if sources:
    with xr.open_dataset(sources[0]) as ref:
        a,b=xr.align(ref.sst.sel(time='2024-11-01'),sst.sel(time='2024-11-01'),join='exact')
        ok=np.isfinite(a.values)&np.isfinite(b.values)
        overlap=dict(mes='2024-11',mesma_mascara=bool(np.array_equal(np.isfinite(a),np.isfinite(b))),
            max_abs_degC=float(np.max(np.abs(a.values[ok]-b.values[ok]))),
            treinamento_original_preservado=True)
manifest=dict(url=URL,pedido=PEDIDO,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    recuperado_em_utc=datetime.now(timezone.utc).isoformat(),atributos=attrs,
    comparacao_sobreposicao=overlap,fontes_alternativas_NCEI=fallback,publicacao_historica_comprovada=False,
    nota='Arquivo consolidado retrospectivo. Timestamp de download não é data de primeira publicação.')
if cached_manifest:
    manifest['recuperado_em_utc'] = cached_manifest['recuperado_em_utc']
    manifest['fontes_alternativas_NCEI'] = cached_manifest.get('fontes_alternativas_NCEI', [])
manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
if overlap:
    assert overlap['mesma_mascara'] and overlap['max_abs_degC']<=1e-6, 'Produto diferente na sobreposição: investigar antes de inferir.'
print('SST pronta: 12 entradas de dezembro/2024 a novembro/2025; treino original preservado.',flush=True)
print('Sobreposição:',overlap)
