"""Piloto ERA5 e entradas de 2025. Não baixa nem lê a chuva-alvo de 2025."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
import numpy as np
import pandas as pd
import xarray as xr

BASE=Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path.cwd()/'outputs'
SAIDA=BASE/'nimbus_avaliacao_2025'/'ERA5'
SAIDA.mkdir(parents=True,exist_ok=True)
SL='reanalysis-era5-single-levels-monthly-means'
PL='reanalysis-era5-pressure-levels-monthly-means'
MAPA_SL={'total_cloud_cover':('tcc','cloud_cover'),
    '2m_temperature':('t2m','t2'),'surface_pressure':('sp','surface_pressure')}
MAPA_PL={'geopotential':('z','geopotential_850'),'relative_humidity':('r','rel_hum_850'),
    'specific_humidity':('q','shum_850'),'temperature':('t','temperature_850'),
    'u_component_of_wind':('u','u_850'),'v_component_of_wind':('v','v_850')}
LAT=np.arange(-60,15.001,.25); LON=np.arange(-90,-24.999,.25)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def jsave(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf-8')

def cliente():
    token=os.environ.get('CDS_API_KEY')
    if not token and Path('/kaggle').is_dir():
        try:
            from kaggle_secrets import UserSecretsClient
            token=UserSecretsClient().get_secret('CDS_API_KEY')
        except Exception:
            raise RuntimeError('Habilite CDS_API_KEY em Add-ons > Secrets. Não cole o token no código.') from None
    if not token: raise RuntimeError('CDS_API_KEY não está configurado.')
    if importlib.util.find_spec('cdsapi') is None:
        subprocess.check_call([sys.executable,'-m','pip','install','--quiet','cdsapi>=0.7.7,<0.8'])
    import cdsapi
    return cdsapi.Client(url='https://cds.climate.copernicus.eu/api',key=token,
        quiet=True,debug=False,timeout=120,retry_max=3,sleep_max=30)

def obter(c,dataset,variaveis,datas,nome):
    datas=pd.DatetimeIndex(datas)
    assert len(set(datas.year))==1
    pedido=dict(product_type=['monthly_averaged_reanalysis'],variable=list(variaveis),
        year=[str(datas[0].year)],month=[f'{m:02d}' for m in datas.month],time=['00:00'],
        area=[15,-90,-60,-25],grid=[.25,.25],data_format='netcdf',download_format='unarchived')
    if dataset==PL: pedido['pressure_level']=['850']
    path=SAIDA/(nome+'.nc'); side=path.with_suffix('.json')
    if path.exists():
        m=json.loads(side.read_text(encoding='utf-8'))
        assert m['dataset']==dataset and m['pedido']==pedido and m['sha256']==sha(path)
        return path
    print('CDS:',nome,flush=True)
    parcial=path.with_suffix('.partial.nc')
    try: c.retrieve(dataset,pedido,str(parcial))
    except Exception as e:
        raise RuntimeError(f'Pedido CDS {nome} interrompido ({type(e).__name__}). Confira acesso, licença e conexão; cache preservado.') from None
    assert parcial.stat().st_size>0
    parcial.replace(path)
    jsave(side,dict(dataset=dataset,pedido=pedido,sha256=sha(path),
        recuperado_em_utc=datetime.now(timezone.utc).isoformat()))
    return path

def ler_normalizado(path,mapa):
    caminhos=[Path(path)]
    if zipfile.is_zipfile(path):
        destino=Path(path).with_suffix(''); destino.mkdir(exist_ok=True)
        caminhos=[]
        with zipfile.ZipFile(path) as z:
            for membro in z.infolist():
                if not membro.filename.endswith('.nc'): continue
                p=(destino/membro.filename).resolve()
                assert p.is_relative_to(destino.resolve()), 'Caminho inválido no ZIP.'
                p.parent.mkdir(parents=True,exist_ok=True)
                p.write_bytes(z.read(membro)); caminhos.append(p)
        assert caminhos
    partes=[]
    for p in caminhos:
        with xr.open_dataset(p) as fonte: parte=fonte.load()
        # O CDS separa médias mensais: chuva às 06 UTC e campos instantâneos
        # às 00 UTC. Normalizar cada calendário mensal ANTES de unir evita
        # quatro linhas com NaNs para dois meses. Não calcular média de horários.
        eixo='valid_time' if 'valid_time' in parte.dims else 'time'
        datas=pd.DatetimeIndex(parte[eixo].values)
        meses=datas.to_period('M').to_timestamp()
        assert meses.is_unique and (datas.day==1).all(), 'Arquivo não contém uma média por mês.'
        if eixo!='time': parte=parte.rename({eixo:'time'})
        partes.append(parte.assign_coords(time=meses).sortby('time'))
    d=xr.merge(partes,compat='no_conflicts',join='exact')
    rename={a:b for a,b in [('valid_time','time'),('latitude','lat'),('longitude','lon')] if a in d.dims}
    d=d.rename(rename)
    if 'expver' in d.dims:
        assert 1 in d.expver.values
        d=d.sel(expver=1,drop=True)
    if 'pressure_level' in d.dims:
        assert d.sizes['pressure_level']==1 and float(d.pressure_level[0])==850
        d=d.sel(pressure_level=850,drop=True)
    for dim in set(d.dims)-{'time','lat','lon'}:
        assert d.sizes[dim]==1, f'Dimensão não prevista {dim}.'
        d=d.isel({dim:0},drop=True)
    d=d.assign_coords(lon=((d.lon+180)%360)-180).sortby('lat').sortby('lon')
    assert np.array_equal(d.lat,LAT) and np.array_equal(d.lon,LON), 'Grade incompatível; não interpolar silenciosamente.'
    t=pd.DatetimeIndex(d.time.values).to_period('M').to_timestamp()
    assert t.is_unique
    d=d.assign_coords(time=t).sortby('time')
    out={}
    for _,(short,name) in mapa.items():
        assert short in d, f'{short} ausente.'
        a=d[short].transpose('time','lat','lon')
        assert np.isfinite(a.values).all()
        if name=='tp':
            assert a.attrs.get('units') in ('m','m/day','m day**-1')
            a=a*1000.; a.attrs['units']='mm/day'
        out[name]=a.astype('float32')
    return xr.Dataset(out)

def piloto(c):
    datas=pd.to_datetime(['2022-01-01','2022-07-01'])
    a=ler_normalizado(obter(c,SL,[*MAPA_SL,'total_precipitation'],datas,'piloto_superficie_2022'),
        {**MAPA_SL,'total_precipitation':('tp','tp')})
    b=ler_normalizado(obter(c,PL,MAPA_PL,datas,'piloto_pressao_2022'),MAPA_PL)
    novo=xr.merge([a,b],compat='equals')
    if Path('/kaggle/input').is_dir():
        candidates=list(Path('/kaggle/input').rglob('treino_tp.nc'))
    else: candidates=[Path.cwd()/'data/raw/treino_tp.nc']
    assert len(candidates)==1
    pasta=candidates[0].parent
    rows=[]
    for v in novo.data_vars:
        p=pasta/f'treino_{v}.nc'
        assert p.is_file(), f'Arquivo oficial do piloto ausente: {p}'
        with xr.open_dataset(p) as ref:
            assert v in ref
            x,y=xr.align(ref[v].sel(time=datas).transpose('time','lat','lon'),novo[v],join='exact')
            delta=x.values.astype('float64')-y.values.astype('float64')
            rows.append(dict(variavel=v,max_abs=float(np.abs(delta).max()),
                rmse=float(np.sqrt(np.mean(delta**2))),
                compativel=bool(np.allclose(x,y,rtol=1e-5,atol=1e-6))))
    jsave(SAIDA/'compatibilidade_piloto.json',dict(datas=list(map(str,datas)),resultados=rows,
        aprovado=all(r['compativel'] for r in rows),origem='CDS versus arquivos oficiais; anterior a 2025'))
    assert all(r['compativel'] for r in rows), 'Piloto ERA5 diverge: investigar antes de avançar. Alvos 2025 não foram lidos.'
    print('Piloto de compatibilidade aprovado para 10 variáveis.',flush=True)

def executar():
    c=cliente(); piloto(c)
    origens=pd.date_range('2024-12-01','2025-11-01',freq='MS')
    partes=[]
    for ano in sorted(set(origens.year)):
        ts=origens[origens.year==ano]
        a=ler_normalizado(obter(c,SL,MAPA_SL,ts,f'atmos_superficie_{ano}'),MAPA_SL)
        b=ler_normalizado(obter(c,PL,MAPA_PL,ts,f'atmos_pressao_{ano}'),MAPA_PL)
        partes.append(xr.merge([a,b],compat='equals'))
    d=xr.concat(partes,dim='time').sortby('time')
    assert pd.DatetimeIndex(d.time.values).equals(origens)
    d.to_netcdf(SAIDA/'atmosfera_origens_2025.nc')
    jsave(SAIDA/'manifesto_2025.json',dict(sha256=sha(SAIDA/'atmosfera_origens_2025.nc'),
        origens=list(map(str,origens)),chuva_alvo_2025_lida=False,
        natureza='entradas ERA5 finais, retrospectivas; nao disponiveis no fim de T-1'))
    print('ERA5: entradas de 2025 preparadas. Chuva-alvo continua fechada.')

if __name__=='__main__': executar()
