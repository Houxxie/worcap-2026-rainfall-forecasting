"""Abre a chuva de 2025 uma única vez, após conferir as previsões congeladas."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import numpy as np
import pandas as pd
import xarray as xr

def carregar_modulo(nome,path):
    spec=importlib.util.spec_from_file_location(nome,path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

def executar():
    codigo=Path(__file__).resolve().parent
    era=carregar_modulo('aquisicao_era5',codigo/'adquirir_era5_2025.py')
    raiz=era.BASE/'nimbus_avaliacao_2025'
    saida=raiz/'previsoes_congeladas'
    registro=json.loads((saida/'registro_previsoes.json').read_text())
    arquivo=saida/'previsoes_2025.nc'
    assert era.sha(arquivo)==registro['sha256'], 'Previsões modificadas após congelamento.'
    assert era.sha(raiz/'protocolo_congelado.json')==registro['protocolo_sha256']
    assert registro['alvos_2025_lidos'] is False
    assert json.loads((raiz/'ERA5/compatibilidade_piloto.json').read_text())['aprovado']
    with xr.open_dataset(arquivo) as ds: preds=ds.load()
    datas=pd.date_range('2025-01-01','2025-12-01',freq='MS')
    assert pd.DatetimeIndex(preds.time.values).equals(datas)
    assert set(preds.data_vars)=={'controle','sst_8pcs','climatologia'}
    assert preds.sizes=={'time':12,'lat':301,'lon':261}
    assert all(np.isfinite(preds[v]).all() for v in preds.data_vars)
    # Este é o primeiro ponto em que respostas da reserva podem ser recuperadas.
    c=era.cliente()
    path=era.obter(c,era.SL,['total_precipitation'],datas,'alvos_chuva_2025')
    # Marca consulta ANTES da primeira leitura; retomada é reprodução, nunca uma nova reserva.
    marker=raiz/'reserva_consultada.json'
    if not marker.exists(): era.jsave(marker,dict(primeira_leitura_utc=datetime.now(timezone.utc).isoformat(),
        protocolo_sha256=registro['protocolo_sha256'],previsoes_sha256=registro['sha256']))
    obs=era.ler_normalizado(path,{'total_precipitation':('tp','tp')}).tp
    obs,preds=xr.align(obs,preds,join='exact')
    rows=[]
    masks={'dominio_inteiro':np.ones(obs.sizes['lat'],bool),
        'lat_menor_que_menos35':obs.lat.values < -35,
        'lat_menos35_a_menos15':(obs.lat.values>=-35)&(obs.lat.values < -15),
        'lat_maior_igual_menos15':obs.lat.values>=-15}
    for modelo in preds.data_vars:
        for i,t in enumerate(datas):
            e=preds[modelo].values[i].astype('float64')-obs.values[i].astype('float64')
            for nome,mask in masks.items():
                err=e[mask]; w=np.broadcast_to(np.cos(np.deg2rad(obs.lat.values[mask]))[:,None],err.shape)
                rows.append(dict(modelo=modelo,mes=str(t.date()),regiao=nome,n=err.size,
                    sse=float((err**2).sum()),soma_erro=float(err.sum()),
                    soma_abs=float(np.abs(err).sum()),sse_area=float((w*err**2).sum()),peso_area=float(w.sum())))
    d=pd.DataFrame(rows); d.to_csv(raiz/'metricas_2025_mensais_regioes.csv',index=False)
    g=d.groupby(['modelo','regiao'])[['n','sse','soma_erro','soma_abs','sse_area','peso_area']].sum().reset_index()
    g['rmse']=np.sqrt(g.sse/g.n);g['mae']=g.soma_abs/g.n;g['vies']=g.soma_erro/g.n
    g['rmse_area']=np.sqrt(g.sse_area/g.peso_area)
    g.to_csv(raiz/'metricas_2025.csv',index=False)
    print(g[g.regiao=='dominio_inteiro'][['modelo','rmse','mae','vies','rmse_area']].to_string(index=False))
    print('2025 agora está consultado. Resultados retrospectivos de um ano; não constituem certificação operacional.')

if __name__=='__main__': executar()
