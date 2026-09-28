"""Captura e audita a previsão CFSv2 do próximo mês, preservando os originais."""
from pathlib import Path
import tempfile
import pandas as pd
from registro import Registro, deslocar, exigir
from coletar import baixar
import adaptador_cfsv2 as a


def preparar(registro, alvo, atualizar=False):
    r=registro if isinstance(registro,Registro) else Registro(registro)
    origem=deslocar(alvo,-1)
    ano,mm=map(int,origem.split('-'))
    m=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')[mm-1]
    d=f'(0000%201%20{m}%20{ano})'
    intervalo=f'S/{d}/{d}/RANGE/'
    urls=[a.FONTE+intervalo+'L/1.5/VALUES/Y/-61/16/RANGE/X/269/336/RANGE/data.nc',
        'https://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.REALTIME_ENSEMBLE/.FLXF/sampleS/'+intervalo+'data.nc']
    recibos=[]
    for u in urls:
        antigos=[e for e in r.eventos() if e['tipo']=='fonte_recebida' and
            e['dados']['fonte']=='cfsv2' and e['dados']['transporte'].get('url')==u]
        e=antigos[-1] if antigos and not atualizar else baixar(r,'cfsv2',u,'cfsv2_bruto',alvo)
        exigir(e is not None,'CFSv2 não adquirido; manter previsão bloqueada.')
        recibos.append(e)
    with tempfile.TemporaryDirectory() as td:
        paths=[]
        for i,e in enumerate(recibos):
            p=Path(td)/f'original_{i}.nc';p.write_bytes(r.ler_objeto(e['dados']['objeto']));paths.append(p)
        ts=pd.DatetimeIndex([origem+'-01'])
        tabela=a.tabela_inicializacoes(paths[1],ts)
        try:
            ds,auditoria=a.processar_bloco(paths[0],ts,tabela)
        except ValueError as erro:
            campos=a.carregar_membros(paths[0],ts)
            import numpy as np
            nvalidos=np.isfinite(campos[0]).sum(axis=(-1,-2))
            info=dict(fonte='cfsv2',mes_origem=origem,mes_alvo=alvo,
                motivo=str(erro),pais=[e['id'] for e in recibos],
                pixels_finitos_por_membro={str(i+1):int(n) for i,n in enumerate(nvalidos)},
                membros_esperados=(np.flatnonzero(tabela[ts[0]]['mascara'])+1).tolist(),
                objetos=[],acao='bloquear; nenhuma exceção ou média parcial aplicada')
            r.adicionar('auditoria_falhou',info)
            print('CFSv2 bloqueado:',info['motivo'],flush=True)
            return None
        exigir(pd.DatetimeIndex(ds.time.values).strftime('%Y-%m').tolist()==[alvo],'Alvo CFSv2 incorreto.')
        p=Path(td)/'cfsv2.nc';ds.to_netcdf(p)
        e=r.derivado('cfsv2',p.read_bytes(),dict(validado=True,meses=[origem],meses_alvo=[alvo],
            unidades='mm/day',lead=1.5,auditoria=auditoria),[e['id'] for e in recibos],Path(a.__file__))
        print('CFSv2 auditado:',origem,'→',alvo,';',auditoria[0]['membros'],'membros; id',e['id'][:12],flush=True)
        return e


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--registro',default='outputs/acompanhamento');p.add_argument('--alvo',required=True)
    p.add_argument('--atualizar',action='store_true',help='Reconsultar a fonte e preservar nova versão dos originais.')
    args=p.parse_args();preparar(args.registro,args.alvo,args.atualizar)
