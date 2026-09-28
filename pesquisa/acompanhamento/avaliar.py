"""Verificação posterior das previsões congeladas; sem selecionar modelos."""
from pathlib import Path
import json
import tempfile
from dados import validar_previsao, metricas, conferir_observacao_bruta
from registro import Registro, agora, canonico, deslocar, exigir, mes


def avaliar(registro, alvo, observacao, evidencia):
    r = registro if isinstance(registro,Registro) else Registro(registro)
    exigir(agora() >= mes(deslocar(alvo,4)), 'A verificação primária aguarda T+4 e produto final.')
    p = r.protocolo()
    previsoes = [e for e in r.eventos() if e['tipo']=='previsao_congelada' and e['dados']['mes_alvo']==alvo]
    exigir(len(previsoes)==1, 'Sem previsão previamente congelada para este mês.')
    previsao = previsoes[0]
    exigir(evidencia.get('produto')=='ERA5 monthly_averaged_reanalysis' and
           evidencia.get('expver')==1 and evidencia.get('mes_alvo')==alvo,
           'Exigir metadados do produto final e mês correto.')
    exigir(evidencia.get('url','').startswith('https://cds.climate.copernicus.eu/'), 'Fonte da verificação inesperada.')
    raw = Path(evidencia['arquivo_bruto']).read_bytes()
    auditoria=conferir_observacao_bruta(evidencia['arquivo_bruto'],observacao,alvo)
    with r.transacao(), tempfile.TemporaryDirectory() as td:
        fp = Path(td)/'previsao.nc'
        fp.write_bytes(r.ler_objeto(previsao['dados']['previsao']))
        validar_previsao(fp,alvo)
        linhas = metricas(fp,observacao)
        anteriores = [e for e in r.eventos() if e['tipo']=='verificacao' and e['dados']['mes_alvo']==alvo]
        ob = r.guardar(Path(observacao).read_bytes())
        bruto = r.guardar(raw)
        ev = {k:v for k,v in evidencia.items() if k!='arquivo_bruto'}
        ev['conferencia_bytes']=auditoria
        return r._adicionar('verificacao',dict(mes_alvo=alvo,protocolo=p['id'],previsao=previsao['id'],
            revisao=len(anteriores),primaria=not anteriores,metricas=linhas,evidencia=ev,
            observacao=ob,bruto=bruto,objetos=[ob,bruto]))


def painel(registro):
    r = registro if isinstance(registro,Registro) else Registro(registro)
    p = r.protocolo()['dados']['plano']
    ev = r.eventos(conferir_objetos=True)
    atual = p['primeiro_mes']
    rows=[]
    while atual <= p['ultimo_mes']:
        preds=[e for e in ev if e['tipo']=='previsao_congelada' and e['dados']['mes_alvo']==atual]
        obs=[e for e in ev if e['tipo']=='verificacao' and e['dados']['mes_alvo']==atual and e['dados']['primaria']]
        estado=('avaliado' if obs else 'emitido_aguardando_observacao' if preds else
                'nao_emitido_prazo_encerrado' if agora()>=mes(atual) else 'aguardando_emissao')
        rows.append(dict(mes_alvo=atual,estado=estado,previsao=preds[0]['id'] if preds else None,
                         verificacao=obs[0]['id'] if obs else None))
        atual=deslocar(atual,1)
    resumo=dict(meses_planejados=len(rows),meses_emitidos=sum(x['previsao'] is not None for x in rows),
        meses_avaliados=sum(x['verificacao'] is not None for x in rows),meses=rows)
    # Agregar SSE e n; não usar média simples de RMSE mensal.
    sums={}
    for e in ev:
        if e['tipo']!='verificacao' or not e['dados']['primaria']: continue
        for a in e['dados']['metricas']:
            chave=a['modelo']+'/'+a['regiao']
            z=sums.setdefault(chave,{k:0 for k in ['n','sse','soma_erro','soma_erro_absoluto','sse_area','peso_area']})
            for k in z: z[k]+=a[k]
    for z in sums.values():
        z.update(rmse=(z['sse']/z['n'])**.5,mae=z['soma_erro_absoluto']/z['n'],
                 vies=z['soma_erro']/z['n'],rmse_area=(z['sse_area']/z['peso_area'])**.5)
    resumo['metricas_primarias']=sums
    return resumo
