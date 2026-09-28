"""Relata métricas já calculadas; não treina, seleciona ou lê novos alvos."""
from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def executar(raiz):
    raiz = Path(raiz)
    protocolo = json.loads((raiz/'protocolo_congelado.json').read_text())
    registro = json.loads((raiz/'previsoes_congeladas/registro_previsoes.json').read_text())
    consulta = json.loads((raiz/'reserva_consultada.json').read_text())
    digest = hashlib.sha256((raiz/'protocolo_congelado.json').read_bytes()).hexdigest()
    assert registro['protocolo_sha256'] == consulta['protocolo_sha256'] == digest
    assert consulta['previsoes_sha256'] == registro['sha256']
    assert pd.Timestamp(registro['criado_em_utc']) <= pd.Timestamp(consulta['primeira_leitura_utc'])
    g = pd.read_csv(raiz/'metricas_2025.csv')
    d = pd.read_csv(raiz/'metricas_2025_mensais_regioes.csv')
    d['mes'] = pd.to_datetime(d['mes'])
    assert not d.duplicated(['modelo','regiao','mes']).any()
    assert set(d.modelo) == {'controle','sst_8pcs','climatologia'}
    assert set(d.mes) == set(pd.date_range('2025-01-01','2025-12-01',freq='MS'))
    cols = ['n','sse','soma_erro','soma_abs','sse_area','peso_area']
    agregado = d.groupby(['modelo','regiao'])[cols].sum().sort_index()
    original = g.set_index(['modelo','regiao']).sort_index()
    assert agregado.index.equals(original.index)
    assert np.allclose(agregado, original[cols], rtol=1e-12, atol=1e-8)
    assert np.allclose(np.sqrt(original.sse/original.n),original.rmse,rtol=1e-12)
    dominio = g[g.regiao=='dominio_inteiro'].set_index('modelo')
    assert (dominio.n == 942732).all()
    mensal = d[d.regiao=='dominio_inteiro'].copy()
    mensal['rmse'] = np.sqrt(mensal.sse/mensal.n)
    meses = mensal.pivot(index='mes',columns='modelo',values='rmse').sort_index()
    delta = float(dominio.loc['sst_8pcs','rmse']-dominio.loc['controle','rmse'])
    ganho = -100*delta/float(dominio.loc['controle','rmse'])
    melhores = int((meses.sst_8pcs < meses.controle).sum())
    nomes = {'controle':'Referência híbrida','sst_8pcs':'Híbrido + 8 PCs de SST','climatologia':'Climatologia do treino'}
    resumo = dict(protocolo_sha256=digest,previsoes_sha256=registro['sha256'],
        natureza='retrospectiva com produtos consolidados',treino=protocolo['treino_alvos'],
        avaliacao=protocolo['teste_alvos'],n=942732,
        rmse_controle=float(dominio.loc['controle','rmse']),
        rmse_sst=float(dominio.loc['sst_8pcs','rmse']),delta_rmse=delta,
        ganho_percentual=ganho,meses_com_melhora=melhores,meses=12,
        ajuste_apos_consulta=False,agregacao_mensal_conferida=True)
    (raiz/'resumo_2025.json').write_text(json.dumps(resumo,indent=2,ensure_ascii=False),encoding='utf-8')
    linhas = '\n'.join(f"| {nomes[m]} | {dominio.loc[m,'rmse']:.6f} | {dominio.loc[m,'mae']:.6f} | {dominio.loc[m,'vies']:+.6f} | {dominio.loc[m,'rmse_area']:.6f} |" for m in nomes)
    texto = f'''# Nimbus PA — resultado retrospectivo de 2025

Treino fixo: janeiro/1993–dezembro/2022. Avaliação: os 12 meses de 2025,
942.732 pontos-meses na grade completa. Parâmetros, pesos e transformação SST
foram definidos antes de adquirir as respostas de 2025.

| Modelo | RMSE | MAE | Viés | RMSE ponderado por área |
|---|---:|---:|---:|---:|
{linhas}

Unidades: mm/dia. Viés = previsão menos alvo. O RMSE global é calculado
por sqrt(SSE/n), e não pela média dos RMSEs mensais.

**Delta SST − referência: {delta:+.6f} mm/dia ({ganho:+.3f}% de redução do RMSE).
Meses com menor RMSE: {melhores}/12.**

Este resultado mede um ano adicional, previamente não consultado nos registros
do projeto. Não demonstra melhora universal nem operação em tempo real.
2025 agora está consultado e não pode servir para ajustar o modelo e continuar
sendo apresentado como teste inédito.

A precipitação-alvo é a reanálise ERA5. A avaliação mede concordância com esse
produto. As entradas mensais consolidadas T−1 não estavam necessariamente
publicadas ao fim de T−1; consulte `codigo/auditoria_disponibilidade.md`.
Uma versão operacional exige novo treinamento com defasagens e versões dos
dados compatíveis com o instante de emissão.

Protocolo SHA-256: `{digest}`.
As previsões foram gravadas e identificadas por hash antes da primeira leitura
dos alvos. A agregação das tabelas mensais foi conferida novamente no relatório.
'''
    (raiz/'resultado_2025.md').write_text(texto,encoding='utf-8')
    fig,axs = plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for modelo,cor in [('controle','#386cb0'),('sst_8pcs','#df6b26')]:
        axs[0].plot(range(1,13),meses[modelo],'-o',label=nomes[modelo],color=cor,markersize=4)
    axs[0].set(xticks=range(1,13),xlabel='Mês de 2025',ylabel='RMSE (mm/dia)',title='Comparação mensal')
    axs[0].legend(fontsize=9);axs[0].grid(alpha=.2)
    reg = g.pivot(index='regiao',columns='modelo',values='rmse')
    ordem=['lat_menor_que_menos35','lat_menos35_a_menos15','lat_maior_igual_menos15']
    diferencas=(reg.sst_8pcs-reg.controle).loc[ordem]
    axs[1].barh(['Sul de 35°S','35°S a 15°S','Norte de 15°S'],diferencas,
        color=['#27866c' if v<0 else '#c85454' for v in diferencas])
    axs[1].axvline(0,color='#444',linewidth=.8)
    axs[1].set(xlabel='Delta RMSE: SST − referência (mm/dia)',title='Diagnóstico por faixa de latitude')
    axs[1].grid(axis='x',alpha=.2)
    fig.suptitle('Avaliação retrospectiva de 2025 — modelos congelados até 2022',fontsize=13)
    fig.savefig(raiz/'comparacao_2025.png',dpi=170);plt.close(fig)
    inclusoes=['protocolo_congelado.json','reserva_consultada.json','resumo_2025.json',
        'resultado_2025.md','comparacao_2025.png','metricas_2025.csv',
        'metricas_2025_mensais_regioes.csv','previsoes_congeladas/registro_previsoes.json',
        'ERA5/compatibilidade_piloto.json','ERA5/manifesto_2025.json',
        'SEAS5/manifesto_2025.json','CFSv2/manifesto_2025.json','SST/manifesto_2025.json']
    inclusoes += [str(p.relative_to(raiz)) for p in (raiz/'codigo').rglob('*')
        if p.is_file() and p.suffix in {'.py','.md','.json','.csv'}]
    with zipfile.ZipFile(raiz/'nimbus_relatorio_2025.zip','w',zipfile.ZIP_DEFLATED) as z:
        for rel in inclusoes:
            if (raiz/rel).is_file():z.write(raiz/rel,rel)
    print(texto)
    return resumo


if __name__=='__main__':
    base=Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path.cwd()/'outputs'
    executar(base/'nimbus_avaliacao_2025')
