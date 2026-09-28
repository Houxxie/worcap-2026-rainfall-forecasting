"""Conferência independente das tabelas exportadas da execução real no Kaggle.

Não reabre alvos, treina modelos ou altera o protocolo registrado.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

projeto = Path(__file__).resolve().parents[2]
destino = projeto / 'outputs/nimbus_avaliacao_2025/relatorios_kaggle'
registro = json.loads((destino/'previsoes_congeladas/registro_previsoes.json').read_text())
consulta = json.loads((destino/'reserva_consultada.json').read_text())
resumo = json.loads((destino/'resumo_2025.json').read_text())
digest = hashlib.sha256((Path(__file__).parent/'protocolo_congelado.json').read_bytes()).hexdigest()
assert digest == resumo['protocolo_sha256'] == registro['protocolo_sha256'] == consulta['protocolo_sha256']
assert registro['sha256'] == resumo['previsoes_sha256'] == consulta['previsoes_sha256']
assert pd.Timestamp(registro['criado_em_utc']) < pd.Timestamp(consulta['primeira_leitura_utc'])
g = pd.read_csv(destino/'metricas_2025.csv').set_index(['modelo','regiao']).sort_index()
m = pd.read_csv(destino/'metricas_2025_mensais_regioes.csv')
assert not m.duplicated(['modelo','regiao','mes']).any()
assert set(m.mes) == set(pd.date_range('2025-01-01','2025-12-01',freq='MS').strftime('%Y-%m-%d'))
cols = ['n','sse','soma_erro','soma_abs','sse_area','peso_area']
agr = m.groupby(['modelo','regiao'])[cols].sum().sort_index()
assert agr.index.equals(g.index)
assert np.allclose(agr, g[cols], rtol=1e-12, atol=1e-8)
for nome, calculo in [('rmse',np.sqrt(g.sse/g.n)),('mae',g.soma_abs/g.n),('vies',g.soma_erro/g.n),('rmse_area',np.sqrt(g.sse_area/g.peso_area))]:
    assert np.allclose(g[nome],calculo,rtol=1e-12,atol=1e-12), nome
assert g.xs('dominio_inteiro',level='regiao').n.eq(942732).all()
m['rmse'] = np.sqrt(m.sse/m.n)
mensal = m[m.regiao.eq('dominio_inteiro')].pivot(index='mes',columns='modelo',values='rmse')
assert int((mensal.sst_8pcs < mensal.controle).sum()) == 9
piloto = json.loads((destino/'ERA5/compatibilidade_piloto.json').read_text())
assert piloto['aprovado'] and len(piloto['resultados']) == 10
assert all(r['compativel'] and r['max_abs'] == 0 for r in piloto['resultados'])
fig, ax = plt.subplots(figsize=(9,4.5),layout='constrained')
ax.plot(range(1,13),mensal.controle,'-o',label='Referência híbrida',color='#386cb0')
ax.plot(range(1,13),mensal.sst_8pcs,'-o',label='Híbrido + 8 PCs de SST',color='#df6b26')
ax.set(xticks=range(1,13),xlabel='Mês de 2025',ylabel='RMSE (mm/dia)',title='2025: avaliação retrospectiva com ERA5 como alvo')
ax.legend(); ax.grid(alpha=.2)
fig.savefig(destino/'comparacao_mensal_2025.png',dpi=180);plt.close(fig)
verificacao = dict(status='PASS',origem='oito relatórios lidos da saída visível do Kaggle',
    protocolo_sha256=digest,cronologia_conferida=True,agregacao_mensal_conferida=True,
    metricas_recalculadas_a_partir_das_somas=True,piloto_10_variaveis_exato=True,
    limitacao='Modelos e NetCDFs completos permanecem no Kaggle; hashes dos relatórios são coerentes, mas os bytes das previsões não foram baixados para conferência local.')
(destino/'verificacao_exportacao.json').write_text(json.dumps(verificacao,indent=2,ensure_ascii=False),encoding='utf-8')
estado = dict(data='2026-09-28',notebook='https://www.kaggle.com/code/houxie/nimbus-pa-refer-ncia-h-brida-e-pesquisa-sst/edit',
    versao_salva=6,nome_versao='2025 concluído: referência e SST auditadas',
    status_salvamento_observado='Successful',outputs_incluidos=True,
    protocolo_sha256=digest,preparacao_kaggle='concluida',inferencia_fixa='concluida',avaliacao_2025='concluida',
    pendentes=[],alvos_2025_adquiridos=True,alvos_2025_consultados=True,
    natureza='retrospectiva com produtos consolidados',resultados_2025=resumo,
    previsoes_congeladas_utc=registro['criado_em_utc'],primeira_leitura_alvos_utc=consulta['primeira_leitura_utc'],
    preservacao_kaggle='nimbus_avaliacao_2025_preservada.zip; 98 arquivos; aproximadamente 75 MiB',
    preservacao_local='oito relatórios exportados e verificados; download do ZIP pelo navegador expirou',
    auditoria_disponibilidade='concluida; defasagens operacionais ainda precisam ser revistas e validadas',
    proxima_etapa='Novo protocolo de treinamento com entradas disponíveis na data de emissão; sem usar 2025 como reserva inédita novamente.')
(Path(__file__).parent/'estado_execucao.json').write_text(json.dumps(estado,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(verificacao,indent=2,ensure_ascii=False))
