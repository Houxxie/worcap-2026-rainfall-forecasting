"""Consolida os relatórios reais baixados do Kaggle; não escolhe parâmetros."""
from pathlib import Path
import json
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / 'resultados_kaggle'
r = json.loads((DATA/'resultado.json').read_text(encoding='utf-8'))
assert r['status'] == 'comparacao_completa'
g = pd.read_csv(DATA/'metricas_globais.csv').set_index('modelo')
rows = ['| Modelo | RMSE | MAE | Viés | RMSE por área |', '|---|---:|---:|---:|---:|']
for key, name in [('climatologia','Climatologia'),('controle','Híbrido de referência'),('sst_8pcs','Híbrido + oito PCs SST')]:
    rows.append('| '+name+' | '+' | '.join(f'{g.loc[key,c]:.9f}' for c in ['rmse','mae','vies','rmse_area'])+' |')
detail=[]
for label,key in [('blocos','bloco'),('anos','ano'),('meses_calendario','mes'),('faixas_latitude','regiao')]:
    t=pd.read_csv(DATA/f'metricas_{label}.csv').pivot(index=key,columns='modelo',values='rmse')
    t['delta_sst_menos_controle']=t.sst_8pcs-t.controle
    t.to_csv(DATA/f'comparacao_{label}.csv')
    detail.append(f'### {label.replace("_"," ").capitalize()}\n\n| Período ou faixa | Controle | SST | Diferença |\n|---|---:|---:|---:|')
    detail.extend(f'| {i} | {x.controle:.9f} | {x.sst_8pcs:.9f} | {x.delta_sst_menos_controle:+.9f} |' for i,x in t.iterrows())
gain=100*(r['rmse_controle']-r['rmse_sst'])/r['rmse_controle']
text=f'''# Nimbus PA — comparação SST executada em 28/09/2026

Execução real no [notebook privado do Kaggle](https://www.kaggle.com/code/houxie/nimbus-pa-refer-ncia-h-brida-e-pesquisa-sst/edit).
Pasta da execução: `{r['saida']}`.

## Resultado

{chr(10).join(rows)}

Diferença SST menos referência: **{r['delta_rmse']:+.9f} mm/dia**.
Redução relativa do RMSE: **{gain:.4f}%**.
O SST melhorou {r['consistencia']['blocos']['melhores']} de sete blocos e
{r['consistencia']['anos']['melhores']} de 14 anos.
RMSE por área pondera cada ponto por cosseno da latitude; todas as métricas
usam o domínio completo da grade, inclusive áreas oceânicas.

## O que foi comparado

A mesma solução híbrida recebe oito PCs de SST nos dois componentes LightGBM.
O ridge local fica igual, assim como exemplos, períodos, referências climáticas,
hiperparâmetros, semente, escalas dos resíduos e pesos 37,5% + 37,5% + 25%.
As climatologias de SST, máscara oceânica e PCA são estimadas somente no treino
de cada bloco. A SST é de T−1. Não houve busca de pesos ou de componentes após ver os resultados.

## Reprodução da referência

Os sete blocos históricos foram reproduzidos dentro da tolerância original de 1e-7.
Os componentes de árvores e a média deles reproduziram os hashes originais.
O CSV híbrido apresentou 532 diferenças em 1.885.464 valores, cada uma com módulo
0,000001 mm/dia. Por isso a referência foi confirmada numericamente, sem declarar
identidade byte a byte. O CSV arquivado foi usado apenas para auditoria.
O diagnóstico local encontrou pequenas diferenças nos coeficientes e nas médias
dos preditores da regressão; a causa exata na cadeia numérica não foi isolada.

## Limites e próximo passo

2007–2020 são anos já consultados durante o desenvolvimento. Esta comparação é
exploratória, não um teste independente nem prova de melhora nos próximos anos.
A data mensal T−1 de SST consolidada também não comprova publicação até o fim de T−1.
Antes de promover o candidato, precisamos definir um período não usado na escolha
de modelos e conferir a disponibilidade real das fontes em cada emissão.
Manter a referência e o candidato separados permite essa avaliação sem refazer a seleção.

Os modelos e as previsões completas por ponto estão na saída salva do Kaggle.
Os relatórios baixados, protocolos, coeficientes PCA e auditorias estão em `resultados_kaggle/`.
A entrega original da competição permanece preservada.

## Diagnósticos

{chr(10).join(detail)}
'''
(HERE/'RESULTADOS.md').write_text(text,encoding='utf-8')
print(json.dumps(dict(rmse_controle=r['rmse_controle'],rmse_sst=r['rmse_sst'],ganho_percentual=gain,
    consistencia=r['consistencia'],globais=g.reset_index().to_dict('records')),indent=2,ensure_ascii=False))
