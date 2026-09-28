# Comparação retrospectiva com SST — 28/09/2026

Objetivo: confirmar a referência híbrida da Nimbus PA e medir a contribuição de SST
aos seus dois componentes de árvores. Não usar o antigo placar para escolher o candidato.

## Referência congelada

- Dois LightGBM: 29 e 31 preditores, 300 árvores, 31 folhas, taxa 0,05 e semente 42.
- Componente ridge local com alpha 0,1.
- Mistura fixa: 37,5% para cada árvore e 25% para ridge.
- Escalas dos resíduos: 0,900 e 0,875.
- Até 30 anos de treino, 5.000 pontos por mês.
- Sete blocos temporais de 24 meses, de 2007 a 2020.
- RMSE histórico esperado: 1,7489511692458355.
- SHA-256 do CSV de referência: `e98e8954debb8f7c764665a31804ec5512a1c2578c1ed4a24e50c51dedfb429b`.

## Uma alteração

Acrescentar oito PCs de SST às duas árvores: 37 e 39 preditores.
O ridge fica idêntico. Não reajustar número de PCs, pesos, sementes ou hiperparâmetros
depois de observar esta comparação.

O snapshot NOAA ERSSTv5 já foi preservado. A análise usa somente janeiro/1982 a
novembro/2020. A entrada para o alvo T é a SST de T−1; a PCA aprende apenas dos
mesmos meses de origem correspondentes aos alvos de treino de cada bloco.

## Comparação pareada

Reutilizar os modelos históricos da execução de referência recém-confirmada.
Conferir hashes das 29 features compartilhadas, dos alvos e das amostras.
Recalcular as previsões do controle e confirmar suas métricas mensais arquivadas.
Salvar os coeficientes da PCA, modelos candidatos, previsões completas e diagnósticos.

Avaliar RMSE, MAE e viés no domínio inteiro, RMSE com ponderação por área,
blocos, anos, meses do calendário e faixas de latitude. Diferenças por pixel são
diagnósticas; pixels e meses não são tratados como réplicas estatísticas independentes.

## Critério de interpretação

Uma redução agregada do erro precisa ser examinada junto com sua distribuição
temporal e espacial. Mesmo se o candidato ganhar, estes anos já foram consultados
no desenvolvimento: não declarar melhora generalizável ou desempenho futuro comprovado.
Definir posteriormente uma avaliação realmente independente com observações compatíveis
e disponibilidade das entradas auditada. As versões consolidadas de SST, reanálises
e índices podem ter latência e revisões que precisam de investigação operacional.

O notebook da competição e as entregas originais não são alterados por estes arquivos.
A primeira versão salva no Kaggle registra a reprodução; a extensão é uma etapa de pesquisa.

## Conferência executada em 28/09

Os sete blocos reproduziram as métricas históricas dentro da tolerância original de
1e-7. Os dois CSVs de árvores e a média reproduziram os hashes exatos. No híbrido,
532 de 1.885.464 valores diferiram em 0,000001 mm/dia; RMSE entre previsões de
1,6797579e-8 mm/dia. A causa numérica exata ainda não foi isolada.
Essa execução é uma reprodução numérica, sem identidade byte a byte. A cópia
de pesquisa agora registra ambos os hashes e compara integralmente o CSV arquivado,
com tolerâncias explícitas. Não altera os valores para forçar coincidência.
Os arquivos da competição continuam intactos. Detalhes em
`auditoria_referencia_kaggle.json` e `exportar_referencia.py`.
