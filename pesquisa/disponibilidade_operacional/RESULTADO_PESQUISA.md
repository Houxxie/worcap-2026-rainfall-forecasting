# Nimbus PA — resultado do experimento de disponibilidade

Execução concluída em 28/09/2026, no Kaggle, com o protocolo registrado antes dos ajustes:
`5a531778786b49ca9cd8ac337cdb3297ac580d67f78d5f278492e41ff6aa243a`.

## Resultado e decisão

| Modelo | RMSE (mm/dia) | MAE (mm/dia) | Viés (mm/dia) |
|---|---:|---:|---:|
| Híbrido com defasagens | 1,753399 | 1,047213 | +0,004916 |
| Mesmo híbrido + oito PCs de SST | 1,752762 | 1,048663 | +0,017217 |

A redução de RMSE é 0,000637 mm/dia (0,0363%). A SST melhora em cinco dos sete
blocos, mas apenas seis dos catorze anos. MAE e magnitude do viés pioram.
**Este experimento não justifica substituir a referência pelo tratamento com SST.**
O tratamento permanece como candidato de pesquisa, com seus resultados preservados.

## O que foi comparado

Ambos os híbridos foram treinados novamente com os mesmos exemplos, arquitetura,
pesos, semente e limite de trinta anos. Para prever T, o calendário usa atmosfera T−4,
índices T−3, SST T−2 e inicialização das previsões sazonais T−1. A PCA usa somente o
treinamento. Foram avaliados todos os 78.561 pontos em 168 meses, entre 2007 e 2020.
Não houve busca de defasagens, pesos ou número de componentes.

Essa é uma comparação retrospectiva em anos já usados no desenvolvimento.
Os dados consolidados e hindcasts não documentam as versões efetivamente disponíveis
em cada data; a publicação histórica do agregado CFSv2/IRI ainda não foi comprovada.
Defasar as fontes reduz uma hipótese otimista de disponibilidade, mas não transforma
este teste em uma simulação operacional certificada. O teste anterior de 2025 não
foi carregado nem reavaliado nesta etapa.

## Conferência e preservação

- Seis testes sintéticos temporais aprovados, além das auditorias nos sete blocos reais.
- No Kaggle, 133 arquivos conferidos por tamanho e SHA-256.
- Cópia local dos relatórios: 49 arquivos conferidos por tamanho e SHA-256.
- Agregações de erros globais, regionais, por ano e por bloco recalculadas independentemente.
- Calendários, cortes de treinamento e 2.016 linhas mensais-regionais conferidos.
- A conferência final não ajusta modelos.

Os modelos e grandes arrays ficam nos outputs do notebook. O ZIP local contém os
relatórios e auditorias, sem esses arquivos volumosos. Consulte `relatorios_kaggle/resultado.md`,
`relatorios_kaggle/conferencia_independente.json` e `estado_execucao.json`.

## Próxima prioridade proposta

Preparar registros verificáveis de publicação/recebimento e versão das fontes, com
bloqueio quando faltar uma entrada. Depois, congelar o procedimento e acompanhar
previsões em um período ainda não consultado. Isso permite avaliar capacidade futura
com mais segurança antes de ampliar arquitetura ou fazer novas buscas de parâmetros.
