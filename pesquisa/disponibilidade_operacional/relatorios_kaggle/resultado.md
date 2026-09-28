# Nimbus PA — comparação com fontes defasadas

Avaliação 2007–2020, em 13.198.248 pontos-meses por modelo. Sete blocos históricos;
ambos os híbridos foram treinados novamente com os mesmos exemplos e limites temporais.

| Modelo | RMSE | MAE | Viés | RMSE por área |
|---|---:|---:|---:|---:|
| Híbrido com defasagens | 1.753399 | 1.047213 | +0.004916 | 1.832106 |
| Híbrido com defasagens + 8 PCs SST | 1.752762 | 1.048663 | +0.017217 | 1.831263 |
| Climatologia | 1.850645 | 1.100536 | -0.005280 | 1.938385 |

Unidades: mm/dia. **Delta de RMSE SST menos controle: -0.000637**.
SST melhora em **5/7 blocos** e **6/14 anos**.
As métricas globais usam somas de erros, não médias de RMSEs.

Atmosfera ERA5 T−4, índices T−3, SST T−2, previsões sazonais inicializadas em T−1.
O treino de cada bloco termina em setembro anterior ao primeiro janeiro previsto;
há no máximo 360 meses. Árvores, ridge, pesos e escalas ficam fixos. A PCA é ajustada
somente às SSTs associadas aos meses de treinamento e não recebe chuva como entrada.

**Interpretação:** sensibilidade retrospectiva em anos já usados no desenvolvimento.
Arquivos consolidados e hindcasts não reconstituem as versões publicadas em cada data.
A chegada do agregado CFSv2/IRI continua sem calendário histórico comprovado. Por isso,
esta execução não libera previsões operacionais e não garante melhora futura.
2025 não foi carregado, treinado ou reavaliado nesta etapa.

Cada bloco conserva modelos, PCA, climatologias, regressão, previsões e auditorias.
O protocolo foi registrado antes do primeiro ajuste desta comparação; não houve busca de lags ou pesos.
