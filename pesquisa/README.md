# Pesquisa após a competição

O objetivo passa a ser avaliar previsão mensal em meses futuros, além de reproduzir
a solução do hackathon. O [código histórico](../competicao/README.md) permanece
disponível em sua própria versão.

## Referência atual

**`hibrido-defasado-v1`**: dois LightGBM e uma ridge local, pesos fixos
0,375/0,375/0,25, máximo de trinta anos de treino. Usa atmosfera T−4, índices
oceânicos T−3 e previsões sazonais inicializadas em T−1. Não acrescenta PCs de SST.

Essa referência foi treinada e avaliada em sete blocos históricos. Ainda falta
materializar e integrar seu pacote final para emissão prospectiva. “Referência”
significa o procedimento escolhido para as próximas comparações; não certifica
superioridade universal nem disponibilidade operacional das fontes.

O [notebook de comparação](disponibilidade_operacional/Nimbus_PA_defasagens_e_SST.ipynb)
executa a referência e o candidato com SST nas mesmas condições. O
[protocolo](disponibilidade_operacional/protocolo.json), o
[guia de execução](disponibilidade_operacional/LEIA_ME.md) e o
[resultado](disponibilidade_operacional/RESULTADO_PESQUISA.md) documentam o experimento.

## Experimentos registrados

| Estudo | Evidência | Decisão |
|---|---|---|
| [SST global + PCA](sst_pca/LEIA_ME_SST.md) | Investigação de oito componentes oceânicos | Conservar como linha de pesquisa |
| [Avaliação retrospectiva de 2025](avaliacao_2025/LEIA_ME.md) | RMSE 1,684157 → 1,667615 com SST; entradas consolidadas T−1 | Período já consultado; não comprova operação histórica |
| [Comparação com defasagens](disponibilidade_operacional/RESULTADO_PESQUISA.md) | RMSE 1,753399 → 1,752762; SST piorou MAE e magnitude do viés | Manter controle sem PCs como referência |
| [Registro prospectivo](acompanhamento/README.md) | Coleta iniciada; bloqueio real de entrada CFSv2 incompleta | Nenhuma previsão real emitida até o registro de 28/09/2026 |

Os estudos têm calendários e amostras diferentes. Resultados de uma linha não
devem ser usados como comparação direta contra outra. O acompanhamento futuro
só começa a produzir evidência preditiva depois de congelar previsões antes dos
meses-alvo e obter as observações posteriores.

## Quando mudar a referência

1. Registrar hipótese, entradas, período e critérios antes de consultar resultados novos.
2. Executar controle e candidato nos mesmos meses, pontos e cortes de informação.
3. Examinar RMSE, MAE, viés, cobertura, estabilidade temporal e disponibilidade das fontes.
4. Registrar a decisão, inclusive quando for manter a referência.
5. Só então atualizar o catálogo e criar uma versão; preservar o procedimento anterior.

Um resultado negativo pode entrar no repositório com código e relatório. Versionar
o projeto é registrar o trabalho, e não publicar somente os testes que ganharam.
