# A origem: Hackathon WorCAP 2026

A equipe Nimbus PA ficou em **9º lugar no resultado oficial**. Esta seção apresenta
a solução híbrida entregue à banca e conserva a reprodução das duas submissões
finais. O desenvolvimento posterior está na [seção de pesquisa](../pesquisa/README.md).

## Código para conhecer e executar

- [Notebook explicado, preparado para leitura no GitHub](Nimbus_PA_WorCAP_2026.ipynb).
- [Script original da entrega](../entregas/Nimbus_PA_homologacao/WorCAP_modelo_hibrido_reproducao.py).
- [Notebook original arquivado](../entregas/Nimbus_PA_homologacao/WorCAP_modelo_hibrido_apresentacao.ipynb).
- [Instruções originais de execução](../entregas/Nimbus_PA_homologacao/LEIA_ME.md).

O notebook de apresentação tem as mesmas células de código e explicação do original.
A preparação remove apenas outputs, contadores de execução e metadados de sessão.
Seu hash de arquivo é diferente; a igualdade das células é verificada por
`scripts/conferir_versoes.py`. Os resultados ficam nos relatórios arquivados,
sem aparentar uma execução recente.

## Duas saídas reproduzidas

| Solução | Score público informado | Saída |
|---|---:|---|
| Média dos dois LightGBM | 1,65183 | `componentes/media_arvores.csv` |
| Dois LightGBM + ridge local, mistura 75%/25% | 1,65354 | `submission_hibrida.csv` |

O híbrido foi escolhido como ponto de partida da pesquisa porque a equipe relatou
melhor desempenho em 2024. A média das árvores teve o melhor score público entre
essas duas soluções. Preservamos ambas e explicitamos o critério, em vez de usar
um único rótulo de “melhor” para avaliações diferentes.

Os hashes esperados estão em
[identificacao_submissoes.json](../entregas/Nimbus_PA_homologacao/identificacao_submissoes.json).
O pacote registra seus testes em
[verificacao_local.json](../entregas/Nimbus_PA_homologacao/verificacao_local.json):
a combinação e a ridge foram conferidas com previsões/dados reais; o treinamento
completo das árvores reais não foi reexecutado localmente nessa preparação.
Conferir arquivos agora não equivale a treinar novamente o modelo.

## Método preservado

Treino final: janeiro/1993 a dezembro/2022. Dois LightGBM de anomalias de chuva,
previsões sazonais SEAS5/CFSv2, índices oceânicos, variáveis atmosféricas e ridge
por ponto de grade. Mistura fixa de 0,375/0,375/0,25. O notebook explica as etapas
e confere entradas e CSVs esperados, sem depender de submissões antigas.

O código histórico conserva suas escolhas originais. A
[nota temporal enviada com a solução](../entregas/Nimbus_PA_homologacao/NOTA_TEMPORAL.md)
expõe a diferença entre mês observado e disponibilidade de publicação. Corrigir
isso na pesquisa produz outro modelo e não modifica o arquivo histórico.

## Preservação e futura versão

O [manifesto de preservação](manifesto_preservacao.json) usa o inventário do pacote
de homologação local. Ele não comprova o recebimento do e-mail pela organização.
Os originais ficam em `entregas/Nimbus_PA_homologacao/` sem alterações.

O identificador histórico e a tag Git são **`competicao-2026`**. Essa tag identifica
a importação da entrega preservada. O commit usa a data real de importação, sem
simular um histórico de commits que ainda não existia durante a competição.
