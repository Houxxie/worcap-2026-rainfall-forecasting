# Nimbus PA — avaliação independente retrospectiva de 2025

Esta etapa compara a referência híbrida com a mesma arquitetura acrescida de oito componentes de SST. Mantém treino em janeiro/1993–dezembro/2022, parâmetros, amostras e pesos. O objetivo é medir extrapolação temporal, sem selecionar uma nova configuração usando 2025.

2021–2022 já havia sido consultado em setembro de 2026; os anos 2023–2024 também influenciaram escolhas pelo placar. Por isso, nenhum deles é denominado reserva inédita nesta etapa.

## Situação em 28/09/2026

- Protocolo registrado antes da aquisição dos alvos de 2025; hash em `protocolo.sha256`.
- CFSv2: 12 meses preparados, membros e datas conferidos; duas médias do servidor reproduzidas a partir dos membros.
- SST: 12 meses de entrada preparados, mais novembro/2024 para conferência; valores e máscara da sobreposição coincidem com o snapshot original.
- Índices oceânicos: 12 entradas extraídas do mesmo snapshot público documentado no projeto; o arquivo de treino permanece preservado.
- Os testes locais de datas, hashes, normalização de grade e conversão de unidades passaram. Todos os scripts e células Python compilam.
- A execução completa no Kaggle terminou. O piloto de janeiro/julho de 2022 reproduziu exatamente as dez variáveis dos arquivos oficiais, incluindo a precipitação convertida para mm/dia.
- As previsões foram congeladas às 17:14:30 UTC e a primeira leitura dos alvos ocorreu às 17:17:28 UTC, em 28/09/2026. Nenhum parâmetro foi alterado depois da consulta.
- Em 2025, o RMSE caiu de **1,684156550** para **1,667614989 mm/dia**, redução de **0,982%**, com melhora em **9 dos 12 meses**. O MAE passou de 1,033102321 para 1,016537528, e o viés de +0,122877599 para +0,040781611 mm/dia.
- O RMSE melhorou nas três faixas de latitude avaliadas; ao sul de 35°S, a diferença foi mínima (−0,000196 mm/dia). Houve piora mensal em março, junho e outubro.
- 2025 agora está consultado. Qualquer ajuste orientado por esse resultado pertence ao desenvolvimento; não poderá reapresentar 2025 como teste inédito.

Os testes locais não substituem a execução completa no ambiente Kaggle fixado. O ambiente local tem versões diferentes das de treinamento e foi usado somente para preparação e verificação.

## Executar no Kaggle

O [notebook Nimbus PA — referência híbrida e pesquisa SST](https://www.kaggle.com/code/houxie/nimbus-pa-refer-ncia-h-brida-e-pesquisa-sst/edit) contém a execução completa e os relatórios. A versão 5 preserva as previsões anteriores à leitura dos alvos. O estado do salvamento final está em `estado_execucao.json`. A exportação local dos oito relatórios foi obtida da saída visível da execução no Kaggle, em `outputs/nimbus_avaliacao_2025/relatorios_kaggle`; os NetCDFs e modelos completos permanecem nas saídas do Kaggle e no ZIP de preservação de 75 MiB.

O notebook existente recebeu as três etapas finais: preparação, congelamento das previsões e avaliação. Também há uma versão independente em `Nimbus_PA_avaliacao_2025.ipynb`.

1. Manter os inputs privados da competição, SEAS5, CFSv2, snapshot SST e referência de auditoria já anexados ao notebook.
2. Manter o ambiente original do Kaggle e Internet ativada para obter dados públicos e dependências.
3. Habilitar o segredo `CDS_API_KEY` no notebook. Não colar a chave nas células ou saídas. Os termos dos datasets ERA5 e sazonais precisam estar aceitos na conta CDS; se houver erro de licença, a etapa deve parar para que o titular confira.
4. Executar somente a nova célula de preparação. O cache será validado e reaproveitado. Primeiro, o piloto de janeiro/julho de 2022 compara dez variáveis ERA5 com os arquivos oficiais, incluindo precipitação e conversão de unidades. Se divergir, investigar antes de ler 2025.
5. Executar a inferência. Ela ajusta as duas configurações fixas, salva modelos, PCA, ridge, previsões e hashes. Não abre os alvos de 2025.
6. Executar a avaliação. Ela verifica o congelamento antes de adquirir precipitação ERA5 de 2025, registra que a reserva foi consultada e calcula RMSE, MAE, viés e RMSE ponderado por área. A avaliação mensal e por faixas de latitude acompanha o resultado global.
7. Salvar a versão com as saídas. Não escolher outros pesos ou parâmetros a partir desse resultado e continuar chamando 2025 de teste inédito.

O protocolo e os scripts incorporados não são sobrescritos silenciosamente. Uma alteração posterior exige uma nova versão documentada, principalmente depois da primeira leitura da reserva.

## Interpretação operacional

Leia `auditoria_disponibilidade.md`. A média mensal ERA5 de T−1 chega depois que T−1 termina. Os índices HadISST têm atraso maior; a SST consolidada também exige verificar publicação e revisões. O SEAS5 tem calendário nominal compatível, enquanto o agregado específico do IRI exige comprovação de chegada.

Portanto, 2025 foi um teste **retrospectivo com produtos consolidados**, tendo a reanálise ERA5 como alvo. Uma previsão emitida antes do mês começar exige outro treinamento com entradas efetivamente disponíveis na data de emissão. Esse teste não certifica a versão atual como operacional. A próxima etapa deve corrigir e auditar essas defasagens antes de ampliar a complexidade do modelo.

## Arquivos principais

- `protocolo_congelado.json`: definição anterior à consulta da reserva.
- `auditoria_disponibilidade.md`: fontes e prazos de disponibilidade.
- `prever_modelos_2025.py`: funções congeladas e comparação com SST.
- `adquirir_*_2025.py`: aquisição e validação das entradas.
- `avaliar_reserva.py`: leitura protegida dos alvos e métricas.
- `proveniencia_codigo.json`: origem dos trechos preservados.
- `verificacao_local.json`: resultado dos testes de preparação.

A preparação salva os dados em `/kaggle/working/nimbus_avaliacao_2025`. Localmente, os dados públicos conferidos estão em `outputs/nimbus_avaliacao_2025`, na raiz do projeto. As entregas originais da competição continuam preservadas.
