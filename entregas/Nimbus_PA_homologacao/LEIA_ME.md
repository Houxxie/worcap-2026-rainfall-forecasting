# Hackathon WorCAP 2026 — Nimbus PA

Equipe no Kaggle: **Nimbus PA**.

Este pacote contém o notebook e o script que treinam os componentes a partir dos dados e reproduzem duas soluções submetidas. A execução já calcula ambas; não depende de CSV de submissão anterior.

| Solução | Score público informado pelo participante | Arquivo gerado dentro da pasta da execução |
|---|---:|---|
| Média de dois modelos LightGBM, com previsões sazonais | 1,65183 | `componentes/media_arvores.csv` |
| Combinação de árvores e regressão ridge por ponto | 1,65354 | `submission_hibrida.csv` |

Os nomes de saída são descritivos. Os hashes em `identificacao_submissoes.json` identificam o conteúdo dos CSVs originalmente enviados, independentemente do nome. A seleção final é aquela registrada no Kaggle; este pacote não altera submissões nem sua seleção.

## Reproduzir no Kaggle

1. Importe `WorCAP_modelo_hibrido_apresentacao.ipynb` em um notebook novo.
2. Anexe os dados oficiais da competição em Add Input. Os treze arquivos e seus hashes estão em `referencia_auditada.json`, campo `entradas`.
3. Extraia o ZIP no computador. Em Upload → New Dataset, envie as pastas `dados_preparados/SEAS5` e `dados_preparados/CFSv2` (o manifesto e os três NetCDFs de cada fonte). Anexe o dataset ao notebook. Caso esses mesmos arquivos já estejam em seus Inputs, basta reutilizá-los. Deve haver somente uma cópia de cada manifesto nos Inputs.
4. Use o ambiente de referência descrito em `requirements.txt`: Python 3.12, NumPy 2.0.2, pandas 2.3.3, xarray 2025.12.0 e LightGBM 4.6.0. O notebook confere versões e hashes e não instala pacotes automaticamente.
5. Mantenha `REEXECUTAR_VALIDACAO=False` e execute Run All. Isso ajusta os dois LightGBM finais e a regressão ridge, exporta os CSVs e verifica a identidade com os arquivos submetidos. Internet, token e GPU não são necessários.
6. As saídas ficam em `/kaggle/working/worcap_banca_hibrida/<execucao>/`. Os dois CSVs da tabela acima são verificados por SHA-256. Nenhum arquivo é enviado automaticamente ao Kaggle.

Para repetir também os sete blocos de validação de 2007–2020, altere somente `REEXECUTAR_VALIDACAO=True` antes de Run All. Os pesos e hiperparâmetros permanecem fixos. Os relatórios já visíveis no notebook/HTML são registros auditados de execuções anteriores, não resultados de uma nova execução nesta máquina.

## Conteúdo e auditoria

- `DADOS_EXTERNOS.md`: lista de todas as fontes externas e links de localização.
- `NOTA_TEMPORAL.md`: alinhamento das entradas e ressalvas sobre disponibilidade histórica.
- `calendario_fontes_2023_2024.csv`: datas de referência por mês previsto.
- `conferencia_temporal.json`: resultado da conferência de calendários e hashes.
- `verificacao_local.json`: escopo dos testes de reprodução já realizados.
- `dados_preparados/NOAA`: cópia legível das séries incorporadas no código e sua proveniência; o notebook usa o mesmo conteúdo incorporado.
- `dados_preparados/SEAS5` e `dados_preparados/CFSv2`: arquivos efetivamente utilizados e manifestos da preparação.

O treinamento LightGBM completo com os arquivos atmosféricos reais não foi repetido localmente nesta preparação. A combinação de previsões reais, a regressão ridge final e a identidade dos CSVs foram verificadas; o notebook completo foi executado em uma grade sintética de teste. A reprodução completa com os dados oficiais é feita no Kaggle e exige os hashes esperados.

O pacote não inclui os dados oficiais da competição nem os CSVs de aproximadamente 58 MB. Ele contém o código que os gera e os dados externos preparados. Os GRIBs brutos e downloads anuais citados nos manifestos não são necessários para esse treinamento e não estão duplicados no ZIP.

**Sobre a regra temporal:** datas de observação/inicialização anteriores ao alvo foram conferidas, mas isso não certifica a disponibilidade histórica de cada arquivo. A nota temporal registra, em particular, o atraso de publicação do HadISST e o uso retrospectivo dos índices NOAA. Esta entrega não contém uma declaração de homologação ou de ausência garantida de vazamento.
