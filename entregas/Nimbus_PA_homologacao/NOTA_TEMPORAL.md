# Nota sobre a informação temporal — Nimbus PA

Esta nota descreve o que o código verifica e o que os registros disponíveis não permitem certificar. Não é uma declaração de aprovação das regras pela organização.

## Cortes e seleção de entradas

Para cada mês-alvo T de janeiro/2023 a dezembro/2024:

| Componente | Dados utilizados |
|---|---|
| Ajuste dos modelos e referências finais | Alvos de janeiro/1993 a dezembro/2022; parâmetros ficam fixos durante 2023–2024 |
| Atmosfera | Variáveis oficiais com `time_origem=T−1`; não são usadas as variáveis atmosféricas observadas do próprio T |
| Índices NOAA | Valores mensais datados T−1; recentragem com meses do treino |
| SEAS5 | Previsão de T inicializada em T−1, segundo mês de previsão |
| CFSv2 | Previsão de T na emissão nominal T−1; todas as datas de inicialização de seus membros são anteriores ao início de T |
| Combinação | Pesos fixos; nenhum ajuste contra observações de chuva de 2023–2024 |

As datas dos 24 meses estão em `calendario_fontes_2023_2024.csv`. Os seis NetCDFs preparados tiveram seus hashes e calendários conferidos. Alterar valores NOAA datados T ou posteriores não altera a feature de T nem a referência de treino nos 24 casos testados.

Nos sete blocos históricos, o corte de ajuste precede a validação, as referências são recalculadas dentro do treino e a feature de climatologia da chuva exclui o próprio exemplo de treinamento (leave-one-out). Esses cuidados não transformam versões retrospectivas de dados em versões publicadas naquela época.

## Disponibilidade histórica: ressalva material

**O mês de referência de uma observação não é sua data de publicação.** As séries NOAA foram baixadas em setembro/2026, não foram preservadas as versões publicadas em cada data de previsão e não foi aplicada uma defasagem adicional baseada na data de divulgação do fornecedor.

O [Met Office, produtor do HadISST](https://www.metoffice.gov.uk/hadobs/hadisst/), informa que no início de cada mês acrescenta os campos do mês retrasado. Portanto, existe uma incompatibilidade potencial entre usar os índices HadISST datados T−1 e exigir que os valores publicados já estivessem acessíveis até o fim de T−1. A documentação de publicação é evidência de um risco concreto de disponibilidade; esta conferência não estabelece as datas efetivas de publicação de cada índice em 2023–2024. A recentragem mensal no treino não corrige esse atraso nem elimina revisões retrospectivas.

Os hindcasts sazonais também são simulações retrospectivas: sua inicialização antiga não significa publicação naquele ano. O [ECMWF descreve a implantação operacional do SEAS5 e os reforecasts](https://www.ecmwf.int/en/newsletter/154/meteorology/ecmwfs-new-long-range-forecasting-system-seas5). O manifesto CFSv2 registra a transição nominal para operacional em abril/2011. Para o teste, as inicializações verificadas precedem os meses-alvo, mas o pacote não contém um arquivo de todas as versões publicadas historicamente para comprovar ausência de revisões.

Assim, as checagens demonstram alinhamento temporal por mês de observação/inicialização e isolamento do ajuste em relação às observações de chuva de teste. **Não demonstram disponibilidade operacional estrita de todas as fontes até cada prazo.** Deve-se esclarecer com a comissão se séries retrospectivas do mês anterior e hindcasts são admitidos na interpretação de “em tese estaria disponível”. Se a exigência for estritamente a publicação histórica, a disponibilidade dos índices NOAA em T−1 precisa ser avaliada pela organização.

## Seleção e reprodução

Os sete blocos foram consultados ao longo do desenvolvimento. O período 2021–2022 já foi consultado em estudo anterior e não é apresentado como avaliação independente; no ajuste final destas soluções ele pertence ao treino. Pontuações do leaderboard público também foram consultadas. Não se afirma avaliação cega de 2023 nem conhecimento do resultado privado.

O pacote preserva o método dos CSVs já submetidos, inclusive o snapshot NOAA e os arredondamentos. Não substitui silenciosamente os dados por uma fonte ou defasagem diferente, o que produziria outra solução. Nenhum novo modelo é apresentado como se fosse o originalmente enviado.
