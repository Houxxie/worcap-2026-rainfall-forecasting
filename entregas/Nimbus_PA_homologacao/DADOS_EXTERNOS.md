# Dados externos — Nimbus PA

As duas soluções usam as mesmas três famílias de dados externos. Os dados atmosféricos e a chuva de treinamento pertencem à competição e são anexados separadamente.

## 1. Índices oceânicos — NOAA Physical Sciences Laboratory

Quatro séries mensais de anomalia de temperatura da superfície do mar, em graus Celsius. Para prever o mês T, o código consulta somente a linha datada T−1 e subtrai a referência mensal estimada no treino. Não utiliza média móvel centrada nem consulta valores datados T ou posteriores na feature de T.

| Índice utilizado | Arquivo de origem |
|---|---|
| Niño 3.4, série longa HadISST | [nino34.long.anom.data](https://psl.noaa.gov/data/timeseries/month/data/nino34.long.anom.data) |
| Niño 1+2, série longa HadISST | [nino12.long.anom.data](https://psl.noaa.gov/data/timeseries/month/data/nino12.long.anom.data) |
| Tropical Northern Atlantic, TNA | [tna.data](https://psl.noaa.gov/data/correlation/tna.data) |
| Tropical Southern Atlantic, TSA | [tsa.data](https://psl.noaa.gov/data/correlation/tsa.data) |

[Catálogo NOAA PSL](https://psl.noaa.gov/data/timeseries/month/). O snapshot usado foi capturado em 21/09/2026 e contém dezembro/1976 a novembro/2024. Os hashes originais, rodapés e metadados estão em `dados_preparados/NOAA/fontes_noaa.json`. A tabela exata está em `indices_noaa.csv` e incorporada em ambos os formatos de código.

O fornecedor identifica HadISST nos metadados das séries. Fonte subjacente: [Met Office HadISST](https://www.metoffice.gov.uk/hadobs/hadisst/); Rayner et al. (2003), DOI [10.1029/2002JD002670](https://doi.org/10.1029/2002JD002670). São séries retrospectivas, com ressalva de atraso de publicação detalhada na nota temporal.

## 2. Previsão sazonal de precipitação — ECMWF / Copernicus C3S

- Dataset: [Seasonal forecast monthly statistics on single levels](https://cds.climate.copernicus.eu/datasets/seasonal-monthly-single-levels).
- Identificador: `seasonal-monthly-single-levels`; centro ECMWF; sistema `51`; produto `monthly_mean`; variável `total_precipitation`; `leadtime_month=2`.
- Referência: DOI [10.24381/cds.68dd14c3](https://doi.org/10.24381/cds.68dd14c3).
- Inicialização no mês T−1 para a previsão do mês T. A precipitação média prevista pelo ensemble é convertida de m/s para mm/dia e interpolada espacialmente para a grade oficial. A anomalia é calculada com referência mensal estimada somente no treino.
- Alvos preparados: janeiro/1982 a dezembro/2024; divisão em desenvolvimento, ajuste final e teste.
- Os arquivos usados estão em `dados_preparados/SEAS5`. O manifesto registra as requisições, os hashes, a contagem de membros e as correções de negativos residuais. Os dados são previsões sazonais, não observações de chuva de teste.
- Documentação do sistema: [SEAS5 — ECMWF](https://www.ecmwf.int/en/newsletter/154/meteorology/ecmwfs-new-long-range-forecasting-system-seas5).

## 3. Previsão sazonal de precipitação — NCEP CFSv2 / NMME / IRI

- Fonte utilizada: [IRI Data Library — NCEP CFSv2 PENTAD_SAMPLES_FULL, prec](https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.NCEP-CFSv2/.HINDCAST/.PENTAD_SAMPLES_FULL/.prec/).
- Catálogo complementar: [NOAA CPC — dados NMME](https://www.cpc.ncep.noaa.gov/products/NMME/data.html).
- Seleção: emissão nominal S=T−1, lead IRI `1.5`, média dos membros com cobertura completa; unidade final mm/dia.
- Alvos preparados: fevereiro/1982 a dezembro/2024. Janeiro/1982 não foi preenchido. O código confere também as datas de inicialização dos membros, e não apenas o rótulo mensal S.
- Os arquivos usados estão em `dados_preparados/CFSv2`. O manifesto contém URLs exatas de aquisição, hashes, transição entre hindcast e operacional e a exceção auditada do membro 17 na emissão agosto/2019.
- A média e sua anomalia mensal são interpoladas para a grade oficial. A referência da anomalia usa somente os meses de treino.
- Referência NMME: Kirtman et al. (2014), DOI [10.1175/BAMS-D-12-00050.1](https://doi.org/10.1175/BAMS-D-12-00050.1).

## Dados oficiais, para localização

[Competição no Kaggle](https://www.kaggle.com/competitions/previsao-climatica-de-precipitacao-sobre-a-america-do-sul/data). Os arquivos incluem a chuva de treinamento, o alvo deslocado, nove variáveis atmosféricas históricas, as features de teste e o sample submission. Os hashes estão no notebook e em `referencia_auditada.json`.

As soluções deste pacote não utilizam SST global com PCA, redes CNN/U-Net nem chuva observada de 2023–2024. A lista acima descreve os dados que efetivamente entram nos CSVs reproduzidos.
