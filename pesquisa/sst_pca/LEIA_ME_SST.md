# Snapshot de SST para pesquisa Nimbus PA

Fonte: NOAA Physical Sciences Laboratory, ERSSTv5.
https://psl.noaa.gov/data/gridded/data.noaa.ersst.v5.html

Arquivo: `ersstv5_4graus_198201_202411.nc`.
Temperatura mensal da superfície do mar em graus Celsius, janeiro de 1982 a novembro de 2024.
Recorte de 60°S a 60°N, todas as longitudes, subamostrado de 2° para 4° (não é média espacial de células).
É o snapshot já utilizado durante a competição, preservado para permitir comparações reproduzíveis.

Pedido original: `https://psl.noaa.gov/thredds/ncss/grid/Datasets/noaa.ersst.v5/sst.mnmean.nc`
com `var=sst`, `north=60`, `south=-60`, `west=0`, `east=358`, `horizStride=2`,
`time_start=1982-01-01T00:00:00Z`, `time_end=2024-11-01T00:00:00Z`,
`timeStride=1` e `accept=netcdf4`.

A experiência de pesquisa usa somente janeiro de 1982 a novembro de 2020.
Em cada bloco, máscara oceânica, climatologias, centragem e PCA são ajustadas
exclusivamente nos meses de treino. Para prever o alvo T, utiliza-se a SST de T−1.
Os oito componentes e os pesos do modelo foram definidos antes de executar a comparação.

Esta é uma avaliação retrospectiva. A data de referência T−1 não comprova
que a versão mensal consolidada estava publicada até o fim de T−1. Uma avaliação
operacional exigirá verificar a latência e as versões disponíveis em cada emissão.
Os blocos de 2007–2020 já foram usados no desenvolvimento e não são um teste independente.
