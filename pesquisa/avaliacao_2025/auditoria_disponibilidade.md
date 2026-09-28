# Disponibilidade dos dados e teste de 2025

Verificação em 28/09/2026. As fontes abaixo são documentação dos produtores. Datas nominais,
datas de aquisição e datas de publicação têm significados distintos.

## Período de avaliação

2007–2020 foi usado repetidamente para escolher modelos. A reserva 2021–2022 foi consultada
em 17/09/2026 e depois entrou no ajuste final. Os placares de 2023–2024 também foram
consultados. Esses períodos continuam úteis para diagnóstico, mas não são uma nova reserva.

**2025 é a nova reserva retrospectiva**, conforme os registros disponíveis do projeto.
O protocolo foi gravado antes de adquirir seus alvos. Ambos os modelos serão ajustados
a janeiro/1993–dezembro/2022, com os parâmetros e pesos já definidos. Isso mede a extrapolação
dos modelos congelados, incluindo o intervalo de dois anos após o treino. Atualizar o treino
até 2024 seria outro experimento. Nenhum valor observado de 2025 foi lido nesta preparação.

## O que estaria disponível antes do início do mês T

| Entrada | Uso atual | Evidência e consequência |
|---|---|---|
| Nove campos atmosféricos ERA5 | Média mensal T−1 | ERA5T mensal chega aproximadamente cinco dias após o mês terminar; ERA5 final substitui ERA5T cerca de dois meses depois. T−1 completo não está disponível ao fim de T−1. |
| Niño 1+2, Niño 3.4, TNA, TSA | Índices mensais T−1, snapshot NOAA derivado de HadISST | O produtor HadISST acrescenta o mês anterior ao último no início de cada novo mês. Ao fim de T−1, T−2 ainda não deve ser presumido disponível; T−3 é uma defasagem inicial conservadora a confirmar na fonte dos índices. |
| Oito PCs ERSSTv5 | SST mensal T−1 | Produto mensal consolidado, baseado no mês completo. Não foi localizada garantia de publicação até o último instante desse próprio mês. T−1 fica reprovado para essa finalidade; a publicação efetiva e as revisões precisam ser registradas. |
| SEAS5 sistema 51 | Inicialização dia 1 de T−1, previsão de T, forecastMonth=2 | O CDS publica a contribuição ECMWF no dia 6 às 12 UTC do mês de inicialização. O calendário nominal cabe antes de T. Para cada emissão operacional, ainda conferir a chegada do arquivo e os membros. |
| CFSv2 IRI PENTAD_SAMPLES_FULL | Emissão nominal T−1; lead=1,5 | Datas reais de todos os membros são verificadas. Isso não comprova a publicação do agregado no IRI. A documentação consultada não estabelece prazo garantido para esse endpoint específico. Não aplicar ao IRI a agenda de publicação do C3S. |
| Climatologias, máscara e PCA | Estimadas nos dados de treino | O cálculo não consulta a reserva. Ainda é necessário distinguir os produtos retrospectivos de suas versões disponíveis na época. |
| Chuva ERA5 do alvo | Apenas avaliação | Pode ser consolidada posteriormente para verificar a previsão; nunca entra na geração da previsão daquele mês. |

Fontes:

- [ECMWF — ERA5 data documentation, versão atual](https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5%2Bdata%2Bdocumentation).
- [Met Office — HadISST](https://www.metoffice.gov.uk/hadobs/hadisst/).
- [NOAA — ERSST](https://www.ncei.noaa.gov/products/extended-reconstructed-sst) e [arquivos mensais ERSSTv5](https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/netcdf/).
- [ECMWF — calendário de publicação C3S](https://confluence.ecmwf.int/spaces/CKB/pages/104239050/Summary%2Bof%2Bavailable%2Bdata).
- [NOAA CPC — acesso NMME](https://www.cpc.ncep.noaa.gov/products/NMME/data.html) e [produto IRI utilizado](https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.NCEP-CFSv2/.HINDCAST/.PENTAD_SAMPLES_FULL/.prec/).

## Revisões são relevantes

HadISST registra correção do campo de janeiro/2025 em 05/06/2025. O arquivo consolidado
recuperado hoje pode, portanto, diferir do que existia em uma emissão histórica. A listagem
NCEI ERSSTv5 consultada mostra arquivos de vários anos com a mesma atualização em setembro/2026:
esse timestamp não é prova da primeira publicação. Não se inventou uma agenda diária para ERSST.

Os dados ERA5 distribuídos pela competição são reanálise, incluindo a precipitação-alvo.
Logo, avaliar contra ERA5 mede concordância com esse produto; não equivale a verificar
chuva diretamente em pluviômetros.

## Interpretação e trabalho operacional

O teste de 2025 compara os dois métodos sob **o mesmo contrato retrospectivo**. Se houver
ganho, ele será evidência em um ano ainda não consultado, sem demonstrar operação real
ou validade universal nos anos futuros. Após a primeira avaliação, 2025 estará consultado.

Uma versão emitida antes do mês começar precisa ser treinada novamente com defasagens
compatíveis: por exemplo, ERA5T e ERSST de T−2 e índices HadISST de T−3, sujeitos à
verificação de chegada. Esses lags são uma política candidata, não uma prova de disponibilidade.
Se usarmos apenas ERA5 final, a defasagem precisa considerar a consolidação de cerca de dois meses.
SEAS5 pode manter T−1; CFSv2 exige confirmação de chegada do agregado escolhido.

O contrato de produção deve exigir `disponivel_em <= emitido_em`, registrar versão, URL,
hash e aquisição, e interromper a emissão quando a condição falhar ou for desconhecida.
Um fallback precisa ter modelo próprio treinado previamente. Não completar uma fonte
faltante com dados futuros nem alterar silenciosamente o significado de T−1.
