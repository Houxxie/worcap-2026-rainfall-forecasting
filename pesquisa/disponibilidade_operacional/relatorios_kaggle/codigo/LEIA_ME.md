# Nimbus PA — fontes defasadas e SST global

Este notebook compara dois híbridos de precipitação mensal com o mesmo calendário de informações.
Um utiliza atmosfera, índices oceânicos, SEAS5 e CFSv2; o outro acrescenta **oito componentes principais
da temperatura da superfície do mar (SST)** aos dois componentes LightGBM.

O objetivo é verificar se a SST ainda contribui quando os produtos mensais não são tratados como
disponíveis imediatamente ao fechar o mês. É uma etapa de pesquisa, com arquivos consolidados.

## Calendário fixado antes do treinamento

| Informação para prever o mês T | Mês usado | Exemplo: janeiro/2007 |
|---|---|---|
| Atmosfera ERA5 consolidada | T−4 | Setembro/2006 |
| Índices NOAA derivados de HadISST | T−3 | Outubro/2006 |
| SST ERSSTv5 | T−2 | Novembro/2006 |
| Inicialização SEAS5/CFSv2 | T−1 | Dezembro/2006 |
| Chuva mais recente permitida no treinamento | T−4 do primeiro alvo do bloco | Setembro/2006 |

A avaliação cobre **sete blocos de 24 meses entre 2007 e 2020**. Em cada bloco, o modelo fica fixo;
as entradas são atualizadas mês a mês, usando o calendário acima. Isso não significa emitir todas
as 24 previsões em uma única data. O treinamento termina em setembro anterior ao início do bloco,
com no máximo 360 meses e início comum em março/1982. A primeira SST é janeiro/1982.

As climatologias de chuva, atmosfera e índices usam janelas móveis de 360 meses, cada qual nas
datas reais de sua fonte. SEAS5, CFSv2 e PCA usam os meses associados aos exemplos disponíveis.
A feature climatológica da chuva e o resíduo de treino excluem a própria observação (leave-one-out).

## Modelo e comparação

- Dois LightGBMs: 300 árvores, 31 folhas, taxa 0,05, semente 42 e 5.000 pontos por mês.
- Regressão ridge local entre as duas previsões sazonais, penalização 0,1.
- Mistura fixa: 37,5% de cada árvore e 25% da regressão. Escalas dos resíduos: 0,9 e 0,875.
- A regressão e as colunas originais de treinamento são idênticas entre os dois tratamentos.
- PCA: oito componentes, SVD completa, anomalias mensais, peso sqrt(cos(latitude)), sem padronizar
  por desvio de cada pixel. Máscara oceânica, climatologias, centro e componentes são ajustados só no treino.
- Não há busca de hiperparâmetros, pesos, quantidade de PCs ou defasagens nesta execução.

Cada versão é treinada novamente. Avaliamos todos os 78.561 pontos em todos os 168 meses:
13.198.248 erros por modelo. A métrica principal é sqrt(SSE/n), em mm/dia. MAE, viés, RMSE com
peso de área, meses, anos, blocos e faixas de latitude ajudam a localizar ganhos e perdas.
Os modelos ajustam e avaliam precipitação ERA5, uma reanálise, não pluviômetros independentes.

## O que a política resolve e o que ainda falta

As defasagens são **hipóteses conservadoras, não comprovantes de publicação**. ERA5T mensal chega
cerca de cinco dias após o mês; ERA5 final substitui o preliminar aproximadamente dois meses depois.
Usar T−4 deixa margem para a consolidação. HadISST adiciona o mês anterior ao último no início
de cada novo mês; T−3 deixa margem adicional. ERSST T−2 é uma hipótese de disponibilidade a
conferir em cada emissão, sem garantia universal de data neste notebook.

SEAS5 via C3S tem agenda nominal de publicação no dia 6, 12 UTC. **A agenda C3S não comprova
a publicação do agregado CFSv2 no endpoint IRI usado aqui.** Seus membros têm inicializações
anteriores a T, porém falta um arquivo histórico das datas de chegada desse agregado.

Os dados finais podem ter revisões posteriores e os hindcasts foram produzidos retrospectivamente.
Portanto, a comparação mede sensibilidade às defasagens, e não uma operação histórica certificada.
Uma futura emissão deverá comprovar a chegada e a versão de cada arquivo antes do instante
de previsão; a função de verificação bloqueia datas ausentes ou posteriores. Este notebook
**não emite previsão de produção**. A autenticação dos registros exige um coletor auditável.

2007–2020 já foi usado para desenvolver modelos; não é reserva inédita. O teste anterior de 2025
permanece arquivado e **não é carregado nesta etapa**. Não há decisão baseada no placar Kaggle.

## Execução no Kaggle

Anexe os mesmos quatro Inputs: dados oficiais WorCAP, `worcap-seas5-dados`, `worcap-cfsv2-dados`
e o snapshot que contém `ersstv5_4graus_198201_202411.nc`. Não precisa de token nem novos downloads.
Use CPU e o ambiente da referência: NumPy 2.0.2, pandas 2.3.3, xarray 2025.12.0, LightGBM 4.6.0.

Execute todas as células em ordem. A primeira materializa código e protocolo, a segunda executa
testes temporais sintéticos, e a seguinte treina e avalia. Cada bloco grava modelos, PCA,
climatologias, coeficientes ridge, previsões e auditorias. Retomar a mesma pasta exige hashes
idênticos de código, dados, ambiente e arquivos de cada bloco já concluído.

Os resultados ficam em `/kaggle/working/nimbus_defasagens_4_3_2`. Salvar uma versão concluída
com outputs preserva modelos e previsões; o ZIP de relatórios contém tabelas, documentação e auditorias.
Os artefatos da competição e o notebook anterior não são alterados.

## Fontes e rastreabilidade

- [ERA5 — documentação ECMWF](https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5%2Bdata%2Bdocumentation).
- [HadISST — Met Office](https://www.metoffice.gov.uk/hadobs/hadisst/).
- [ERSST — NOAA NCEI](https://www.ncei.noaa.gov/products/extended-reconstructed-sst).
  Mantemos ERSSTv5 por comparabilidade, embora a página atual já apresente v6.
- [C3S — dados e calendário](https://confluence.ecmwf.int/spaces/CKB/pages/104239050/Summary%2Bof%2Bavailable%2Bdata).
- [CFSv2 — IRI Data Library](https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.NCEP-CFSv2/.HINDCAST/.PENTAD_SAMPLES_FULL/.prec/).

Fontes documentais consultadas em 28/09/2026. Os manifests existentes conferem produto, unidade,
grade e membros sazonais. Os cinco meses NOAA julho–novembro/1976 são recuperados dos arquivos
brutos preservados; o período sobreposto é conferido exatamente contra o snapshot original.
Não se atualiza silenciosamente nenhuma série ou versão de dados.
