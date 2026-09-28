# Nimbus PA — comparação SST executada em 28/09/2026

Execução real no [notebook privado do Kaggle](https://www.kaggle.com/code/houxie/nimbus-pa-refer-ncia-h-brida-e-pesquisa-sst/edit).
Pasta da execução: `/kaggle/working/nimbus_pesquisa_sst/20260928T141021Z`.

## Resultado

| Modelo | RMSE | MAE | Viés | RMSE por área |
|---|---:|---:|---:|---:|
| Climatologia | 1.849642004 | 1.100117106 | -0.004361159 | 1.937260652 |
| Híbrido de referência | 1.748951169 | 1.045834189 | 0.011350132 | 1.827144907 |
| Híbrido + oito PCs SST | 1.747381721 | 1.045664370 | 0.017678434 | 1.825333943 |

Diferença SST menos referência: **-0.001569448 mm/dia**.
Redução relativa do RMSE: **0.0897%**.
O SST melhorou 5 de sete blocos e
9 de 14 anos.
RMSE por área pondera cada ponto por cosseno da latitude; todas as métricas
usam o domínio completo da grade, inclusive áreas oceânicas.

## O que foi comparado

A mesma solução híbrida recebe oito PCs de SST nos dois componentes LightGBM.
O ridge local fica igual, assim como exemplos, períodos, referências climáticas,
hiperparâmetros, semente, escalas dos resíduos e pesos 37,5% + 37,5% + 25%.
As climatologias de SST, máscara oceânica e PCA são estimadas somente no treino
de cada bloco. A SST é de T−1. Não houve busca de pesos ou de componentes após ver os resultados.

## Reprodução da referência

Os sete blocos históricos foram reproduzidos dentro da tolerância original de 1e-7.
Os componentes de árvores e a média deles reproduziram os hashes originais.
O CSV híbrido apresentou 532 diferenças em 1.885.464 valores, cada uma com módulo
0,000001 mm/dia. Por isso a referência foi confirmada numericamente, sem declarar
identidade byte a byte. O CSV arquivado foi usado apenas para auditoria.
O diagnóstico local encontrou pequenas diferenças nos coeficientes e nas médias
dos preditores da regressão; a causa exata na cadeia numérica não foi isolada.

## Limites e próximo passo

2007–2020 são anos já consultados durante o desenvolvimento. Esta comparação é
exploratória, não um teste independente nem prova de melhora nos próximos anos.
A data mensal T−1 de SST consolidada também não comprova publicação até o fim de T−1.
Antes de promover o candidato, precisamos definir um período não usado na escolha
de modelos e conferir a disponibilidade real das fontes em cada emissão.
Manter a referência e o candidato separados permite essa avaliação sem refazer a seleção.

Os modelos e as previsões completas por ponto estão na saída salva do Kaggle.
Os relatórios baixados, protocolos, coeficientes PCA e auditorias estão em `resultados_kaggle/`.
A entrega original da competição permanece preservada.

## Diagnósticos

### Blocos

| Período ou faixa | Controle | SST | Diferença |
|---|---:|---:|---:|
| A | 1.734979566 | 1.734113634 | -0.000865932 |
| B | 1.799163720 | 1.796438181 | -0.002725539 |
| C | 1.735990024 | 1.737343241 | +0.001353217 |
| H1 | 1.727823282 | 1.726237330 | -0.001585952 |
| H2 | 1.795970130 | 1.797738701 | +0.001768571 |
| H3 | 1.747182015 | 1.744372128 | -0.002809887 |
| H4 | 1.699258309 | 1.692957327 | -0.006300982 |
### Anos

| Período ou faixa | Controle | SST | Diferença |
|---|---:|---:|---:|
| 2007 | 1.726586769 | 1.724920986 | -0.001665782 |
| 2008 | 1.729058911 | 1.727552671 | -0.001506240 |
| 2009 | 1.824717708 | 1.820989345 | -0.003728364 |
| 2010 | 1.766754849 | 1.774183383 | +0.007428534 |
| 2011 | 1.743557090 | 1.739496160 | -0.004060930 |
| 2012 | 1.750799435 | 1.749234505 | -0.001564930 |
| 2013 | 1.709453841 | 1.705926442 | -0.003527399 |
| 2014 | 1.689001235 | 1.679888092 | -0.009113143 |
| 2015 | 1.697552971 | 1.687392382 | -0.010160589 |
| 2016 | 1.771615676 | 1.779608705 | +0.007993029 |
| 2017 | 1.717715630 | 1.718312674 | +0.000597044 |
| 2018 | 1.877081031 | 1.871304847 | -0.005776184 |
| 2019 | 1.688388904 | 1.689688162 | +0.001299258 |
| 2020 | 1.782320296 | 1.783725593 | +0.001405297 |
### Meses calendario

| Período ou faixa | Controle | SST | Diferença |
|---|---:|---:|---:|
| 1 | 1.915205933 | 1.917797271 | +0.002591339 |
| 2 | 1.905653950 | 1.911826656 | +0.006172706 |
| 3 | 1.804977520 | 1.794639779 | -0.010337741 |
| 4 | 1.756560730 | 1.761224361 | +0.004663631 |
| 5 | 1.927506703 | 1.932972336 | +0.005465633 |
| 6 | 1.650723279 | 1.641596611 | -0.009126668 |
| 7 | 1.580213701 | 1.580140368 | -0.000073333 |
| 8 | 1.531173978 | 1.518503942 | -0.012670036 |
| 9 | 1.455265148 | 1.455194206 | -0.000070942 |
| 10 | 1.607201559 | 1.603459711 | -0.003741848 |
| 11 | 1.877427483 | 1.879383021 | +0.001955537 |
| 12 | 1.885634146 | 1.878768422 | -0.006865724 |
### Faixas latitude

| Período ou faixa | Controle | SST | Diferença |
|---|---:|---:|---:|
| dominio_inteiro | 1.748951169 | 1.747381721 | -0.001569448 |
| lat_maior_igual_menos15 | 2.247386297 | 2.244899117 | -0.002487179 |
| lat_menor_que_menos35 | 1.067703766 | 1.068752682 | +0.001048916 |
| lat_menos35_a_menos15 | 1.563524023 | 1.561428671 | -0.002095352 |
