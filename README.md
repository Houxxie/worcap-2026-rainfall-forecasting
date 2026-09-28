# Nimbus PA — previsão mensal de precipitação

O Nimbus PA nasceu no **Hackathon WorCAP 2026**, no qual a equipe ficou em **9º
lugar no resultado oficial**. O projeto investiga previsão mensal de precipitação
sobre a América do Sul, combinando aprendizado de máquina, previsões sazonais e
informações oceânicas e atmosféricas.

O código da competição permanece preservado. A pesquisa posterior passa a ter
outro objetivo: avaliar previsões para meses futuros usando informações que
tenham realmente chegado antes da emissão. **O projeto está em pesquisa; ainda
não é um serviço operacional de previsão.**

## Duas portas de entrada

| Quero conhecer… | Comece aqui |
|---|---|
| A solução enviada à banca | [Versão da competição](competicao/README.md) · [Notebook explicado](competicao/Nimbus_PA_WorCAP_2026.ipynb) |
| O modelo em pesquisa e suas avaliações | [Pesquisa atual](pesquisa/README.md) · [Notebook de comparação](pesquisa/disponibilidade_operacional/Nimbus_PA_defasagens_e_SST.ipynb) |
| A coleta para previsões futuras | [Registro de fontes e congelamento](pesquisa/acompanhamento/README.md) |

Essas partes coexistem. Uma versão nova não apaga a entrega da competição, e um
experimento novo não substitui automaticamente a referência. O [histórico de
versões](CHANGELOG.md) e o [catálogo de modelos](VERSOES.json) registram essa distinção.

## Como o modelo funciona

Dois modelos LightGBM estimam anomalias de chuva: um utiliza SEAS5 e o outro
acrescenta CFSv2. Uma regressão ridge por ponto combina as previsões sazonais.
A previsão híbrida usa pesos fixos de **37,5% / 37,5% / 25%**, respectivamente.
Climatologias e transformações são estimadas no treinamento, com janela máxima
de trinta anos. O alvo é precipitação ERA5, uma reanálise, em mm/dia.

Na competição, as entradas mensais eram alinhadas a T−1. A pesquisa atual testa
atmosfera T−4, índices oceânicos T−3 e inicializações sazonais T−1. Essas defasagens
não comprovam datas históricas de publicação. Por isso, o acompanhamento novo
registra separadamente mês de referência, versão e horário de recebimento do arquivo.

## Resultados e decisão atual

| Contexto | Resultado | Interpretação |
|---|---|---|
| Competição: média dos dois LightGBM | Score público informado: **1,65183** | Melhor resultado público entre as duas soluções finais |
| Competição: híbrido com ridge local | Score público informado: **1,65354** | Ponto de partida da pesquisa; melhor em 2024 segundo o relato da equipe |
| Pesquisa: híbrido com fontes defasadas | RMSE **1,753399** | Referência atual na comparação histórica de 2007–2020 |
| Pesquisa: mesma referência + oito PCs de SST | RMSE **1,752762** | Redução de 0,0363%, com piora de MAE e viés; candidato em investigação |

O resultado oficial foi **9º lugar, score 1,80114**, conforme comunicado da
organização apresentado pela equipe. Ele não deve ser confundido com os scores
públicos. Tampouco se deve comparar diretamente um score da competição com o
RMSE de outro período e calendário de informações.

2007–2020 já foi consultado no desenvolvimento. A [avaliação de 2025](pesquisa/avaliacao_2025/LEIA_ME.md)
também permanece documentada como retrospectiva, com dados consolidados. Nenhum
desses resultados demonstra, sozinho, melhoria nos anos futuros.

**Referência mantida:** híbrido sem PCs adicionais de SST, com fontes defasadas.
A avaliação histórica foi concluída; o pacote final para inferência prospectiva
ainda precisa ser ajustado e integrado. Em 28/09/2026 não havia previsão real
congelada, e o CFSv2 recebido estava incompleto. O [estado inicial do
registro](pesquisa/acompanhamento/estado_inicial.json) conserva essa situação.

## Reproduzir e consultar os dados

Os notebooks têm instruções próprias para Kaggle e ambientes de referência.
A reprodução da competição exige os arquivos oficiais e os snapshots externos
com os hashes documentados. Apenas clonar o código não baixa esses arquivos nem
reproduz o treinamento. Consulte o [guia de reprodução e dados](docs/REPRODUCAO.md).

Fontes: dados oficiais WorCAP, ERA5, SEAS5/C3S, índices NOAA PSL, CFSv2/IRI e,
nos experimentos de SST, ERSSTv5. A [lista da competição](entregas/Nimbus_PA_homologacao/DADOS_EXTERNOS.md)
e os protocolos identificam produtos, links, unidades e transformações.
Dados brutos, modelos ajustados e grandes saídas ficam fora do Git.

## Evolução e autoria

- **Competição:** código entregue e referências preservados, inclusive limitações temporais.
- **Pesquisa:** experimentos com hipótese, configuração, avaliação e decisão registradas.
- **Previsões futuras:** recibos de fontes, congelamento antes do mês-alvo e verificação posterior.

Veja [como serão organizadas as versões no GitHub](docs/PUBLICACAO_GITHUB.md).
O repositório [Houxxie/nimbus-pa](https://github.com/Houxxie/nimbus-pa) foi criado
como privado para a revisão inicial. A licença do código é [MIT](LICENSE);
as fontes de dados mantêm suas próprias condições, descritas em [NOTICE.md](NOTICE.md).

Desenvolvimento da equipe **Nimbus PA**, com assistência de IA em código,
depuração, organização e documentação. Fontes de dados e bibliotecas mantêm
suas atribuições. Experimentos sem ganho também fazem parte do histórico.
