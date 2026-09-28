# Nimbus PA — acompanhamento prospectivo

Esta etapa registra dados recebidos e prepara o congelamento das previsões mensais.
O objetivo é medir desempenho em meses futuros com as versões de entrada realmente
recebidas antes da emissão. O acompanhamento inicial cobre outubro/2026–setembro/2027.

**Estado atual: coleta iniciada; nenhuma previsão real emitida.** Os testes do
congelamento usam dados sintéticos em pastas temporárias e não entram no registro real.
Ainda é necessário materializar o pacote final do híbrido com defasagens e conectar
as aquisições ERA5/SEAS5 autenticadas ao registro. O CFSv2 de setembro/2026 recebido
em 28/09 foi bloqueado por ausência dos membros esperados 21–24.

## Calendário e referência

| Informação para T | Mês utilizado |
|---|---|
| Atmosfera ERA5 final | T−4 |
| Niño 1+2, Niño 3.4, TNA e TSA | T−3 |
| SEAS5 e CFSv2 | Inicialização/emissão nominal T−1, previsão de T |
| ERSSTv5 | T−2; arquivado para pesquisa, sem entrar no modelo de referência |

O plano mantém o híbrido de duas árvores e ridge local, com pesos 0,375/0,375/0,25.
O pacote final deve ser reajustado uma única vez na janela janeiro/1993–dezembro/2022,
com as defasagens do protocolo. Essa janela fixa reaproveita o histórico oficial já
preparado e permite avaliar a degradação de um modelo sem atualização durante o ano.
Não significa que esse seja o último mês de chuva disponível em 2026.
Reajuste anual ou treino mais recente será outro experimento, registrado antes de seus resultados.

Não serão usados como pacote final os modelos dos folds históricos ou os modelos
antigos com T−1: eles têm calendários e/ou cortes diferentes. O registro recusa um
pacote incompleto, mas sua existência não substitui a auditoria do treinamento/inferência.

## O que o registro garante

- Preserva cada versão por SHA-256, sem sobrescrever os bytes de outra versão.
- Usa horário UTC da coleta real; não aceita data de recebimento retroativa.
- Armazena URL e cabeçalhos HTTP selecionados, sem cookies ou credenciais.
- Mantém `primeira_publicacao_utc=null` quando a primeira publicação é desconhecida.
- Encadeia eventos por hash e confere os arquivos antes da emissão.
- Exige todas as fontes obrigatórias, o mês correto e a entrada aprovada pela auditoria.
- Uma falha não autoriza preencher lacunas, trocar o mês ou reduzir o ensemble.
- Recusa previsão depois de iniciado o mês, outra previsão no mesmo mês, alteração de
  fontes entre inferência e congelamento, grade diferente, NaN ou precipitação negativa.
- Congela simultaneamente o híbrido e a climatologia do mesmo treinamento.
- A verificação posterior conserva observação, versão e métricas; revisões não apagam
  a primeira verificação. O painel inclui os meses sem emissão, evitando seleção de meses favoráveis.

Receber os bytes antes do mês previsto demonstra que estavam acessíveis nessa coleta;
não exige inventar a data de primeira publicação. Arquivos transformados conservam os
IDs dos originais e o código do adaptador. Isso é mais preciso que chamar `Last-Modified`
de data de publicação ou usar a inicialização do modelo como horário de chegada do arquivo.

## Limites

Este é um registro local, não um serviço com assinatura e carimbo de tempo independente.
Um administrador pode reescrever toda a cadeia ou apagar sua ponta. Para evidência externa,
poderemos ancorar o hash da previsão e da cadeia em uma versão privada hospedada, antes de T,
ou em armazenamento com retenção. A data de um commit Git definida no computador, sozinha,
não é esse comprovante. Nenhuma publicação nem ancoragem externa foi feita nesta etapa.

ERA5 final é reanálise; avaliar contra ela não equivale a medir erro em pluviômetros.
Não há garantia de entrega das fontes. Sem os arquivos corretos, o mês ficará como não emitido.
Há um plano e mecanismos de bloqueio; ainda não há um serviço de previsão em produção.

## Executar localmente

Na raiz do projeto, em Python 3.12 com `requirements.txt` desta pasta:

```bash
python pesquisa/acompanhamento/testar_registro.py
python pesquisa/acompanhamento/coletar.py --alvo 2026-10 --cfsv2
python pesquisa/acompanhamento/preparar_cfsv2.py --alvo 2026-10
```

O registro fica em `outputs/acompanhamento`, fora do Git. Para uma nova consulta ao
CFSv2, use `preparar_cfsv2.py --alvo 2026-10 --atualizar`. Cada consulta preserva o
horário verdadeiro e os bytes encontrados. Não reutiliza uma falha em cache indefinidamente.
Os comandos são pontuais; **não há agendamento automático**.

## Kaggle

Importe `Nimbus_PA_registro_prospectivo.ipynb`. A fase de coleta pública requer Internet
e nenhuma senha. Para manter a cadeia em outra sessão, anexe a saída salva anterior;
o notebook importa o registro anterior completo, mantendo seus horários. Se a pasta
estiver ausente, a nova execução começa outra cadeia e não comprova a coleta anterior.

Salvar os outputs continua sendo necessário. Reiniciar uma sessão sem guardar o registro
pode perder os únicos comprovantes locais dos downloads e das previsões.

## Integração restante para a primeira emissão

1. Adquirir ERA5 final T−4 e SEAS5 T−1 no CDS; registrar imediatamente os originais,
   validar unidades/grade/membros e preservar os recibos pais durante a transformação.
2. Reexecutar a auditoria CFSv2 quando chegar uma versão completa. Não aprovar uma exceção
   a partir de resultados do mês-alvo.
3. Ajustar e auditar o pacote final da referência no calendário definido em `plano.json`.
4. Executar a inferência somente com os IDs de `Registro.prontidao(T)`. Registrar hashes
   do modelo, código, ambiente, entradas e arquivo de saída. `Registro.congelar(...)`
   faz as verificações finais sem ler chuva observada do alvo.
5. Após T+4, obter a primeira versão ERA5 final verificada e executar a avaliação primária.
   O período de espera é uma política do projeto, não uma promessa de data do produtor.

O bloqueio atual está em `estado_inicial.json`. O arquivo é uma fotografia de 28/09/2026,
não uma consulta automática ao estado atual da fonte.

## Fontes

- [Índices mensais NOAA PSL](https://psl.noaa.gov/data/timeseries/month/); URLs exatas em `coletar.py`.
- [ERSSTv5 — NCEI](https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/netcdf/).
- [CFSv2 — IRI](https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.NCEP-CFSv2/.HINDCAST/.PENTAD_SAMPLES_FULL/.prec/).
- [ERA5 — ECMWF](https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5%2Bdata%2Bdocumentation).
- [SEAS5/C3S — CDS](https://cds.climate.copernicus.eu/datasets/seasonal-monthly-single-levels).

O adaptador CFSv2 deriva das funções auditadas do próprio projeto; a origem e os hashes
constam em `proveniencia.json`. Nenhum código do repositório de outro competidor foi usado.
