# Reparo da leitura mensal ERA5 — 28/09/2026

O primeiro piloto real de janeiro e julho de 2022 retornou um ZIP com dois NetCDFs:

- `data_stream-moda_stepType-avgad.nc`: precipitação (`tp`), com rótulos mensais no dia 1 às 06 UTC;
- `data_stream-moda_stepType-avgua.nc`: cobertura de nuvens, temperatura de 2 m e pressão de superfície, com rótulos no dia 1 às 00 UTC.

A leitura inicial unia esses horários antes de normalizar o calendário mensal. Isso criava quatro linhas, duas por mês, com campos ausentes. O controle de unicidade interrompeu a execução antes de consultar qualquer alvo de 2025.

A correção normaliza cada arquivo mensal separadamente e depois exige alinhamento exato para uni-los. Verifica uma única entrada por mês em cada arquivo. Não faz média dos dois horários, não preenche dados e não altera os valores, a conversão de unidades ou a tolerância do piloto.

Um teste sintético reproduziu o ZIP com horários 00/06 UTC e confirmou duas linhas mensais completas, mantendo os valores das duas variáveis. Os demais testes de preparação também passaram.

A migração preserva o código anterior em `codigo/historico`, exige seu hash conhecido e registra os hashes antes/depois. Só pode ocorrer antes de congelar previsões ou consultar a reserva. O protocolo estatístico continua com SHA-256 `0c4762c5b8f81da5395394c7dd04829d576143d9fa5de733dccacaf6ea14f0c3`.
