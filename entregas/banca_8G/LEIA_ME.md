# Previsão híbrida de precipitação mensal

O notebook explica a solução completa, passo a passo: dados, alinhamento temporal, climatologia, dois componentes LightGBM, regressão ridge por ponto, combinação, validação e exportação.

1. Importe `WorCAP_modelo_hibrido_apresentacao.ipynb` em um notebook novo do Kaggle.
2. Anexe a competição, `worcap-seas5-dados` e `worcap-cfsv2-dados`. Não precisa de CSV antigo, backup de previsões, Internet, token ou GPU.
3. Mantenha `REEXECUTAR_VALIDACAO=False` para ajustar os três componentes finais e reproduzir o CSV. Use `True` se desejar refazer também os sete blocos históricos.
4. Execute Run All e confira a mensagem de reprodução confirmada. O CSV fica em Output, na pasta `worcap_banca_hibrida/<execucao>/submission_hibrida.csv`.

O HTML permite leitura imediata das explicações, gráficos e resultados auditados; não representa uma nova execução do treinamento. No notebook, as tabelas históricas estão identificadas como registros auditados, e as células de treinamento devem ser executadas no Kaggle.

O score público documentado é **1,65354**, informado pelo participante. O hash completo esperado é `e98e8954debb8f7c764665a31804ec5512a1c2578c1ed4a24e50c51dedfb429b`.

As versões exigidas são NumPy 2.0.2, pandas 2.3.3, xarray 2025.12.0 e LightGBM 4.6.0 em Python 3.12. Se alguma conferência falhar, o código interrompe a reprodução sem declarar equivalência.

O pacote ZIP inclui os dados sazonais preparados e os seus manifestos. Caso os dois datasets sazonais não estejam disponíveis no Kaggle, extraia `dados_preparados`, use Upload → New Dataset para os arquivos dessas pastas e anexe o dataset criado. O código procura os manifestos recursivamente. Os arquivos oficiais da competição são anexados pela própria competição e não estão duplicados no pacote.

O ZIP contém código, documentação, registros de verificação e os dados sazonais necessários; não inclui o CSV de submissão de aproximadamente 58 MB. Ele será gerado ao executar o notebook. Nenhum token ou credencial é necessário ou incluído.
