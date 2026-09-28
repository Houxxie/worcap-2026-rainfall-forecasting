# Nimbus PA — solução da competição WorCAP 2026

Este marco preserva o código da equipe Nimbus PA entregue para homologação.
A equipe ficou em 9º lugar no resultado oficial, conforme comunicado da organização.
Os arquivos foram importados ao Git em 28/09/2026; esta não é uma reconstrução
retroativa do histórico de commits da competição.

## Reproduzir

- [Notebook explicado](competicao/Nimbus_PA_WorCAP_2026.ipynb), com as mesmas células do original e sem outputs de sessão.
- [Instruções e ambiente](entregas/Nimbus_PA_homologacao/LEIA_ME.md).
- [Script original](entregas/Nimbus_PA_homologacao/WorCAP_modelo_hibrido_reproducao.py).
- [Fontes externas](entregas/Nimbus_PA_homologacao/DADOS_EXTERNOS.md).
- [Limitações de disponibilidade temporal](entregas/Nimbus_PA_homologacao/NOTA_TEMPORAL.md).
- [Hashes dos dois CSVs](entregas/Nimbus_PA_homologacao/identificacao_submissoes.json).

O código reproduz a média de dois LightGBM (score público informado de 1,65183)
e a mistura dessa média com ridge local (1,65354). O híbrido é o ponto de partida
da pesquisa posterior. Esses scores públicos não são o score oficial final de 1,80114.

Os dados oficiais e os seis NetCDFs sazonais não estão incluídos neste Git.
Os manifestos preservam os hashes dos snapshots necessários. A nota local
LEIA_ANTES_DE_ENVIAR.txt integra o arquivo de trabalho e não é apresentada como
um anexo efetivamente enviado à banca.

Os testes originais estão documentados; a importação ao Git não reexecuta treino.
Código original sob [MIT](LICENSE); condições e atribuições dos dados permanecem
com seus fornecedores. Desenvolvimento da equipe Nimbus PA com assistência de IA.
