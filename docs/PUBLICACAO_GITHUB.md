# Organização das versões no GitHub

Vamos manter **pastas e versões juntas**. As pastas ajudam quem chega ao projeto
a encontrar a solução histórica e a pesquisa atual. As tags preservam um retrato
do repositório em um marco específico. Uma não substitui a outra.

## Estrutura preparada

```text
README.md                         apresentação e escolha do percurso
CHANGELOG.md                      mudanças e decisões
VERSOES.json                      referência atual, candidatos e marcos
competicao/                       apresentação da solução entregue
entregas/Nimbus_PA_homologacao/    arquivo original preservado
pesquisa/                         referência e estudos posteriores
  disponibilidade_operacional/   comparação com defasagens e SST
  acompanhamento/                registro e avaliação de meses futuros
docs/                             reprodução e organização
scripts/conferir_versoes.py        conferência local, sem treinamento
```

Outras pastas locais guardam o desenvolvimento anterior. Não é necessário mover
ou renomear esses arquivos agora: os caminhos também fazem parte da reprodução.
O README apresenta os percursos principais sem exigir que o leitor conheça os
números internos dos experimentos da competição.

## Histórico inicial e versões

1. O primeiro commit importa a entrega preservada com a data real da importação.
2. A tag `competicao-2026` marca esse estado; ela não deve ser sobrescrita depois.
3. O commit seguinte organiza pesquisa, catálogo e acompanhamento, com a tag
   `pesquisa-v0.1.0`. São versões do código, sem novos treinamentos.
4. Continuar em `main` com código revisado e documentado. Usar branches para
   mudanças isoladas, por exemplo `codex/sst-disponibilidade`.
5. Registrar nas releases o que mudou, os dados, os testes e a decisão sobre a referência.

O repositório escolhido é [Houxxie/nimbus-pa](https://github.com/Houxxie/nimbus-pa),
inicialmente **privado**, com licença **MIT** para o código. A
[preservação da entrega](../competicao/manifesto_preservacao.json) permite verificar
os arquivos históricos independentemente do nome do commit ou da tag.

## Versão do projeto não é colocação do modelo

Uma release pode melhorar documentação, testes ou aquisição sem melhorar RMSE.
O [catálogo](../VERSOES.json) distingue a versão do software do modelo usado como
referência. Atualmente, o controle com defasagens continua como referência; SST
é candidato experimental. Os testes sem ganho também serão documentados.

## Antes do primeiro envio

- Revisar uma lista explícita de arquivos. Não adicionar toda a pasta de trabalho indiscriminadamente.
- Manter fora do Git NetCDFs, GRIBs, arrays, modelos, CSVs completos de submissão e registros brutos.
- Conferir notebooks e documentos quanto a tokens, URLs temporárias autenticadas e caminhos pessoais.
- Preservar o original da entrega; cópias preparadas para leitura devem documentar suas transformações.
- Manter a licença MIT escolhida e as atribuições das fontes.
- Preservar a visibilidade privada escolhida até uma decisão explícita de abrir o projeto.

O `.gitignore` ajuda, mas não substitui essa revisão nem remove arquivos já
adicionados ao índice. Um clone sem os snapshots de dados não equivale a uma
reprodução completa; o [guia de dados](REPRODUCAO.md) explicita essa dependência.

O [registro do primeiro envio](../publicacao/README.md) descreve a seleção e suas
conferências. Publicar uma versão pública será uma decisão posterior da equipe.
