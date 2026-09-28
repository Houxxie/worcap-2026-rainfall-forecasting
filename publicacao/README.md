# Primeiro envio privado

Escolhas confirmadas pela equipe em 28/09/2026:

- Conta e repositório: **Houxxie/nimbus-pa**.
- Visibilidade inicial: **privado**.
- Licença do código: **MIT**; fontes e dados continuam com suas condições próprias.

A [lista de arquivos](arquivos_primeiro_envio.json) é explícita: apenas seus caminhos
entram no primeiro envio. A seleção inclui código, notebooks, proveniência, pequenos
snapshots tabulares de índices públicos e relatórios. Exclui dados oficiais e sazonais
binários, modelos ajustados, CSVs completos, credenciais, registros brutos de coleta,
rascunhos locais e capturas do navegador.

Os originais da banca preservam outputs científicos estáticos; a cópia de apresentação
e os notebooks de pesquisa não contêm outputs de execução. As seis imagens selecionadas
são figuras científicas, não capturas de sessões. A revisão verificou os quatro desenhos
distintos; as outras duas imagens são cópias de conteúdo idêntico.

## Conferências

```bash
python scripts/conferir_versoes.py
python scripts/revisar_primeiro_envio.py
```

Na cópia local, após preparar o índice Git para o primeiro envio completo:

```bash
python scripts/revisar_primeiro_envio.py --conferir-indice
```

O último comando compara também os bytes de cada arquivo no índice com a seleção.
As verificações examinam hashes, links, sintaxe dos notebooks e padrões de credenciais,
URLs autenticadas e caminhos pessoais, inclusive em conteúdos gzip/base64 embutidos.
Não são uma prova de ausência de todo tipo possível de informação sensível.
O inventário completo resultante fica em `outputs/publicacao`, fora do Git.

## Marcos do histórico

`competicao-2026` identifica a importação atual da entrega histórica, com seus
arquivos originais e notebook para leitura. `pesquisa-v0.1.0` identifica a primeira
organização da pesquisa e do registro prospectivo. As datas Git são as datas reais
da importação; não simulam commits feitos durante a competição.

Esses marcos versionam o código. Não representam novos treinamentos, reavaliações
de placar ou certificação de um serviço operacional. Os dados grandes necessários
à reprodução continuam descritos no [guia de dados](../docs/REPRODUCAO.md).
