"""Verifica arquivos e documentação das versões; não treina nem acessa a rede."""
from pathlib import Path
from urllib.parse import unquote
import argparse
import hashlib
import json
import re


ROOT = Path(__file__).resolve().parents[1]


def exigir(ok, mensagem):
    if not ok:
        raise ValueError(mensagem)


def caminho(relativo):
    p = (ROOT / relativo).resolve()
    exigir(p.is_relative_to(ROOT), f'Caminho fora do projeto: {relativo}')
    return p


def ler(relativo):
    return json.loads(caminho(relativo).read_text(encoding='utf-8-sig'))


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def conferir(exigir_dados=False):
    manifesto = ler('competicao/manifesto_preservacao.json')
    ausentes = []
    originais = 0
    for item in manifesto['arquivos']:
        p = caminho(item['arquivo'])
        if not p.is_file() and item['dado_binario'] and not exigir_dados:
            ausentes.append(item['arquivo'])
            continue
        exigir(p.is_file(), f'Arquivo arquivado ausente: {item["arquivo"]}')
        exigir(p.stat().st_size == item['bytes'], f'Tamanho alterado: {item["arquivo"]}')
        exigir(sha(p) == item['sha256'], f'Hash alterado: {item["arquivo"]}')
        originais += 1

    copia = manifesto['notebook_preparado']
    exigir(sha(caminho(copia['arquivo'])) == copia['sha256'], 'Cópia preparada alterada.')
    exigir(sha(caminho(copia['origem'])) == copia['sha256_origem'], 'Notebook original alterado.')
    original, preparado = ler(copia['origem']), ler(copia['arquivo'])
    exigir(len(original['cells']) == len(preparado['cells']), 'Número de células diferente.')
    codigo = 0
    for i, (a, b) in enumerate(zip(original['cells'], preparado['cells'])):
        exigir(a['cell_type'] == b['cell_type'], f'Tipo alterado na célula {i}.')
        exigir(a['source'] == b['source'], f'Conteúdo alterado na célula {i}.')
        exigir(b.get('metadata') == {}, f'Metadados de sessão na célula {i}.')
        if b['cell_type'] == 'code':
            exigir(b['outputs'] == [] and b['execution_count'] is None,
                   f'Saídas ou contador de execução na célula {i}.')
            compile(''.join(b['source']), f'competicao/celula_{i}', 'exec')
            codigo += 1
    exigir(set(preparado['metadata']) <= {'kernelspec', 'language_info'},
           'Metadados adicionais no notebook preparado.')

    catalogo = ler('VERSOES.json')
    ids = [v['id'] for v in catalogo['versoes']]
    exigir(len(ids) == len(set(ids)), 'IDs de versões repetidos.')
    exigir(catalogo['referencia_pesquisa'] in ids, 'Referência não catalogada.')
    artefatos = set()
    for versao in catalogo['versoes']:
        for k in ('guia', 'notebook', 'identificacao_csv', 'estado_datado'):
            if k in versao:
                exigir(caminho(versao[k]).is_file(), f'Entrada ausente: {versao[k]}')
        if 'deriva_de' in versao:
            exigir(versao['deriva_de'] in ids, 'Origem de candidato não catalogada.')
        for item in versao['artefatos']:
            p = caminho(item['arquivo'])
            exigir(p.is_file() and sha(p) == item['sha256'],
                   f'Artefato da versão alterado: {item["arquivo"]}')
            artefatos.add(item['arquivo'])

    guias = ['README.md', 'CHANGELOG.md', 'competicao/README.md', 'pesquisa/README.md',
             'docs/PUBLICACAO_GITHUB.md', 'docs/REPRODUCAO.md']
    links = 0
    for nome in guias:
        p = caminho(nome)
        texto = p.read_text(encoding='utf-8')
        for href in re.findall(r'\[[^\]]+\]\(([^)]+)\)', texto):
            if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', href) or href.startswith('#'):
                continue
            destino = (p.parent / unquote(href.split('#')[0])).resolve()
            exigir(destino.is_relative_to(ROOT) and destino.is_file(),
                   f'Link local ausente: {nome} -> {href}')
            links += 1

    # Escopo explícito: guias novos e cópia preparada, não uma auditoria de todo o workspace.
    padroes = [r'kkb-production|jupyter-proxy|eyJhbGci', r'C:[/\\]Users[/\\]',
               r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----']
    for nome in guias + [copia['arquivo']]:
        texto = caminho(nome).read_text(encoding='utf-8')
        exigir(not any(re.search(x, texto) for x in padroes),
               f'Revisão de informação de sessão necessária: {nome}')

    resumo = dict(originais_sha256_conferidos=originais,
                  dados_binarios_ausentes=ausentes,
                  celulas_preservadas=len(preparado['cells']),
                  celulas_codigo_com_sintaxe_valida=codigo,
                  artefatos_catalogados_conferidos=len(artefatos),
                  links_locais_conferidos=links,
                  treinamento_reexecutado=False, acesso_a_rede=False)
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    if ausentes:
        print('Código conferido; os dados ausentes ainda são necessários para reprodução completa.')
    return resumo


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exigir-dados', action='store_true')
    args = parser.parse_args()
    conferir(args.exigir_dados)
