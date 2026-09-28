"""Audita a lista explícita do primeiro envio, sem imprimir conteúdo sensível."""
from pathlib import Path
from urllib.parse import unquote
import argparse
import base64
import gzip
import hashlib
import io
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
LISTA = ROOT / 'publicacao/arquivos_primeiro_envio.json'
MAX_BYTES = 2_000_000
MAX_DECODE = 20_000_000
PADROES = {
    'url_sessao': r'https?://[^\s"<>]*?(?:kkb-production|jupyter-proxy)[^\s"<>]*',
    'jwt': r'eyJhbGci[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+',
    'caminho_pessoal': r'[A-Z]:[/\\]+Users[/\\]+',
    'chave_privada': r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
    'token_github': r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})',
    'chave_aws': r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
}


def exigir(ok, msg):
    if not ok:
        raise ValueError(msg)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def fontes_textuais(texto):
    """Inspeciona também gzip/base64, incluindo camadas embutidas em JSON."""
    fila = [('texto', texto, 0)]
    vistos = set()
    while fila:
        origem, txt, depth = fila.pop()
        h = sha(txt.encode())
        if h in vistos:
            continue
        vistos.add(h)
        yield origem, txt
        exigir(depth <= 5, 'Camadas compactadas excedem o limite de inspeção.')
        for i, value in enumerate(set(re.findall(r'H4sI[A-Za-z0-9+/=]{40,}', txt))):
            try:
                raw = base64.b64decode(value, validate=True)
                with gzip.GzipFile(fileobj=io.BytesIO(raw)) as f:
                    data = f.read(MAX_DECODE + 1)
                decoded = data.decode('utf-8')
            except (ValueError, UnicodeError, OSError, EOFError):
                continue
            exigir(len(data) <= MAX_DECODE, 'Conteúdo descompactado excede o limite.')
            fila.append((origem + f'/gzip_{i}', decoded, depth + 1))


def revisar(conferir_indice=False):
    nomes = json.loads(LISTA.read_text(encoding='utf-8'))['arquivos']
    exigir(nomes == sorted(set(nomes)), 'A lista deve ser única e ordenada.')
    selected = set(nomes)
    bloqueios, arquivos = [], []
    camadas, links = 0, 0
    for nome in nomes:
        p = (ROOT/nome).resolve()
        exigir(p.is_relative_to(ROOT) and p.is_file(), f'Arquivo inválido: {nome}')
        exigir(not any(x in p.relative_to(ROOT).parts for x in ['.git','outputs','tmp','__pycache__']),
               f'Pasta excluída na seleção: {nome}')
        exigir(p.suffix not in {'.nc','.grib','.grb','.npy','.npz','.zip','.pkl','.joblib'},
               f'Binário científico na seleção: {nome}')
        exigir(not p.name.startswith('submission') and p.name not in {'.env','.cdsapirc','kaggle.json'},
               f'Arquivo excluído na seleção: {nome}')
        raw = p.read_bytes()
        exigir(len(raw) <= MAX_BYTES, f'Arquivo acima de 2 MB: {nome}')
        arquivos.append(dict(arquivo=nome,bytes=len(raw),sha256=sha(raw)))
        if p.suffix == '.png':
            continue  # Figuras científicas revisadas separadamente; nenhuma captura de sessão.
        text = raw.decode('utf-8-sig')
        for origem, txt in fontes_textuais(text):
            camadas += origem != 'texto'
            for tipo, regex in PADROES.items():
                for m in re.finditer(regex, txt):
                    bloqueios.append(dict(arquivo=nome,camada=origem,tipo=tipo,
                                          linha=txt.count('\n',0,m.start())+1))
        if p.suffix == '.md':
            for href in re.findall(r'\[[^\]]+\]\(([^)]+)\)', text):
                if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', href) or href.startswith('#'):
                    continue
                target = (p.parent/unquote(href.split('#')[0])).resolve()
                exigir(target.is_relative_to(ROOT), f'Link fora do projeto em {nome}')
                exigir(target.relative_to(ROOT).as_posix() in selected,
                       f'Link fora da seleção: {nome} -> {href}')
                links += 1
        if p.suffix == '.ipynb':
            nb = json.loads(text)
            for i, cell in enumerate(nb['cells']):
                if cell['cell_type'] == 'code':
                    compile(''.join(cell['source']),f'{nome}/cell_{i}','exec')
            # Originais preservam outputs científicos; cópias e pesquisa não têm outputs.
            if not nome.startswith('entregas/'):
                exigir(all(not c.get('outputs') for c in nb['cells']),f'Outputs inesperados: {nome}')

    if conferir_indice:
        raw = subprocess.check_output(['git','ls-files','-z'],cwd=ROOT)
        indexed = set(raw.decode('utf-8').strip('\0').split('\0')) if raw else set()
        exigir(indexed == selected, 'O índice Git difere da lista revisada.')
        for a in arquivos:
            data = subprocess.check_output(['git','show',':'+a['arquivo']],cwd=ROOT)
            exigir(sha(data)==a['sha256'],f'Bytes do índice diferentes: {a["arquivo"]}')

    resumo = dict(schema='nimbus_revisao_primeiro_envio_v1',arquivos=len(arquivos),
                  bytes=sum(x['bytes'] for x in arquivos),camadas_gzip_inspecionadas=int(camadas),
                  links_locais=links,bloqueios=bloqueios,indice_conferido=conferir_indice,
                  limite='Verificação por padrões e revisão dirigida; não é prova de ausência de qualquer segredo.',
                  inventario=arquivos)
    destino=ROOT/'outputs/publicacao/revisao_primeiro_envio.json'
    destino.parent.mkdir(parents=True,exist_ok=True)
    destino.write_text(json.dumps(resumo,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in resumo.items() if k!='inventario'},ensure_ascii=False,indent=2))
    exigir(not bloqueios, 'Revisão encontrou bloqueios; consulte o relatório, sem publicar.')
    return resumo


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--conferir-indice',action='store_true')
    revisar(p.parse_args().conferir_indice)
