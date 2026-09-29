"""Acquire public sources as separate versions without publishing or issuing a forecast."""
from pathlib import Path
from urllib.parse import urlparse
import argparse
import json
import requests
from registry import Registro, agora, deslocar, exigir
from data_contracts import indices_psl, inspecionar_netcdf
FONTES = {'nino34': 'https://psl.noaa.gov/data/timeseries/month/data/nino34.long.anom.data', 'nino12': 'https://psl.noaa.gov/data/timeseries/month/data/nino12.long.anom.data', 'tna': 'https://psl.noaa.gov/data/correlation/tna.data', 'tsa': 'https://psl.noaa.gov/data/correlation/tsa.data'}
HOSTS = {'psl.noaa.gov', 'www.ncei.noaa.gov', 'iridl.ldeo.columbia.edu'}

def baixar(registro, fonte, url, perfil, esperado=None):
    exigir(urlparse(url).scheme == 'https' and urlparse(url).hostname in HOSTS, "Use the producer's authorized HTTPS endpoint.")
    inicio = agora().isoformat()
    try:
        with requests.Session() as s:
            s.headers['User-Agent'] = 'WorCAP-rainfall-research/1.0 (prospective-data-capture)'
            with s.get(url, stream=True, timeout=(20, 90), allow_redirects=False) as r:
                exigir(r.status_code == 200, f'HTTP {r.status_code}')
                partes, tamanho = ([], 0)
                for chunk in r.iter_content(128 * 1024):
                    tamanho += len(chunk)
                    exigir(tamanho <= 32 * 1024 * 1024, 'Response exceeds the 32 MiB limit.')
                    partes.append(chunk)
                b = b''.join(partes)
                exigir(bool(b), 'Empty response.')
                try:
                    meta = indices_psl(b) if perfil == 'psl' else inspecionar_netcdf(b, perfil, esperado)
                except Exception as erro_validacao:
                    meta = dict(validado=False, meses=[], erro_inspecao=type(erro_validacao).__name__, pendencia='Response preserved in quarantine; do not use for inference.')
                transporte = dict(url=url, metodo='HTTPS GET', iniciado_em_utc=inicio, resposta_em_utc=agora().isoformat(), status_http=r.status_code, cabecalhos={k: r.headers[k] for k in ['Date', 'Last-Modified', 'ETag', 'Content-Type', 'Content-Length'] if k in r.headers}, last_modified_nao_e_primeira_publicacao=True)
        e = registro.recibo(fonte, b, meta, transporte)
        print(f'{fonte}: received; validated={meta['validado']}; {len(b):,} bytes; id={e['id'][:12]}', flush=True)
        return e
    except Exception as erro:
        registro.adicionar('coleta_falhou', dict(fonte=fonte, url=url, iniciado_em_utc=inicio, erro_tipo=type(erro).__name__, objetos=[]))
        print(f'{fonte}: acquisition rejected ({type(erro).__name__}). No substitution with data from a different month.', flush=True)
        return None

def executar(pasta, plano_path, alvo, incluir_cfsv2=False):
    r = Registro(pasta)
    plano = json.loads(Path(plano_path).read_text(encoding='utf-8'))
    r.iniciar(plano)
    r.prontidao(alvo)
    saidas = [baixar(r, n, u, 'psl') for n, u in FONTES.items()]
    t = deslocar(alvo, -2).replace('-', '')
    u = f'https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/netcdf/ersst.v5.{t}.nc'
    saidas.append(baixar(r, 'ersstv5', u, 'ersstv5', deslocar(alvo, -2)))
    if incluir_cfsv2:
        origem = deslocar(alvo, -1)
        ano, m = map(int, origem.split('-'))
        mon = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')[m - 1]
        data = f'(0000%201%20{mon}%20{ano})'
        raiz = 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.NCEP-CFSv2/.HINDCAST/.PENTAD_SAMPLES_FULL/.prec/'
        u = raiz + f'S/{data}/{data}/RANGE/L/1.5/VALUES/Y/-61/16/RANGE/X/269/336/RANGE/data.nc'
        saidas.append(baixar(r, 'cfsv2', u, 'cfsv2_bruto', alvo))
    r.exportar_resumo(Path(pasta) / 'resumo_registro.json')
    estado = r.prontidao(alvo)
    (Path(pasta) / 'prontidao.json').write_text(json.dumps(estado, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(estado, indent=2, ensure_ascii=False), flush=True)
    return estado
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--registro', default='outputs/prospective')
    p.add_argument('--plano', default=str(Path(__file__).with_name('plan.json')))
    p.add_argument('--alvo', required=True)
    p.add_argument('--cfsv2', action='store_true')
    a = p.parse_args()
    executar(a.registro, a.plano, a.alvo, a.cfsv2)
