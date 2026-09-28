# %% Configuração — executar em um notebook novo, com Internet ligada
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import quote, urlparse
import hashlib, importlib.util, json, os, shutil, subprocess, sys, time

faltantes = [p for m, p in [('netCDF4', 'netCDF4>=1.6,<2'),
    ('cftime', 'cftime>=1.6,<2'), ('requests', 'requests>=2.31,<3')]
    if importlib.util.find_spec(m) is None]
if faltantes:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', *faltantes])
import numpy as np
import pandas as pd
import xarray as xr
import cftime
import requests
from IPython.display import display, FileLink

BASE = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path.cwd()/'outputs'
SAIDA = BASE/'worcap_CFSv2_dados'
# Para retomar, anexe a saída anterior em Input. Normalmente não precisa editar.
PASTA_CACHE_MANUAL = None
INICIO_ALVOS, FIM_ALVOS = '1982-02-01', '2024-12-01'
LEAD = 1.5  # centro do mês seguinte ao mês nominal de emissão; NÃO é o lead 2 do CDS
LAT = np.arange(-61, 17, dtype=np.float32)
LON_360 = np.arange(269, 337, dtype=np.float32)  # -91 a -24; margem para grade oficial
SCHEMA = 'worcap_cfsv2_pentad_media_auditada_v1'
POLITICA_MEMBROS = 'ensemble_completo_ou_excecao_201908_m17_auditada_v2'
# Exceção confirmada nos campos individuais; não libera lacunas em outros membros/meses.
EXCECOES_MEMBROS = {'2019-08-01': (17,)}
ROOT = 'https://iridl.ldeo.columbia.edu/SOURCES/'
MODELO = ROOT+'.Models/.NMME/.NCEP-CFSv2/'
FONTE = MODELO+'.HINDCAST/.PENTAD_SAMPLES_FULL/.prec/'
FONTES_DATAS = {
    'hindcast': ROOT+'.NOAA/.NCEP/.EMC/.CFSv2/.ENSEMBLE/.FLXF/sampleS/',
    'operacional': ROOT+'.NOAA/.NCEP/.EMC/.CFSv2/.REALTIME_ENSEMBLE/.FLXF/sampleS/',
}
MESES_EN = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
for sub in ('brutos', 'metadados', 'processados'):
    (SAIDA/sub).mkdir(parents=True, exist_ok=True)
VERSOES = dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
              xarray=xr.__version__, requests=requests.__version__, cftime=cftime.__version__)
print('WorCAP — CFSv2/NMME: aquisição e auditoria; sem treinamento.')
print('Saída:', SAIDA, '| Ambiente:', VERSOES)
print('Downloads públicos: nenhum token, conta ou Input obrigatório.')


# %% Downloads com cache, hashes e retomada
def exigir(ok, mensagem):
    if not bool(ok):
        raise ValueError(mensagem)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for parte in iter(lambda: f.read(1024*1024), b''):
            h.update(parte)
    return h.hexdigest()


def json_salvar(path, obj):
    path = Path(path)
    parcial = path.with_name(path.name+'.tmp')
    parcial.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    parcial.replace(path)


def intervalo_url(datas):
    datas = pd.DatetimeIndex(datas)
    def data_iri(t):
        return quote(f'(0000 1 {MESES_EN[t.month-1]} {t.year})', safe='()')
    return f'S/{data_iri(datas[0])}/{data_iri(datas[-1])}/RANGE/'


def url_campos(datas):
    return (FONTE+intervalo_url(datas)+f'L/{LEAD}/VALUES/'
            'Y/-61/16/RANGE/X/269/336/RANGE/data.nc')


def url_agregado(datas, produto):
    # A mesma média aritmética, calculada no servidor, reduz o tráfego e o tempo de serialização.
    # A cobertura é conferida SEPARADAMENTE por membro: número de pixels finitos e não negativos.
    operacao = {'media': '%5BM%5D/average/',
                'cobertura': '0/masklt/0/mul/1/add/%5BX/Y%5D/sum/'}[produto]
    return url_campos(datas).removesuffix('data.nc')+operacao+'data.nc'


def validar_arquivo(path, tipo):
    with Path(path).open('rb') as f:
        assinatura = f.read(8)
    if tipo == 'nc':
        exigir(assinatura[:3] == b'CDF' or assinatura == b'\x89HDF\r\n\x1a\n',
               'A resposta não é NetCDF. Pode ser uma página de login ou falha do servidor.')
        with xr.open_dataset(path, decode_times=False, engine='netcdf4') as ds:
            exigir(len(ds.variables) > 0, 'NetCDF vazio.')
    else:
        exigir(Path(path).read_text(encoding='utf-8').lstrip().startswith('Attributes {'),
               'Metadados DAS inválidos; download cancelado.')


def conferir_cache(path, url, tipo):
    meta_path = path.with_suffix(path.suffix+'.json')
    if not path.exists() or not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    exigir(meta.get('url') == url and meta.get('schema') == SCHEMA,
           f'Cache de outra requisição: {path}. Não será reutilizado.')
    exigir(meta.get('sha256') == sha256(path), f'Hash de cache divergente: {path}.')
    validar_arquivo(path, tipo)
    return meta


def baixar(url, nome, pasta='brutos', tipo='nc'):
    exigir(urlparse(url).scheme == 'https' and urlparse(url).hostname == 'iridl.ldeo.columbia.edu',
           'Host de download não autorizado pelo protocolo.')
    path = SAIDA/pasta/nome
    meta = conferir_cache(path, url, tipo)
    if meta:
        return path, meta
    raizes = [Path(PASTA_CACHE_MANUAL)] if PASTA_CACHE_MANUAL else []
    if Path('/kaggle/input').is_dir():
        raizes.append(Path('/kaggle/input'))
    for raiz in raizes:
        for candidato in sorted(raiz.rglob(nome)):
            if candidato.resolve() == path.resolve():
                continue
            m = conferir_cache(candidato, url, tipo)
            if m:
                shutil.copy2(candidato, path)
                shutil.copy2(candidato.with_suffix(candidato.suffix+'.json'),
                             path.with_suffix(path.suffix+'.json'))
                print('Recuperado do Input:', nome, flush=True)
                return path, m
    parcial = path.with_suffix(path.suffix+'.part')
    for tentativa in range(3):
        try:
            with requests.get(url, stream=True, timeout=(20, 180),
                    headers={'User-Agent': 'WorCAP-CFSv2-research/1.0'}) as resp:
                if '/auth/' in resp.url or resp.status_code in (401, 403):
                    raise ValueError('A IRI exigiu autenticação nesta requisição. '
                        'Pare e envie apenas esta mensagem de erro; não coloque credenciais no código.')
                resp.raise_for_status()
                exigir(urlparse(resp.url).hostname == 'iridl.ldeo.columbia.edu', 'Redirecionamento inesperado.')
                n = 0
                with parcial.open('wb') as f:
                    for b in resp.iter_content(64*1024):
                        n += len(b)
                        exigir(n <= 64*1024*1024, 'Resposta maior que o limite de 64 MiB por requisição.')
                        f.write(b)
                validar_arquivo(parcial, tipo)
                meta = dict(schema=SCHEMA, url=url, url_resposta=resp.url,
                    recuperado_em_utc=datetime.now(timezone.utc).isoformat(), bytes=n,
                    sha256=sha256(parcial), content_type=resp.headers.get('Content-Type'),
                    last_modified=resp.headers.get('Last-Modified'))
            parcial.replace(path)
            json_salvar(path.with_suffix(path.suffix+'.json'), meta)
            return path, meta
        except requests.RequestException as exc:
            if tentativa == 2:
                raise RuntimeError(f'Falha de rede em {nome}. Reexecute para retomar o cache. '
                                   f'Tipo: {type(exc).__name__}') from None
            print(f'Rede indisponível em {nome}; tentativa {tentativa+2}/3.', flush=True)
            time.sleep(2**(tentativa+1))


# %% Datas nominais, calendário da fonte e datas reais de inicialização
def meses_de_S(s):
    exigir(str(s.attrs.get('units', '')).strip() == 'months since 1960-01-01', 'Unidade de S inesperada.')
    exigir(str(s.attrs.get('calendar')) in ('360', '360_day'), 'Calendário de S inesperado.')
    valores = np.asarray(s.values, dtype=np.float64)
    exigir(valores.ndim == 1 and np.isfinite(valores).all(), 'S inválido.')
    exigir(np.equal(valores, np.rint(valores)).all(), 'S contém mês fracionário.')
    return pd.DatetimeIndex([(pd.Period('1960-01', freq='M')+int(v)).to_timestamp() for v in valores])


def tabela_inicializacoes(path, origens):
    with xr.open_dataset(path, decode_times=False, engine='netcdf4') as d:
        ds = d.load()
    exigir(set(ds.data_vars) == {'sampleS'} and set(ds.sampleS.dims) == {'S','M'},
           'Formato da tabela sampleS inesperado.')
    exigir(meses_de_S(ds.S).equals(pd.DatetimeIndex(origens)), 'Cobertura de sampleS divergente.')
    exigir(np.array_equal(ds.M.values, np.arange(1,29)), 'IDs de membros da tabela inesperados.')
    exigir(ds.sampleS.attrs.get('units') == 'days since 1960-01-01', 'Unidade de sampleS inesperada.')
    exigir(str(ds.sampleS.attrs.get('calendar')) in ('365', '365_day', 'noleap'), 'Calendário sampleS inesperado.')
    resultado = {}
    for k, origem in enumerate(origens):
        numeros = ds.sampleS.transpose('S','M').values[k].astype(np.float64)
        exigir(not np.isinf(numeros).any(), 'Data de inicialização infinita.')
        mascara = np.isfinite(numeros)
        n = 28 if origem.month == 11 else 24
        exigir(np.array_equal(mascara, np.arange(28) < n),
               f'Tabela de membros divergente em {origem.date()}: esperados {n}.')
        datas_cf = cftime.num2date(numeros[mascara], 'days since 1960-01-01', calendar='365_day')
        # Converte os RÓTULOS ano/mês/dia; não trata calendário sem bissextos como gregoriano.
        datas = pd.DatetimeIndex([pd.Timestamp(t.year,t.month,t.day) for t in datas_cf])
        alvo = origem+pd.offsets.MonthBegin(1)
        exigir((datas < alvo).all(), f'Inicialização invade mês-alvo: {origem.date()}.')
        exigir(datas.min() >= origem-pd.Timedelta(days=40) and
               datas.max() <= origem+pd.Timedelta(days=7), 'Janela de inicialização pentadal inesperada.')
        exigir(len(set(datas)) == n//4, 'Esperados quatro membros por data de inicialização.')
        exigir(np.all(np.unique(datas.values, return_counts=True)[1] == 4), 'Agrupamento pentadal divergente.')
        resultado[origem] = dict(mascara=mascara, datas=datas, numeros=numeros,
            fase_fonte='hindcast' if origem < pd.Timestamp('2011-04-01') else 'operacional')
    return resultado


def carregar_membros(path, origens):
    with xr.open_dataset(path, decode_times=False, engine='netcdf4') as d:
        ds = d.load()
    exigir(set(ds.data_vars) == {'prec'}, 'Arquivo deve conter somente precipitação prevista prec.')
    exigir(set(ds.prec.dims) == {'S','L','M','Y','X'}, 'Dimensões inesperadas.')
    exigir(meses_de_S(ds.S).equals(pd.DatetimeIndex(origens)), 'Meses do NetCDF não coincidem com o pedido.')
    exigir(ds.sizes['L'] == 1 and float(ds.L.values[0]) == LEAD, 'Lead incorreto; exigido L=1.5.')
    exigir(ds.L.attrs.get('units') == 'months' and float(ds.L.attrs.get('pointwidth', -1)) == 1,
           'Exigida previsão mensal, não média de vários meses.')
    exigir(ds.prec.attrs.get('units') == 'mm/day', 'Unidade inesperada; nenhuma conversão será presumida.')
    exigir(ds.prec.attrs.get('standard_name') in ('precipitation_rate','lwe_precipitation_rate'),
           'Variável não identificada como taxa de precipitação.')
    exigir(np.array_equal(ds.M.values, np.arange(1,29)), 'IDs de membros incompletos ou duplicados.')
    exigir(np.array_equal(ds.Y.values, LAT) and np.array_equal(ds.X.values, LON_360),
           'Grade nativa divergente do recorte de 1 grau solicitado.')
    exigir(ds.X.attrs.get('units') == 'degree_east' and ds.Y.attrs.get('units') == 'degree_north',
           'Unidades geográficas divergentes.')
    valores = ds.prec.transpose('S','L','M','Y','X').values[:,0].astype(np.float64)
    # IRIDL também usa file_missing_value, que não é atributo CF padrão.
    marcador = ds.prec.attrs.get('file_missing_value')
    if marcador is not None and np.isfinite(float(marcador)):
        valores[valores == float(marcador)] = np.nan
    return valores


def membros_utilizaveis(membros, origem, info):
    esperados = info['mascara']
    exigir(np.isnan(membros[~esperados]).all(), f'Membro fora da tabela com dados em {origem}.')
    completos = np.isfinite(membros).all(axis=(-1,-2))
    faltantes = tuple(int(i+1) for i in np.flatnonzero(esperados & ~completos))
    if faltantes:
        exigir(faltantes == EXCECOES_MEMBROS.get(str(origem.date())),
               f'Membros ausentes/parciais não previstos em {origem.date()}: {faltantes}.')
        exigir(np.isnan(membros[np.array(faltantes)-1]).all(),
               'A exceção exige membro inteiramente ausente; lacunas parciais ou infinitos não são aceitos.')
    usados = esperados & completos
    exigir((membros[usados] >= 0).all(), f'Precipitação negativa em {origem.date()}. Investigar.')
    datas = info['datas'][usados[esperados]]
    return usados, datas, faltantes


def processar_bloco(path, origens, tabela):
    origens = pd.DatetimeIndex(origens)
    membros = carregar_membros(path, origens)
    hash_bruto = sha256(path)
    medias, linhas, datas_min, datas_max, ns, fases = [], [], [], [], [], []
    for k, origem in enumerate(origens):
        info = tabela[origem]
        validos, datas, faltantes = membros_utilizaveis(membros[k], origem, info)
        v = membros[k,validos]
        exigir(np.isfinite(v).all(), f'Membro previsto ausente/parcial em {origem}. Não usar média parcial.')
        exigir((v >= 0).all(), f'Precipitação negativa em {origem}: mínimo {v.min():.12g} mm/day. Investigar.')
        media = v.mean(axis=0, dtype=np.float64).astype(np.float32)
        alvo = origem+pd.offsets.MonthBegin(1)
        medias.append(media); ns.append(int(validos.sum()))
        datas_min.append(datas.min()); datas_max.append(datas.max())
        fases.append(0 if info['fase_fonte'] == 'hindcast' else 1)
        linhas.append(dict(time_alvo=str(alvo.date()), time_origem=str(origem.date()),
            lead=LEAD, membros=ns[-1], membro_ids=','.join(map(str,np.flatnonzero(validos)+1)),
            membros_esperados=int(info['mascara'].sum()), membros_indisponiveis_ids=list(faltantes),
            politica_membros=POLITICA_MEMBROS,
            inicializacao_mais_antiga=str(datas.min().date()),
            inicializacao_mais_recente=str(datas.max().date()),
            inicializacoes_anteriores_ao_alvo=True, fase_fonte=info['fase_fonte'],
            minimo_membros_mm_day=float(v.min()), maximo_membros_mm_day=float(v.max()),
            minimo_media_mm_day=float(media.min()), maximo_media_mm_day=float(media.max()),
            media_espacial_mm_day=float(media.mean(dtype=np.float64)),
            membros_ausentes_estruturais=28-int(info['mascara'].sum()), pixels_ausentes_membros_validos=0,
            negativos_corrigidos=0, arquivo_bruto=path.name, sha256_bruto=hash_bruto))
    ds = xr.Dataset({
        'cfsv2_tp_media': (('time','lat','lon'), np.stack(medias)),
        'n_membros': ('time', np.array(ns,dtype=np.int16)),
        'time_origem': ('time', origens.values),
        'inicializacao_mais_antiga': ('time', pd.DatetimeIndex(datas_min).values),
        'inicializacao_mais_recente': ('time', pd.DatetimeIndex(datas_max).values),
        'fase_fonte': ('time',np.array(fases,dtype=np.int8))},
        coords={'time': (origens.to_period('M')+1).to_timestamp(), 'lat': LAT, 'lon': LON_360-360},
        attrs=dict(schema=SCHEMA, fonte=FONTE, modelo='NCEP-CFSv2',
            amostragem='PENTAD_SAMPLES_FULL; 24 membros, 28 em novembro',
            produto='previsao mensal; media aritmetica local dos membros completos disponiveis',
            politica_membros=POLITICA_MEMBROS,
            lead_iri=LEAD, referencia_temporal='S nominal no mes anterior ao alvo; datas reais auditadas',
            observacoes_de_chuva_utilizadas='nenhuma', climatologia_aplicada='nenhuma',
            grade='nativa de 1 grau; interpolacao fica para o notebook de modelagem'))
    ds.cfsv2_tp_media.attrs = dict(units='mm/day', long_name='Precipitacao prevista: media dos membros CFSv2')
    ds.lat.attrs['units'], ds.lon.attrs['units'] = 'degrees_north', 'degrees_east'
    ds.fase_fonte.attrs = dict(flag_values=np.array([0,1],dtype=np.int8), flag_meanings='hindcast operacional')
    return ds, linhas


def processar_agregados(path_media, path_cobertura, origens, tabela):
    origens = pd.DatetimeIndex(origens)
    with xr.open_dataset(path_media, decode_times=False, engine='netcdf4') as d:
        media_ds = d.load()
    with xr.open_dataset(path_cobertura, decode_times=False, engine='netcdf4') as d:
        cobertura = d.load()
    for d in (media_ds, cobertura):
        exigir(set(d.data_vars) == {'prec'}, 'Variável de resposta inesperada.')
        exigir(meses_de_S(d.S).equals(origens), 'Datas da média ou cobertura divergentes.')
        exigir(d.sizes['L'] == 1 and float(d.L.values[0]) == LEAD, 'Lead da resposta divergente.')
        exigir(d.L.attrs.get('units') == 'months' and float(d.L.attrs.get('pointwidth',-1)) == 1,
               'Produto não é uma previsão mensal.')
    exigir(set(media_ds.prec.dims) == {'S','L','Y','X'}, 'Dimensões da média divergentes.')
    exigir(media_ds.prec.attrs.get('units') == 'mm/day', 'Unidade da média divergente.')
    exigir(np.array_equal(media_ds.Y.values,LAT) and np.array_equal(media_ds.X.values,LON_360),
           'Grade da média divergente.')
    exigir(set(cobertura.prec.dims) == {'S','L','M'} and
           np.array_equal(cobertura.M.values,np.arange(1,29)), 'Cobertura por membro divergente.')
    # O atributo de unidade herdado pela expressão de cobertura não é interpretado como chuva.
    # 0*prec+1 vale 1 em cada pixel válido; a soma espacial conta pixels, não precipitação.
    contagens = cobertura.prec.transpose('S','L','M').values[:,0].astype(np.float64)
    medias = media_ds.prec.transpose('S','L','Y','X').values[:,0].astype(np.float64)
    exigir(np.isfinite(medias).all() and (medias >= 0).all(), 'Média negativa, ausente ou não finita.')
    linhas, ns, mins, maxs, fases = [], [], [], [], []
    hm, hc = sha256(path_media), sha256(path_cobertura)
    for k, origem in enumerate(origens):
        info = tabela[origem]; esperados = info['mascara']; validos = esperados.copy(); datas = info['datas']
        ruins = tuple(int(i+1) for i in np.flatnonzero(esperados & (contagens[k] != len(LAT)*len(LON_360))))
        excecao = None
        if ruins:
            exigir(ruins == EXCECOES_MEMBROS.get(str(origem.date())),
                   f'Cobertura incompleta não prevista em {origem.date()}: membros {ruins}.')
            print(f'{origem:%Y-%m}: conferindo exceção do membro 17 nos campos individuais...', flush=True)
            path_raw, meta_raw = baixar(url_campos(pd.DatetimeIndex([origem])),
                                       f'cfsv2_excecao_membros_{origem:%Y%m}.nc')
            raw = carregar_membros(path_raw, pd.DatetimeIndex([origem]))[0]
            validos, datas, faltantes = membros_utilizaveis(raw, origem, info)
            exigir(faltantes == ruins, 'Campos individuais e contagens de cobertura discordam.')
            media_local = raw[validos].mean(axis=0, dtype=np.float64)
            exigir(np.allclose(media_local, medias[k], rtol=2e-7, atol=2e-6),
                   'Média da IRI diverge da média dos 23 membros completos. Parar e investigar.')
            excecao = dict(membros_indisponiveis_ids=list(faltantes),
                arquivo=str(path_raw.relative_to(SAIDA)), **meta_raw,
                maximo_delta_media_mm_day=float(np.max(np.abs(media_local-medias[k]))))
        vazios = contagens[k,~esperados]
        exigir(np.all(np.isnan(vazios) | (vazios == 0)), 'Membro inesperado contém pixels válidos.')
        ns.append(int(validos.sum())); mins.append(datas.min()); maxs.append(datas.max())
        fases.append(0 if info['fase_fonte']=='hindcast' else 1)
        linhas.append(dict(time_alvo=str((origem+pd.offsets.MonthBegin(1)).date()),
            time_origem=str(origem.date()), lead=LEAD, membros=ns[-1],
            membros_esperados=int(esperados.sum()), membros_indisponiveis_ids=list(ruins),
            politica_membros=POLITICA_MEMBROS, auditoria_excecao=excecao,
            membro_ids=','.join(map(str,np.flatnonzero(validos)+1)),
            inicializacao_mais_antiga=str(datas.min().date()), inicializacao_mais_recente=str(datas.max().date()),
            inicializacoes_anteriores_ao_alvo=True, fase_fonte=info['fase_fonte'],
            pixels_validos_por_membro=len(LAT)*len(LON_360), membros_com_cobertura_completa=ns[-1],
            minimo_media_mm_day=float(medias[k].min()), maximo_media_mm_day=float(medias[k].max()),
            media_espacial_mm_day=float(medias[k].mean()), negativos_corrigidos=0,
            arquivo_media=path_media.name, sha256_media=hm,
            arquivo_cobertura=path_cobertura.name, sha256_cobertura=hc))
    ds=xr.Dataset({
        'cfsv2_tp_media': (('time','lat','lon'),medias.astype(np.float32)),
        'n_membros': ('time',np.array(ns,dtype=np.int16)),
        'time_origem': ('time',origens.values),
        'inicializacao_mais_antiga': ('time',pd.DatetimeIndex(mins).values),
        'inicializacao_mais_recente': ('time',pd.DatetimeIndex(maxs).values),
        'fase_fonte': ('time',np.array(fases,dtype=np.int8))},
        coords={'time':(origens.to_period('M')+1).to_timestamp(),'lat':LAT,'lon':LON_360-360},
        attrs=dict(schema=SCHEMA,fonte=FONTE,modelo='NCEP-CFSv2',lead_iri=LEAD,
            produto='media aritmetica dos membros calculada pela IRI; cobertura completa auditada por membro',
            politica_membros=POLITICA_MEMBROS,
            amostragem='PENTAD_SAMPLES_FULL; 24 membros, 28 em novembro',
            observacoes_de_chuva_utilizadas='nenhuma',climatologia_aplicada='nenhuma',
            grade='nativa de 1 grau; sem interpolacao'))
    ds.cfsv2_tp_media.attrs=dict(units='mm/day',long_name='Precipitacao prevista: media dos membros CFSv2')
    ds.lat.attrs['units'],ds.lon.attrs['units']='degrees_north','degrees_east'
    ds.fase_fonte.attrs=dict(flag_values=np.array([0,1],dtype=np.int8),flag_meanings='hindcast operacional')
    return ds,linhas


def salvar_nc(ds, path):
    path = Path(path)
    parcial = path.with_suffix('.tmp.nc')
    ds.to_netcdf(parcial, engine='netcdf4',
        encoding={'cfsv2_tp_media': {'zlib':True, 'complevel':4, 'dtype':'float32'}})
    with xr.open_dataset(parcial, engine='netcdf4') as conferido:
        xr.testing.assert_equal(ds, conferido.load())
    parcial.replace(path)


# %% Catálogo e tabela de inicializações — pequenos downloads públicos
ALVOS = pd.date_range(INICIO_ALVOS, FIM_ALVOS, freq='MS')
ORIGENS = (ALVOS.to_period('M')-1).to_timestamp()
exigir(len(ALVOS) == 515, 'Este protocolo prepara 515 meses-alvo: fev/1982 a dez/2024.')
fontes_meta = {}
for nome, url in {
    'cfsv2_prec': FONTE+'dods.das',
    'cfsv2_hindcast': MODELO+'.HINDCAST/.PENTAD_SAMPLES/dods.das',
    'cfsv2_operacional': MODELO+'.FORECAST/.PENTAD_SAMPLES/dods.das',
}.items():
    path, meta = baixar(url, nome+'.das', 'metadados', 'das')
    fontes_meta[nome] = dict(arquivo=str(path.relative_to(SAIDA)), **meta)

TABELA = {}
for fase, datas in [('hindcast', ORIGENS[ORIGENS < '2011-04-01']),
                    ('operacional', ORIGENS[ORIGENS >= '2011-04-01'])]:
    url = FONTES_DATAS[fase]+intervalo_url(datas)+'data.nc'
    path, meta = baixar(url, 'cfsv2_datas_'+fase+'.nc', 'metadados')
    TABELA.update(tabela_inicializacoes(path, datas))
    fontes_meta['datas_'+fase] = dict(arquivo=str(path.relative_to(SAIDA)), **meta)
exigir(set(TABELA) == set(ORIGENS), 'Datas de inicialização incompletas.')
linhas_datas = []
for origem, info in TABELA.items():
    for membro, numero, data in zip(np.flatnonzero(info['mascara'])+1,
                                    info['numeros'][info['mascara']], info['datas']):
        linhas_datas.append(dict(time_origem=str(origem.date()),
            time_alvo=str((origem+pd.offsets.MonthBegin(1)).date()), membro=int(membro),
            inicializacao_data=str(data.date()), sampleS_original=float(numero),
            calendario_original='365_day', unidade_original='days since 1960-01-01',
            fase_fonte=info['fase_fonte']))
pd.DataFrame(linhas_datas).to_csv(SAIDA/'inicializacoes_por_membro.csv', index=False)
print('Tabela de inicializações validada:', len(TABELA), 'meses; todos os membros anteriores ao alvo.')
print('Alvos:', ALVOS[0].date(), 'a', ALVOS[-1].date(), '| Lead IRI:', LEAD)


# %% Baixar média e cobertura por membro, por ano de emissão; retomável
BLOCOS, AUDITORIA, PEDIDOS = [], [], []
for ano in sorted(set(ORIGENS.year)):
    datas = ORIGENS[ORIGENS.year == ano]
    print(f'{ano}: {len(datas)} emissões; conferindo cobertura dos membros...', flush=True)
    nome = f'cfsv2_L1p5_{ano}_{datas[0].month:02d}-{datas[-1].month:02d}'
    path_c, meta_c = baixar(url_agregado(datas,'cobertura'), nome+'_cobertura.nc')
    print(f'{ano}: obtendo média mensal do ensemble...', flush=True)
    path_m, meta_m = baixar(url_agregado(datas,'media'), nome+'_media.nc')
    ds, linhas = processar_agregados(path_m,path_c,datas,TABELA)
    nc = SAIDA/'processados'/f'cfsv2_media_{ano}.nc'
    salvar_nc(ds, nc)
    BLOCOS.append(nc)
    AUDITORIA.extend(linhas)
    PEDIDOS.append(dict(ano=ano,media=dict(arquivo=str(path_m.relative_to(SAIDA)),**meta_m),
        cobertura=dict(arquivo=str(path_c.relative_to(SAIDA)),**meta_c),
        arquivo_processado=str(nc.relative_to(SAIDA)),sha256_processado=sha256(nc)))
    print(f'{ano}: concluído; {min(ds.n_membros.values)}–{max(ds.n_membros.values)} membros; '
          'datas, unidade e grade conferidas.', flush=True)


# %% Reproduzir localmente a média em dois meses: início do histórico e fim do teste
AUDITORIA_PILOTOS = []
for origem in [ORIGENS[0],ORIGENS[-1]]:
    datas=pd.DatetimeIndex([origem])
    path,meta=baixar(url_campos(datas),f'cfsv2_piloto_membros_{origem:%Y%m}.nc')
    local,_=processar_bloco(path,datas,TABELA)
    with xr.open_dataset(SAIDA/'processados'/f'cfsv2_media_{origem.year}.nc') as anual:
        servidor=anual.cfsv2_tp_media.sel(time=local.time).values
    esperado=local.cfsv2_tp_media.values
    exigir(np.allclose(esperado,servidor,rtol=2e-7,atol=2e-6),
           f'Média da IRI diverge da média local em {origem:%Y-%m}.')
    AUDITORIA_PILOTOS.append(dict(time_origem=str(origem.date()),
        maximo_delta_mm_day=float(np.max(np.abs(esperado-servidor))),
        arquivo=str(path.relative_to(SAIDA)),**meta))
print('Pilotos: média do servidor reproduzida localmente com os membros individuais.')


# %% Consolidar desenvolvimento, ajuste final e teste; criar manifesto
partes = []
for path in BLOCOS:
    with xr.open_dataset(path, engine='netcdf4') as ds:
        partes.append(ds.load())
COMPLETO = xr.concat(partes, dim='time').sortby('time')
exigir(pd.DatetimeIndex(COMPLETO.time.values).equals(ALVOS), 'Meses duplicados ou faltantes na consolidação.')
exigir(pd.DatetimeIndex(COMPLETO.time_origem.values).equals(ORIGENS), 'Deslocamento temporal incorreto.')
exigir(np.isfinite(COMPLETO.cfsv2_tp_media.values).all(), 'Valores finais não finitos.')
AUDITORIA = pd.DataFrame(AUDITORIA).sort_values('time_alvo').reset_index(drop=True)
AUDITORIA.to_csv(SAIDA/'auditoria_mensal.csv', index=False)
arquivos = {}
for fase, inicio, fim, quantidade in [
    ('desenvolvimento','1982-02-01','2020-12-01',467),
    ('somente_ajuste_final','2021-01-01','2022-12-01',24),
    ('teste','2023-01-01','2024-12-01',24),
]:
    ds = COMPLETO.sel(time=slice(inicio,fim))
    exigir(ds.sizes['time'] == quantidade, f'Cobertura incorreta em {fase}.')
    path = SAIDA/f'cfsv2_{fase}.nc'
    salvar_nc(ds, path)
    arquivos[fase] = dict(arquivo=path.name, sha256=sha256(path), bytes=path.stat().st_size,
        inicio_alvos=inicio, fim_alvos=fim, meses=quantidade)
manifesto = dict(schema=SCHEMA, criado_em_utc=datetime.now(timezone.utc).isoformat(),
    ambiente=VERSOES, modelo='NCEP-CFSv2', fonte=FONTE, lead_iri=LEAD,
    unidade='mm/day', inicio_alvos=INICIO_ALVOS, fim_alvos=FIM_ALVOS,
    meses=515, transicao_origem_operacional='2011-04-01',
    temporalidade='S nominal T-1; todas as datas sampleS dos membros sao anteriores ao mes T',
    limitacao_datas='sampleS informa datas agrupadas, sem distinguir os quatro horarios do dia; '
        'a auditoria exige o dia inteiro anterior ao alvo',
    retrospectivas='hindcasts sao simulacoes retrospectivas; a data de inicializacao nao e a data '
        'de publicacao historica do arquivo',
    anomalias='nao calculadas; a climatologia CFSv2 sera estimada somente no treino de cada fold',
    interpolacao='nao aplicada; grade nativa 1 grau com margem para grade oficial',
    janeiro_1982='nao disponivel com S=T-1; nao preenchido',
    treinamento='nenhum nesta etapa; futura comparacao pareada com M6, janela maxima de 30 anos',
    chuva_observada_utilizada=False, arquivos=arquivos, metadados=fontes_meta, requisicoes=PEDIDOS,
    pilotos_membros=AUDITORIA_PILOTOS,
    politica_membros=POLITICA_MEMBROS,
    excecoes_permitidas=EXCECOES_MEMBROS,
    excecoes_aplicadas=[r for r in AUDITORIA.to_dict('records') if r.get('membros_indisponiveis_ids')],
    auditoria_mensal_sha256=sha256(SAIDA/'auditoria_mensal.csv'),
    inicializacoes_sha256=sha256(SAIDA/'inicializacoes_por_membro.csv'),
    fontes_documentacao=[
        'https://www.cpc.ncep.noaa.gov/products/NMME/data.html',
        'https://iridl.ldeo.columbia.edu/auth/notice',
        'https://github.com/iri-pycpt/PyCPT2-Seasonal-Forecast-User-Guide/blob/main/configuration.md',
        'https://github.com/iri-pycpt/pycpt/blob/62b90ccf4cb543f2a5680a5c553bcf4ca6b846c0/pycpt/src/cptdl/catalog.py'],
    creditos='NCEP CFSv2 / NMME; IRI Data Library; NOAA, NSF, NASA, DOE, NCEP, IRI e NCAR. '
        'Kirtman et al. (2014), DOI 10.1175/BAMS-D-12-00050.1.')
json_salvar(SAIDA/'cfsv2_manifesto.json', manifesto)
display(pd.DataFrame(arquivos).T)
display(AUDITORIA.head())
print('PRONTO:', SAIDA)
print('515 meses validados; nenhum mês preenchido; nenhuma observação de chuva utilizada.')
print('Salve a versão concluída. No futuro notebook de treinamento, anexe esta saída em Add Input > Notebook.')
print('O M6 e os arquivos do SEAS5 continuam sendo necessários na próxima etapa.')
if Path('/kaggle/working').is_dir():
    display(FileLink(os.path.relpath(SAIDA/'cfsv2_manifesto.json', Path.cwd())))
