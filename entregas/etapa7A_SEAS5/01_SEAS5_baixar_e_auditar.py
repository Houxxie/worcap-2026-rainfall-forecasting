# %% Configuração: apenas aquisição de previsões, sem chuva observada ou treinamento
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, sys, importlib.util, subprocess
import numpy as np
import pandas as pd
import xarray as xr
from IPython.display import display, FileLink

# Internet deve estar ligada neste notebook de aquisição.
# O token fica somente no Kaggle Secrets, com nome CDS_API_KEY.
DEPENDENCIAS = {'cdsapi': 'cdsapi>=0.7.7,<0.8', 'eccodes': 'eccodes>=2.40,<3',
                'netCDF4': 'netCDF4>=1.6,<2'}
faltantes = [spec for mod, spec in DEPENDENCIAS.items() if importlib.util.find_spec(mod) is None]
if faltantes:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', *faltantes])
import cdsapi
import eccodes as ec

DATASET = 'seasonal-monthly-single-levels'
SYSTEM = '51'
LEAD = 2
# Margem de 1 grau para interpolar até os limites da grade oficial sem extrapolar.
AREA = [16, -91, -61, -24]  # norte, oeste, sul, leste
INICIO_ALVOS = '1982-01-01'
FIM_ALVOS = '2024-12-01'
BASE = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path.cwd()/'outputs'
SAIDA = BASE/'worcap_SEAS5_dados'
SAIDA.mkdir(parents=True, exist_ok=True)
(SAIDA/'brutos').mkdir(exist_ok=True)
(SAIDA/'anuais').mkdir(exist_ok=True)
# Se precisar retomar em outra sessão, anexe a saída anterior em Input.
PASTA_CACHE_MANUAL = None
SCHEMA = 'worcap_seas5_v1'
FONTE = 'https://cds.climate.copernicus.eu/datasets/seasonal-monthly-single-levels'

def exigir(ok, mensagem):
    if not bool(ok):
        raise ValueError(mensagem)

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def json_salvar(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding='utf-8')

def requisicao(origens):
    origens = pd.DatetimeIndex(origens)
    exigir(len(set(origens.year)) == 1, 'Cada requisição deve conter um único ano de inicialização.')
    return dict(originating_centre='ecmwf', system=SYSTEM,
                variable=['total_precipitation'], product_type=['monthly_mean'],
                year=[str(origens[0].year)], month=[f'{m:02d}' for m in origens.month],
                leadtime_month=[str(LEAD)], area=AREA, data_format='grib')

def validar_metadados(meta):
    exigir(str(meta['system']) == SYSTEM and str(meta['origin']) == 'ecmf',
           'Arquivo não corresponde ao ECMWF SEAS5 sistema 51. Não misturar com sistema 5.')
    exigir(str(meta['type']) == 'fcmean' and str(meta['stream']) == 'msmm',
           'Exigida média mensal de membro; não usar reanálise, climatologia pronta ou anomalia pronta.')
    exigir(int(meta['paramId']) == 172228, 'Exigida taxa média de precipitação prevista, paramId 172228.')
    exigir(str(meta['units']).replace(' ', '') in ('ms**-1', 'ms^-1', 'ms-1', 'm/s'),
           f"Unidade inesperada: {meta['units']}. Nenhuma conversão será adivinhada.")
    inicializacao = pd.Timestamp(str(int(meta['dataDate'])))
    exigir(inicializacao.day == 1 and int(meta['dataTime']) == 0, 'Inicialização deve ser dia 1, 00 UTC.')
    exigir(int(meta['forecastMonth']) == LEAD == 2, 'Lead diferente de 2.')
    alvo = (inicializacao.to_period('M')+1).to_timestamp()
    exigir(int(meta['verifyingMonth']) == alvo.year*100+alvo.month,
           'verifyingMonth diverge de T. Não usar validityDate como mês-alvo: pode apontar ao fim do intervalo.')
    exigir(inicializacao < alvo, 'Previsão não foi inicializada antes do alvo.')
    n = int(meta['numberOfForecastsInEnsemble'])
    # Alguns GRIBs fornecidos pelo CDS trazem zero neste campo do cabeçalho.
    # Zero não é uma contagem utilizável: conferir os IDs reais ao terminar o mês.
    exigir(n in (0, 25, 51), f'Quantidade declarada de membros não prevista: {n}.')
    exigir(0 <= int(meta['number']) < (n or 51), 'Identificador de membro fora do conjunto esperado.')
    return inicializacao, alvo, n


def conferir_membros(inicio, ids, declarados):
    inicio = pd.Timestamp(inicio)
    ids, declarados = set(ids), set(declarados)
    informados = declarados - {0}
    exigir(len(informados) <= 1, 'Cabeçalhos divergem sobre o tamanho do ensemble.')
    n = len(ids)
    # ECMWF: hindcasts 1981–2016 = 25; operação desde novembro/2017 = 51.
    # Jan–out/2017 é a transição: aceitar apenas conjuntos completos 0..24 ou 0..50.
    permitidos = {25} if inicio.year <= 2016 else ({51} if inicio >= pd.Timestamp('2017-11-01') else {25, 51})
    exigir(n in permitidos and ids == set(range(n)),
           f'Ensemble incompleto/inesperado em {inicio.date()}: {n} membros, '
           f'IDs={sorted(ids)}; tamanhos permitidos={sorted(permitidos)}. Não completar artificialmente.')
    exigir(not informados or informados == {n},
           f'Contagem real {n} diverge do cabeçalho {sorted(informados)} em {inicio.date()}.')
    return n

def tratar_negativos_precipitacao(valores, contexto, packing_error=None):
    # Limite operacional deste notebook; não é um limite teórico do packingError.
    # A taxa mensal pode herdar erros de etapas anteriores de acumulação/conversão.
    # 0,01 mm/dia equivale a no máximo 0,31 mm em um mês, por membro/pixel.
    # Não truncar valores positivos e não modificar o GRIB original.
    fator, tolerancia_mm_day = 86400000.0, 0.01
    valores = np.asarray(valores, dtype=np.float64)
    exigir(np.isfinite(valores).all(), f'Valores SEAS5 não finitos: {contexto}.')
    indices = np.flatnonzero(valores.ravel() < 0)
    negativos = valores.ravel()[indices]
    minimo_mm_day = float(negativos.min()*fator) if len(indices) else 0.0
    exigir(minimo_mm_day >= -tolerancia_mm_day,
           f'Precipitação negativa acima da tolerância: {minimo_mm_day:.10g} mm/day; '
           f'limite=-{tolerancia_mm_day} mm/day; {contexto}. Investigar; não ampliar automaticamente.')
    pe = float(packing_error) if packing_error is not None else None
    if pe is not None and not np.isfinite(pe):
        pe = None
    registro = dict(politica='negativos_ate_0.01_mm_day_para_zero_v1',
        tolerancia_mm_day=tolerancia_mm_day, quantidade=int(len(indices)),
        minimo_mm_day_antes=minimo_mm_day,
        soma_incrementos_mm_day=float(-negativos.sum()*fator),
        packingError_mm_day=pe*fator if pe is not None else None,
        indices_flat_grib=indices.tolist(), valores_originais_m_s=negativos.tolist())
    if len(indices):
        valores = valores.copy()
        valores.flat[indices] = 0.0
    return valores, registro


def campo_regular(latitudes, longitudes, valores):
    latitudes = np.asarray(latitudes, dtype=np.float64)
    longitudes = (np.asarray(longitudes, dtype=np.float64)+180) % 360-180
    valores = np.asarray(valores, dtype=np.float64)
    lat, lon = np.unique(latitudes), np.unique(longitudes)
    exigir(len(lat)*len(lon) == len(valores), 'Não é uma grade regular completa.')
    iy, ix = np.searchsorted(lat, latitudes), np.searchsorted(lon, longitudes)
    exigir(len(np.unique(iy*len(lon)+ix)) == len(valores), 'Coordenadas duplicadas no GRIB.')
    out = np.empty((len(lat), len(lon)), dtype=np.float64)
    out[iy, ix] = valores
    exigir(np.isfinite(out).all(), 'SEAS5 contém valores não finitos.')
    exigir((out >= 0).all(), 'Precipitação prevista negativa. Investigar antes de continuar.')
    return lat, lon, out

def auditar_dataset(ds, esperado=None):
    exigir(ds.attrs.get('schema') == SCHEMA, 'Versão de artefato SEAS5 desconhecida.')
    exigir(ds.attrs.get('dataset') == DATASET and str(ds.attrs.get('system')) == SYSTEM,
           'Proveniência SEAS5 inválida.')
    exigir(ds.attrs.get('product_type') == 'monthly_mean' and int(ds.attrs.get('leadtime_month')) == 2,
           'Produto ou lead incorreto.')
    exigir(ds.seas5_tp_media.attrs.get('units') == 'mm/day', 'SEAS5 deve estar em mm/day após conversão.')
    exigir(ds.seas5_tp_media.dims == ('time', 'lat', 'lon'), 'Dimensões SEAS5 inesperadas.')
    datas = pd.DatetimeIndex(ds.time.values)
    exigir(datas.is_unique and datas.is_monotonic_increasing, 'Alvos duplicados/desordenados.')
    exigir(datas.equals(pd.date_range(datas.min(), datas.max(), freq='MS')), 'Alvos SEAS5 com lacunas.')
    if esperado is not None:
        exigir(datas.equals(pd.DatetimeIndex(esperado)), 'Calendário SEAS5 não coincide com a requisição.')
    exigir(pd.DatetimeIndex(ds.time_origem.values).equals((datas.to_period('M')-1).to_timestamp()),
           'SEAS5 não está inicializado exatamente em T−1.')
    exigir(np.isin(ds.n_membros.values, [25, 51]).all(), 'Conjunto de membros incompleto.')
    for eixo in ['lat','lon']:
        v = ds[eixo].values
        exigir(len(v)>1 and np.all(np.diff(v)==1), f'Grade nativa de 1 grau incorreta: {eixo}.')
    exigir(ds.lat.min() <= -60 and ds.lat.max() >= 15 and ds.lon.min() <= -90 and ds.lon.max() >= -25,
           'Recorte não cobre a grade oficial. Não extrapolar.')
    exigir(np.isfinite(ds.seas5_tp_media.values).all() and (ds.seas5_tp_media.values >= 0).all(),
           'Valores SEAS5 inválidos.')

def decodificar_grib(path, origens):
    origens = pd.DatetimeIndex(origens)
    conjuntos, somas, membros, esperados, auditoria = {}, {}, {}, {}, []
    correcoes = {}
    lat_ref = lon_ref = None
    keys = ['system','origin','type','stream','paramId','units','dataDate','dataTime',
            'forecastMonth','verifyingMonth','number','numberOfForecastsInEnsemble']
    with Path(path).open('rb') as f:
        while True:
            g = ec.codes_grib_new_from_file(f)
            if g is None:
                break
            try:
                meta = {k: ec.codes_get(g,k) for k in keys}
                inicio, alvo, n = validar_metadados(meta)
                exigir(inicio in origens, f'Inicialização não solicitada: {inicio}.')
                exigir(int(ec.codes_get(g,'numberOfMissing')) == 0, 'Membro GRIB contém pixels ausentes.')
                try:
                    packing_error = ec.codes_get(g, 'packingError')
                except ec.CodesInternalError:
                    packing_error = None
                valores, ajuste = tratar_negativos_precipitacao(ec.codes_get_values(g),
                    f'origem={inicio.date()}, alvo={alvo.date()}, membro={meta["number"]}', packing_error)
                ajuste['membro'] = int(meta['number'])
                lat, lon, campo = campo_regular(ec.codes_get_array(g,'latitudes'),
                    ec.codes_get_array(g,'longitudes'), valores)
                if lat_ref is None:
                    lat_ref, lon_ref = lat, lon
                exigir(np.array_equal(lat,lat_ref) and np.array_equal(lon,lon_ref),
                       'Grade mudou entre membros ou meses.')
                if alvo not in somas:
                    somas[alvo] = np.zeros_like(campo)
                    membros[alvo] = set()
                    esperados[alvo] = set()
                    conjuntos[alvo] = inicio
                    correcoes[alvo] = []
                membro = int(meta['number'])
                exigir(membro not in membros[alvo], 'Membro duplicado no mesmo mês.')
                esperados[alvo].add(n)
                membros[alvo].add(membro)
                somas[alvo] += campo
                if ajuste['quantidade']:
                    correcoes[alvo].append(ajuste)
            finally:
                ec.codes_release(g)
    destinos = (origens.to_period('M')+1).to_timestamp()
    exigir(set(somas) == set(destinos), 'Faltam meses ou existem meses extras no arquivo baixado.')
    mapas, contagens = [], []
    for alvo in destinos:
        n = conferir_membros(conjuntos[alvo], membros[alvo], esperados[alvo])
        # Fonte: taxa mensal média em m/s. Resultado: mm/dia médios, não total do mês.
        mapa = (somas[alvo]/n * 1000.0 * 86400.0).astype(np.float32)
        mapas.append(mapa)
        contagens.append(n)
        ajustes = correcoes[alvo]
        n_corrigidos = sum(a['quantidade'] for a in ajustes)
        if n_corrigidos:
            print(f'{alvo.date()}: {n_corrigidos} negativos residuais zerados nos membros; '
                  f'mínimo={min(a["minimo_mm_day_antes"] for a in ajustes):.9g} mm/day. '
                  'Correções registradas na auditoria.', flush=True)
        auditoria.append(dict(time_alvo=str(alvo.date()), time_origem=str(conjuntos[alvo].date()),
            forecastMonth=2, verifyingMonth=alvo.year*100+alvo.month, membros=n,
            membros_declarados_cabecalho=sorted(esperados[alvo]),
            membros_ids=sorted(membros[alvo]), contagem_validada_por_ids=True,
            politica_negativos='negativos_ate_0.01_mm_day_para_zero_v1',
            tolerancia_negativos_mm_day=0.01, negativos_corrigidos=n_corrigidos,
            correcoes_negativas=ajustes,
            incremento_medio_ensemble_mm_day=sum(a['soma_incrementos_mm_day'] for a in ajustes)/(n*mapa.size),
            minimo_mm_day=float(mapa.min()), maximo_mm_day=float(mapa.max()), media_mm_day=float(mapa.mean())))
    ds = xr.Dataset({'seas5_tp_media': (('time','lat','lon'), np.stack(mapas)),
                     'n_membros': ('time', np.asarray(contagens,dtype=np.int16))},
        coords={'time':destinos, 'time_origem':('time',origens.values), 'lat':lat_ref, 'lon':lon_ref},
        attrs=dict(schema=SCHEMA,dataset=DATASET,system=SYSTEM,originating_centre='ecmwf',
                   product_type='monthly_mean',leadtime_month=2,paramId=172228,
                   units_origem='m s**-1',fator_para_mm_day=86400000.0,
                   fonte=FONTE,descricao='Previsão SEAS5, não observação/reanálise do alvo.',
                   hindcasts='Retrospectivos; não são vintages operacionais da década de 1980.'))
    ds.seas5_tp_media.attrs.update(units='mm/day', long_name='Precipitação prevista: média de todos os membros')
    auditar_dataset(ds,destinos)
    return ds, auditoria

def verificar_catalogo(origens):
    import requests
    url = 'https://cds.climate.copernicus.eu/api/catalogue/v1/collections/'+DATASET
    r = requests.get(url,timeout=60); r.raise_for_status(); meta=r.json()
    url2 = next(l['href'] for l in meta['links'] if l['rel']=='constraints')
    r = requests.get(url2,timeout=60); r.raise_for_status(); restricoes=r.json()
    json_salvar(SAIDA/'catalogo.json',meta)
    json_salvar(SAIDA/'restricoes_catalogo.json',restricoes)
    for origem in origens:
        pedido=dict(originating_centre='ecmwf',system=SYSTEM,variable='total_precipitation',
                    product_type='monthly_mean',year=str(origem.year),month=f'{origem.month:02d}',leadtime_month='2')
        exigir(any(all(v in linha.get(k,[]) for k,v in pedido.items()) for linha in restricoes),
               f'Catálogo não oferece a combinação solicitada em {origem.date()}. Não substituir sistema ou datas.')
    print('Catálogo confirmou todas as inicializações solicitadas para o sistema 51.',flush=True)

def cliente_cds():
    # O segredo não é impresso, gravado no notebook ou incluído no manifesto.
    token = os.environ.get('CDS_API_KEY')
    if not token and Path('/kaggle').is_dir():
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret('CDS_API_KEY')
        except Exception:
            raise RuntimeError('Crie e habilite o secret CDS_API_KEY em Add-ons → Secrets. '
                               'Use o Personal Access Token do CDS e aceite a licença do dataset no site.') from None
    if not token:
        raise RuntimeError('Defina CDS_API_KEY no ambiente local, ou execute no Kaggle com esse Secret habilitado.')
    return cdsapi.Client(url='https://cds.climate.copernicus.eu/api',key=token,
                         quiet=True,debug=False,timeout=120,retry_max=3,sleep_max=30)

# %% Download com retomada e auditoria de cada ano
ALVOS = pd.date_range(INICIO_ALVOS,FIM_ALVOS,freq='MS')
ORIGENS = (ALVOS.to_period('M')-1).to_timestamp()
verificar_catalogo(ORIGENS)
print('Alvos:',ALVOS.min().date(),'a',ALVOS.max().date(),'| Inicializações:',ORIGENS.min().date(),'a',ORIGENS.max().date())
print('Somente previsões. Nenhuma observação de chuva é lida neste notebook.')
CACHE = []
if PASTA_CACHE_MANUAL:
    CACHE.append(Path(PASTA_CACHE_MANUAL))
elif Path('/kaggle/input').is_dir():
    # Saídas parciais ainda não têm o manifesto final; localizar também os caches anuais.
    CACHE = sorted(set([p.parent for p in Path('/kaggle/input').rglob('seas5_51_manifesto.json')]+
        [p.parent.parent for p in Path('/kaggle/input').rglob('seas5_51_*.json') if p.parent.name=='anuais']))
cliente = None
partes, fontes, linhas = [], [], []
for ano in sorted(set(ORIGENS.year)):
    datas = ORIGENS[ORIGENS.year==ano]
    pedido = requisicao(datas)
    chave = hashlib.sha256(json.dumps(pedido,sort_keys=True).encode()).hexdigest()[:12]
    nome = f'seas5_51_{ano}_{chave}'
    nc, side = SAIDA/'anuais'/f'{nome}.nc', SAIDA/'anuais'/f'{nome}.json'
    pares = [(nc,side)]+[(p/'anuais'/nc.name,p/'anuais'/side.name) for p in CACHE]
    reutilizado = False
    for candidato, registro in pares:
        if candidato.is_file() and registro.is_file():
            m = json.loads(registro.read_text(encoding='utf-8'))
            exigir(m['request']==pedido and m['sha256_nc']==sha256(candidato), 'Cache alterado ou incompatível.')
            with xr.open_dataset(candidato) as x:
                ds=x.load()
            auditar_dataset(ds,(datas.to_period('M')+1).to_timestamp())
            if candidato != nc:
                import shutil
                shutil.copyfile(candidato,nc); shutil.copyfile(registro,side)
            auditoria = m['auditoria']
            reutilizado=True
            break
    if not reutilizado:
        bruto = SAIDA/'brutos'/f'{nome}.grib'
        pedido_path = bruto.with_suffix('.request.json')
        if bruto.is_file():
            exigir(pedido_path.is_file() and json.loads(pedido_path.read_text())==pedido, 'GRIB existente sem requisição correspondente.')
        else:
            if cliente is None:
                cliente = cliente_cds()
            print('Solicitando',ano,'|',len(datas),'inicializações. A fila do CDS pode demorar.',flush=True)
            parcial = bruto.with_suffix('.partial')
            try:
                cliente.retrieve(DATASET,pedido,str(parcial))
            except Exception as exc:
                # A exceção original pode conter informação de autenticação: não a propagar.
                raise RuntimeError(f'CDS interrompeu o download de {ano} ({type(exc).__name__}). '
                    'Confira token, licença aceita e Internet. Reexecute para reutilizar os anos concluídos.') from None
            exigir(parcial.stat().st_size>0, 'Download vazio.')
            parcial.replace(bruto)
            json_salvar(pedido_path,pedido)
        ds,auditoria = decodificar_grib(bruto,datas)
        ds.to_netcdf(nc,engine='netcdf4',encoding={'seas5_tp_media':{'zlib':True,'complevel':4}})
        m = dict(request=pedido,sha256_grib=sha256(bruto),sha256_nc=sha256(nc),
                 arquivo_bruto=bruto.name,arquivo_nc=nc.name,auditoria=auditoria)
        json_salvar(side,m)
    partes.append(ds)
    fontes.append(m)
    linhas.extend(auditoria)
    print(ano,'concluído', '(cache)' if reutilizado else '',flush=True)

# %% Artefatos separados: desenvolvimento, uso exclusivo no ajuste final e teste
todo = xr.concat(partes,dim='time',combine_attrs='identical').sortby('time')
auditar_dataset(todo,ALVOS)
arquivos = {}
recortes = {'desenvolvimento':('1982-01-01','2020-12-01'),
            'somente_ajuste_final':('2021-01-01','2022-12-01'),
            'teste':('2023-01-01','2024-12-01')}
for uso,(inicio,fim) in recortes.items():
    ds=todo.sel(time=slice(inicio,fim)).copy()
    ds.attrs['uso']=uso
    caminho=SAIDA/f'seas5_51_{uso}.nc'
    ds.to_netcdf(caminho,engine='netcdf4',encoding={'seas5_tp_media':{'zlib':True,'complevel':4}})
    arquivos[uso]=dict(arquivo=caminho.name,sha256=sha256(caminho),inicio=inicio,fim=fim,meses=ds.sizes['time'])
pd.DataFrame(linhas).to_csv(SAIDA/'auditoria_mensal.csv',index=False)
manifesto=dict(schema=SCHEMA,dataset=DATASET,system=SYSTEM,product_type='monthly_mean',
    variable='total_precipitation',leadtime_month=2,paramId=172228,area=AREA,
    criado_utc=datetime.now(timezone.utc).isoformat(),fonte=FONTE,arquivos=arquivos,
    requisicoes=fontes,observacoes_chuva_lidas=False,metricas_reserva_calculadas=False,
    unidades_saida='mm/day',fator_conversao=86400000.0,
    nota='Hindcasts retrospectivos; inicialização T−1 e verifyingMonth T. Sem climatologia/anomalia pronta do CDS.')
json_salvar(SAIDA/'seas5_51_manifesto.json',manifesto)
display(pd.DataFrame(linhas).head())
print('PRONTO:',SAIDA)
print('Salve esta versão. No notebook 02, anexe a saída desta versão em Add Input → Notebook.')
print('O notebook 02 precisa também da competição em Input; não precisa de Internet ou token.')
display(FileLink(str(SAIDA/'seas5_51_manifesto.json')))
