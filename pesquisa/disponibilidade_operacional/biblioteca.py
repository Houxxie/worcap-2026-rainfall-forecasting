from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter
import base64, gzip, io, json, hashlib, gc, platform, importlib.metadata
import numpy as np
import pandas as pd
import xarray as xr
import lightgbm as lgb
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from IPython.display import display, Markdown, Image, FileLink
DATASET, SYSTEM, SCHEMA = 'seasonal-monthly-single-levels', '51', 'worcap_seas5_v1'
PASTA_SEAS5_MANUAL = PASTA_CFSV2_MANUAL = None


PONTOS_POR_MES = 5000
ANOS_TREINO = 30
ARVORES = 300
SEMENTE = 42
CORTE_FINAL = '2022-12-01'
FIM_DESENVOLVIMENTO = '2020-12-01'
PARAMETROS = {'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.05, 'num_leaves': 31, 'min_data_in_leaf': 150, 'lambda_l2': 10.0, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'num_threads': 4, 'seed': SEMENTE, 'deterministic': True, 'force_col_wise': True, 'verbosity': -1}
BLOCOS = [{'nome': 'H1', 'corte': '2006-12-01', 'inicio': '2007-01-01'}, {'nome': 'H2', 'corte': '2008-12-01', 'inicio': '2009-01-01'}, {'nome': 'H3', 'corte': '2010-12-01', 'inicio': '2011-01-01'}, {'nome': 'H4', 'corte': '2012-12-01', 'inicio': '2013-01-01'}, {'nome': 'A', 'corte': '2014-12-01', 'inicio': '2015-01-01'}, {'nome': 'B', 'corte': '2016-12-01', 'inicio': '2017-01-01'}, {'nome': 'C', 'corte': '2018-12-01', 'inicio': '2019-01-01'}]
VARIAVEIS = ['cloud_cover', 'geopotential_850', 'rel_hum_850', 'shum_850', 'surface_pressure', 't2', 'temperature_850', 'u_850', 'v_850']
INDICES_NOMES = ['nino34', 'nino12', 'tna', 'tsa']
FONTES_INDICES = {'fonte': 'NOAA Physical Sciences Laboratory', 'captura_utc': '2026-09-21T23:00:40.933121+00:00', 'unidade': 'degC', 'tipo': 'indices mensais de anomalia de temperatura da superficie do mar', 'interpretacao_temporal': 'Versoes retrospectivas, com mes de observacao estritamente T-1; uso de dados externos confirmado pelo participante. Nao e arquivo de vintages em tempo real.', 'inicio': '1976-12-01', 'fim': '2024-11-01', 'meses': 576, 'colunas': ['time_origem', 'nino34', 'nino12', 'tna', 'tsa'], 'fontes': [{'indice': 'nino34', 'url': 'https://psl.noaa.gov/data/timeseries/month/data/nino34.long.anom.data', 'arquivo_bruto': 'nino34.data', 'sha256_bruto': '46e501b91f624ef85b525d73fbb427f9d04d30411ba6e5c7cff579b35fb4ff01', 'inicio_declarado': 1870, 'fim_declarado': 2026, 'rodape': '-99.99\n  NINA34\n 5N-5S 170W-120W \n HadISST \n  Anomaly from 1981-2010\n https://psl.noaa.gov/data/timeseries/month/\n  units=degC'}, {'indice': 'nino12', 'url': 'https://psl.noaa.gov/data/timeseries/month/data/nino12.long.anom.data', 'arquivo_bruto': 'nino12.data', 'sha256_bruto': 'b13ad5ad085cd13e70ec5d82662b509d522204e38a8eefd133f14be270e51928', 'inicio_declarado': 1870, 'fim_declarado': 2026, 'rodape': '-99.99\n  NINA12\n 0N-10S 270W-280E \n HadISST \n  Anomaly from 1981-2010\n https://psl.noaa.gov/data/timeseries/month/\n  units=degC'}, {'indice': 'tna', 'url': 'https://psl.noaa.gov/data/correlation/tna.data', 'arquivo_bruto': 'tna.data', 'sha256_bruto': '387c4125f377e57bcb683a849aefcd76da7d7f3cdcb2ac68ab086ceadf300e8b', 'inicio_declarado': 1948, 'fim_declarado': 2026, 'rodape': '-99.99\n  TNAa\n Tropical Northern Atlantic Index\n SST Anom 5.5N-23.5N; 15W-57.5W\n  Using HadISST1.1: Climo 1991-2020\n  NOAA/PSL\n  https://psl.noaa.gov/data/timeseries/month/'}, {'indice': 'tsa', 'url': 'https://psl.noaa.gov/data/correlation/tsa.data', 'arquivo_bruto': 'tsa.data', 'sha256_bruto': '700c0dffda0f19c1254ba482808b345eaa0538f5bf09fc3bc025f46a5fbe3a69', 'inicio_declarado': 1948, 'fim_declarado': 2026, 'rodape': '-99.99\n  TSAa\n  NOAA/PSL\n  https://psl.noaa.gov/data/timeseries/month/'}], 'sha256_csv': 'b6f444fbec48681c1e5b46a5c22aa197ad91c7fbbdaff82c3ed3a9c63f7c15a4', 'transformacao_no_modelo': 'Recentragem mensal em cada fold, usando somente as 360 origens de treino; sem padronizacao, medias moveis ou indices adicionais.'}
INDICES_CSV_GZIP_BASE64 = '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'
INDICES_CSV_SHA256 = 'b6f444fbec48681c1e5b46a5c22aa197ad91c7fbbdaff82c3ed3a9c63f7c15a4'
SCHEMA_CFSV2 = 'worcap_cfsv2_pentad_media_auditada_v1'
POLITICA_CFSV2 = 'ensemble_completo_ou_excecao_201908_m17_auditada_v2'
FONTE_CFSV2 = 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.NCEP-CFSv2/.HINDCAST/.PENTAD_SAMPLES_FULL/.prec/'
FEATURE_SEAS5 = 'seas5_tp_prevista_media_T_emitida_Tmenos1'
FEATURE_SEAS5_ANOM = 'seas5_tp_anomalia_mensal_treino_fold'
FEATURE_CFSV2 = 'cfsv2_tp_prevista_media_T_emitida_Tmenos1'
FEATURE_CFSV2_ANOM = 'cfsv2_tp_anomalia_mensal_treino_fold'
NOMES_MODELOS = ['arvores_seas5', 'arvores_multissistema']
FASES_CFSV2 = dict(desenvolvimento=('1982-02-01', '2020-12-01', 467), somente_ajuste_final=('2021-01-01', '2022-12-01', 24), teste=('2023-01-01', '2024-12-01', 24))
MOS_ALPHA = 0.1
MOS_FRACAO = 0.25

def exigir(condicao, mensagem):
    if not bool(condicao):
        raise ValueError(mensagem)

def sha256(arquivo):
    digest = hashlib.sha256()
    with Path(arquivo).open('rb') as stream:
        for bloco in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(bloco)
    return digest.hexdigest()

def salvar_json(nome, conteudo):
    caminho = SAIDA / nome
    caminho.write_text(json.dumps(conteudo, indent=2, ensure_ascii=False, default=str, allow_nan=False), encoding='utf-8')
    return caminho

def localizar_dados(pasta_manual=None):
    obrigatorios = ('treino_tp.nc', 'treino_tp_alvo.nc', 'teste_features.nc', 'sample_submission.csv')
    if pasta_manual is not None:
        candidatas = [Path(pasta_manual)]
    elif Path('/kaggle/input').is_dir():
        candidatas = sorted({p.parent for p in Path('/kaggle/input').rglob(obrigatorios[0])})
    else:
        candidatas = [Path.cwd() / 'data' / 'raw', Path.cwd().parent / 'data' / 'raw']
    validas = sorted({p.resolve() for p in candidatas if all(((p / nome).is_file() for nome in obrigatorios))})
    exigir(len(validas) == 1, f'Não foi encontrada uma única pasta com os quatro arquivos oficiais. Anexe a competição em Input > Add Input ou defina PASTA_DADOS_MANUAL. Pastas completas encontradas: {validas}')
    return validas[0]

def conferir_campo(campo, referencia, datas):
    exigir(set(campo.dims) == {'time', 'lat', 'lon'}, 'Dimensões atmosféricas inesperadas.')
    for eixo in ('lat', 'lon'):
        exigir(np.array_equal(campo[eixo].values, referencia[eixo].values), f'Grade atmosférica desalinhada: {eixo}.')
    exigir(pd.DatetimeIndex(campo.time.values).equals(pd.DatetimeIndex(datas)), 'Calendário atmosférico desalinhado.')

def ajustar_clima_atmosfera(campos, origens):
    origens = pd.DatetimeIndex(origens)
    resultado = {}
    for nome in VARIAVEIS:
        campo = campos[nome]
        indices = pd.DatetimeIndex(campo.time.values).get_indexer(origens)
        exigir((indices >= 0).all(), 'Meses de ajuste atmosférico ausentes.')
        mapas = []
        for mes in range(1, 13):
            ids = indices[origens.month == mes]
            exigir(len(ids) > 0, 'Mês sem histórico atmosférico.')
            valores = campo.values[ids]
            exigir(np.isfinite(valores).all(), f'{nome}: valores de treino não finitos.')
            mapas.append(valores.mean(axis=0, dtype=np.float64).astype(np.float32))
        resultado[nome] = np.stack(mapas)
    return resultado

def reconstruir_chuva(clima_tp, anomalias, peso):
    peso = float(peso)
    exigir(np.isfinite(peso) and peso >= 0, 'Peso deve ser finito e não negativo.')
    meses = pd.DatetimeIndex(anomalias.time.values).month.to_numpy() - 1
    valores = np.maximum(0, clima_tp.values[meses] + peso * anomalias.values)
    exigir(np.isfinite(valores).all(), 'Reconstrução da chuva produziu valores inválidos.')
    return xr.DataArray(valores.astype(np.float32), dims=anomalias.dims, coords=anomalias.coords, name='tp_mm_day', attrs={'units': 'mm/day'})

def auditar_chuva_desenv():
    with xr.open_dataset(PASTA / 'treino_tp.nc') as ds, xr.open_dataset(PASTA / 'treino_tp_alvo.nc') as da:
        bruto, alvo = (ds.tp, da.tp_alvo)
        esperado = pd.date_range('1940-01-01', CORTE_FINAL, freq='MS')
        exigir(set(bruto.dims) == set(alvo.dims) == {'time', 'lat', 'lon'}, 'Dimensões inesperadas da precipitação.')
        bruto, alvo = (bruto.transpose('time', 'lat', 'lon'), alvo.transpose('time', 'lat', 'lon'))
        exigir(bruto.shape == alvo.shape == (996, 301, 261), 'Grade/calendário inesperados.')
        exigir(pd.DatetimeIndex(bruto.time.values).equals(esperado), 'Calendário oficial divergente.')
        exigir(bruto.attrs.get('units') == alvo.attrs.get('units') == 'mm/day', 'Chuva deve estar em mm/day; nenhuma conversão será aplicada.')
        for eixo in ['time', 'lat', 'lon']:
            exigir(np.array_equal(bruto[eixo].values, alvo[eixo].values), f'Alvo desalinhado em {eixo}.')
        for eixo in ['lat', 'lon']:
            exigir(np.all(np.diff(bruto[eixo].values) == 0.25), f'Grade inválida: {eixo}.')
        chuva = bruto.sel(time=slice(None, FIM_DESENVOLVIMENTO)).load()
        exigir(np.isfinite(chuva.values).all() and (chuva.values >= 0).all(), 'Precipitação histórica inválida.')
        for i in range(0, chuva.sizes['time'] - 1, 24):
            j = min(i + 24, chuva.sizes['time'] - 1)
            exigir(np.array_equal(alvo.isel(time=slice(i, j)).values, chuva.values[i + 1:j + 1]), 'tp_alvo[M] não coincide com tp[M+1].')
    return chuva

def carregar_atmosfera(referencia, inicio, fim):
    resultado, metadados = ({}, [])
    esperado_completo = pd.date_range('1940-01-01', CORTE_FINAL, freq='MS')
    esperado_recorte = pd.date_range(inicio, fim, freq='MS')
    for nome in VARIAVEIS:
        with xr.open_dataset(PASTA / f'treino_{nome}.nc') as ds:
            bruto = ds[nome].transpose('time', 'lat', 'lon')
            conferir_campo(bruto, referencia, esperado_completo)
            campo = bruto.sel(time=slice(inicio, fim)).astype('float32').load()
            conferir_campo(campo, referencia, esperado_recorte)
            exigir(np.isfinite(campo.values).all(), f'{nome}: atmosfera não finita.')
            resultado[nome] = campo
            metadados.append(dict(variavel=nome, atributos=dict(bruto.attrs), inicio=str(inicio), fim=str(fim), minimo=float(campo.min()), maximo=float(campo.max())))
        print('Carregado:', nome, campo.shape, flush=True)
    return (resultado, metadados)

def carregar_indices_incorporados():
    dados = gzip.decompress(base64.b64decode(INDICES_CSV_GZIP_BASE64))
    exigir(hashlib.sha256(dados).hexdigest() == INDICES_CSV_SHA256, 'Hash das series NOAA incorporadas divergente.')
    tabela = pd.read_csv(io.BytesIO(dados), parse_dates=['time_origem'])
    exigir(tabela.columns.tolist() == ['time_origem'] + INDICES_NOMES, 'Colunas dos indices oceanicos divergentes.')
    tabela = tabela.set_index('time_origem')
    conferir_indices(tabela)
    esperado = pd.date_range(FONTES_INDICES['inicio'], FONTES_INDICES['fim'], freq='MS')
    exigir(tabela.index.equals(esperado), 'Snapshot NOAA tem lacunas ou datas alteradas.')
    return (tabela, dados)

def conferir_indices(tabela):
    exigir(tabela.columns.tolist() == INDICES_NOMES, 'Ordem dos quatro indices alterada.')
    datas = pd.DatetimeIndex(tabela.index)
    exigir(len(datas) > 0 and datas.is_unique and datas.is_monotonic_increasing, 'Datas NOAA vazias, duplicadas ou fora de ordem.')
    exigir(datas.equals(pd.date_range(datas.min(), datas.max(), freq='MS')), 'Calendario NOAA precisa ser mensal, continuo e no primeiro dia.')
    exigir(np.isfinite(tabela.to_numpy(dtype=np.float64)).all(), 'Indice NOAA ausente/infinito. Nao interpolar nem preencher com meses futuros.')

def auditar_dataset(ds, esperado=None):
    exigir(ds.attrs.get('schema') == SCHEMA, 'Versão de artefato SEAS5 desconhecida.')
    exigir(ds.attrs.get('dataset') == DATASET and str(ds.attrs.get('system')) == SYSTEM, 'Proveniência SEAS5 inválida.')
    exigir(ds.attrs.get('product_type') == 'monthly_mean' and int(ds.attrs.get('leadtime_month')) == 2, 'Produto ou lead incorreto.')
    exigir(ds.seas5_tp_media.attrs.get('units') == 'mm/day', 'SEAS5 deve estar em mm/day após conversão.')
    exigir(ds.seas5_tp_media.dims == ('time', 'lat', 'lon'), 'Dimensões SEAS5 inesperadas.')
    datas = pd.DatetimeIndex(ds.time.values)
    exigir(datas.is_unique and datas.is_monotonic_increasing, 'Alvos duplicados/desordenados.')
    exigir(datas.equals(pd.date_range(datas.min(), datas.max(), freq='MS')), 'Alvos SEAS5 com lacunas.')
    if esperado is not None:
        exigir(datas.equals(pd.DatetimeIndex(esperado)), 'Calendário SEAS5 não coincide com a requisição.')
    exigir(pd.DatetimeIndex(ds.time_origem.values).equals((datas.to_period('M') - 1).to_timestamp()), 'SEAS5 não está inicializado exatamente em T−1.')
    exigir(np.isin(ds.n_membros.values, [25, 51]).all(), 'Conjunto de membros incompleto.')
    for eixo in ['lat', 'lon']:
        v = ds[eixo].values
        exigir(len(v) > 1 and np.all(np.diff(v) == 1), f'Grade nativa de 1 grau incorreta: {eixo}.')
    exigir(ds.lat.min() <= -60 and ds.lat.max() >= 15 and (ds.lon.min() <= -90) and (ds.lon.max() >= -25), 'Recorte não cobre a grade oficial. Não extrapolar.')
    exigir(np.isfinite(ds.seas5_tp_media.values).all() and (ds.seas5_tp_media.values >= 0).all(), 'Valores SEAS5 inválidos.')

def localizar_seas5():
    if PASTA_SEAS5_MANUAL:
        candidatos = [Path(PASTA_SEAS5_MANUAL) / 'seas5_51_manifesto.json']
    else:
        candidatos = []
        for raiz in [Path('/kaggle/input'), Path('/kaggle/working/worcap_SEAS5_dados'), Path.cwd() / 'outputs/worcap_SEAS5_dados']:
            if raiz.is_dir():
                candidatos.extend(raiz.rglob('seas5_51_manifesto.json'))
    candidatos = sorted(set((p.resolve() for p in candidatos if p.is_file())))
    exigir(len(candidatos) == 1, 'Anexe o dataset preparado SEAS5 em Input, ou defina PASTA_SEAS5_MANUAL. É necessário encontrar exatamente um manifesto SEAS5.')
    caminho = candidatos[0]
    m = json.loads(caminho.read_text(encoding='utf-8'))
    exigir(m.get('schema') == SCHEMA and m.get('system') == SYSTEM and (m.get('dataset') == DATASET), 'Manifesto SEAS5 incompatível.')
    exigir(m.get('leadtime_month') == 2 and m.get('paramId') == 172228 and (m.get('product_type') == 'monthly_mean') and (m.get('variable') == 'total_precipitation'), 'Produto SEAS5 não corresponde ao experimento.')
    return (caminho.parent, m)

def carregar_seas5(uso, referencia):
    item = MANIFESTO_SEAS5['arquivos'][uso]
    caminho = PASTA_SEAS5 / item['arquivo']
    exigir(caminho.is_file() and sha256(caminho) == item['sha256'], f'SEAS5 ausente/alterado: {uso}.')
    with xr.open_dataset(caminho) as ds:
        bruto = ds.load()
    auditar_dataset(bruto, pd.date_range(item['inicio'], item['fim'], freq='MS'))
    exigir(bruto.attrs.get('uso') == uso, 'Uso do arquivo SEAS5 divergente.')
    grade = bruto.seas5_tp_media.interp(lat=referencia.lat, lon=referencia.lon, method='linear').astype(np.float32)
    exigir(np.isfinite(grade.values).all() and (grade.values >= 0).all(), 'Interpolação SEAS5 inválida.')
    for coord in ['lat', 'lon']:
        exigir(np.array_equal(grade[coord].values, referencia[coord].values), 'Grade SEAS5 desalinhada.')
    grade.attrs.update(units='mm/day', system=SYSTEM, leadtime_month=2)
    return grade

def valores_seas5(seas, destino, pontos=None):
    destino = pd.Timestamp(destino)
    exigir(pd.DatetimeIndex(seas.time.values).is_unique, 'Alvos SEAS5 duplicados.')
    i = pd.DatetimeIndex(seas.time.values).get_indexer([destino])[0]
    exigir(i >= 0, f'Previsão SEAS5 ausente para {destino.date()}.')
    origem = pd.Timestamp(seas.time_origem.values[i])
    exigir(origem == (destino.to_period('M') - 1).to_timestamp() and origem < destino, 'SEAS5 usa inicialização incorreta ou posterior ao limite.')
    v = seas.values[i].reshape(-1)
    if pontos is not None:
        v = v[pontos]
    exigir(np.isfinite(v).all(), 'Feature SEAS5 não finita.')
    return v

def amostrar_m5_documentado(chuva, atmosfera, seas, corte):
    destinos = calendario_pareado(corte)
    origens = (destinos.to_period('M') - 1).to_timestamp()
    ref_origens, ref_alvos = calendario_pares(atmosfera[VARIAVEIS[0]].time.values, corte, 30)
    clima, somas, contagens = estatisticas_climatologia(chuva, corte, 30)
    exigir(np.all(contagens == 30), 'Climatologia de chuva não tem exatamente 30 anos.')
    ca = ajustar_clima_atmosfera(atmosfera, ref_origens)
    ca['_clima_indices'] = ajustar_clima_indices(INDICES_OC, ref_origens, corte)
    exigir(set(destinos).issubset(set(ref_alvos)), 'Exemplo fora da janela de referência.')
    ngrade = chuva.sizes['lat'] * chuva.sizes['lon']
    n = min(PONTOS_POR_MES, ngrade)
    X = np.empty((len(destinos) * n, 28), dtype=np.float32)
    y = np.empty(len(X), dtype=np.float32)
    ids = np.empty((len(destinos), n), dtype=np.int32)
    for i, t in enumerate(destinos):
        pontos = np.random.default_rng(np.random.SeedSequence([SEMENTE, t.year, t.month])).choice(ngrade, size=n, replace=False)
        ids[i] = pontos
        basicas = matriz_mes(atmosfera, ca, clima, t, pontos)
        obs = chuva.sel(time=t).values.reshape(-1)[pontos]
        basicas[:, 22] = ((somas[t.month - 1].reshape(-1)[pontos] - obs.astype(np.float64)) / 29).astype(np.float32)
        sl = slice(i * n, (i + 1) * n)
        X[sl, :27] = basicas
        X[sl, 27] = valores_seas5(seas, t, pontos)
        y[sl] = obs - basicas[:, 22]
    exigir(np.isfinite(X).all() and np.isfinite(y).all(), 'Treino não finito.')
    exigir((origens < destinos).all() and destinos.max() <= pd.Timestamp(corte), 'Corte temporal violado.')
    return (X, y, ids, destinos, clima, ca, somas, contagens)

def valores_anomalia_seas5(seas, clima_seas, destino, pontos=None):
    for eixo in ['lat', 'lon']:
        exigir(np.array_equal(seas[eixo].values, clima_seas[eixo].values), 'Climatologia SEAS5 desalinhada.')
    bruto = valores_seas5(seas, destino, pontos)
    referencia = clima_seas.sel(month=pd.Timestamp(destino).month).values.reshape(-1)
    if pontos is not None:
        referencia = referencia[pontos]
    anomalia = (bruto - referencia).astype(np.float32)
    exigir(np.isfinite(anomalia).all(), 'Anomalia SEAS5 não finita.')
    return anomalia

def amostrar_arvores_seas5(chuva, atmosfera, seas, corte):
    x5, y, ids, destinos, clima, ca, somas, contagens = amostrar_m5_documentado(chuva, atmosfera, seas, corte)
    clima_seas = ajustar_clima_seas5(seas, corte)
    X = np.empty((len(x5), 29), dtype=np.float32)
    X[:, :28] = x5
    n = ids.shape[1]
    for i, t in enumerate(destinos):
        X[i * n:(i + 1) * n, 28] = valores_anomalia_seas5(seas, clima_seas, t, ids[i])
    exigir(np.array_equal(X[:, :28], x5), 'As 28 features do base_seas5 foram modificadas.')
    exigir(np.isfinite(X).all() and np.isfinite(y).all(), 'Treino não finito.')
    del x5
    return (X, y, ids, destinos, clima, ca, somas, contagens, clima_seas)

def validar_manifesto_cfsv2(m):
    exigir(m.get('schema') == SCHEMA_CFSV2 and m.get('modelo') == 'NCEP-CFSv2' and (m.get('fonte') == FONTE_CFSV2), 'Manifesto CFSv2 incompatível.')
    exigir(m.get('lead_iri') == 1.5 and m.get('unidade') == 'mm/day' and (m.get('chuva_observada_utilizada') is False), 'Produto ou temporalidade CFSv2 incorretos.')
    exigir(m.get('meses') == 515 and m.get('inicio_alvos') == '1982-02-01' and (m.get('fim_alvos') == '2024-12-01'), 'Cobertura CFSv2 divergente.')
    exigir(m.get('politica_membros') == POLITICA_CFSV2 and m.get('excecoes_permitidas') == {'2019-08-01': [17]}, 'Política de membros CFSv2 divergente.')
    aplicadas = m.get('excecoes_aplicadas', [])
    exigir(isinstance(aplicadas, list) and len(aplicadas) <= 1, 'Exceções CFSv2 não previstas.')
    if aplicadas:
        r = aplicadas[0]
        exigir(r.get('time_origem') == '2019-08-01' and r.get('time_alvo') == '2019-09-01' and (r.get('membros') == 23) and (r.get('membros_esperados') == 24) and (r.get('membros_indisponiveis_ids') == [17]), 'Exceção CFSv2 incorreta.')
        a = r.get('auditoria_excecao', {})
        exigir(a.get('membros_indisponiveis_ids') == [17] and np.isfinite(a.get('maximo_delta_media_mm_day', np.nan)) and (0 <= a['maximo_delta_media_mm_day'] <= 1e-05), 'Exceção sem média individual conferida.')
    for fase, (inicio, fim, n) in FASES_CFSV2.items():
        item = m.get('arquivos', {}).get(fase, {})
        exigir(item.get('arquivo') == f'cfsv2_{fase}.nc' and item.get('inicio_alvos') == inicio and (item.get('fim_alvos') == fim) and (item.get('meses') == n), 'Partição CFSv2 incorreta.')

def localizar_cfsv2():
    if PASTA_CFSV2_MANUAL:
        candidatos = [Path(PASTA_CFSV2_MANUAL) / 'cfsv2_manifesto.json']
    else:
        candidatos = []
        for raiz in [Path('/kaggle/input'), Path('/kaggle/working/worcap_CFSv2_dados'), Path.cwd() / 'outputs/worcap_CFSv2_dados']:
            if raiz.is_dir():
                candidatos.extend(raiz.rglob('cfsv2_manifesto.json'))
    candidatos = sorted(set((p.resolve() for p in candidatos if p.is_file())))
    exigir(len(candidatos) == 1, 'Anexe a saída concluída do notebook CFSv2 em Input > Notebook ou defina PASTA_CFSV2_MANUAL; necessário exatamente um manifesto CFSv2.')
    caminho = candidatos[0]
    m = json.loads(caminho.read_text(encoding='utf-8'))
    validar_manifesto_cfsv2(m)
    return (caminho.parent, m)

def auditar_cfsv2(ds, uso, m):
    inicio, fim, n = FASES_CFSV2[uso]
    datas = pd.date_range(inicio, fim, freq='MS')
    origens = (datas.to_period('M') - 1).to_timestamp()
    exigir(ds.attrs.get('schema') == SCHEMA_CFSV2 and ds.attrs.get('modelo') == 'NCEP-CFSv2' and (ds.attrs.get('fonte') == FONTE_CFSV2) and (ds.attrs.get('lead_iri') == 1.5), 'Proveniência do NetCDF CFSv2 divergente.')
    exigir(ds.attrs.get('politica_membros') == POLITICA_CFSV2, 'NetCDF CFSv2 sem política corrigida.')
    exigir(ds.cfsv2_tp_media.dims == ('time', 'lat', 'lon') and ds.cfsv2_tp_media.attrs.get('units') == 'mm/day', 'Dimensões/unidade CFSv2 inválidas.')
    exigir(pd.DatetimeIndex(ds.time.values).equals(datas), 'Datas CFSv2 faltantes, extras ou desordenadas.')
    exigir(pd.DatetimeIndex(ds.time_origem.values).equals(origens), 'CFSv2 não foi emitido em T-1.')
    exigir(np.array_equal(ds.lat.values, np.arange(-61, 17)) and np.array_equal(ds.lon.values, np.arange(-91, -23)), 'Grade nativa CFSv2 divergente.')
    for var in ['n_membros', 'inicializacao_mais_antiga', 'inicializacao_mais_recente', 'fase_fonte', 'time_origem']:
        exigir(ds[var].dims == ('time',), f'Coordenada CFSv2 inválida: {var}.')
    dmin = pd.DatetimeIndex(ds.inicializacao_mais_antiga.values)
    dmax = pd.DatetimeIndex(ds.inicializacao_mais_recente.values)
    exigir(not dmin.hasnans and (not dmax.hasnans) and (dmin <= dmax).all() and (dmax < datas).all() and (dmin >= origens - pd.Timedelta(days=40)).all() and (dmax <= origens + pd.Timedelta(days=7)).all(), 'Inicializações CFSv2 fora da janela permitida.')
    esperados = np.where(origens.month == 11, 28, 24)
    if m['excecoes_aplicadas'] and pd.Timestamp('2019-09-01') in datas:
        esperados[datas == pd.Timestamp('2019-09-01')] = 23
    exigir(np.array_equal(ds.n_membros.values, esperados), 'Número de membros CFSv2 difere do manifesto.')
    exigir(np.array_equal(ds.fase_fonte.values, (origens >= pd.Timestamp('2011-04-01')).astype(np.int8)), 'Transição hindcast/operacional CFSv2 incorreta.')
    v = ds.cfsv2_tp_media.values
    exigir(np.isfinite(v).all() and (v >= 0).all(), 'Precipitação prevista CFSv2 inválida.')

def carregar_cfsv2(uso, referencia):
    item = MANIFESTO_CFSV2['arquivos'][uso]
    path = PASTA_CFSV2 / item['arquivo']
    exigir(path.is_file() and sha256(path) == item['sha256'], f'CFSv2 ausente/alterado: {uso}.')
    with xr.open_dataset(path) as d:
        bruto = d.load()
    auditar_cfsv2(bruto, uso, MANIFESTO_CFSV2)
    grade = bruto.cfsv2_tp_media.interp(lat=referencia.lat, lon=referencia.lon, method='linear').astype(np.float32)
    exigir(np.isfinite(grade.values).all() and (grade.values >= 0).all(), 'Interpolação CFSv2 inválida; não extrapolar.')
    for eixo in ['lat', 'lon']:
        exigir(np.array_equal(grade[eixo].values, referencia[eixo].values), 'Grade CFSv2 desalinhada.')
    for coord in ['time_origem', 'n_membros', 'inicializacao_mais_antiga', 'inicializacao_mais_recente', 'fase_fonte']:
        grade = grade.assign_coords({coord: ('time', bruto[coord].values)})
    grade.attrs.update(units='mm/day', lead_iri=1.5)
    salvar_json(f'auditoria_CFSv2_{uso}.json', dict(arquivo=str(path), sha256=item['sha256'], meses=bruto.sizes['time'], minimo=float(grade.min()), maximo=float(grade.max()), membros_usados=np.unique(bruto.n_membros.values).tolist(), interpolacao='bilinear sem extrapolacao'))
    return grade

def ajustar_clima_previsao(previsoes, corte, nome):
    treino = calendario_pareado(corte)
    datas = pd.DatetimeIndex(previsoes.time.values)
    exigir(datas.is_unique and datas.is_monotonic_increasing and treino.isin(datas).all(), f'Calendário incompleto/desordenado: {nome}.')
    sel = previsoes.sel(time=treino)
    exigir(pd.DatetimeIndex(sel.time_origem.values).equals((treino.to_period('M') - 1).to_timestamp()), f'Origem incorreta na climatologia {nome}.')
    v = sel.values
    exigir(np.isfinite(v).all() and (v >= 0).all(), f'Valores inválidos na climatologia {nome}.')
    n = np.array([(treino.month == m).sum() for m in range(1, 13)], dtype=np.int16)
    exigir((n > 0).all() and n.max() <= 30, 'Climatologia de previsão excede 30 anos ou mês ausente.')
    medias = np.stack([v[treino.month == m].mean(axis=0, dtype=np.float64) for m in range(1, 13)]).astype(np.float32)
    return xr.DataArray(medias, dims=('month', 'lat', 'lon'), coords={'month': np.arange(1, 13), 'lat': previsoes.lat, 'lon': previsoes.lon, 'n_meses': ('month', n)}, name=f'climatologia_{nome}', attrs=dict(units='mm/day', inicio=str(treino[0].date()), corte=str(pd.Timestamp(corte).date()), meses_treino=len(treino), anos_min=int(n.min()), anos_max=int(n.max()), referencia='somente preditores nos meses do treino pareado; uma media de ensemble por ano/mes'))

def ajustar_clima_seas5(seas, corte):
    return ajustar_clima_previsao(seas, corte, 'SEAS5')

def valores_cfsv2(cfs, destino, pontos=None):
    destino = pd.Timestamp(destino)
    exigir(pd.Timestamp(cfs.inicializacao_mais_recente.sel(time=destino).values) < destino, 'Inicialização real CFSv2 invade o alvo.')
    return valores_seas5(cfs, destino, pontos)

def valores_anomalia_cfsv2(cfs, clima_cfs, destino, pontos=None):
    for eixo in ['lat', 'lon']:
        exigir(np.array_equal(cfs[eixo], clima_cfs[eixo]), 'Climatologia CFSv2 desalinhada.')
    bruto = valores_cfsv2(cfs, destino, pontos)
    ref = clima_cfs.sel(month=pd.Timestamp(destino).month).values.ravel()
    if pontos is not None:
        ref = ref[pontos]
    anom = (bruto - ref).astype(np.float32)
    exigir(np.isfinite(anom).all(), 'Anomalia CFSv2 inválida.')
    return anom

def amostrar_arvores_multissistema(chuva, atmosfera, seas, cfs, corte):
    x6, y, ids, destinos, clima, ca, somas, contagens, cs = amostrar_arvores_seas5(chuva, atmosfera, seas, corte)
    cc = ajustar_clima_previsao(cfs, corte, 'CFSv2')
    X = np.empty((len(x6), 31), dtype=np.float32)
    X[:, :29] = x6
    n = ids.shape[1]
    for i, t in enumerate(destinos):
        sl = slice(i * n, (i + 1) * n)
        X[sl, 29] = valores_cfsv2(cfs, t, ids[i])
        X[sl, 30] = valores_anomalia_cfsv2(cfs, cc, t, ids[i])
    exigir(np.array_equal(X[:, :29], x6), '29 features do controle alteradas.')
    exigir(np.isfinite(X).all() and np.isfinite(y).all(), 'Matriz de treino inválida.')
    return (X, y, ids, destinos, clima, ca, somas, contagens, cs, cc)

def prever_par(modelo, nome, atmosfera, seas, cfs, ca, clima, cs, cc, destinos):
    destinos = pd.DatetimeIndex(destinos)
    exigir(nome in NOMES_MODELOS, 'Modelo desconhecido.')
    exigir((destinos > pd.Timestamp(cs.attrs['corte'])).all() and cs.attrs['corte'] == cc.attrs['corte'], 'Inferência fora do período posterior ao corte.')
    out = np.empty((len(destinos), clima.sizes['lat'], clima.sizes['lon']), dtype=np.float32)
    for i, t in enumerate(destinos):
        x = np.column_stack([matriz_mes(atmosfera, ca, clima, t), valores_seas5(seas, t), valores_anomalia_seas5(seas, cs, t)]).astype(np.float32)
        if nome == 'arvores_multissistema':
            x = np.column_stack([x, valores_cfsv2(cfs, t), valores_anomalia_cfsv2(cfs, cc, t)]).astype(np.float32)
        exigir(x.shape[1] == (29 if nome == 'arvores_seas5' else 31) and np.isfinite(x).all(), 'Inferência inválida.')
        out[i] = modelo.predict(x).reshape(out.shape[1:])
    exigir(np.isfinite(out).all(), 'Previsões inválidas.')
    return xr.DataArray(out, dims=('time', 'lat', 'lon'), coords={'time': destinos, 'lat': clima.lat, 'lon': clima.lon}, attrs={'units': 'mm/day'})

def mos_fit(chuva, seas, cfs, corte):
    dates = calendario_pareado(corte)
    r = chuva.sel(time=dates).transpose('time', 'lat', 'lon').values.reshape(len(dates), -1)
    s = seas.sel(time=dates).transpose('time', 'lat', 'lon').values.reshape(len(dates), -1)
    c = cfs.sel(time=dates).transpose('time', 'lat', 'lon').values.reshape(len(dates), -1)
    for pre in [seas, cfs]:
        assert pd.DatetimeIndex(pre.sel(time=dates).time_origem.values).equals((dates.to_period('M') - 1).to_timestamp())
    assert (pd.DatetimeIndex(cfs.sel(time=dates).inicializacao_mais_recente.values) < dates).all()
    baseline = estatisticas_climatologia(chuva, corte, 30)[0]
    shape = baseline.shape[1:]
    ng = r.shape[1]
    coef = np.zeros((2, ng), np.float64)
    mu = np.zeros((2, 12, ng), np.float64)
    for lo in range(0, ng, 4096):
        hi = min(lo + 4096, ng)
        y = r[:, lo:hi].astype(np.float64)
        x = s[:, lo:hi].astype(np.float64)
        z = c[:, lo:hi].astype(np.float64)
        assert np.isfinite(y).all() and np.isfinite(x).all() and np.isfinite(z).all()
        for m in range(1, 13):
            ix = dates.month == m
            mu[0, m - 1, lo:hi] = x[ix].mean(0)
            mu[1, m - 1, lo:hi] = z[ix].mean(0)
            x[ix] -= mu[0, m - 1, lo:hi]
            z[ix] -= mu[1, m - 1, lo:hi]
            y[ix] -= y[ix].mean(0)
        sx = np.sqrt(np.mean(x * x, axis=0))
        sz = np.sqrt(np.mean(z * z, axis=0))
        sx = np.where(sx > 1e-12, sx, 1.0)
        sz = np.where(sz > 1e-12, sz, 1.0)
        x /= sx
        z /= sz
        a = np.mean(x * x, axis=0) + MOS_ALPHA
        b = np.mean(x * z, axis=0)
        d = np.mean(z * z, axis=0) + MOS_ALPHA
        u = np.mean(x * y, axis=0)
        v = np.mean(z * y, axis=0)
        det = a * d - b * b
        assert (det > 0).all()
        coef[0, lo:hi] = (d * u - b * v) / det / sx
        coef[1, lo:hi] = (a * v - b * u) / det / sz
    assert np.isfinite(coef).all()
    return dict(coef=coef.reshape(2, *shape), mu=mu.reshape(2, 12, *shape), baseline=baseline.values, lat=baseline.lat.values, lon=baseline.lon.values, corte=str(pd.Timestamp(corte).date()), inicio=str(dates[0].date()), meses=len(dates))

def mos_predict(fit, seas, cfs, dates):
    dates = pd.DatetimeIndex(dates)
    assert (dates > pd.Timestamp(fit['corte'])).all()
    for pre in [seas, cfs]:
        assert np.array_equal(pre.lat, fit['lat']) and np.array_equal(pre.lon, fit['lon'])
    out = []
    for t in dates:
        s = valores_seas5(seas, t).reshape(fit['baseline'].shape[1:]).astype(np.float64)
        c = valores_cfsv2(cfs, t).reshape(s.shape).astype(np.float64)
        anom = fit['coef'][0] * (s - fit['mu'][0, t.month - 1]) + fit['coef'][1] * (c - fit['mu'][1, t.month - 1])
        out.append(np.maximum(0, fit['baseline'][t.month - 1].astype(np.float64) + anom))
    return np.stack(out)

def matriz_mes(campos, clima_atmos, clima_tp, destino, pontos=None):
    """
    Atmosfera T−4; índices T−3; meses reais preservados.

    A climatologia da chuva é integral por padrão.
    Somente amostrar_treino substitui sua coluna pelo leave-one-out.
    """
    destino = pd.Timestamp(destino)
    origem = origem_mensal([destino], LAG_ATMOSFERA)[0]
    nlat = clima_tp.sizes['lat']
    nlon = clima_tp.sizes['lon']
    pontos = np.arange(nlat * nlon) if pontos is None else np.asarray(pontos)
    exigir(pontos.ndim == 1 and np.issubdtype(pontos.dtype, np.integer), 'Índices inválidos.')
    exigir(((pontos >= 0) & (pontos < nlat * nlon)).all(), 'Ponto fora da grade.')
    X = np.empty((len(pontos), len(FEATURES)), dtype=np.float32)
    for j, nome in enumerate(VARIAVEIS):
        campo = campos[nome]
        i = pd.DatetimeIndex(campo.time.values).get_indexer([origem])[0]
        exigir(i >= 0, f'Origem {origem.date()} ausente em {nome}.')
        valores = campo.values[i].reshape(-1)[pontos]
        clima_origem = clima_atmos[nome][origem.month - 1].reshape(-1)[pontos]
        X[:, j] = valores
        X[:, j + len(VARIAVEIS)] = valores - clima_origem
    X[:, 18] = clima_tp.lat.values[pontos // nlon]
    X[:, 19] = clima_tp.lon.values[pontos % nlon]
    X[:, 20] = np.sin(2 * np.pi * destino.month / 12)
    X[:, 21] = np.cos(2 * np.pi * destino.month / 12)
    X[:, 22] = clima_tp.values[destino.month - 1].reshape(-1)[pontos]
    exigir(origem < destino, 'Violação temporal: origem não é anterior ao alvo.')
    valores_oceano = valores_indices_mes(INDICES_OC, clima_atmos['_clima_indices'], destino)
    X[:, 23:] = valores_oceano[None, :]
    exigir(np.isfinite(X).all(), 'Atributos contêm valores ausentes ou infinitos.')
    return X

SST_COMPONENTES = 8
SST_NOME = 'ersstv5_4graus_198201_202411.nc'
SST_HASH = '908034f6418833302ea432f25850ad2982d1114b747b7b298e26a350eaf2891d'

def sst_auditar_arquivo(path):
    with xr.open_dataset(path) as ds:
        exigir('sst' in ds and ds.sst.dims==('time','lat','lon'),'SST: variável/dimensões incorretas.')
        exigir(ds.sst.attrs.get('units')=='degC','SST deve estar em graus Celsius.')
        exigir('ERSSTv5' in str(ds.attrs.get('title','')) and
               str(ds.attrs.get('product_version',''))=='Version 5','Produto diferente de ERSSTv5.')
        t=pd.DatetimeIndex(ds.time.values)
        exigir(t.equals(pd.date_range('1982-01-01','2024-11-01',freq='MS')),'SST: calendário incompleto ou diferente.')
        exigir(np.array_equal(ds.lat.values,np.arange(60,-61,-4)) and
               np.array_equal(ds.lon.values,np.arange(0,360,4)),'SST: esperado recorte 60S–60N em grade de 4 graus.')
        sst=ds.sst.astype('float32').load()
        atributos={k:str(v) for k,v in ds.attrs.items()}
    v=sst.values;validos=v[np.isfinite(v)]
    exigir(validos.size>0 and not np.isinf(v).any() and validos.min()>=-1.801 and validos.max()<=45,
           'SST: valores fora da faixa física do produto.')
    # Os NaNs terrestres serão excluídos por máscara ajustada apenas no treino de cada fold.
    return sst,dict(arquivo=Path(path).name,sha256=sha256(path),bytes=Path(path).stat().st_size,
        inicio_origens=str(t[0].date()),fim_origens=str(t[-1].date()),meses=len(t),grade=[31,90],
        minimo=float(validos.min()),maximo=float(validos.max()),atributos=atributos,
        referencia='https://psl.noaa.gov/data/gridded/data.noaa.ersst.v5.html',
        origem='NOAA PSL / ERSSTv5; snapshot retrospectivo, sem garantia de versão operacional histórica')

def sst_ajustar_pca(sst,alvos_treino):
    alvos_treino=pd.DatetimeIndex(alvos_treino)
    exigir(alvos_treino.is_unique and alvos_treino.equals(pd.date_range(alvos_treino.min(),alvos_treino.max(),freq='MS')),
           'Calendário de ajuste da PCA inválido.')
    origens=origem_mensal(alvos_treino, LAG_SST)
    treino=sst.sel(time=origens).transpose('time','lat','lon')
    bruto=treino.values.reshape(len(origens),-1).astype(np.float64)
    mascara=np.isfinite(bruto).all(axis=0)
    exigir(mascara.any(),'Nenhum ponto SST oceânico completo no treino.')
    mensal=np.full((12,bruto.shape[1]),np.nan,dtype=np.float64)
    contagens=np.array([(origens.month==m).sum() for m in range(1,13)])
    exigir((contagens>=2).all(),'Climatologia SST mensal insuficiente.')
    for m in range(1,13):mensal[m-1,mascara]=bruto[origens.month==m][:,mascara].mean(axis=0)
    anom=bruto[:,mascara]-mensal[origens.month-1][:,mascara]
    ativos=np.flatnonzero(mascara)
    mascara[ativos[np.std(anom,axis=0)<=1e-8]]=False
    latitudes=np.repeat(sst.lat.values,sst.sizes['lon'])
    pesos=np.sqrt(np.cos(np.deg2rad(latitudes[mascara]))).astype(np.float64)
    exigir(int(mascara.sum())>SST_COMPONENTES and len(origens)>SST_COMPONENTES,'PCA sem dimensões suficientes.')
    matriz=(bruto[:,mascara]-mensal[origens.month-1][:,mascara])*pesos
    exigir(np.isfinite(matriz).all(),'Matriz de PCA inválida.')
    with threadpool_limits(limits=1):
        pca=PCA(n_components=SST_COMPONENTES,svd_solver='full').fit(matriz)
    return dict(mascara=mascara,clima_mensal=mensal[:,mascara],pesos_area=pesos,
        centro=pca.mean_,componentes=pca.components_,variancia_explicada=pca.explained_variance_ratio_,
        contagens_mensais=contagens,origens_treino=origens.values,lat=sst.lat.values.copy(),lon=sst.lon.values.copy())

def sst_transformar(sst,estado,alvos):
    alvos=pd.DatetimeIndex(alvos);origens=origem_mensal(alvos, LAG_SST)
    exigir(np.array_equal(sst.lat,estado['lat']) and np.array_equal(sst.lon,estado['lon']),'Grade SST mudou.')
    v=sst.sel(time=origens).values.reshape(len(origens),-1).astype(np.float64)[:,estado['mascara']]
    exigir(np.isfinite(v).all(),'SST ausente no oceano selecionado pelo treino; não imputar usando futuro.')
    x=(v-estado['clima_mensal'][origens.month-1])*estado['pesos_area']-estado['centro']
    with threadpool_limits(limits=1):scores=x@estado['componentes'].T
    exigir(scores.shape==(len(alvos),SST_COMPONENTES) and np.isfinite(scores).all(),'Scores PCA inválidos.')
    return scores.astype(np.float32)

def pesquisa_prever_sst(modelo, nome, atmosfera, seas, cfs, ca, clima, cs, cc, datas, scores):
    datas = pd.DatetimeIndex(datas)
    exigir((datas > pd.Timestamp(cs.attrs['corte'])).all(), 'Inferência invade treino.')
    exigir(scores.shape == (len(datas), 8), 'Scores SST desalinhados.')
    out = np.empty((len(datas), clima.sizes['lat'], clima.sizes['lon']), dtype=np.float32)
    expected = (FEATURES_SEAS5 if nome == 'arvores_seas5' else FEATURES_MULTISSISTEMA) + SST_FEATURES
    exigir(modelo.feature_name() == expected, 'Ordem de features SST incorreta.')
    for i, t in enumerate(datas):
        x = np.column_stack([matriz_mes(atmosfera, ca, clima, t),
            valores_seas5(seas, t), valores_anomalia_seas5(seas, cs, t)]).astype(np.float32)
        if nome == 'arvores_multissistema':
            x = np.column_stack([x, valores_cfsv2(cfs, t), valores_anomalia_cfsv2(cfs, cc, t)]).astype(np.float32)
        x = np.column_stack([x, np.broadcast_to(scores[i], (len(x), 8))]).astype(np.float32)
        exigir(x.shape[1] == len(expected) and np.isfinite(x).all(), 'Features SST inválidas.')
        out[i] = modelo.predict(x).reshape(out.shape[1:])
    return xr.DataArray(out, dims=('time', 'lat', 'lon'),
        coords=dict(time=datas, lat=clima.lat, lon=clima.lon), attrs=dict(units='mm/day'))

def pesquisa_diagnosticos(obs, previsoes, bloco):
    """Dados completos, sem amostragem de pixels e sem mascarar erros."""
    linhas = []
    lat = obs.lat.values
    regioes = {'dominio_inteiro': np.ones(lat.size, dtype=bool),
        'lat_menor_que_menos35': lat < -35,
        'lat_menos35_a_menos15': (lat >= -35) & (lat < -15),
        'lat_maior_igual_menos15': lat >= -15}
    for modelo, pred in previsoes.items():
        obs, pred = xr.align(obs, pred, join='exact')
        for i, t in enumerate(pd.DatetimeIndex(obs.time.values)):
            erro = pred.values[i].astype(np.float64) - obs.values[i].astype(np.float64)
            exigir(np.isfinite(erro).all(), 'Erro não finito nos diagnósticos.')
            for regiao, mask in regioes.items():
                if not mask.any():
                    continue
                e = erro[mask]
                w = np.broadcast_to(np.cos(np.deg2rad(lat[mask]))[:, None], e.shape)
                linhas.append(dict(modelo=modelo, bloco=bloco, mes_alvo=str(t.date()),
                    ano=t.year, mes=t.month, regiao=regiao, n=e.size,
                    sse=float(np.square(e).sum()), soma_erro=float(e.sum()),
                    soma_erro_absoluto=float(np.abs(e).sum()),
                    sse_area=float((w * np.square(e)).sum()), peso_area=float(w.sum())))
    return pd.DataFrame(linhas)

def pesquisa_resumir(mensais, grupos):
    cols = ['n', 'sse', 'soma_erro', 'soma_erro_absoluto', 'sse_area', 'peso_area']
    r = mensais.groupby(grupos, as_index=False)[cols].sum()
    r['rmse'] = np.sqrt(r.sse / r.n)
    r['mae'] = r.soma_erro_absoluto / r.n
    r['vies'] = r.soma_erro / r.n
    r['rmse_area'] = np.sqrt(r.sse_area / r.peso_area)
    return r

"""Política temporal candidata; não certifica vintages nem datas de publicação."""
LAG_ATMOSFERA = 4
LAG_INDICES = 3
LAG_SST = 2
LAG_CHUVA_TREINO = 4
INICIO_COMUM = pd.Timestamp('1982-03-01')
FEATURES = ([v + '_lag4' for v in VARIAVEIS]
    + [v + '_lag4_anomalia' for v in VARIAVEIS]
    + ['latitude', 'longitude', 'mes_alvo_sin', 'mes_alvo_cos', 'clima_tp_alvo']
    + [n + '_lag3_anomalia_fold' for n in INDICES_NOMES])
FEATURES_SEAS5 = FEATURES + [FEATURE_SEAS5, FEATURE_SEAS5_ANOM]
FEATURES_MULTISSISTEMA = FEATURES_SEAS5 + [FEATURE_CFSV2, FEATURE_CFSV2_ANOM]
SST_FEATURES = [f'sst_lag2_pc_{i:02d}_treino_fold' for i in range(1, 9)]


def origem_mensal(alvos, lag):
    datas = pd.DatetimeIndex(alvos)
    exigir(lag >= 1 and datas.equals(datas.to_period('M').to_timestamp()), 'Datas/lag inválidos.')
    return (datas.to_period('M') - lag).to_timestamp()


def janela_referencia(corte, anos=30):
    corte = pd.Timestamp(corte)
    exigir(corte.day == 1, 'Corte precisa identificar um mês.')
    return pd.date_range(end=corte, periods=12 * anos, freq='MS')


def calendario_pareado(corte):
    datas = janela_referencia(corte)
    datas = datas[datas >= INICIO_COMUM]
    exigir(24 <= len(datas) <= 360, 'Janela comum inválida.')
    return datas


def calendario_pares(datas_disponiveis, corte, anos=30):
    alvos = janela_referencia(corte, anos)
    origens = origem_mensal(alvos, LAG_ATMOSFERA)
    disponiveis = pd.DatetimeIndex(datas_disponiveis)
    exigir(disponiveis.is_unique and origens.isin(disponiveis).all(), 'Faltam origens atmosféricas.')
    return origens, alvos


def estatisticas_climatologia(tp, corte, anos=30):
    datas = janela_referencia(corte, anos)
    treino = tp.sel(time=datas).transpose('time', 'lat', 'lon')
    exigir(np.isfinite(treino.values).all(), 'Chuva de treino incompleta.')
    somas = np.stack([treino.values[datas.month == m].sum(axis=0, dtype=np.float64)
                      for m in range(1, 13)])
    contagens = np.array([(datas.month == m).sum() for m in range(1, 13)], dtype=np.int64)
    exigir((contagens == anos).all(), 'Climatologia precisa de todos os meses.')
    clima = xr.DataArray((somas / contagens[:, None, None]).astype(np.float32),
        dims=('month', 'lat', 'lon'), coords=dict(month=np.arange(1, 13), lat=tp.lat, lon=tp.lon),
        attrs=dict(units='mm/day', ajuste_inicio=str(datas[0].date()), ajuste_fim=str(datas[-1].date())))
    return clima, somas, contagens


def ajustar_clima_indices(tabela, origens_atmosfera, corte):
    # Calendário próprio: índices T−3, enquanto atmosfera utiliza T−4.
    alvos = janela_referencia(corte)
    exigir(pd.DatetimeIndex(origens_atmosfera).equals(origem_mensal(alvos, LAG_ATMOSFERA)),
           'Referência atmosférica fora do protocolo.')
    origens = origem_mensal(alvos, LAG_INDICES)
    valores = tabela.loc[origens, INDICES_NOMES].to_numpy(dtype=np.float64)
    exigir(np.isfinite(valores).all(), 'Índices de treino incompletos.')
    return np.stack([valores[origens.month == m].mean(axis=0, dtype=np.float64) for m in range(1, 13)])


def valores_indices_mes(tabela, clima_indices, destino):
    origem = origem_mensal([destino], LAG_INDICES)[0]
    valores = tabela.loc[origem, INDICES_NOMES].to_numpy(dtype=np.float64)
    exigir(np.isfinite(valores).all(), 'Índice indisponível; não preencher.')
    return (valores - clima_indices[origem.month - 1]).astype(np.float32)


def exigir_disponibilidade_real(registros, emitido_em, fontes_necessarias):
    """Porta para futura produção; não é usada para certificar este retrospectivo.

    O registro precisa se referir aos bytes recebidos, não à data nominal da previsão.
    Valida metadados fornecidos; sua autenticidade exige um coletor auditável.
    """
    emissao = pd.Timestamp(emitido_em)
    exigir(emissao.tzinfo is not None, 'Emissão exige timezone.')
    exigir(len(registros) == len(fontes_necessarias), 'Quantidade de fontes incorreta.')
    exigir({r.get('fonte') for r in registros} == set(fontes_necessarias), 'Fonte faltante/duplicada.')
    for r in registros:
        for chave in ['disponivel_em', 'recebido_em', 'url', 'sha256', 'versao']:
            exigir(bool(r.get(chave)), 'Disponibilidade desconhecida: ' + r.get('fonte', '?') + '/' + chave)
        recebido, publicado = pd.Timestamp(r['recebido_em']), pd.Timestamp(r['disponivel_em'])
        exigir(recebido.tzinfo is not None and publicado.tzinfo is not None, 'Datas exigem timezone.')
        exigir(publicado <= recebido <= emissao, 'Fonte chegou após a emissão ou cronologia inválida.')
        exigir(len(r['sha256']) == 64 and set(r['sha256']) <= set('0123456789abcdef'), 'SHA-256 inválido.')
    return True


def calendario_emissoes(datas, corte):
    datas = pd.DatetimeIndex(datas)
    limite = origem_mensal([datas[0]], LAG_CHUVA_TREINO)[0]
    exigir(pd.Timestamp(corte) <= limite, 'Chuva de treino recente demais para a primeira emissão.')
    return pd.DataFrame(dict(time_alvo=datas, limite_emissao=datas - pd.Timedelta(nanoseconds=1),
        atmosfera=origem_mensal(datas, LAG_ATMOSFERA), indices=origem_mensal(datas, LAG_INDICES),
        sst=origem_mensal(datas, LAG_SST), inicializacao_sazonal=origem_mensal(datas, 1),
        ultimo_alvo_treino=pd.Timestamp(corte), disponibilidade_historica_comprovada=False))


def auditar_invariancia_defasada(chuva, atmosfera, seas, cfs, sst, corte, datas, clima, ca, cs, cc, pca):
    """Muta cópias pequenas reais; dados após cada limite não podem afetar T."""
    primeiro = pd.Timestamp(datas[0])
    pontos = np.array([0, 1, chuva.sizes['lon']], dtype=np.int32)
    # Recorte espacial pequeno mantém todas as datas para testar os ajustes.
    ch = chuva.isel(lat=slice(0, 2), lon=slice(0, 2)).copy(deep=True)
    at = {k: v.isel(lat=slice(0, 2), lon=slice(0, 2)).copy(deep=True) for k, v in atmosfera.items()}
    alvos_ref = janela_referencia(corte)
    orig_at = origem_mensal(alvos_ref, LAG_ATMOSFERA)
    c0, _, _ = estatisticas_climatologia(ch, corte)
    a0 = ajustar_clima_atmosfera(at, orig_at)
    a0['_clima_indices'] = ajustar_clima_indices(INDICES_OC, orig_at, corte)
    x0 = matriz_mes(at, a0, c0, primeiro)
    ch.values[pd.DatetimeIndex(ch.time.values) > pd.Timestamp(corte)] = 123456
    c1, _, _ = estatisticas_climatologia(ch, corte)
    exigir(np.array_equal(c0.values, c1.values), 'Chuva após corte alterou climatologia.')
    for v in at.values():
        v.values[pd.DatetimeIndex(v.time.values) > origem_mensal([primeiro], LAG_ATMOSFERA)[0]] = 7654321
    a1 = ajustar_clima_atmosfera(at, orig_at)
    a1['_clima_indices'] = a0['_clima_indices']
    exigir(all(np.array_equal(a0[k], a1[k]) for k in a0), 'Futuro alterou climatologia atmosférica.')
    exigir(np.array_equal(x0, matriz_mes(at, a1, c1, primeiro)), 'Atmosfera recente/futura alterou T.')
    ni = INDICES_OC.copy(deep=True)
    ni.loc[ni.index > origem_mensal([primeiro], LAG_INDICES)[0], :] = 99999
    ci = ajustar_clima_indices(ni, orig_at, corte)
    exigir(np.array_equal(ci, a0['_clima_indices']), 'Índices futuros alteraram ajuste.')
    exigir(np.array_equal(valores_indices_mes(ni, ci, primeiro), x0[0, 23:]), 'Índice futuro alterou feature.')
    for pre, nome in [(seas, 'SEAS5'), (cfs, 'CFSv2')]:
        pequeno = pre.isel(lat=slice(0, 2), lon=slice(0, 2)).copy(deep=True)
        antes = ajustar_clima_previsao(pequeno, corte, nome)
        pequeno.values[pd.DatetimeIndex(pequeno.time.values) > pd.Timestamp(corte)] = 12345
        exigir(np.array_equal(antes, ajustar_clima_previsao(pequeno, corte, nome)), 'Futuro alterou clima sazonal.')
    copia = sst.copy(deep=True)
    copia.values[pd.DatetimeIndex(copia.time.values) > origem_mensal([primeiro], LAG_SST)[0]] = 12345
    novo = sst_ajustar_pca(copia, calendario_pareado(corte))
    for k in ['mascara', 'clima_mensal', 'centro', 'componentes', 'origens_treino']:
        exigir(np.array_equal(novo[k], pca[k]), 'SST futura alterou PCA: ' + k)
    exigir(np.array_equal(sst_transformar(copia, novo, [primeiro]), sst_transformar(sst, pca, [primeiro])),
           'SST mais recente que T−2 alterou previsão de T.')
    return dict(chuva_pos_corte_invariante=True, atmosfera_apos_Tmenos4_invariante=True,
        indices_apos_Tmenos3_invariantes=True, sst_apos_Tmenos2_invariante=True,
        climas_sazonais_somente_treino=True, publicacao_historica_comprovada=False)

