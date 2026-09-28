# %% Configuração M4 preservada e snapshot NOAA incorporado
from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter
from itertools import zip_longest
import gc
import base64
import gzip
import io
import hashlib
import importlib.metadata
import json
import platform
import numpy as np
import pandas as pd
import xarray as xr
import lightgbm as lgb
import matplotlib.pyplot as plt
from IPython.display import display, FileLink

PASTA_DADOS_MANUAL = None
# Deixe None para encontrar as mesmas saídas 8B/8C usadas na 8D.
CAP_PASTA_8B_MANUAL = None
CAP_CSV_8C_MANUAL = None
# Em caso de interrupção, informe a pasta 8F impressa no início e execute a 8F novamente.
CAP_PASTA_SAIDA_RETOMAR = None
SST_ARQUIVO_MANUAL = None

PONTOS_POR_MES = 5000
ANOS_TREINO = 30
ARVORES = 300
SEMENTE = 42
PESO_M0 = 0.875
PESOS = (np.arange(61, dtype=np.float64) / 40.0).tolist()
CORTE_FINAL = '2022-12-01'
FIM_DESENVOLVIMENTO = '2020-12-01'
TOL_AUDITORIA = 1e-6
# As referências por período foram publicadas com nove casas decimais.
TOL_COMPARACAO = 1e-8
PARAMETROS = {'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.05, 'num_leaves': 31, 'min_data_in_leaf': 150, 'lambda_l2': 10.0, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'num_threads': 4, 'seed': SEMENTE, 'deterministic': True, 'force_col_wise': True, 'verbosity': -1}
BLOCOS = [{'nome': 'H1', 'corte': '2006-12-01', 'inicio': '2007-01-01'}, {'nome': 'H2', 'corte': '2008-12-01', 'inicio': '2009-01-01'}, {'nome': 'H3', 'corte': '2010-12-01', 'inicio': '2011-01-01'}, {'nome': 'H4', 'corte': '2012-12-01', 'inicio': '2013-01-01'}, {'nome': 'A', 'corte': '2014-12-01', 'inicio': '2015-01-01'}, {'nome': 'B', 'corte': '2016-12-01', 'inicio': '2017-01-01'}, {'nome': 'C', 'corte': '2018-12-01', 'inicio': '2019-01-01'}]
VARIAVEIS = ['cloud_cover', 'geopotential_850', 'rel_hum_850', 'shum_850', 'surface_pressure', 't2', 'temperature_850', 'u_850', 'v_850']
FEATURES_M0 = VARIAVEIS + [v + '_anomalia' for v in VARIAVEIS] + ['latitude', 'longitude', 'mes_alvo_sin', 'mes_alvo_cos', 'clima_tp_alvo']
INDICES_NOMES = ['nino34', 'nino12', 'tna', 'tsa']
FEATURES_OCEANO = [n + '_lag1_anomalia_fold' for n in INDICES_NOMES]
FEATURES = FEATURES_M0 + FEATURES_OCEANO
ORDEM_BLOCOS = [b['nome'] for b in BLOCOS]
REF_M0_GLOBAL = {'rmse': 1.8101121800554918, 'mae': 1.085263304,
                 'vies': 0.022914405, 'peso': 0.875, 'n': 13198248}
REF_CLIMA_GLOBAL = 1.8496420039184214
# Somente números dos relatórios da Etapa 2; não são dados de treino.
COL_REF = ['rmse_climatologia', 'mae_climatologia', 'vies_climatologia',
           'rmse_M0', 'mae_M0', 'vies_M0']
REF_FOLD = pd.DataFrame([['H1', 1.839845926, 1.075422198, -0.089676034, 1.780676831, 1.052738061, -0.065322037], ['H2', 1.985533781, 1.17764983, -0.08012667, 1.89938271, 1.144985077, -0.005764244], ['H3', 1.833986724, 1.103440925, -0.076567408, 1.796554028, 1.089670267, -0.03648091], ['H4', 1.721495863, 1.024299361, 0.024806943, 1.707686158, 1.020427292, 0.025005878], ['A', 1.854691975, 1.125081751, 0.131241513, 1.816861444, 1.110338007, 0.142916522], ['B', 1.853953344, 1.097407801, -0.078870291, 1.84693707, 1.094939451, -0.043868717], ['C', 1.848450902, 1.09751788, 0.138663834, 1.816897806, 1.083744973, 0.143914343]], columns=['bloco'] + COL_REF)
REF_ANO = pd.DataFrame([[2007, 1.842494728, 1.077941774, 0.00183693, 1.790923888, 1.057678144, -0.009944569], [2008, 1.837193305, 1.072902621, -0.181188997, 1.770370463, 1.047797978, -0.120699505], [2009, 2.022061463, 1.192506261, -0.074923059, 1.942703508, 1.164179748, -0.024245971], [2010, 1.948321388, 1.162793399, -0.085330281, 1.855050521, 1.125790405, 0.012717483], [2011, 1.852994051, 1.084233803, -0.212546637, 1.794078445, 1.068334992, -0.127252253], [2012, 1.814780332, 1.122648046, 0.05941182, 1.799026204, 1.111005543, 0.054290434], [2013, 1.721092997, 1.034856288, 0.020723158, 1.702893681, 1.03280552, 0.033810877], [2014, 1.721898635, 1.013742434, 0.028890727, 1.712465223, 1.008049065, 0.016200879], [2015, 1.846480311, 1.121117686, 0.129572222, 1.79034488, 1.094301357, 0.125278846], [2016, 1.862867443, 1.129045816, 0.132910804, 1.842996535, 1.126374657, 0.160554197], [2017, 1.785312205, 1.081442101, -0.139569874, 1.756382116, 1.070597014, -0.078965828], [2018, 1.92014227, 1.1133735, -0.018170709, 1.933255012, 1.119281888, -0.008771606], [2019, 1.808213478, 1.059386025, 0.158010997, 1.770023859, 1.042689764, 0.154691493], [2020, 1.887830897, 1.135649734, 0.119316671, 1.862592499, 1.124800182, 0.133137192]], columns=['ano'] + COL_REF)
FONTE_CODIGO_ETAPA2_SHA256 = '7695f2302636e4d3089062a76523e91ec0d64a66b320b35a443940cd1b691c9a'

FONTES_INDICES = {'fonte': 'NOAA Physical Sciences Laboratory', 'captura_utc': '2026-09-21T23:00:40.933121+00:00', 'unidade': 'degC', 'tipo': 'indices mensais de anomalia de temperatura da superficie do mar', 'interpretacao_temporal': 'Versoes retrospectivas, com mes de observacao estritamente T-1; uso de dados externos confirmado pelo participante. Nao e arquivo de vintages em tempo real.', 'inicio': '1976-12-01', 'fim': '2024-11-01', 'meses': 576, 'colunas': ['time_origem', 'nino34', 'nino12', 'tna', 'tsa'], 'fontes': [{'indice': 'nino34', 'url': 'https://psl.noaa.gov/data/timeseries/month/data/nino34.long.anom.data', 'arquivo_bruto': 'nino34.data', 'sha256_bruto': '46e501b91f624ef85b525d73fbb427f9d04d30411ba6e5c7cff579b35fb4ff01', 'inicio_declarado': 1870, 'fim_declarado': 2026, 'rodape': '-99.99\n  NINA34\n 5N-5S 170W-120W \n HadISST \n  Anomaly from 1981-2010\n https://psl.noaa.gov/data/timeseries/month/\n  units=degC'}, {'indice': 'nino12', 'url': 'https://psl.noaa.gov/data/timeseries/month/data/nino12.long.anom.data', 'arquivo_bruto': 'nino12.data', 'sha256_bruto': 'b13ad5ad085cd13e70ec5d82662b509d522204e38a8eefd133f14be270e51928', 'inicio_declarado': 1870, 'fim_declarado': 2026, 'rodape': '-99.99\n  NINA12\n 0N-10S 270W-280E \n HadISST \n  Anomaly from 1981-2010\n https://psl.noaa.gov/data/timeseries/month/\n  units=degC'}, {'indice': 'tna', 'url': 'https://psl.noaa.gov/data/correlation/tna.data', 'arquivo_bruto': 'tna.data', 'sha256_bruto': '387c4125f377e57bcb683a849aefcd76da7d7f3cdcb2ac68ab086ceadf300e8b', 'inicio_declarado': 1948, 'fim_declarado': 2026, 'rodape': '-99.99\n  TNAa\n Tropical Northern Atlantic Index\n SST Anom 5.5N-23.5N; 15W-57.5W\n  Using HadISST1.1: Climo 1991-2020\n  NOAA/PSL\n  https://psl.noaa.gov/data/timeseries/month/'}, {'indice': 'tsa', 'url': 'https://psl.noaa.gov/data/correlation/tsa.data', 'arquivo_bruto': 'tsa.data', 'sha256_bruto': '700c0dffda0f19c1254ba482808b345eaa0538f5bf09fc3bc025f46a5fbe3a69', 'inicio_declarado': 1948, 'fim_declarado': 2026, 'rodape': '-99.99\n  TSAa\n  NOAA/PSL\n  https://psl.noaa.gov/data/timeseries/month/'}], 'sha256_csv': 'b6f444fbec48681c1e5b46a5c22aa197ad91c7fbbdaff82c3ed3a9c63f7c15a4', 'transformacao_no_modelo': 'Recentragem mensal em cada fold, usando somente as 360 origens de treino; sem padronizacao, medias moveis ou indices adicionais.'}
INDICES_CSV_SHA256 = 'b6f444fbec48681c1e5b46a5c22aa197ad91c7fbbdaff82c3ed3a9c63f7c15a4'
INDICES_CSV_GZIP_BASE64 = (
    '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'
)


INICIO=perf_counter()
EXECUCAO=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
BASE=Path('/kaggle/working') if Path('/kaggle/input').is_dir() else Path.cwd()/'outputs'
SAIDA=Path(CAP_PASTA_SAIDA_RETOMAR) if CAP_PASTA_SAIDA_RETOMAR else BASE/'worcap_M10_SST_PCA_8F'/EXECUCAO
SAIDA.mkdir(parents=True,exist_ok=bool(CAP_PASTA_SAIDA_RETOMAR))
VERSOES={v:importlib.metadata.version(v) for v in ['numpy','pandas','xarray','lightgbm']}
print('WorCAP — Etapa 8F; M10: M6 + 8 PCs de SST; 37 features; 300 árvores; máximo 30 anos')
print('Python:',platform.python_version(),'| Versões:',VERSOES,'| Saída:',SAIDA)
pd.set_option('display.float_format',lambda x:f'{x:.9f}')

# %% Funções preservadas; exportação oficial auditada
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

def mostrar_tabela(tabela):
    with pd.option_context('display.max_columns', None, 'display.max_rows', 100, 'display.precision', 9):
        display(tabela)

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

def auditar_contrato(tp, alvo, teste):
    exigir(set(tp.dims) == {'time', 'lat', 'lon'}, 'Dimensões inesperadas da chuva.')
    exigir(set(alvo.dims) == set(tp.dims), 'Dimensões inesperadas do alvo.')
    tp = tp.transpose('time', 'lat', 'lon')
    alvo = alvo.transpose('time', 'lat', 'lon')
    datas = pd.DatetimeIndex(tp.time.values)
    destinos = pd.DatetimeIndex(teste.time.values)
    exigir(datas.equals(pd.date_range('1940-01-01', '2022-12-01', freq='MS')), 'O calendário de treino diverge de janeiro/1940 a dezembro/2022.')
    exigir(destinos.equals(pd.date_range('2023-01-01', '2024-12-01', freq='MS')), 'O calendário de teste diverge de janeiro/2023 a dezembro/2024.')
    exigir(tp.shape == (996, 301, 261), 'Grade ou quantidade de meses inesperada.')
    exigir(tp.attrs.get('units') == alvo.attrs.get('units') == 'mm/day', 'A unidade deve ser mm/day; não aplicar conversão automática.')
    for coord in ('time', 'lat', 'lon'):
        exigir(np.array_equal(tp[coord].values, alvo[coord].values), f'Coordenada {coord} diferente entre chuva e alvo.')
    for coord in ('lat', 'lon'):
        exigir(np.array_equal(tp[coord].values, teste[coord].values), f'Coordenada {coord} diferente entre treino e teste.')
        exigir(np.all(np.diff(tp[coord].values) == 0.25), f'Ordem/intervalo incorreto de {coord}.')
    origem = pd.DatetimeIndex(teste.time_origem.values).to_period('M')
    exigir(origem.equals(destinos.to_period('M') - 1), 'Origem do teste não é o mês anterior.')
    exigir(np.array_equal(teste.lag_meses.values, np.arange(1, 25)), 'Defasagens inesperadas.')
    tp.load()
    exigir(np.isfinite(tp.values).all() and (tp.values >= 0).all(), 'Chuva de treino não finita ou negativa; investigar antes de continuar.')
    for inicio in range(0, len(datas) - 1, 24):
        fim = min(inicio + 24, len(datas) - 1)
        resposta = alvo.isel(time=slice(inicio, fim)).values
        exigir(np.array_equal(resposta, tp.values[inicio + 1:fim + 1]), 'O alvo não corresponde exatamente à chuva do mês seguinte.')
    exigir(np.isnan(alvo.isel(time=-1).values).all(), 'Último alvo deveria estar vazio.')
    exigir(np.isnan(teste.tp_alvo.values).all(), 'Alvos de teste deveriam estar vazios.')
    ancora = teste.tp_ultima_obs.transpose('time', 'lat', 'lon').values
    exigir(np.all(ancora == tp.isel(time=-1).values[None]), 'A referência de chuva do teste diverge de dezembro/2022.')
    return {'meses_treino': len(datas), 'pares_supervisionados_validos': len(datas) - 1, 'meses_teste': len(destinos), 'pontos_por_mes': tp.sizes['lat'] * tp.sizes['lon'], 'unidade': 'mm/day', 'alvo_ja_deslocado': True, 'chuva_teste_congelada_em': '2022-12', 'contrato_conferido': True}

def estatisticas_climatologia(tp, corte, anos=None):
    """Mesma implementação da Etapa 1: somas em float64 e mapas em float32."""
    corte = pd.Timestamp(corte)
    exigir(corte.month == 12 and corte.day == 1, 'Esta versão usa cortes no primeiro dia de dezembro.')
    datas = pd.DatetimeIndex(tp.time.values)
    exigir(datas.is_unique, 'Calendário de chuva duplicado.')
    inicio = datas.min() if anos is None else pd.Timestamp(corte.year - anos + 1, 1, 1)
    permitir = (datas >= inicio) & (datas <= corte)
    exigir(permitir.any(), 'Histórico de ajuste vazio.')
    calendario_esperado = pd.date_range(inicio, corte, freq='MS')
    exigir(datas[permitir].equals(calendario_esperado), 'A janela da climatologia não contém todos os meses esperados.')
    somas = []
    contagens = []
    mapas = []
    for mes in range(1, 13):
        indices = np.flatnonzero(permitir & (datas.month == mes))
        exigir(len(indices) > 0, f'Sem observações de treino para o mês {mes}.')
        valores = tp.isel(time=indices).transpose('time', 'lat', 'lon').values
        exigir(np.isfinite(valores).all(), 'Climatologia recebeu dados ausentes.')
        soma = valores.sum(axis=0, dtype=np.float64)
        contagem = len(indices)
        somas.append(soma)
        contagens.append(contagem)
        mapas.append((soma / contagem).astype(np.float32))
    clima = xr.DataArray(np.stack(mapas), dims=('month', 'lat', 'lon'), coords={'month': np.arange(1, 13), 'lat': tp.lat, 'lon': tp.lon}, name='climatologia_mm_day', attrs={'units': 'mm/day', 'ajuste_inicio': str(inicio.date()), 'ajuste_fim': str(corte.date()), 'janela_anos': 'todo_historico' if anos is None else anos})
    return (clima, np.stack(somas), np.asarray(contagens, dtype=np.int64))

def ajustar_climatologia(tp, corte, anos=None):
    clima, _, _ = estatisticas_climatologia(tp, corte, anos)
    return clima

def metricas_mensais(observado, previsto, modelo, bloco, corte):
    exigir(pd.DatetimeIndex(observado.time.values).isin(pd.date_range('2007-01-01','2020-12-01',freq='MS')).all(), 'Métricas fora de 2007–2020 são proibidas nesta execução.')
    observado, previsto = xr.align(observado, previsto, join='exact')
    observado = observado.transpose('time', 'lat', 'lon')
    previsto = previsto.transpose('time', 'lat', 'lon')
    datas = pd.DatetimeIndex(observado.time.values)
    exigir(((datas >= pd.Timestamp('2007-01-01')) & (datas <= pd.Timestamp('2020-12-01'))).all(), 'Esta etapa só permite calcular métricas de 2007–2020.')
    reais = observado.values
    previsoes = previsto.values
    registros = []
    for i, data in enumerate(datas):
        real = reais[i]
        pred = previsoes[i]
        exigir(np.isfinite(real).all() and np.isfinite(pred).all(), 'Métrica não pode ocultar NaNs ou infinitos.')
        erro = pred.astype(np.float64) - real.astype(np.float64)
        sse = float(np.square(erro).sum())
        n = int(erro.size)
        registros.append({'modelo': modelo, 'bloco': bloco, 'corte': str(pd.Timestamp(corte).date()), 'mes_alvo': str(data.date()), 'ano': int(data.year), 'ano_no_bloco': int(data.year - pd.Timestamp(corte).year), 'sse': sse, 'n': n, 'soma_erro': float(erro.sum()), 'soma_erro_absoluto': float(np.abs(erro).sum()), 'rmse': float(np.sqrt(sse / n))})
    return pd.DataFrame(registros)

def agregar_metricas(mensais, grupos):
    tabela = mensais.groupby(grupos, as_index=False)[['sse', 'n', 'soma_erro', 'soma_erro_absoluto']].sum()
    tabela['rmse'] = np.sqrt(tabela.sse / tabela.n)
    tabela['vies'] = tabela.soma_erro / tabela.n
    tabela['mae'] = tabela.soma_erro_absoluto / tabela.n
    return tabela

def indices_dos_ids(ids, previsto):
    partes = ids.str.split('_', expand=True)
    exigir(partes.shape[1] == 4, 'ID precisa conter ano, mês, latitude e longitude.')
    padrao = '\\d{4}_(?:0[1-9]|1[0-2])_-?\\d+\\.\\d{2}_-?\\d+\\.\\d{2}'
    exigir(ids.str.fullmatch(padrao).fillna(False).all(), 'ID oficial malformado ou ausente.')
    ano = partes[0].astype(np.int64).to_numpy()
    mes = partes[1].astype(np.int64).to_numpy()
    datas = pd.DatetimeIndex(previsto.time.values)
    chaves_tempo = pd.Index(datas.year * 12 + datas.month - 1)
    ti = chaves_tempo.get_indexer(ano * 12 + mes - 1)
    yi = pd.Index(previsto.lat.values).get_indexer(partes[2].astype(float).to_numpy())
    xi = pd.Index(previsto.lon.values).get_indexer(partes[3].astype(float).to_numpy())
    exigir((ti >= 0).all() and (yi >= 0).all() and (xi >= 0).all(), 'Há um ID cuja data ou coordenada não existe nas previsões.')
    return (ti, yi, xi)

def salvar_submissao(sample_path, previsto, destino, tamanho_bloco=150000):
    sample_path = Path(sample_path)
    destino = Path(destino)
    exigir(sample_path.resolve() != destino.resolve(), 'Nunca sobrescrever o CSV oficial.')
    exigir(set(previsto.dims) == {'time', 'lat', 'lon'}, 'Dimensões da previsão inválidas.')
    exigir(previsto.attrs.get('units') == 'mm/day', 'Previsão deve estar em mm/day.')
    previsto = previsto.transpose('time', 'lat', 'lon')
    for coord in ('time', 'lat', 'lon'):
        exigir(pd.Index(previsto[coord].values).is_unique, f'Coordenada {coord} duplicada.')
    exigir(pd.DatetimeIndex(previsto.time.values).to_period('M').is_unique, 'Mais de uma previsão para o mesmo mês.')
    valores = previsto.values
    exigir(np.isfinite(valores).all() and (valores >= 0).all(), 'Previsões precisam ser finitas e não negativas.')
    exigir(pd.read_csv(sample_path, nrows=0).columns.tolist() == ['id', 'tp_mm_day'], 'Colunas do CSV oficial diferentes de id,tp_mm_day.')
    hash_original = sha256(sample_path)
    vistos = np.zeros(valores.size, dtype=bool)
    linhas = 0
    temporario = destino.with_suffix('.partial.csv')
    destino.parent.mkdir(parents=True, exist_ok=True)
    with pd.read_csv(sample_path, dtype={'id': str}, chunksize=tamanho_bloco) as leitor:
        for amostra in leitor:
            ti, yi, xi = indices_dos_ids(amostra.id, previsto)
            indices = np.ravel_multi_index((ti, yi, xi), valores.shape)
            exigir(len(np.unique(indices)) == len(indices) and (not vistos[indices].any()), 'IDs ou combinações de coordenadas duplicados no CSV oficial.')
            vistos[indices] = True
            saida = amostra.copy()
            saida['tp_mm_day'] = valores[ti, yi, xi]
            saida.to_csv(temporario, index=False, mode='w' if linhas == 0 else 'a', header=linhas == 0, float_format='%.6f')
            linhas += len(saida)
    exigir(linhas == valores.size and vistos.all(), 'CSV não cobre exatamente toda a previsão.')
    contagem = 0
    with pd.read_csv(sample_path, dtype={'id': str}, chunksize=tamanho_bloco) as original, pd.read_csv(temporario, dtype={'id': str}, chunksize=tamanho_bloco) as gravado:
        for amostra, saida in zip_longest(original, gravado):
            exigir(amostra is not None and saida is not None, 'Número de linhas alterado na escrita.')
            exigir(saida.columns.tolist() == ['id', 'tp_mm_day'], 'Colunas extras na saída.')
            exigir(amostra.id.equals(saida.id), 'Texto ou ordem dos IDs foi alterado.')
            ti, yi, xi = indices_dos_ids(amostra.id, previsto)
            exigir(np.isfinite(saida.tp_mm_day).all() and (saida.tp_mm_day >= 0).all(), 'CSV escrito contém previsão inválida.')
            exigir(np.allclose(saida.tp_mm_day, valores[ti, yi, xi], atol=5.1e-07, rtol=0), 'Valores escritos não correspondem às previsões por coordenada.')
            contagem += len(saida)
    exigir(contagem == linhas, 'Quantidade de linhas divergente após reabertura.')
    exigir(sha256(sample_path) == hash_original, 'O CSV oficial foi alterado durante a execução.')
    temporario.replace(destino)
    return {'arquivo': destino.name, 'linhas': linhas, 'ids_preservados_integralmente': True, 'ordem_preservada_integralmente': True, 'cobertura_completa': True, 'valores_conferidos_apos_escrita': True, 'minimo_mm_day': float(valores.min()), 'maximo_mm_day': float(valores.max()), 'sample_sha256': hash_original, 'submissao_sha256': sha256(destino), 'bytes': destino.stat().st_size}

def calendario_pares(datas_disponiveis, corte, anos=30):
    corte = pd.Timestamp(corte)
    exigir(corte.month == 12 and corte.day == 1, 'Corte precisa ser dezembro, dia 1.')
    destinos = pd.date_range(pd.Timestamp(corte.year - anos + 1, 1, 1), corte, freq='MS')
    origens = (destinos.to_period('M') - 1).to_timestamp()
    disponiveis = pd.DatetimeIndex(datas_disponiveis)
    exigir(disponiveis.is_unique, 'Calendário atmosférico duplicado.')
    exigir((disponiveis.get_indexer(origens) >= 0).all(), 'Faltam meses de entrada para o treino.')
    return (origens, destinos)

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

def matriz_mes(campos, clima_atmos, clima_tp, destino, pontos=None):
    """
    Atmosfera do mês anterior ao alvo.

    A climatologia da chuva é integral por padrão.
    Somente amostrar_treino substitui sua coluna pelo leave-one-out.
    """
    destino = pd.Timestamp(destino)
    origem = (destino.to_period('M') - 1).to_timestamp()
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
    # As 23 primeiras colunas permanecem nas mesmas posições do M0.
    exigir(origem < destino, 'Violação temporal: origem não é anterior ao alvo.')
    # Uma unica linha mensal NOAA; nao depende de observacoes do mes-alvo.
    valores_oceano = valores_indices_mes(INDICES_OC, clima_atmos['_clima_indices'], destino)
    X[:, 23:] = valores_oceano[None, :]
    exigir(np.isfinite(X).all(), 'Atributos contêm valores ausentes ou infinitos.')
    return X

def reconstruir_chuva(clima_tp, anomalias, peso):
    peso = float(peso)
    exigir(np.isfinite(peso) and peso >= 0, 'Peso deve ser finito e não negativo.')
    meses = pd.DatetimeIndex(anomalias.time.values).month.to_numpy() - 1
    valores = np.maximum(0, clima_tp.values[meses] + peso * anomalias.values)
    exigir(np.isfinite(valores).all(), 'Reconstrução da chuva produziu valores inválidos.')
    return xr.DataArray(valores.astype(np.float32), dims=anomalias.dims, coords=anomalias.coords, name='tp_mm_day', attrs={'units': 'mm/day'})

def auditar_chuva_desenv():
    # Antes da seleção, só materializa observações até dez/2020.
    with xr.open_dataset(PASTA / 'treino_tp.nc') as ds, \
         xr.open_dataset(PASTA / 'treino_tp_alvo.nc') as da:
        bruto, alvo = ds.tp, da.tp_alvo
        esperado = pd.date_range('1940-01-01', CORTE_FINAL, freq='MS')
        exigir(set(bruto.dims) == set(alvo.dims) == {'time', 'lat', 'lon'},
               'Dimensões inesperadas da precipitação.')
        bruto, alvo = bruto.transpose('time', 'lat', 'lon'), alvo.transpose('time', 'lat', 'lon')
        exigir(bruto.shape == alvo.shape == (996, 301, 261), 'Grade/calendário inesperados.')
        exigir(pd.DatetimeIndex(bruto.time.values).equals(esperado), 'Calendário oficial divergente.')
        exigir(bruto.attrs.get('units') == alvo.attrs.get('units') == 'mm/day',
               'Chuva deve estar em mm/day; nenhuma conversão será aplicada.')
        for eixo in ['time', 'lat', 'lon']:
            exigir(np.array_equal(bruto[eixo].values, alvo[eixo].values),
                   f'Alvo desalinhado em {eixo}.')
        for eixo in ['lat', 'lon']:
            exigir(np.all(np.diff(bruto[eixo].values) == .25), f'Grade inválida: {eixo}.')
        chuva = bruto.sel(time=slice(None, FIM_DESENVOLVIMENTO)).load()
        exigir(np.isfinite(chuva.values).all() and (chuva.values >= 0).all(),
               'Precipitação histórica inválida.')
        for i in range(0, chuva.sizes['time'] - 1, 24):
            j = min(i + 24, chuva.sizes['time'] - 1)
            exigir(np.array_equal(alvo.isel(time=slice(i, j)).values, chuva.values[i+1:j+1]),
                   'tp_alvo[M] não coincide com tp[M+1].')
    return chuva

def carregar_atmosfera(referencia, inicio, fim):
    resultado, metadados = {}, []
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
            metadados.append(dict(variavel=nome, atributos=dict(bruto.attrs),
                                  inicio=str(inicio), fim=str(fim),
                                  minimo=float(campo.min()), maximo=float(campo.max())))
        print('Carregado:', nome, campo.shape, flush=True)
    return resultado, metadados

def carregar_indices_incorporados():
    dados = gzip.decompress(base64.b64decode(INDICES_CSV_GZIP_BASE64))
    exigir(hashlib.sha256(dados).hexdigest() == INDICES_CSV_SHA256,
           'Hash das series NOAA incorporadas divergente.')
    tabela = pd.read_csv(io.BytesIO(dados), parse_dates=['time_origem'])
    exigir(tabela.columns.tolist() == ['time_origem'] + INDICES_NOMES,
           'Colunas dos indices oceanicos divergentes.')
    tabela = tabela.set_index('time_origem')
    conferir_indices(tabela)
    esperado = pd.date_range(FONTES_INDICES['inicio'], FONTES_INDICES['fim'], freq='MS')
    exigir(tabela.index.equals(esperado), 'Snapshot NOAA tem lacunas ou datas alteradas.')
    return tabela, dados

def conferir_indices(tabela):
    exigir(tabela.columns.tolist() == INDICES_NOMES, 'Ordem dos quatro indices alterada.')
    datas = pd.DatetimeIndex(tabela.index)
    exigir(len(datas) > 0 and datas.is_unique and datas.is_monotonic_increasing,
           'Datas NOAA vazias, duplicadas ou fora de ordem.')
    exigir(datas.equals(pd.date_range(datas.min(), datas.max(), freq='MS')),
           'Calendario NOAA precisa ser mensal, continuo e no primeiro dia.')
    exigir(np.isfinite(tabela.to_numpy(dtype=np.float64)).all(),
           'Indice NOAA ausente/infinito. Nao interpolar nem preencher com meses futuros.')

def ajustar_clima_indices(tabela, origens, corte):
    origens = pd.DatetimeIndex(origens)
    corte = pd.Timestamp(corte)
    alvos = pd.date_range(pd.Timestamp(corte.year - ANOS_TREINO + 1, 1, 1), corte, freq='MS')
    exigir(origens.equals((alvos.to_period('M') - 1).to_timestamp()),
           'Climatologia oceanica nao recebeu exatamente as origens do treino.')
    exigir((origens < corte).all(), 'Climatologia oceanica inclui data fora do treino.')
    indices = tabela.index.get_indexer(origens)
    exigir((indices >= 0).all(), 'Faltam origens NOAA para a climatologia de treino.')
    historico = tabela.iloc[indices].to_numpy(dtype=np.float64)
    exigir(np.isfinite(historico).all(), 'Climatologia oceanica recebeu valores ausentes.')
    medias = []
    for mes in range(1, 13):
        valores = historico[origens.month == mes]
        exigir(len(valores) == ANOS_TREINO, 'Contagem mensal oceanica diferente de 30 anos.')
        medias.append(valores.mean(axis=0, dtype=np.float64))
    return np.stack(medias)

def valores_indices_mes(tabela, clima_indices, destino):
    destino = pd.Timestamp(destino)
    exigir(destino == destino.to_period('M').to_timestamp(), 'Mes-alvo NOAA invalido.')
    origem = (destino.to_period('M') - 1).to_timestamp()
    exigir(origem < destino, 'Indice oceanico nao e estritamente anterior ao alvo.')
    exigir(tabela.index.is_unique, 'Datas NOAA duplicadas.')
    i = tabela.index.get_indexer([origem])[0]
    exigir(i >= 0, f'Indice oceanico em T-1 ausente: {origem.date()}.')
    exigir(clima_indices.shape == (12, 4) and np.isfinite(clima_indices).all(),
           'Climatologia oceanica invalida.')
    valores = tabela.iloc[i].to_numpy(dtype=np.float64)
    exigir(np.isfinite(valores).all(), 'Indice NOAA nao finito.')
    resultado = (valores - clima_indices[origem.month - 1]).astype(np.float32)
    exigir(np.isfinite(resultado).all(), 'Anomalia oceanica nao finita.')
    return resultado

def auditar_causalidade_indices(tabela):
    # Invariancia: alterar T e posteriores nao pode alterar entrada nem climatologia do fold.
    for bloco in BLOCOS:
        corte = pd.Timestamp(bloco['corte'])
        alvos = pd.date_range(pd.Timestamp(corte.year-ANOS_TREINO+1, 1, 1), corte, freq='MS')
        origens = (alvos.to_period('M')-1).to_timestamp()
        ci = ajustar_clima_indices(tabela, origens, corte)
        alvo = pd.Timestamp(bloco['inicio'])
        antes = valores_indices_mes(tabela, ci, alvo)
        alterada = tabela.copy()
        alterada.loc[alterada.index >= alvo, :] = 12345.0
        depois_ci = ajustar_clima_indices(alterada, origens, corte)
        exigir(np.array_equal(ci, depois_ci), 'Dados futuros alteraram climatologia NOAA.')
        exigir(np.array_equal(antes, valores_indices_mes(alterada, depois_ci, alvo)),
               'Dados futuros alteraram as features NOAA.')
    print('Auditoria NOAA: alterar T e meses posteriores nao altera a entrada em T-1 nem o ajuste.')

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



# %% Funções originais M6 preservadas
def localizar_seas5():
    if PASTA_SEAS5_MANUAL:
        candidatos=[Path(PASTA_SEAS5_MANUAL)/'seas5_51_manifesto.json']
    else:
        candidatos=[]
        for raiz in [Path('/kaggle/input'),Path('/kaggle/working/worcap_SEAS5_dados'),Path.cwd()/'outputs/worcap_SEAS5_dados']:
            if raiz.is_dir():
                candidatos.extend(raiz.rglob('seas5_51_manifesto.json'))
    candidatos=sorted(set(p.resolve() for p in candidatos if p.is_file()))
    exigir(len(candidatos)==1,'Anexe a saída salva do notebook 01 em Input → Notebook, '
           'ou defina PASTA_SEAS5_MANUAL. É necessário encontrar exatamente um manifesto SEAS5.')
    caminho=candidatos[0]
    m=json.loads(caminho.read_text(encoding='utf-8'))
    exigir(m.get('schema')==SCHEMA and m.get('system')==SYSTEM and m.get('dataset')==DATASET,
           'Manifesto SEAS5 incompatível.')
    exigir(m.get('leadtime_month')==2 and m.get('paramId')==172228 and
           m.get('product_type')=='monthly_mean' and m.get('variable')=='total_precipitation',
           'Produto SEAS5 não corresponde ao experimento.')
    return caminho.parent,m

def carregar_seas5(uso,referencia):
    item=MANIFESTO_SEAS5['arquivos'][uso]
    caminho=PASTA_SEAS5/item['arquivo']
    exigir(caminho.is_file() and sha256(caminho)==item['sha256'],f'SEAS5 ausente/alterado: {uso}.')
    with xr.open_dataset(caminho) as ds:
        bruto=ds.load()
    auditar_dataset(bruto,pd.date_range(item['inicio'],item['fim'],freq='MS'))
    exigir(bruto.attrs.get('uso')==uso,'Uso do arquivo SEAS5 divergente.')
    # Reamostragem espacial fixa, sem ajuste a observações ou extrapolação.
    grade=bruto.seas5_tp_media.interp(lat=referencia.lat,lon=referencia.lon,method='linear').astype(np.float32)
    exigir(np.isfinite(grade.values).all() and (grade.values>=0).all(),'Interpolação SEAS5 inválida.')
    for coord in ['lat','lon']:
        exigir(np.array_equal(grade[coord].values,referencia[coord].values),'Grade SEAS5 desalinhada.')
    grade.attrs.update(units='mm/day',system=SYSTEM,leadtime_month=2)
    return grade

def valores_seas5(seas,destino,pontos=None):
    destino=pd.Timestamp(destino)
    exigir(pd.DatetimeIndex(seas.time.values).is_unique,'Alvos SEAS5 duplicados.')
    i=pd.DatetimeIndex(seas.time.values).get_indexer([destino])[0]
    exigir(i>=0,f'Previsão SEAS5 ausente para {destino.date()}.')
    origem=pd.Timestamp(seas.time_origem.values[i])
    exigir(origem==(destino.to_period('M')-1).to_timestamp() and origem<destino,
           'SEAS5 usa inicialização incorreta ou posterior ao limite.')
    v=seas.values[i].reshape(-1)
    if pontos is not None:
        v=v[pontos]
    exigir(np.isfinite(v).all(),'Feature SEAS5 não finita.')
    return v

def amostrar_m5_original(chuva,atmosfera,seas,corte):
    destinos=calendario_pareado(corte)
    origens=(destinos.to_period('M')-1).to_timestamp()
    ref_origens,ref_alvos=calendario_pares(atmosfera[VARIAVEIS[0]].time.values,corte,30)
    clima,somas,contagens=estatisticas_climatologia(chuva,corte,30)
    exigir(np.all(contagens==30),'Climatologia de chuva não tem exatamente 30 anos.')
    ca=ajustar_clima_atmosfera(atmosfera,ref_origens)
    ca['_clima_indices']=ajustar_clima_indices(INDICES_OC,ref_origens,corte)
    exigir(set(destinos).issubset(set(ref_alvos)),'Exemplo fora da janela de referência.')
    ngrade=chuva.sizes['lat']*chuva.sizes['lon']
    n=min(PONTOS_POR_MES,ngrade)
    X=np.empty((len(destinos)*n,28),dtype=np.float32)
    y=np.empty(len(X),dtype=np.float32)
    ids=np.empty((len(destinos),n),dtype=np.int32)
    for i,t in enumerate(destinos):
        pontos=np.random.default_rng(np.random.SeedSequence([SEMENTE,t.year,t.month])).choice(ngrade,size=n,replace=False)
        ids[i]=pontos
        basicas=matriz_mes(atmosfera,ca,clima,t,pontos)
        obs=chuva.sel(time=t).values.reshape(-1)[pontos]
        basicas[:,22]=((somas[t.month-1].reshape(-1)[pontos]-obs.astype(np.float64))/29).astype(np.float32)
        sl=slice(i*n,(i+1)*n)
        X[sl,:27]=basicas
        X[sl,27]=valores_seas5(seas,t,pontos)
        y[sl]=obs-basicas[:,22]
    exigir(np.isfinite(X).all() and np.isfinite(y).all(),'Treino não finito.')
    exigir((origens<destinos).all() and destinos.max()<=pd.Timestamp(corte),'Corte temporal violado.')
    return X,y,ids,destinos,clima,ca,somas,contagens

def valores_anomalia_seas5(seas,clima_seas,destino,pontos=None):
    for eixo in ['lat','lon']:
        exigir(np.array_equal(seas[eixo].values,clima_seas[eixo].values),'Climatologia SEAS5 desalinhada.')
    bruto=valores_seas5(seas,destino,pontos)
    referencia=clima_seas.sel(month=pd.Timestamp(destino).month).values.reshape(-1)
    if pontos is not None:
        referencia=referencia[pontos]
    anomalia=(bruto-referencia).astype(np.float32)
    exigir(np.isfinite(anomalia).all(),'Anomalia SEAS5 não finita.')
    # Anomalias negativas são válidas; não truncar a nova feature.
    return anomalia

def amostrar_m6(chuva,atmosfera,seas,corte):
    # A função original do M5 é copiada sem mudanças pelo gerador deste notebook.
    x5,y,ids,destinos,clima,ca,somas,contagens=amostrar_m5_original(chuva,atmosfera,seas,corte)
    clima_seas=ajustar_clima_seas5(seas,corte)
    X=np.empty((len(x5),29),dtype=np.float32)
    X[:,:28]=x5
    n=ids.shape[1]
    for i,t in enumerate(destinos):
        X[i*n:(i+1)*n,28]=valores_anomalia_seas5(seas,clima_seas,t,ids[i])
    exigir(np.array_equal(X[:,:28],x5),'As 28 features do M5 foram modificadas.')
    exigir(np.isfinite(X).all() and np.isfinite(y).all(),'Treino não finito.')
    del x5
    return X,y,ids,destinos,clima,ca,somas,contagens,clima_seas

# %% Protocolo M6 versus M8: duas features CFSv2, máximo 30 anos
PASTA_SEAS5_MANUAL = None
PASTA_CFSV2_MANUAL = None
DATASET, SYSTEM, SCHEMA = 'seasonal-monthly-single-levels', '51', 'worcap_seas5_v1'
SCHEMA_CFSV2 = 'worcap_cfsv2_pentad_media_auditada_v1'
POLITICA_CFSV2 = 'ensemble_completo_ou_excecao_201908_m17_auditada_v2'
FONTE_CFSV2 = 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.NCEP-CFSv2/.HINDCAST/.PENTAD_SAMPLES_FULL/.prec/'
FEATURE_SEAS5 = 'seas5_tp_prevista_media_T_emitida_Tmenos1'
FEATURE_SEAS5_ANOM = 'seas5_tp_anomalia_mensal_treino_fold'
FEATURE_CFSV2 = 'cfsv2_tp_prevista_media_T_emitida_Tmenos1'
FEATURE_CFSV2_ANOM = 'cfsv2_tp_anomalia_mensal_treino_fold'
FEATURES_M5 = FEATURES + [FEATURE_SEAS5]
FEATURES_M6 = FEATURES_M5 + [FEATURE_SEAS5_ANOM]
FEATURES_M8 = FEATURES_M6 + [FEATURE_CFSV2, FEATURE_CFSV2_ANOM]
NOMES_MODELOS = ['M6_controle', 'M8_CFSv2']
PESO_FIXO = 0.9
INICIO_COMUM = pd.Timestamp('1982-02-01')
REF_M6 = dict(rmse=1.756329679,mae=1.051989727,vies=0.020945998,peso=0.9)
REF_M6_FOLD = dict(H1=1.728660453,H2=1.815091075,H3=1.754706763,H4=1.700222391,
                   A=1.747480391,B=1.804956318,C=1.740324103)
IMPORTANCIAS, AUDITORIA_PARES = [], []
FASES_CFSV2 = dict(desenvolvimento=('1982-02-01','2020-12-01',467),
    somente_ajuste_final=('2021-01-01','2022-12-01',24),teste=('2023-01-01','2024-12-01',24))






def calendario_pareado(corte):
    corte=pd.Timestamp(corte)
    inicio=max(pd.Timestamp(corte.year-ANOS_TREINO+1,1,1),INICIO_COMUM)
    datas=pd.date_range(inicio,corte,freq='MS')
    exigir(12<=len(datas)<=360 and datas.max()==corte,'Janela de treino fora do protocolo.')
    return datas


def validar_manifesto_cfsv2(m):
    exigir(m.get('schema')==SCHEMA_CFSV2 and m.get('modelo')=='NCEP-CFSv2'
           and m.get('fonte')==FONTE_CFSV2,'Manifesto CFSv2 incompatível.')
    exigir(m.get('lead_iri')==1.5 and m.get('unidade')=='mm/day'
           and m.get('chuva_observada_utilizada') is False,'Produto ou temporalidade CFSv2 incorretos.')
    exigir(m.get('meses')==515 and m.get('inicio_alvos')=='1982-02-01'
           and m.get('fim_alvos')=='2024-12-01','Cobertura CFSv2 divergente.')
    exigir(m.get('politica_membros')==POLITICA_CFSV2
           and m.get('excecoes_permitidas')=={'2019-08-01':[17]},'Política de membros CFSv2 divergente.')
    aplicadas=m.get('excecoes_aplicadas',[])
    exigir(isinstance(aplicadas,list) and len(aplicadas)<=1,'Exceções CFSv2 não previstas.')
    if aplicadas:
        r=aplicadas[0]
        exigir(r.get('time_origem')=='2019-08-01' and r.get('time_alvo')=='2019-09-01'
               and r.get('membros')==23 and r.get('membros_esperados')==24
               and r.get('membros_indisponiveis_ids')==[17], 'Exceção CFSv2 incorreta.')
        a=r.get('auditoria_excecao',{})
        exigir(a.get('membros_indisponiveis_ids')==[17] and
               np.isfinite(a.get('maximo_delta_media_mm_day',np.nan)) and
               0<=a['maximo_delta_media_mm_day']<=1e-5,'Exceção sem média individual conferida.')
    for fase,(inicio,fim,n) in FASES_CFSV2.items():
        item=m.get('arquivos',{}).get(fase,{})
        exigir(item.get('arquivo')==f'cfsv2_{fase}.nc' and item.get('inicio_alvos')==inicio
               and item.get('fim_alvos')==fim and item.get('meses')==n,'Partição CFSv2 incorreta.')


def localizar_cfsv2():
    if PASTA_CFSV2_MANUAL:
        candidatos=[Path(PASTA_CFSV2_MANUAL)/'cfsv2_manifesto.json']
    else:
        candidatos=[]
        for raiz in [Path('/kaggle/input'),Path('/kaggle/working/worcap_CFSv2_dados'),
                     Path.cwd()/'outputs/worcap_CFSv2_dados']:
            if raiz.is_dir(): candidatos.extend(raiz.rglob('cfsv2_manifesto.json'))
    candidatos=sorted(set(p.resolve() for p in candidatos if p.is_file()))
    exigir(len(candidatos)==1,'Anexe a saída concluída do notebook CFSv2 em Input > Notebook '
           'ou defina PASTA_CFSV2_MANUAL; necessário exatamente um manifesto CFSv2.')
    caminho=candidatos[0];m=json.loads(caminho.read_text(encoding='utf-8'))
    validar_manifesto_cfsv2(m)
    return caminho.parent,m


def auditar_cfsv2(ds,uso,m):
    inicio,fim,n=FASES_CFSV2[uso]
    datas=pd.date_range(inicio,fim,freq='MS');origens=(datas.to_period('M')-1).to_timestamp()
    exigir(ds.attrs.get('schema')==SCHEMA_CFSV2 and ds.attrs.get('modelo')=='NCEP-CFSv2'
           and ds.attrs.get('fonte')==FONTE_CFSV2 and ds.attrs.get('lead_iri')==1.5,
           'Proveniência do NetCDF CFSv2 divergente.')
    exigir(ds.attrs.get('politica_membros')==POLITICA_CFSV2,'NetCDF CFSv2 sem política corrigida.')
    exigir(ds.cfsv2_tp_media.dims==('time','lat','lon') and ds.cfsv2_tp_media.attrs.get('units')=='mm/day',
           'Dimensões/unidade CFSv2 inválidas.')
    exigir(pd.DatetimeIndex(ds.time.values).equals(datas),'Datas CFSv2 faltantes, extras ou desordenadas.')
    exigir(pd.DatetimeIndex(ds.time_origem.values).equals(origens),'CFSv2 não foi emitido em T-1.')
    exigir(np.array_equal(ds.lat.values,np.arange(-61,17)) and
           np.array_equal(ds.lon.values,np.arange(-91,-23)),'Grade nativa CFSv2 divergente.')
    for var in ['n_membros','inicializacao_mais_antiga','inicializacao_mais_recente','fase_fonte','time_origem']:
        exigir(ds[var].dims==('time',),f'Coordenada CFSv2 inválida: {var}.')
    dmin=pd.DatetimeIndex(ds.inicializacao_mais_antiga.values)
    dmax=pd.DatetimeIndex(ds.inicializacao_mais_recente.values)
    exigir(not dmin.hasnans and not dmax.hasnans and (dmin<=dmax).all() and (dmax<datas).all()
           and (dmin>=origens-pd.Timedelta(days=40)).all()
           and (dmax<=origens+pd.Timedelta(days=7)).all(),'Inicializações CFSv2 fora da janela permitida.')
    esperados=np.where(origens.month==11,28,24)
    if m['excecoes_aplicadas'] and pd.Timestamp('2019-09-01') in datas:
        esperados[datas==pd.Timestamp('2019-09-01')]=23
    exigir(np.array_equal(ds.n_membros.values,esperados),'Número de membros CFSv2 difere do manifesto.')
    exigir(np.array_equal(ds.fase_fonte.values,(origens>=pd.Timestamp('2011-04-01')).astype(np.int8)),
           'Transição hindcast/operacional CFSv2 incorreta.')
    v=ds.cfsv2_tp_media.values
    exigir(np.isfinite(v).all() and (v>=0).all(),'Precipitação prevista CFSv2 inválida.')


def carregar_cfsv2(uso,referencia):
    item=MANIFESTO_CFSV2['arquivos'][uso];path=PASTA_CFSV2/item['arquivo']
    exigir(path.is_file() and sha256(path)==item['sha256'],f'CFSv2 ausente/alterado: {uso}.')
    with xr.open_dataset(path) as d: bruto=d.load()
    auditar_cfsv2(bruto,uso,MANIFESTO_CFSV2)
    grade=bruto.cfsv2_tp_media.interp(lat=referencia.lat,lon=referencia.lon,method='linear').astype(np.float32)
    exigir(np.isfinite(grade.values).all() and (grade.values>=0).all(),'Interpolação CFSv2 inválida; não extrapolar.')
    for eixo in ['lat','lon']:
        exigir(np.array_equal(grade[eixo].values,referencia[eixo].values),'Grade CFSv2 desalinhada.')
    for coord in ['time_origem','n_membros','inicializacao_mais_antiga','inicializacao_mais_recente','fase_fonte']:
        grade=grade.assign_coords({coord:('time',bruto[coord].values)})
    grade.attrs.update(units='mm/day',lead_iri=1.5)
    salvar_json(f'auditoria_CFSv2_{uso}.json',dict(arquivo=str(path),sha256=item['sha256'],
        meses=bruto.sizes['time'],minimo=float(grade.min()),maximo=float(grade.max()),
        membros_usados=np.unique(bruto.n_membros.values).tolist(),interpolacao='bilinear sem extrapolacao'))
    return grade


def ajustar_clima_previsao(previsoes,corte,nome):
    treino=calendario_pareado(corte);datas=pd.DatetimeIndex(previsoes.time.values)
    exigir(datas.is_unique and datas.is_monotonic_increasing and treino.isin(datas).all(),
           f'Calendário incompleto/desordenado: {nome}.')
    sel=previsoes.sel(time=treino)
    exigir(pd.DatetimeIndex(sel.time_origem.values).equals((treino.to_period('M')-1).to_timestamp()),
           f'Origem incorreta na climatologia {nome}.')
    v=sel.values
    exigir(np.isfinite(v).all() and (v>=0).all(),f'Valores inválidos na climatologia {nome}.')
    n=np.array([(treino.month==m).sum() for m in range(1,13)],dtype=np.int16)
    exigir((n>0).all() and n.max()<=30,'Climatologia de previsão excede 30 anos ou mês ausente.')
    medias=np.stack([v[treino.month==m].mean(axis=0,dtype=np.float64) for m in range(1,13)]).astype(np.float32)
    return xr.DataArray(medias,dims=('month','lat','lon'),
        coords={'month':np.arange(1,13),'lat':previsoes.lat,'lon':previsoes.lon,'n_meses':('month',n)},
        name=f'climatologia_{nome}',attrs=dict(units='mm/day',inicio=str(treino[0].date()),
        corte=str(pd.Timestamp(corte).date()),meses_treino=len(treino),anos_min=int(n.min()),anos_max=int(n.max()),
        referencia='somente preditores nos meses do treino pareado; uma media de ensemble por ano/mes'))


def ajustar_clima_seas5(seas,corte):
    return ajustar_clima_previsao(seas,corte,'SEAS5')


def valores_cfsv2(cfs,destino,pontos=None):
    # Mesma seleção T-1 do SEAS5, mais a conferência das datas reais dos membros.
    destino=pd.Timestamp(destino)
    exigir(pd.Timestamp(cfs.inicializacao_mais_recente.sel(time=destino).values)<destino,
           'Inicialização real CFSv2 invade o alvo.')
    return valores_seas5(cfs,destino,pontos)


def valores_anomalia_cfsv2(cfs,clima_cfs,destino,pontos=None):
    for eixo in ['lat','lon']:
        exigir(np.array_equal(cfs[eixo],clima_cfs[eixo]),'Climatologia CFSv2 desalinhada.')
    bruto=valores_cfsv2(cfs,destino,pontos)
    ref=clima_cfs.sel(month=pd.Timestamp(destino).month).values.ravel()
    if pontos is not None: ref=ref[pontos]
    anom=(bruto-ref).astype(np.float32)
    exigir(np.isfinite(anom).all(),'Anomalia CFSv2 inválida.')
    return anom


def auditar_invariancia_previsao(previsoes,corte,nome,identificador):
    clima=ajustar_clima_previsao(previsoes,corte,nome)
    datas=pd.DatetimeIndex(previsoes.time.values);i=np.flatnonzero(datas>pd.Timestamp(corte))
    exigir(len(i)>0,'Auditoria precisa de meses após o corte.')
    t=datas[i[0]]
    getter=valores_anomalia_seas5 if nome=='SEAS5' else valores_anomalia_cfsv2
    antes=getter(previsoes,clima,t).copy()
    copia=previsoes.values[i].copy()
    try:
        previsoes.values[i]=12345.
        exigir(np.array_equal(clima,ajustar_clima_previsao(previsoes,corte,nome)),
               'Validação alterou climatologia de treino.')
    finally: previsoes.values[i]=copia
    i=np.flatnonzero(pd.DatetimeIndex(previsoes.time_origem.values)>=t)
    copia=previsoes.values[i].copy()
    try:
        previsoes.values[i]=54321.
        exigir(np.array_equal(antes,getter(previsoes,clima,t)),'Inicialização futura alterou a feature de T.')
    finally: previsoes.values[i]=copia
    salvar_json(f'invariancia_{nome}_{identificador}.json',dict(corte=corte,
        validacao_nao_altera_clima=True,inicializacoes_futuras_nao_alteram_feature=True))


def amostrar_m8(chuva,atmosfera,seas,cfs,corte):
    x6,y,ids,destinos,clima,ca,somas,contagens,cs=amostrar_m6(chuva,atmosfera,seas,corte)
    cc=ajustar_clima_previsao(cfs,corte,'CFSv2')
    X=np.empty((len(x6),31),dtype=np.float32);X[:,:29]=x6;n=ids.shape[1]
    for i,t in enumerate(destinos):
        sl=slice(i*n,(i+1)*n)
        X[sl,29]=valores_cfsv2(cfs,t,ids[i])
        X[sl,30]=valores_anomalia_cfsv2(cfs,cc,t,ids[i])
    exigir(np.array_equal(X[:,:29],x6),'29 features do controle alteradas.')
    exigir(np.isfinite(X).all() and np.isfinite(y).all(),'Matriz de treino inválida.')
    return X,y,ids,destinos,clima,ca,somas,contagens,cs,cc


def treinar_par(chuva,atmosfera,seas,cfs,corte,identificador,modelos=None):
    conferir_protocolo()
    nomes=NOMES_MODELOS if modelos is None else modelos
    X,y,ids,destinos,clima,ca,somas,contagens,cs,cc=amostrar_m8(chuva,atmosfera,seas,cfs,corte)
    hash_x=hashlib.sha256(np.ascontiguousarray(X[:,:29]).tobytes()).hexdigest()
    registro=dict(ajuste=identificador,inicio=str(destinos[0].date()),fim=str(destinos[-1].date()),
        meses=len(destinos),exemplos=len(X),anos_climatologia_chuva=30,
        anos_mensais_clima_SEAS5=cs.n_meses.values.tolist(),anos_mensais_clima_CFSv2=cc.n_meses.values.tolist(),
        sha256_29_features=hash_x,sha256_y=hashlib.sha256(y.tobytes()).hexdigest(),
        sha256_pontos=hashlib.sha256(ids.tobytes()).hexdigest(),
        identidade_pareada='mesmo X[:,:29], y, pontos e referencias',
        cfsv2_bruto_min=float(X[:,29].min()),cfsv2_bruto_max=float(X[:,29].max()),
        cfsv2_anomalia_min=float(X[:,30].min()),cfsv2_anomalia_max=float(X[:,30].max()))
    AUDITORIA_PARES.append(registro);salvar_json(f'auditoria_par_{identificador}.json',registro)
    np.savez_compressed(SAIDA/f'amostras_{identificador}.npz',time_alvo=destinos.values,pontos=ids)
    np.savez_compressed(SAIDA/f'referencias_LOO_NOAA_{identificador}.npz',
        soma_chuva=somas,contagens=contagens,clima_indices=ca['_clima_indices'])
    clima.to_netcdf(SAIDA/f'clima_chuva_{identificador}.nc')
    cs.to_netcdf(SAIDA/f'clima_SEAS5_{identificador}.nc');cc.to_netcdf(SAIDA/f'clima_CFSv2_{identificador}.nc')
    print(f'{identificador} | {len(destinos)} meses; {len(X):,} exemplos; mesmas 29 features no controle',flush=True)
    resultado={}
    for nome in nomes:
        inicio=perf_counter();features=FEATURES_M6 if nome=='M6_controle' else FEATURES_M8
        entrada=np.ascontiguousarray(X[:,:len(features)])
        exigir(hashlib.sha256(np.ascontiguousarray(entrada[:,:29]).tobytes()).hexdigest()==hash_x,
               'Features compartilhadas alteradas.')
        treino=lgb.Dataset(entrada,label=y,feature_name=features)
        modelo=lgb.train(PARAMETROS,treino,num_boost_round=ARVORES)
        exigir(modelo.feature_name()==features,'Ordem de features divergente.')
        modelo.save_model(str(SAIDA/f'modelo_{nome}_{identificador}.txt'))
        imp=pd.DataFrame(dict(modelo=nome,ajuste=identificador,feature=features,gain=modelo.feature_importance('gain')))
        imp['percentual_gain']=100*imp.gain/imp.gain.sum() if imp.gain.sum() else 0.
        imp.to_csv(SAIDA/f'importancias_{nome}_{identificador}.csv',index=False)
        if identificador!='final':IMPORTANCIAS.append(imp)
        resultado[nome]=modelo
        del treino,entrada;gc.collect()
        print(nome,identificador,'concluído em',round(perf_counter()-inicio,1),'s',flush=True)
    del X,y,ids,somas,contagens;gc.collect()
    return resultado,clima,ca,cs,cc


def prever_par(modelo,nome,atmosfera,seas,cfs,ca,clima,cs,cc,destinos):
    destinos=pd.DatetimeIndex(destinos)
    exigir(nome in NOMES_MODELOS,'Modelo desconhecido.')
    exigir((destinos>pd.Timestamp(cs.attrs['corte'])).all() and
           cs.attrs['corte']==cc.attrs['corte'],'Inferência fora do período posterior ao corte.')
    out=np.empty((len(destinos),clima.sizes['lat'],clima.sizes['lon']),dtype=np.float32)
    for i,t in enumerate(destinos):
        x=np.column_stack([matriz_mes(atmosfera,ca,clima,t),valores_seas5(seas,t),
                           valores_anomalia_seas5(seas,cs,t)]).astype(np.float32)
        if nome=='M8_CFSv2':
            x=np.column_stack([x,valores_cfsv2(cfs,t),valores_anomalia_cfsv2(cfs,cc,t)]).astype(np.float32)
        exigir(x.shape[1]==(29 if nome=='M6_controle' else 31) and np.isfinite(x).all(),'Inferência inválida.')
        out[i]=modelo.predict(x).reshape(out.shape[1:])
    exigir(np.isfinite(out).all(),'Previsões inválidas.')
    return xr.DataArray(out,dims=('time','lat','lon'),coords={'time':destinos,'lat':clima.lat,'lon':clima.lon},attrs={'units':'mm/day'})





CAP_IMPLEMENTACAO_SHA256 = 'b9ff29e1b2f8a6cb34e971884e84bb5ce6dc60d9da6268f74f49d5c4fb7859dd'

import os
from zipfile import ZipFile, ZIP_STORED
# %% Leitura auditada de referências e backup
def cap_buscar(nome):
    raizes = [Path('/kaggle/input'), Path('/kaggle/working')] if Path('/kaggle/input').is_dir() else [BASE, Path.cwd()/'data/raw']
    vistos, arquivos = set(), set()
    for raiz in raizes:
        if not raiz.is_dir(): continue
        for pasta, dirs, nomes in os.walk(raiz, followlinks=True):
            real = os.path.realpath(pasta)
            if real in vistos:
                dirs[:] = []
                continue
            vistos.add(real)
            if nome in nomes: arquivos.add((Path(pasta)/nome).resolve())
    return sorted(arquivos)

def cap_localizar_referencia():
    pastas = [Path(CAP_PASTA_8B_MANUAL)] if CAP_PASTA_8B_MANUAL else [p.parent for p in cap_buscar('previsoes_M6_controle_H1.npz')]
    nomes = [f'previsoes_{m}_{b["nome"]}.npz' for b in BLOCOS for m in NOMES_MODELOS]
    nomes += ['manifesto.json', 'entradas.json', 'metricas_mensais.csv']
    validas = sorted({p.resolve() for p in pastas if all((p/n).is_file() for n in nomes)})
    exigir(len(validas) == 1, 'Preciso da saída completa da 8B ou do backup_8B_para_8C.zip EXTRAÍDO. '
           f'Pastas completas: {[str(p) for p in validas]}. Se houver várias, defina CAP_PASTA_8B_MANUAL.')
    return validas[0], nomes

def cap_ler_npz(path, datas, referencia):
    with np.load(path, allow_pickle=False) as z:
        exigir({'time','lat','lon','baseline','anomalia'} <= set(z.files), f'NPZ incompleto: {path}')
        exigir(pd.DatetimeIndex(z['time']).equals(datas), f'Datas incorretas: {path}')
        exigir(np.array_equal(z['lat'],referencia.lat.values) and np.array_equal(z['lon'],referencia.lon.values),
               f'Grade/ordem incorreta: {path}')
        c,a=z['baseline'],z['anomalia']
    exigir(c.shape==a.shape==(len(datas),*CAP_GRADE) and c.dtype==a.dtype==np.float32,'Formato NPZ divergente.')
    exigir(np.isfinite(c).all() and np.isfinite(a).all() and (c>=0).all(),'Valores NPZ inválidos.')
    return c,a

def cap_metricas_peso42(mensais, modelo):
    ref=CAP_METRICAS_8B[(CAP_METRICAS_8B.modelo==modelo)&np.isclose(CAP_METRICAS_8B.peso,CAP_PESOS[modelo],rtol=0,atol=1e-12)]
    ref=ref[ref.bloco.isin(mensais.bloco.unique())].copy()
    ref.mes_alvo=pd.to_datetime(ref.mes_alvo).dt.strftime('%Y-%m-%d')
    ref=ref.sort_values('mes_alvo'); atual=mensais.sort_values('mes_alvo')
    exigir(len(ref)==len(atual) and ref.mes_alvo.tolist()==atual.mes_alvo.tolist(),'Meses da referência 42 incompletos.')
    for k in ['sse','soma_erro','soma_erro_absoluto','n']:
        exigir(np.allclose(ref[k],atual[k],rtol=1e-10,atol=1e-5),f'Semente 42 não reproduziu métrica 8B: {modelo}/{k}')

def cap_chuva_prevista(par,peso,referencia,datas):
    c,a=par
    valores=np.maximum(0,c+float(peso)*a).astype(np.float32)
    return xr.DataArray(valores,dims=('time','lat','lon'),coords=dict(time=datas,lat=referencia.lat,lon=referencia.lon),
        name='tp_mm_day',attrs={'units':'mm/day'})

def cap_csv42(referencia):
    candidatos=[Path(CAP_CSV_8C_MANUAL)] if CAP_CSV_8C_MANUAL else cap_buscar(CAP_NOME_CAMPEAO)
    validos=[]
    for p in candidatos:
        if not p.is_file():continue
        mp=p.parent/'manifesto.json'
        if not mp.is_file():continue
        m=json.loads(mp.read_text(encoding='utf-8'))
        if m.get('protocolo',{}).get('nome')!='M6_M8_media_fixa_50_50':continue
        exigir(m['protocolo']['fracao_M6']==.5 and m['protocolo']['peso_interno_M6']==.9 and
               m['protocolo']['peso_interno_M8']==.875,'CSV 8C usa outra mistura.')
        exigir(abs(m['decisao']['rmse_media']-referencia)<1e-6,'Referência 8C diferente desta avaliação.')
        exigir(sha256(p)==m['auditoria']['submissao_sha256'],'CSV 8C alterado desde a geração.')
        validos.append(p)
    exigir(validos,'Para reutilizar a semente 42 final, preciso do CSV campeão da 8C e seu manifesto.json na mesma pasta. '
           'Anexe ambos pelo Upload > New Dataset, ou execute na sessão 8B/8C atual. Não é necessário retreinar a semente 42.')
    exigir(len({sha256(p) for p in validos})==1,'Há CSVs 8C diferentes; indique CAP_CSV_8C_MANUAL.')
    return validos[0]

def cap_grade_csv(csv,template):
    acumulado=np.empty(template.shape,dtype=np.float64)
    vistos=np.zeros(acumulado.size,dtype=bool)
    colunas=['id','tp_mm_day']
    with pd.read_csv(csv,dtype={'id':str},chunksize=100000) as pred, \
         pd.read_csv(PASTA/'sample_submission.csv',dtype={'id':str},chunksize=100000) as sample:
        for p,s in zip_longest(pred,sample):
            exigir(p is not None and s is not None,'CSV 8C/sample com tamanhos diferentes.')
            exigir(p.columns.tolist()==s.columns.tolist()==colunas and p.id.equals(s.id),'IDs/ordem/colunas do CSV 8C divergentes.')
            ti,yi,xi=indices_dos_ids(s.id,template)
            inds=np.ravel_multi_index((ti,yi,xi),acumulado.shape)
            exigir(len(np.unique(inds))==len(inds) and not vistos[inds].any(),'IDs duplicados na previsão 42.')
            valores=p.tp_mm_day.to_numpy(dtype=np.float64)
            exigir(np.isfinite(valores).all() and (valores>=0).all(),'CSV 8C tem valores inválidos.')
            acumulado[ti,yi,xi]=valores;vistos[inds]=True
    exigir(vistos.all(),'CSV 8C não cobre a grade inteira.')
    return acumulado

def cap_backup():
    destino=CAP_SAIDA/'backup_8F_completo.zip'
    temporario=CAP_SAIDA/'backup_8F_completo.partial.zip'
    with ZipFile(temporario,'w',ZIP_STORED) as z:
        for p in CAP_SAIDA.rglob('*'):
            if p.is_file() and p not in [destino,temporario]: z.write(p,arcname=f'etapa8F/{p.relative_to(CAP_SAIDA).as_posix()}')
        for nome in CAP_NOMES_REF:z.write(CAP_PASTA_8B/nome,arcname=f'referencia8B/{nome}')
        if 'CAP_CAMPEAO' in globals():
            z.write(CAP_CAMPEAO,arcname=f'referencia8C/{CAP_CAMPEAO.name}')
            z.write(CAP_CAMPEAO.parent/'manifesto.json',arcname='referencia8C/manifesto.json')
    temporario.replace(destino)
    print('BAIXE o backup completo para preservar os modelos e previsões:')
    display(FileLink(os.path.relpath(destino,Path.cwd())))

# %% SST NOAA ERSSTv5: aquisição pequena e auditoria, sem chuva observada
import requests
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

SST_COMPONENTES = 8
SST_NOME = 'ersstv5_4graus_198201_202411.nc'
SST_URL = 'https://psl.noaa.gov/thredds/ncss/grid/Datasets/noaa.ersst.v5/sst.mnmean.nc'
SST_PEDIDO = dict(var='sst',north=60,west=0,east=358,south=-60,horizStride=2,
    time_start='1982-01-01T00:00:00Z',time_end='2024-11-01T00:00:00Z',timeStride=1,accept='netcdf4')
SST_FEATURES = [f'sst_lag1_pc_{i:02d}_treino_fold' for i in range(1,SST_COMPONENTES+1)]
FEATURES_M10 = FEATURES_M6 + SST_FEATURES
VERSOES['scikit-learn']=importlib.metadata.version('scikit-learn')


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


def sst_obter():
    cache=BASE/'worcap_SST_8F_dados';cache.mkdir(parents=True,exist_ok=True)
    path=Path(SST_ARQUIVO_MANUAL) if SST_ARQUIVO_MANUAL else cache/SST_NOME
    if not path.is_file() and not SST_ARQUIVO_MANUAL:
        candidatos=cap_buscar(SST_NOME)
        if candidatos:
            hashes={sha256(p) for p in candidatos}
            exigir(len(hashes)==1,'Há fontes SST distintas; defina SST_ARQUIVO_MANUAL.')
            path=candidatos[0]
    if not path.is_file():
        exigir(not SST_ARQUIVO_MANUAL,'SST_ARQUIVO_MANUAL não existe.')
        print('Baixando SST NOAA: recorte de aproximadamente 3,6 MB; Internet deve estar ligada.',flush=True)
        parcial=path.with_suffix('.partial.nc')
        try:
            with requests.get(SST_URL,params=SST_PEDIDO,stream=True,timeout=(15,120)) as resposta:
                resposta.raise_for_status()
                with parcial.open('wb') as f:
                    for bloco in resposta.iter_content(1024*1024):f.write(bloco)
            sst_auditar_arquivo(parcial)
            parcial.replace(path)
        except Exception as exc:
            raise RuntimeError('Falha no download SST. Ative Internet no Kaggle ou anexe o NetCDF fornecido e indique SST_ARQUIVO_MANUAL. Os modelos anteriores estão preservados.') from exc
    sst,meta=sst_auditar_arquivo(path)
    meta.update(url=SST_URL,pedido=SST_PEDIDO)
    salvo=SAIDA/'fonte_SST.json'
    if salvo.is_file():
        exigir(json.loads(salvo.read_text(encoding='utf-8'))['sha256']==meta['sha256'],'Snapshot SST mudou ao retomar.')
    salvar_json('fonte_SST.json',meta)
    print('SST auditada:',sst.shape,'| origens 1982-01 a 2024-11 | graus Celsius.',flush=True)
    return sst,meta,path


def sst_ajustar_pca(sst,alvos_treino):
    alvos_treino=pd.DatetimeIndex(alvos_treino)
    exigir(alvos_treino.is_unique and alvos_treino.equals(pd.date_range(alvos_treino.min(),alvos_treino.max(),freq='MS')),
           'Calendário de ajuste da PCA inválido.')
    origens=(alvos_treino.to_period('M')-1).to_timestamp()
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
    alvos=pd.DatetimeIndex(alvos);origens=(alvos.to_period('M')-1).to_timestamp()
    exigir(np.array_equal(sst.lat,estado['lat']) and np.array_equal(sst.lon,estado['lon']),'Grade SST mudou.')
    v=sst.sel(time=origens).values.reshape(len(origens),-1).astype(np.float64)[:,estado['mascara']]
    exigir(np.isfinite(v).all(),'SST ausente no oceano selecionado pelo treino; não imputar usando futuro.')
    x=(v-estado['clima_mensal'][origens.month-1])*estado['pesos_area']-estado['centro']
    with threadpool_limits(limits=1):scores=x@estado['componentes'].T
    exigir(scores.shape==(len(alvos),SST_COMPONENTES) and np.isfinite(scores).all(),'Scores PCA inválidos.')
    return scores.astype(np.float32)


def sst_auditar_causalidade(sst,alvos_treino,alvos_inferencia,estado):
    # Mudar SST no alvo T e depois não pode alterar o ajuste nem os scores de T-1.
    primeiro=pd.Timestamp(alvos_inferencia[0]);antes=sst_transformar(sst,estado,[primeiro])
    copia=sst.copy(deep=True);datas=pd.DatetimeIndex(copia.time.values)
    copia.values[datas>=primeiro]=12345.
    novo=sst_ajustar_pca(copia,alvos_treino)
    for chave in ['mascara','clima_mensal','pesos_area','centro','componentes','origens_treino']:
        exigir(np.array_equal(estado[chave],novo[chave]),'SST posterior alterou o ajuste: '+chave)
    exigir(np.array_equal(antes,sst_transformar(copia,novo,[primeiro])),'SST de T/futuro alterou a entrada em T-1.')
    return dict(ajuste_somente_treino=True,mutacao_T_e_futuro_nao_altera_entrada_Tmenos1=True,
        ultimo_mes_ajuste=str(pd.Timestamp(estado['origens_treino'][-1]).date()),primeiro_alvo=str(primeiro.date()))


def sst_treinar_prever(chuva,atmosfera,seas,cfs,corte,identificador,datas,atmos_pred,seas_pred):
    # A função de amostragem preservada fornece exatamente X6, y e pontos da 8B.
    Xref,y,ids,alvos,clima,ca,somas,contagens,cs,cc=amostrar_m8(chuva,atmosfera,seas,cfs,corte)
    estado=sst_ajustar_pca(SST_DADOS,alvos)
    audit_pca=sst_auditar_causalidade(SST_DADOS,alvos,datas,estado)
    scores=sst_transformar(SST_DADOS,estado,alvos);scores_pred=sst_transformar(SST_DADOS,estado,datas)
    X=np.empty((len(Xref),len(FEATURES_M10)),dtype=np.float32);X[:,:29]=Xref[:,:29]
    X[:,29:]=np.repeat(scores,ids.shape[1],axis=0)
    exigir(np.array_equal(X[:,:29],Xref[:,:29]) and np.isfinite(X).all(),'M6 foi alterado ou PCA inválida.')
    registro=dict(ajuste=identificador,inicio=str(alvos[0].date()),fim=str(alvos[-1].date()),meses=len(alvos),
        exemplos=len(X),features=FEATURES_M10,anos_climatologia_chuva=30,
        anos_mensais_clima_SEAS5=cs.n_meses.values.tolist(),anos_mensais_clima_CFSv2=cc.n_meses.values.tolist(),
        sha256_29_features=hashlib.sha256(np.ascontiguousarray(X[:,:29]).tobytes()).hexdigest(),
        sha256_y=hashlib.sha256(y.tobytes()).hexdigest(),sha256_pontos=hashlib.sha256(ids.tobytes()).hexdigest(),
        pontos_oceanicos=int(estado['mascara'].sum()),variancia_explicada=estado['variancia_explicada'].tolist(),
        total_variancia_explicada=float(estado['variancia_explicada'].sum()),auditoria_PCA=audit_pca,
        fonte_SST_sha256=SST_META['sha256'])
    antiga=CAP_PASTA_8B/f'auditoria_par_{identificador}.json'
    if identificador!='final' and antiga.is_file():
        ref=json.loads(antiga.read_text(encoding='utf-8'))
        for chave in ['sha256_29_features','sha256_y','sha256_pontos']:
            exigir(registro[chave]==ref[chave],'Amostragem M10 diferente do controle: '+chave)
    salvar_json(f'auditoria_par_{identificador}.json',registro)
    np.savez_compressed(SAIDA/'pca_SST.npz',**estado)
    np.savez_compressed(SAIDA/f'amostras_{identificador}.npz',time_alvo=alvos.values,pontos=ids)
    np.savez_compressed(SAIDA/f'referencias_LOO_NOAA_{identificador}.npz',soma_chuva=somas,contagens=contagens,clima_indices=ca['_clima_indices'])
    clima.to_netcdf(SAIDA/f'clima_chuva_{identificador}.nc');cs.to_netcdf(SAIDA/f'clima_SEAS5_{identificador}.nc')
    for nome,d,pcs in [('treino',alvos,scores),('inferencia',datas,scores_pred)]:
        t=pd.DataFrame(pcs,columns=SST_FEATURES);t.insert(0,'time_alvo',d)
        t.insert(1,'time_origem',(d.to_period('M')-1).to_timestamp());t.to_csv(SAIDA/f'scores_PCA_{nome}.csv',index=False)
    print(f'{identificador}: {len(X):,} exemplos; 37 features; PCA explica {registro["total_variancia_explicada"]:.1%} da variância ponderada do treino.',flush=True)
    del Xref,ids,somas,contagens;gc.collect()
    inicio=perf_counter();treino=lgb.Dataset(X,label=y,feature_name=FEATURES_M10)
    modelo=lgb.train(PARAMETROS,treino,num_boost_round=ARVORES)
    exigir(modelo.feature_name()==FEATURES_M10,'Ordem de features M10 diferente.')
    modelo.save_model(str(SAIDA/f'modelo_M10_SST_PCA_{identificador}.txt'))
    imp=pd.DataFrame(dict(feature=FEATURES_M10,gain=modelo.feature_importance('gain')))
    imp.to_csv(SAIDA/'importancias.csv',index=False)
    del X,y,treino;gc.collect()
    print('M10 ajustado em',round(perf_counter()-inicio,1),'s; iniciando inferência.',flush=True)
    exigir((pd.DatetimeIndex(datas)>pd.Timestamp(cs.attrs['corte'])).all(),'Inferência anterior ao corte.')
    out=np.empty((len(datas),clima.sizes['lat'],clima.sizes['lon']),dtype=np.float32)
    for i,t in enumerate(datas):
        base=np.column_stack([matriz_mes(atmos_pred,ca,clima,t),valores_seas5(seas_pred,t),valores_anomalia_seas5(seas_pred,cs,t)]).astype(np.float32)
        x=np.column_stack([base,np.broadcast_to(scores_pred[i],(len(base),SST_COMPONENTES))]).astype(np.float32)
        exigir(x.shape[1]==37 and np.isfinite(x).all(),'Features de inferência M10 inválidas.')
        out[i]=modelo.predict(x).reshape(out.shape[1:])
    anom=xr.DataArray(out,dims=('time','lat','lon'),coords={'time':datas,'lat':clima.lat,'lon':clima.lon})
    exigir(np.isfinite(out).all(),'Inferência M10 não finita.')
    return anom,clima,registro


SST_DADOS,SST_META,SST_PATH=sst_obter()
# %% Protocolo: M10 com oito PCs de SST e mistura fixa com a 8C
import os
from zipfile import ZipFile, ZIP_STORED

CAP_SAIDA = SAIDA
CAP_PARAMETROS_BASE = dict(PARAMETROS)
CAP_PARAMETROS_NOVOS = dict(PARAMETROS)
CAP_ARVORES = 300
CAP_PESO = .9
CAP_PESOS = {'M6_controle': .9, 'M8_CFSv2': .875}
CAP_GRADE = (301, 261)
CAP_REF_RMSE = 1.753320754
CAP_NOME_CAMPEAO = 'submission_media_M6_0.900_M8_0.875_50_50.csv'
CAP_CANDIDATOS = ['M10_SST_PCA', 'media_M10_8C_50_50']


def cap_protocolo():
    return dict(schema='worcap_8f_sst_pca_m10_v1', implementacao_sha256=CAP_IMPLEMENTACAO_SHA256,
        parametros_referencia=CAP_PARAMETROS_BASE, parametros_M10=CAP_PARAMETROS_NOVOS,
        arvores_M10=CAP_ARVORES, peso_M10=CAP_PESO, semente=42,
        features_M10=FEATURES_M10, pontos_por_mes=PONTOS_POR_MES, anos_maximos=ANOS_TREINO,
        blocos=BLOCOS, inicio_comum=str(INICIO_COMUM.date()),
        climatologias='iguais a 8B; LOO chuva somente no treino; demais transformações preservadas',
        candidatos=CAP_CANDIDATOS, fracao_M10_na_mistura=.5, fracao_8C_na_mistura=.5,
        selecao='menor RMSE global em 2007–2020; empate até 1e-8 favorece mistura; precisa superar 8C por 1e-8',
        sem_busca_parametros=True, sem_busca_pesos=True, sem_escolha_semente=True,
        treino_final='1993-01 a 2022-12; apenas um M10, após congelar a seleção histórica',
        referencia_8B_sha256=CAP_HASHES_REF, entradas=ENTRADAS,
        manifesto_SEAS5_sha256=HASH_MANIFESTO_SEAS5, manifesto_CFSv2_sha256=HASH_MANIFESTO_CFSV2,
        NOAA_sha256=INDICES_CSV_SHA256, SST_sha256=SST_META['sha256'],
        SST_componentes=SST_COMPONENTES, SST_mascara='finito e variância não nula somente no treino',
        SST_climatologia='mensal por ponto; origens dos exemplos de treino; máximo 30 anos',
        SST_transformacao='SST T-1, anomalia mensal, sqrt(cos(lat)), PCA full sem whitening; oito PCs',
        SST_sem_busca_numero_componentes=True, reserva_avaliada=False, envio_automatico=False)


def conferir_protocolo():
    exigir(hashlib.sha256(json.dumps(cap_protocolo(),sort_keys=True).encode()).hexdigest()==CAP_HASH_PROTOCOLO,
           'Protocolo 8F alterado durante a execução.')
    exigir((PONTOS_POR_MES,ANOS_TREINO,SEMENTE,ARVORES)==(5000,30,42,300) and
           PARAMETROS==CAP_PARAMETROS_NOVOS, 'Configuração SST/M6 divergente do teste predefinido.')
    exigir(len(FEATURES_M6)==29 and FEATURES_M8[:29]==FEATURES_M6 and
           FEATURES_M10[:29]==FEATURES_M6 and len(FEATURES_M10)==37 and SST_COMPONENTES==8,'Features/PCA alteradas.')


def cap_ajustar(identificador,corte,datas,chuva,atmosfera,seas,cfs,atmos_inferencia=None,seas_inferencia=None,cfs_inferencia=None):
    global SAIDA
    conferir_protocolo()
    pasta=CAP_SAIDA/'ajustes'/identificador;pasta.mkdir(parents=True,exist_ok=True)
    selo=pasta/'cache_concluido.json'
    assinatura=dict(protocolo_sha256=CAP_HASH_PROTOCOLO,ajuste=identificador,corte=str(corte),
                    datas=[str(d.date()) for d in datas])
    if selo.is_file():
        meta=json.loads(selo.read_text(encoding='utf-8'))
        exigir(meta['assinatura']==assinatura,'Cache de outro protocolo.')
        exigir(all((pasta/n).is_file() and sha256(pasta/n)==h for n,h in meta['arquivos'].items()),'Cache incompleto/alterado.')
        print(f'M10 {identificador}: cache conferido; sem novo treinamento.',flush=True)
    else:
        anterior=SAIDA
        try:
            SAIDA=pasta
            anom,clima,aud=sst_treinar_prever(chuva,atmosfera,seas,cfs,corte,identificador,datas,
                atmosfera if atmos_inferencia is None else atmos_inferencia,
                seas if seas_inferencia is None else seas_inferencia)
            np.savez_compressed(pasta/'previsoes_M10.npz',time=datas.values,lat=clima.lat.values,
                lon=clima.lon.values,baseline=clima.values[datas.month-1],anomalia=anom.values)
            exigir(aud['inicio']==str(calendario_pareado(corte)[0].date()) and aud['fim']==str(pd.Timestamp(corte).date()),'Calendário diferente.')
            arquivos={p.name:sha256(p) for p in pasta.iterdir() if p.is_file() and p.name!='cache_concluido.json'}
            selo.write_text(json.dumps(dict(assinatura=assinatura,arquivos=arquivos,auditoria=aud),
                ensure_ascii=False,indent=2,default=str),encoding='utf-8')
            del anom,clima
        finally:SAIDA=anterior
        gc.collect()
    aud=json.loads(selo.read_text(encoding='utf-8'))['auditoria']
    exigir(aud['meses']==len(calendario_pareado(corte)),'Quantidade de meses divergente.')
    return cap_ler_npz(pasta/'previsoes_M10.npz',datas,chuva),aud



def cap_escolher(tabela):
    valores=tabela.set_index('modelo').rmse
    ref=float(valores['ensemble_8C'])
    escolhido=('media_M10_8C_50_50' if valores['media_M10_8C_50_50']<=valores['M10_SST_PCA']+1e-8
               else 'M10_SST_PCA')
    return dict(modelo=escolhido,rmse=float(valores[escolhido]),rmse_8C=ref,
                gerar_candidato=bool(valores[escolhido]<ref-1e-8))


# %% Entradas preservadas; recuperar as referências 8B e 8C
exigir((PONTOS_POR_MES,ANOS_TREINO,ARVORES,SEMENTE)==(5000,30,300,42),'Configuração original divergente.')
exigir(VERSOES['numpy']=='2.0.2' and VERSOES['lightgbm']=='4.6.0','Use NumPy 2.0.2 e LightGBM 4.6.0, como antes.')
CAP_PASTA_8B,CAP_NOMES_REF=cap_localizar_referencia()
CAP_HASHES_REF={n:sha256(CAP_PASTA_8B/n) for n in CAP_NOMES_REF}
CAP_REF=json.loads((CAP_PASTA_8B/'manifesto.json').read_text(encoding='utf-8'))
CAP_ENTRADAS_REF=json.loads((CAP_PASTA_8B/'entradas.json').read_text(encoding='utf-8'))
exigir(CAP_REF['protocolo']['semente']==42 and CAP_REF['protocolo']['parametros']==CAP_PARAMETROS_BASE,
       'Referência usa outra configuração/semente.')
exigir(CAP_REF['protocolo']['features_controle']==FEATURES_M6 and CAP_REF['protocolo']['features_candidato']==FEATURES_M8,
       'Features de referência diferentes.')
PASTA=localizar_dados(PASTA_DADOS_MANUAL)
PASTA_SEAS5,MANIFESTO_SEAS5=localizar_seas5();PASTA_CFSV2,MANIFESTO_CFSV2=localizar_cfsv2()
HASH_MANIFESTO_SEAS5=sha256(PASTA_SEAS5/'seas5_51_manifesto.json')
HASH_MANIFESTO_CFSV2=sha256(PASTA_CFSV2/'cfsv2_manifesto.json')
exigir(HASH_MANIFESTO_SEAS5==CAP_REF['fonte_seas5_manifesto_sha256'] and
       HASH_MANIFESTO_CFSV2==CAP_REF['fonte_cfsv2_manifesto_sha256'],'Use os mesmos manifestos/dados da 8B.')
ENTRADAS=[dict(nome=e['nome'],sha256=sha256(PASTA/e['nome']),bytes=(PASTA/e['nome']).stat().st_size) for e in CAP_ENTRADAS_REF]
exigir([e['sha256'] for e in ENTRADAS]==[e['sha256'] for e in CAP_ENTRADAS_REF],'Dados oficiais diferentes da 8B.')
CAP_METRICAS_8B=pd.read_csv(CAP_PASTA_8B/'metricas_mensais.csv')
CAP_CAMPEAO=cap_csv42(CAP_REF_RMSE)
PARAMETROS=dict(CAP_PARAMETROS_NOVOS);ARVORES=CAP_ARVORES
import shutil
if SST_PATH.resolve()!=(CAP_SAIDA/SST_NOME).resolve():shutil.copyfile(SST_PATH,CAP_SAIDA/SST_NOME)
CAP_HASH_PROTOCOLO=hashlib.sha256(json.dumps(cap_protocolo(),sort_keys=True).encode()).hexdigest()
if (CAP_SAIDA/'protocolo.json').is_file():
    exigir(json.loads((CAP_SAIDA/'protocolo.json').read_text(encoding='utf-8'))==cap_protocolo(),'Retomada de outro protocolo.')
salvar_json('protocolo.json',cap_protocolo());salvar_json('entradas.json',ENTRADAS)
salvar_json('fontes_SEAS5.json',MANIFESTO_SEAS5);salvar_json('fontes_CFSv2.json',MANIFESTO_CFSV2)
conferir_protocolo()
tp=auditar_chuva_desenv()
origem_min=pd.Timestamp('1976-12-01');origem_max=pd.Timestamp('2020-11-01')
campos,META_DESENV=carregar_atmosfera(tp,origem_min,origem_max)
INDICES_TODOS,CSV_NOAA=carregar_indices_incorporados()
INDICES_OC=INDICES_TODOS.loc[origem_min:origem_max].copy();del INDICES_TODOS
auditar_causalidade_indices(INDICES_OC)
SEAS=carregar_seas5('desenvolvimento',tp);CFS=carregar_cfsv2('desenvolvimento',tp)
for fonte in [SEAS,CFS]:exigir(pd.Timestamp(fonte.time.max().values)<=pd.Timestamp('2020-12-01'),'Reserva carregada antes da seleção.')
print('Sete M10 novos: M6 com oito PCs SST. 300 árvores; 31 folhas; taxa 0,05; seed 42; peso 0,900.')
print('Somente máscara, clima mensal e PCA do TREINO de cada fold. Nenhuma nova busca de pesos.')
print('Comparações predefinidas: M10 sozinho e 50% M10 + 50% 8C. Seleção pelo RMSE histórico; no máximo um CSV.')


# %% Sete folds pareados; peso 0,900 fixo; mistura 50/50 fixa
CAP_REGISTROS=[]
for b in BLOCOS:
    inicio_fold=perf_counter()
    nome,corte=b['nome'],b['corte'];datas=pd.date_range(b['inicio'],periods=24,freq='MS')
    componentes={}
    for modelo in NOMES_MODELOS:
        arquivo=CAP_PASTA_8B/f'previsoes_{modelo}_{nome}.npz'
        exigir(sha256(arquivo)==CAP_HASHES_REF[arquivo.name],'Previsão antiga mudou durante execução.')
        par=cap_ler_npz(arquivo,datas,tp)
        componentes[modelo]=cap_chuva_prevista(par,CAP_PESOS[modelo],tp,datas)
        if modelo=='M6_controle':baseline_ref=par[0]
        else:exigir(np.array_equal(par[0],baseline_ref),'Baselines antigos divergentes.')
        met=metricas_mensais(tp.sel(time=datas),componentes[modelo],modelo,nome,corte)
        cap_metricas_peso42(met,modelo);CAP_REGISTROS.append(met)
    ref=(componentes['M6_controle'].astype(np.float64)+componentes['M8_CFSv2'].astype(np.float64))*.5
    par_m9,aud=cap_ajustar(nome,corte,datas,tp,campos,SEAS,CFS)
    exigir(np.array_equal(par_m9[0],baseline_ref),'Climatologia diferente do controle.')
    m9=cap_chuva_prevista(par_m9,CAP_PESO,tp,datas)
    mistura=.5*(m9.astype(np.float64)+ref)
    for modelo,pred in [('ensemble_8C',ref),('M10_SST_PCA',m9),('media_M10_8C_50_50',mistura)]:
        CAP_REGISTROS.append(metricas_mensais(tp.sel(time=datas),pred,modelo,nome,corte))
        if modelo in CAP_CANDIDATOS:
            pred.attrs['units']='mm/day'
            pred.to_dataset(name='tp_mm_day').to_netcdf(CAP_SAIDA/f'previsao_{modelo}_{nome}.nc')
    parcial=pd.concat(CAP_REGISTROS,ignore_index=True)
    parcial.to_csv(CAP_SAIDA/'metricas_mensais.csv',index=False)
    tabela=agregar_metricas(parcial[parcial.bloco==nome],['modelo'])
    print('Comparação histórica:',nome);mostrar_tabela(tabela[['modelo','rmse','mae','vies']])
    duracao=perf_counter()-inicio_fold
    print(f'{nome}: {duracao/60:.1f} min incluindo treino, inferência e arquivos.',flush=True)
    if nome=='H1':
        print(f'Estimativa aproximada para os seis folds restantes: {6*duracao/60:.0f} min; '
              'o ajuste final e a exportação exigem tempo adicional. Esta estimativa não é garantia.',flush=True)
    del componentes,par,baseline_ref,ref,par_m9,m9,mistura,pred
    gc.collect()


# %% Comparar M10 e mistura; congelar um candidato usando somente 2007–2020
CAP_MENSAIS=pd.concat(CAP_REGISTROS,ignore_index=True)
exigir(len(CAP_MENSAIS)==5*168 and not CAP_MENSAIS.duplicated(['modelo','mes_alvo']).any(),'Métricas incompletas/duplicadas.')
exigir(set(CAP_MENSAIS.ano)==set(range(2007,2021)),'Reserva/teste entrou nas métricas.')
CAP_GLOBAL=agregar_metricas(CAP_MENSAIS,['modelo'])
CAP_FOLDS=agregar_metricas(CAP_MENSAIS,['modelo','bloco']);CAP_ANOS=agregar_metricas(CAP_MENSAIS,['modelo','ano'])
CAP_RMSE8C=float(CAP_GLOBAL.loc[CAP_GLOBAL.modelo=='ensemble_8C','rmse'].iloc[0])
exigir(abs(CAP_RMSE8C-CAP_REF_RMSE)<1e-6,'A referência não reproduz a 8C.')
for tabela,chave in [(CAP_GLOBAL,None),(CAP_FOLDS,'bloco'),(CAP_ANOS,'ano')]:
    for referencia,sufixo in [('ensemble_8C','8C'),('M6_controle','M6')]:
        ref=tabela[tabela.modelo==referencia]
        base=float(ref.rmse.iloc[0]) if chave is None else tabela[chave].map(ref.set_index(chave).rmse)
        tabela[f'delta_rmse_vs_{sufixo}']=tabela.rmse-base
        tabela[f'ganho_percentual_vs_{sufixo}']=-100*tabela[f'delta_rmse_vs_{sufixo}']/base
CAP_RESUMO=[]
for unidade,tabela in [('bloco',CAP_FOLDS),('ano',CAP_ANOS)]:
    for modelo in CAP_CANDIDATOS:
        for referencia in ['8C','M6']:
            d=tabela.loc[tabela.modelo==modelo,f'delta_rmse_vs_{referencia}']
            CAP_RESUMO.append(dict(modelo=modelo,referencia=referencia,unidade=unidade,
                melhores=int((d<-1e-8).sum()),piores=int((d>1e-8).sum()),empates=int((d.abs()<=1e-8).sum()),
                media_delta=float(d.mean()),mediana_delta=float(d.median()),pior_delta=float(d.max()),melhor_delta=float(d.min())))
for nome,tabela in [('comparacao_global',CAP_GLOBAL),('comparacao_blocos',CAP_FOLDS),('comparacao_anos',CAP_ANOS),
                    ('consistencia',pd.DataFrame(CAP_RESUMO))]:
    tabela.to_csv(CAP_SAIDA/f'{nome}.csv',index=False);mostrar_tabela(tabela)
CAP_DECISAO=dict(cap_escolher(CAP_GLOBAL),protocolo_sha256=CAP_HASH_PROTOCOLO,
    candidatos=CAP_CANDIDATOS,consistencia=CAP_RESUMO,peso_M10=CAP_PESO,
    fracao_M10_na_mistura=.5,fracao_8C_na_mistura=.5,aprovacao_automatica=False,reserva_avaliada=False)
salvar_json('decisao_congelada.json',CAP_DECISAO)
CAP_HASH_DECISAO=sha256(CAP_SAIDA/'decisao_congelada.json')
print('Seleção histórica:',CAP_DECISAO['modelo'],'| Gerar CSV?',CAP_DECISAO['gerar_candidato'])
print('Os folds já foram usados antes; esta comparação não garante melhora no período privado.')
del tp,campos,SEAS,CFS
gc.collect()


# %% Um ajuste final M10; exportar somente o candidato selecionado; preservar backup
def cap_final():
    global INDICES_OC,CAP_CAMPEAO
    conferir_protocolo()
    exigir(sha256(CAP_SAIDA/'decisao_congelada.json')==CAP_HASH_DECISAO,'Seleção alterada.')
    decisao=json.loads((CAP_SAIDA/'decisao_congelada.json').read_text(encoding='utf-8'))
    if not decisao['gerar_candidato']:
        print('Nenhum candidato superou a 8C no histórico. Sem ajuste final e sem novo CSV.')
        cap_backup();return
    CAP_CAMPEAO=cap_csv42(CAP_RMSE8C);hash8c=sha256(CAP_CAMPEAO)
    exigir(sha256(PASTA_SEAS5/'seas5_51_manifesto.json')==HASH_MANIFESTO_SEAS5 and
           sha256(PASTA_CFSV2/'cfsv2_manifesto.json')==HASH_MANIFESTO_CFSV2,'Manifestos alterados.')
    with xr.open_dataset(PASTA/'treino_tp.nc') as ds:chuva=ds.tp.transpose('time','lat','lon').load()
    with xr.open_dataset(PASTA/'treino_tp_alvo.nc') as da,xr.open_dataset(PASTA/'teste_features.nc') as teste:
        contrato=auditar_contrato(chuva,da.tp_alvo,teste)
        atmosfera,meta=carregar_atmosfera(chuva,pd.Timestamp('1992-12-01'),pd.Timestamp('2022-12-01'))
        destinos=pd.DatetimeIndex(teste.time.values);atmos_teste={}
        for v in VARIAVEIS:
            bruto=teste[v].transpose('time','lat','lon');conferir_campo(bruto,chuva,destinos)
            campo=bruto.assign_coords(time=teste.time_origem.values).astype('float32').load()
            exigir(pd.DatetimeIndex(campo.time.values).equals((destinos.to_period('M')-1).to_timestamp()),'Origem de teste divergente.')
            exigir(np.isfinite(campo.values).all(),'Atmosfera inválida.')
            exigir(np.allclose(campo.sel(time=CORTE_FINAL),atmosfera[v].sel(time=CORTE_FINAL),rtol=1e-6,atol=1e-7),f'Sobreposição divergente: {v}')
            atmos_teste[v]=campo
    INDICES_OC,_=carregar_indices_incorporados()
    seas=xr.concat([carregar_seas5('desenvolvimento',chuva).sel(time=slice('1993-01-01',None)),carregar_seas5('somente_ajuste_final',chuva)],dim='time')
    cfs=xr.concat([carregar_cfsv2('desenvolvimento',chuva).sel(time=slice('1993-01-01',None)),carregar_cfsv2('somente_ajuste_final',chuva)],dim='time')
    for fonte in [seas,cfs]:exigir(pd.DatetimeIndex(fonte.time.values).equals(pd.date_range('1993-01-01','2022-12-01',freq='MS')),'Histórico final divergente.')
    seas_t=carregar_seas5('teste',chuva);cfs_t=carregar_cfsv2('teste',chuva)
    par,aud=cap_ajustar('final','2022-12-01',destinos,chuva,atmosfera,seas,cfs,atmos_teste,seas_t,cfs_t)
    exigir(aud['inicio']=='1993-01-01' and aud['fim']=='2022-12-01' and aud['meses']==360,'Treino final deve ter 30 anos.')
    exigir(aud['anos_mensais_clima_SEAS5']==[30]*12 and aud['anos_mensais_clima_CFSv2']==[30]*12,'Climatologias finais divergentes.')
    m9=cap_chuva_prevista(par,CAP_PESO,chuva,destinos)
    previsao=m9.astype(np.float64)
    if decisao['modelo']=='media_M10_8C_50_50':
        previsao.values=.5*(previsao.values+cap_grade_csv(CAP_CAMPEAO,previsao))
    else:exigir(decisao['modelo']=='M10_SST_PCA','Candidato não predefinido.')
    exigir(np.isfinite(previsao.values).all() and (previsao.values>=0).all(),'Previsão final inválida.')
    exigir(sha256(CAP_CAMPEAO)==hash8c,'Campeão 8C alterado.')
    previsao.to_netcdf(CAP_SAIDA/'previsoes_selecionadas_2023_2024.nc')
    nome=f'submission_8F_{decisao["modelo"]}_8PCs_37f.csv'
    auditoria=salvar_submissao(PASTA/'sample_submission.csv',previsao,CAP_SAIDA/nome)
    exigir(auditoria['linhas']==1885464 and auditoria['sample_sha256']==next(e['sha256'] for e in ENTRADAS if e['nome']=='sample_submission.csv'),'Sample/cobertura divergentes.')
    salvar_json('auditoria_submissao.json',auditoria)
    salvar_json('manifesto.json',dict(execucao=EXECUCAO,ambiente=VERSOES,protocolo=cap_protocolo(),decisao=decisao,
        auditoria=auditoria,contrato=contrato,CSV_8C=str(CAP_CAMPEAO),CSV_8C_sha256=hash8c,
        ajustes_historicos_novos=7,ajustes_finais_novos=1,reserva_avaliada=False,envio_automatico=False,
        quantizacao='M10 float32; eventual mistura em float64 com CSV 8C de 6 casas; exportação em 6 casas'))
    mostrar_tabela(CAP_GLOBAL);mostrar_tabela(pd.Series(auditoria))
    print('CANDIDATO SELECIONADO:',CAP_SAIDA/nome)
    print('Nenhuma submissão enviada. Arquivo de 1,65183 preservado.')
    display(FileLink(os.path.relpath(CAP_SAIDA/nome,Path.cwd())))
    cap_backup()


cap_final()
