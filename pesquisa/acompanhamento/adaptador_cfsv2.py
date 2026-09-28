"""Auditoria CFSv2 preservada do projeto: datas reais, membros, lead, grade e unidade."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import xarray as xr
import cftime
LEAD = 1.5
LAT = np.arange(-61,17,dtype=np.float32)
LON_360 = np.arange(269,337,dtype=np.float32)
EXCECOES_MEMBROS = {}  # nenhuma excecao nova aprovada neste acompanhamento
POLITICA_MEMBROS = 'todos_membros_esperados_sem_lacunas_v1'
SCHEMA = 'nimbus_cfsv2_prospectivo_v1'
FONTE = 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.NCEP-CFSv2/.HINDCAST/.PENTAD_SAMPLES_FULL/.prec/'

def exigir(ok, mensagem):
    if not bool(ok):
        raise ValueError(mensagem)

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for parte in iter(lambda: f.read(1024 * 1024), b''):
            h.update(parte)
    return h.hexdigest()

def meses_de_S(s):
    exigir(str(s.attrs.get('units', '')).strip() == 'months since 1960-01-01', 'Unidade de S inesperada.')
    exigir(str(s.attrs.get('calendar')) in ('360', '360_day'), 'Calendário de S inesperado.')
    valores = np.asarray(s.values, dtype=np.float64)
    exigir(valores.ndim == 1 and np.isfinite(valores).all(), 'S inválido.')
    exigir(np.equal(valores, np.rint(valores)).all(), 'S contém mês fracionário.')
    return pd.DatetimeIndex([(pd.Period('1960-01', freq='M') + int(v)).to_timestamp() for v in valores])

def tabela_inicializacoes(path, origens):
    with xr.open_dataset(path, decode_times=False, engine='netcdf4') as d:
        ds = d.load()
    exigir(set(ds.data_vars) == {'sampleS'} and set(ds.sampleS.dims) == {'S', 'M'}, 'Formato da tabela sampleS inesperado.')
    exigir(meses_de_S(ds.S).equals(pd.DatetimeIndex(origens)), 'Cobertura de sampleS divergente.')
    exigir(np.array_equal(ds.M.values, np.arange(1, 29)), 'IDs de membros da tabela inesperados.')
    exigir(ds.sampleS.attrs.get('units') == 'days since 1960-01-01', 'Unidade de sampleS inesperada.')
    exigir(str(ds.sampleS.attrs.get('calendar')) in ('365', '365_day', 'noleap'), 'Calendário sampleS inesperado.')
    resultado = {}
    for k, origem in enumerate(origens):
        numeros = ds.sampleS.transpose('S', 'M').values[k].astype(np.float64)
        exigir(not np.isinf(numeros).any(), 'Data de inicialização infinita.')
        mascara = np.isfinite(numeros)
        n = 28 if origem.month == 11 else 24
        exigir(np.array_equal(mascara, np.arange(28) < n), f'Tabela de membros divergente em {origem.date()}: esperados {n}.')
        datas_cf = cftime.num2date(numeros[mascara], 'days since 1960-01-01', calendar='365_day')
        datas = pd.DatetimeIndex([pd.Timestamp(t.year, t.month, t.day) for t in datas_cf])
        alvo = origem + pd.offsets.MonthBegin(1)
        exigir((datas < alvo).all(), f'Inicialização invade mês-alvo: {origem.date()}.')
        exigir(datas.min() >= origem - pd.Timedelta(days=40) and datas.max() <= origem + pd.Timedelta(days=7), 'Janela de inicialização pentadal inesperada.')
        exigir(len(set(datas)) == n // 4, 'Esperados quatro membros por data de inicialização.')
        exigir(np.all(np.unique(datas.values, return_counts=True)[1] == 4), 'Agrupamento pentadal divergente.')
        resultado[origem] = dict(mascara=mascara, datas=datas, numeros=numeros, fase_fonte='hindcast' if origem < pd.Timestamp('2011-04-01') else 'operacional')
    return resultado

def carregar_membros(path, origens):
    with xr.open_dataset(path, decode_times=False, engine='netcdf4') as d:
        ds = d.load()
    exigir(set(ds.data_vars) == {'prec'}, 'Arquivo deve conter somente precipitação prevista prec.')
    exigir(set(ds.prec.dims) == {'S', 'L', 'M', 'Y', 'X'}, 'Dimensões inesperadas.')
    exigir(meses_de_S(ds.S).equals(pd.DatetimeIndex(origens)), 'Meses do NetCDF não coincidem com o pedido.')
    exigir(ds.sizes['L'] == 1 and float(ds.L.values[0]) == LEAD, 'Lead incorreto; exigido L=1.5.')
    exigir(ds.L.attrs.get('units') == 'months' and float(ds.L.attrs.get('pointwidth', -1)) == 1, 'Exigida previsão mensal, não média de vários meses.')
    exigir(ds.prec.attrs.get('units') == 'mm/day', 'Unidade inesperada; nenhuma conversão será presumida.')
    exigir(ds.prec.attrs.get('standard_name') in ('precipitation_rate', 'lwe_precipitation_rate'), 'Variável não identificada como taxa de precipitação.')
    exigir(np.array_equal(ds.M.values, np.arange(1, 29)), 'IDs de membros incompletos ou duplicados.')
    exigir(np.array_equal(ds.Y.values, LAT) and np.array_equal(ds.X.values, LON_360), 'Grade nativa divergente do recorte de 1 grau solicitado.')
    exigir(ds.X.attrs.get('units') == 'degree_east' and ds.Y.attrs.get('units') == 'degree_north', 'Unidades geográficas divergentes.')
    valores = ds.prec.transpose('S', 'L', 'M', 'Y', 'X').values[:, 0].astype(np.float64)
    marcador = ds.prec.attrs.get('file_missing_value')
    if marcador is not None and np.isfinite(float(marcador)):
        valores[valores == float(marcador)] = np.nan
    return valores

def membros_utilizaveis(membros, origem, info):
    esperados = info['mascara']
    exigir(np.isnan(membros[~esperados]).all(), f'Membro fora da tabela com dados em {origem}.')
    completos = np.isfinite(membros).all(axis=(-1, -2))
    faltantes = tuple((int(i + 1) for i in np.flatnonzero(esperados & ~completos)))
    if faltantes:
        exigir(faltantes == EXCECOES_MEMBROS.get(str(origem.date())), f'Membros ausentes/parciais não previstos em {origem.date()}: {faltantes}.')
        exigir(np.isnan(membros[np.array(faltantes) - 1]).all(), 'A exceção exige membro inteiramente ausente; lacunas parciais ou infinitos não são aceitos.')
    usados = esperados & completos
    exigir((membros[usados] >= 0).all(), f'Precipitação negativa em {origem.date()}. Investigar.')
    datas = info['datas'][usados[esperados]]
    return (usados, datas, faltantes)

def processar_bloco(path, origens, tabela):
    origens = pd.DatetimeIndex(origens)
    membros = carregar_membros(path, origens)
    hash_bruto = sha256(path)
    medias, linhas, datas_min, datas_max, ns, fases = ([], [], [], [], [], [])
    for k, origem in enumerate(origens):
        info = tabela[origem]
        validos, datas, faltantes = membros_utilizaveis(membros[k], origem, info)
        v = membros[k, validos]
        exigir(np.isfinite(v).all(), f'Membro previsto ausente/parcial em {origem}. Não usar média parcial.')
        exigir((v >= 0).all(), f'Precipitação negativa em {origem}: mínimo {v.min():.12g} mm/day. Investigar.')
        media = v.mean(axis=0, dtype=np.float64).astype(np.float32)
        alvo = origem + pd.offsets.MonthBegin(1)
        medias.append(media)
        ns.append(int(validos.sum()))
        datas_min.append(datas.min())
        datas_max.append(datas.max())
        fases.append(0 if info['fase_fonte'] == 'hindcast' else 1)
        linhas.append(dict(time_alvo=str(alvo.date()), time_origem=str(origem.date()), lead=LEAD, membros=ns[-1], membro_ids=','.join(map(str, np.flatnonzero(validos) + 1)), membros_esperados=int(info['mascara'].sum()), membros_indisponiveis_ids=list(faltantes), politica_membros=POLITICA_MEMBROS, inicializacao_mais_antiga=str(datas.min().date()), inicializacao_mais_recente=str(datas.max().date()), inicializacoes_anteriores_ao_alvo=True, fase_fonte=info['fase_fonte'], minimo_membros_mm_day=float(v.min()), maximo_membros_mm_day=float(v.max()), minimo_media_mm_day=float(media.min()), maximo_media_mm_day=float(media.max()), media_espacial_mm_day=float(media.mean(dtype=np.float64)), membros_ausentes_estruturais=28 - int(info['mascara'].sum()), pixels_ausentes_membros_validos=0, negativos_corrigidos=0, arquivo_bruto=path.name, sha256_bruto=hash_bruto))
    ds = xr.Dataset({'cfsv2_tp_media': (('time', 'lat', 'lon'), np.stack(medias)), 'n_membros': ('time', np.array(ns, dtype=np.int16)), 'time_origem': ('time', origens.values), 'inicializacao_mais_antiga': ('time', pd.DatetimeIndex(datas_min).values), 'inicializacao_mais_recente': ('time', pd.DatetimeIndex(datas_max).values), 'fase_fonte': ('time', np.array(fases, dtype=np.int8))}, coords={'time': (origens.to_period('M') + 1).to_timestamp(), 'lat': LAT, 'lon': LON_360 - 360}, attrs=dict(schema=SCHEMA, fonte=FONTE, modelo='NCEP-CFSv2', amostragem='PENTAD_SAMPLES_FULL; 24 membros, 28 em novembro', produto='previsao mensal; media aritmetica local dos membros completos disponiveis', politica_membros=POLITICA_MEMBROS, lead_iri=LEAD, referencia_temporal='S nominal no mes anterior ao alvo; datas reais auditadas', observacoes_de_chuva_utilizadas='nenhuma', climatologia_aplicada='nenhuma', grade='nativa de 1 grau; interpolacao fica para o notebook de modelagem'))
    ds.cfsv2_tp_media.attrs = dict(units='mm/day', long_name='Precipitacao prevista: media dos membros CFSv2')
    ds.lat.attrs['units'], ds.lon.attrs['units'] = ('degrees_north', 'degrees_east')
    ds.fase_fonte.attrs = dict(flag_values=np.array([0, 1], dtype=np.int8), flag_meanings='hindcast operacional')
    return (ds, linhas)
