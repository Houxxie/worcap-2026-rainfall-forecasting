"""CFSv2 audit: actual dates, members, lead time, grid and units."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import xarray as xr
import cftime
LEAD = 1.5
LAT = np.arange(-61, 17, dtype=np.float32)
LON_360 = np.arange(269, 337, dtype=np.float32)
EXCECOES_MEMBROS = {}
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
    exigir(str(s.attrs.get('units', '')).strip() == 'months since 1960-01-01', 'Unexpected S units.')
    exigir(str(s.attrs.get('calendar')) in ('360', '360_day'), 'Unexpected S calendar.')
    valores = np.asarray(s.values, dtype=np.float64)
    exigir(valores.ndim == 1 and np.isfinite(valores).all(), 'Invalid S.')
    exigir(np.equal(valores, np.rint(valores)).all(), 'S contains a fractional month.')
    return pd.DatetimeIndex([(pd.Period('1960-01', freq='M') + int(v)).to_timestamp() for v in valores])

def tabela_inicializacoes(path, origens):
    with xr.open_dataset(path, decode_times=False, engine='netcdf4') as d:
        ds = d.load()
    exigir(set(ds.data_vars) == {'sampleS'} and set(ds.sampleS.dims) == {'S', 'M'}, 'Unexpected sampleS table format.')
    exigir(meses_de_S(ds.S).equals(pd.DatetimeIndex(origens)), 'sampleS coverage mismatch.')
    exigir(np.array_equal(ds.M.values, np.arange(1, 29)), 'Unexpected table member IDs.')
    exigir(ds.sampleS.attrs.get('units') == 'days since 1960-01-01', 'Unexpected sampleS units.')
    exigir(str(ds.sampleS.attrs.get('calendar')) in ('365', '365_day', 'noleap'), 'Unexpected sampleS calendar.')
    resultado = {}
    for k, origem in enumerate(origens):
        numeros = ds.sampleS.transpose('S', 'M').values[k].astype(np.float64)
        exigir(not np.isinf(numeros).any(), 'Infinite initialization date.')
        mascara = np.isfinite(numeros)
        n = 28 if origem.month == 11 else 24
        exigir(np.array_equal(mascara, np.arange(28) < n), f'Member table mismatch in {origem.date()}: expected {n}.')
        datas_cf = cftime.num2date(numeros[mascara], 'days since 1960-01-01', calendar='365_day')
        datas = pd.DatetimeIndex([pd.Timestamp(t.year, t.month, t.day) for t in datas_cf])
        alvo = origem + pd.offsets.MonthBegin(1)
        exigir((datas < alvo).all(), f'Initialization overlaps the target month: {origem.date()}.')
        exigir(datas.min() >= origem - pd.Timedelta(days=40) and datas.max() <= origem + pd.Timedelta(days=7), 'Unexpected pentad initialization window.')
        exigir(len(set(datas)) == n // 4, 'Four members per initialization date are required.')
        exigir(np.all(np.unique(datas.values, return_counts=True)[1] == 4), 'Pentad grouping mismatch.')
        resultado[origem] = dict(mascara=mascara, datas=datas, numeros=numeros, fase_fonte='hindcast' if origem < pd.Timestamp('2011-04-01') else 'operacional')
    return resultado

def carregar_membros(path, origens):
    with xr.open_dataset(path, decode_times=False, engine='netcdf4') as d:
        ds = d.load()
    exigir(set(ds.data_vars) == {'prec'}, 'The file must contain only predicted rainfall prec.')
    exigir(set(ds.prec.dims) == {'S', 'L', 'M', 'Y', 'X'}, 'Unexpected dimensions.')
    exigir(meses_de_S(ds.S).equals(pd.DatetimeIndex(origens)), 'NetCDF months differ from the request.')
    exigir(ds.sizes['L'] == 1 and float(ds.L.values[0]) == LEAD, 'Incorrect lead; L=1.5 is required.')
    exigir(ds.L.attrs.get('units') == 'months' and float(ds.L.attrs.get('pointwidth', -1)) == 1, 'A monthly forecast is required, not a multimonth average.')
    exigir(ds.prec.attrs.get('units') == 'mm/day', 'Unexpected units; no conversion will be assumed.')
    exigir(ds.prec.attrs.get('standard_name') in ('precipitation_rate', 'lwe_precipitation_rate'), 'Variable is not identified as a precipitation rate.')
    exigir(np.array_equal(ds.M.values, np.arange(1, 29)), 'Incomplete or duplicate member IDs.')
    exigir(np.array_equal(ds.Y.values, LAT) and np.array_equal(ds.X.values, LON_360), 'Native grid differs from the requested one-degree subset.')
    exigir(ds.X.attrs.get('units') == 'degree_east' and ds.Y.attrs.get('units') == 'degree_north', 'Geographic unit mismatch.')
    valores = ds.prec.transpose('S', 'L', 'M', 'Y', 'X').values[:, 0].astype(np.float64)
    marcador = ds.prec.attrs.get('file_missing_value')
    if marcador is not None and np.isfinite(float(marcador)):
        valores[valores == float(marcador)] = np.nan
    return valores

def membros_utilizaveis(membros, origem, info):
    esperados = info['mascara']
    exigir(np.isnan(membros[~esperados]).all(), f'A member outside the table contains data in {origem}.')
    completos = np.isfinite(membros).all(axis=(-1, -2))
    faltantes = tuple((int(i + 1) for i in np.flatnonzero(esperados & ~completos)))
    if faltantes:
        exigir(faltantes == EXCECOES_MEMBROS.get(str(origem.date())), f'Unexpected missing or partial members in {origem.date()}: {faltantes}.')
        exigir(np.isnan(membros[np.array(faltantes) - 1]).all(), 'The exception requires a completely absent member; partial gaps and infinities are rejected.')
    usados = esperados & completos
    exigir((membros[usados] >= 0).all(), f'Negative precipitation in {origem.date()}. Investigate before proceeding.')
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
        exigir(np.isfinite(v).all(), f'Missing or partial expected member in {origem}. Do not use a partial ensemble mean.')
        exigir((v >= 0).all(), f'Negative precipitation in {origem}: minimum {v.min():.12g} mm/day. Investigate before proceeding.')
        media = v.mean(axis=0, dtype=np.float64).astype(np.float32)
        alvo = origem + pd.offsets.MonthBegin(1)
        medias.append(media)
        ns.append(int(validos.sum()))
        datas_min.append(datas.min())
        datas_max.append(datas.max())
        fases.append(0 if info['fase_fonte'] == 'hindcast' else 1)
        linhas.append(dict(time_alvo=str(alvo.date()), time_origem=str(origem.date()), lead=LEAD, membros=ns[-1], membro_ids=','.join(map(str, np.flatnonzero(validos) + 1)), membros_esperados=int(info['mascara'].sum()), membros_indisponiveis_ids=list(faltantes), politica_membros=POLITICA_MEMBROS, inicializacao_mais_antiga=str(datas.min().date()), inicializacao_mais_recente=str(datas.max().date()), inicializacoes_anteriores_ao_alvo=True, fase_fonte=info['fase_fonte'], minimo_membros_mm_day=float(v.min()), maximo_membros_mm_day=float(v.max()), minimo_media_mm_day=float(media.min()), maximo_media_mm_day=float(media.max()), media_espacial_mm_day=float(media.mean(dtype=np.float64)), membros_ausentes_estruturais=28 - int(info['mascara'].sum()), pixels_ausentes_membros_validos=0, negativos_corrigidos=0, arquivo_bruto=path.name, sha256_bruto=hash_bruto))
    ds = xr.Dataset({'cfsv2_tp_media': (('time', 'lat', 'lon'), np.stack(medias)), 'n_membros': ('time', np.array(ns, dtype=np.int16)), 'time_origem': ('time', origens.values), 'inicializacao_mais_antiga': ('time', pd.DatetimeIndex(datas_min).values), 'inicializacao_mais_recente': ('time', pd.DatetimeIndex(datas_max).values), 'fase_fonte': ('time', np.array(fases, dtype=np.int8))}, coords={'time': (origens.to_period('M') + 1).to_timestamp(), 'lat': LAT, 'lon': LON_360 - 360}, attrs=dict(schema=SCHEMA, fonte=FONTE, modelo='NCEP-CFSv2', amostragem='PENTAD_SAMPLES_FULL; 24 members, 28 in November', produto='monthly forecast; local arithmetic mean of complete available members', politica_membros=POLITICA_MEMBROS, lead_iri=LEAD, referencia_temporal='nominal S in the preceding month; actual initialization dates audited', observacoes_de_chuva_utilizadas='nenhuma', climatologia_aplicada='nenhuma', grade='native one-degree grid; interpolation is performed in the modeling notebook'))
    ds.cfsv2_tp_media.attrs = dict(units='mm/day', long_name='Predicted precipitation: CFSv2 ensemble mean')
    ds.lat.attrs['units'], ds.lon.attrs['units'] = ('degrees_north', 'degrees_east')
    ds.fase_fonte.attrs = dict(flag_values=np.array([0, 1], dtype=np.int8), flag_meanings='hindcast operacional')
    return (ds, linhas)
