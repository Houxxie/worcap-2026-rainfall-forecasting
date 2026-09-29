"""Inspect acquired bytes and validate forecast field contracts."""
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import xarray as xr
from registry import exigir, ContratoError, mes
LAT = np.arange(-60, 15.001, 0.25)
LON = np.arange(-90, -24.999, 0.25)

def indices_psl(dados):
    linhas = dados.decode('utf-8-sig').strip().splitlines()
    exigir(len(linhas) > 3 and len(linhas[0].split()) == 2, 'Invalid PSL header.')
    a, b = [int(x) for x in linhas[0].split()]
    exigir(1800 <= a <= b <= 2200, 'Invalid declared years.')
    exigir(len(linhas) > b - a + 2, 'Incomplete PSL series.')
    matriz = []
    for ano, linha in zip(range(a, b + 1), linhas[1:b - a + 2]):
        v = linha.split()
        exigir(len(v) == 13 and int(v[0]) == ano, 'Invalid PSL years or months.')
        matriz.extend(((f'{ano:04d}-{m:02d}', float(x)) for m, x in enumerate(v[1:], 1)))
    sentinela = float(linhas[b - a + 2].strip())
    exigir(sentinela < -90, 'Unexpected missing-value marker.')
    pares = [(t, v) for t, v in matriz if v != sentinela]
    exigir(pares and all((np.isfinite(v) and abs(v) < 20 for _, v in pares)), 'Index outside the expected range.')
    return dict(validado=True, formato='monthly NOAA PSL', unidades='degC', meses=[t for t, _ in pares], ultimo_mes=pares[-1][0], ausentes=len(matriz) - len(pares), rodape='\n'.join(linhas[b - a + 3:]))

def inspecionar_netcdf(dados, perfil, esperado=None):
    exigir(dados[:3] == b'CDF' or dados[:8] == b'\x89HDF\r\n\x1a\n', 'Response is not NetCDF.')
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / 'entrada.nc'
        path.write_bytes(dados)
        with xr.open_dataset(path, decode_times=perfil != 'cfsv2_bruto') as ds:
            if perfil == 'ersstv5':
                exigir(ds.attrs.get('product_version') == 'Version v5', 'Different ERSST version.')
                exigir('sst' in ds and ds.sst.attrs.get('units') == 'degree_C', 'Different SST variable or units.')
                ts = pd.DatetimeIndex(ds.time.values)
                meses = ts.strftime('%Y-%m').tolist()
                exigir(len(meses) == 1 and meses[0] == esperado, 'SST month does not match the request.')
                exigir(ds.sizes.get('lat') == 89 and ds.sizes.get('lon') == 180, 'Unexpected ERSST grid.')
                exigir(np.array_equal(ds.lat, np.arange(-88, 90, 2)) and np.array_equal(ds.lon, np.arange(0, 360, 2)), 'Unexpected ERSST coordinates.')
                vals = ds.sst.values
                f = vals[np.isfinite(vals)]
                exigir(f.size > 1000 and (not np.isinf(vals).any()) and (f.min() >= -1.801) and (f.max() <= 45), 'Invalid SST field.')
                return dict(validado=True, meses=meses, versao='ERSSTv5', unidades='degree_C', dimensoes=dict(ds.sizes), minimo=float(f.min()), maximo=float(f.max()))
            exigir(perfil == 'cfsv2_bruto', 'Unknown inspection profile.')
            return dict(validado=False, meses=[], dimensoes=dict(ds.sizes), variaveis=list(ds.data_vars), pendencia='Audit real initialization dates, members, lead 1.5, units and grid before inference.')

def validar_previsao(path, alvo):
    with xr.open_dataset(path) as ds:
        exigir(set(ds.data_vars) == {'precipitacao', 'climatologia'}, 'Both forecast and control climatology are required.')
        exigir(set(ds.dims) == {'time', 'lat', 'lon'} and ds.sizes['time'] == 1, 'Invalid dimensions.')
        exigir(np.array_equal(ds.lat, LAT) and np.array_equal(ds.lon, LON), 'Forecast grid differs from the reference.')
        t = pd.DatetimeIndex(ds.time.values)
        exigir(t.equals(pd.DatetimeIndex([mes(alvo).replace(tzinfo=None)])), 'Incorrect forecast target.')
        for n in ds.data_vars:
            a = ds[n]
            exigir(a.dims == ('time', 'lat', 'lon') and a.attrs.get('units') == 'mm/day', 'Invalid output dimension order or units.')
            exigir(np.isfinite(a.values).all() and (a.values >= 0).all(), 'Prediction contains missing or negative values.')
        return dict(pontos=int(ds.precipitacao.size), minimo=float(ds.precipitacao.min()), maximo=float(ds.precipitacao.max()), unidades='mm/day')

def metricas(previsao, verdade):
    """Require exact alignment; do not mask observation, grid or unit errors."""
    with xr.open_dataset(previsao) as a, xr.open_dataset(verdade) as b:
        exigir(set(b.data_vars) == {'tp'}, 'Observation must contain only tp.')
        exigir(b.tp.attrs.get('units') == 'mm/day' and b.tp.dims == ('time', 'lat', 'lon'), 'Invalid observation contract.')
        exigir(np.isfinite(b.tp.values).all() and (b.tp.values >= 0).all(), 'Incomplete or invalid observation.')
        linhas = []
        for nome in ('precipitacao', 'climatologia'):
            p, y = xr.align(a[nome], b.tp, join='exact')
            exigir(p.shape == y.shape, 'Observation does not cover the complete prediction.')
            erro = np.asarray(p.values, dtype=np.float64) - np.asarray(y.values, dtype=np.float64)
            area = np.broadcast_to(np.cos(np.deg2rad(y.lat.values))[None, :, None], erro.shape)
            for regiao, mask in [('dominio', np.ones(len(y.lat), dtype=bool)), ('sul_35', y.lat.values < -35), ('35_a_15', (y.lat.values >= -35) & (y.lat.values < -15)), ('norte_15', y.lat.values >= -15)]:
                e, w = (erro[:, mask, :], area[:, mask, :])
                exigir(e.size > 0, 'Empty region.')
                sse, n = (float((e * e).sum()), int(e.size))
                linhas.append(dict(modelo=nome, regiao=regiao, n=n, sse=sse, soma_erro=float(e.sum()), soma_erro_absoluto=float(np.abs(e).sum()), rmse=float(np.sqrt(sse / n)), mae=float(np.abs(e).mean()), vies=float(e.mean()), sse_area=float((e * e * w).sum()), peso_area=float(w.sum()), rmse_area=float(np.sqrt((e * e * w).sum() / w.sum()))))
        return linhas

def conferir_observacao_bruta(bruto, normalizado, alvo):
    """Validate ERA5 monthly conversion; reject unknown format or expver."""
    with xr.open_dataset(bruto) as ds, xr.open_dataset(normalizado) as n:
        exigir('tp' in ds and set(n.data_vars) == {'tp'}, 'Missing rainfall variable tp.')
        a = ds.tp
        exigir(a.attrs.get('units') in {'m', 'm/day', 'm day**-1'}, 'Unexpected raw ERA5 units.')
        stream = a.attrs.get('GRIB_stream', ds.attrs.get('GRIB_stream'))
        exigir(stream == 'moda', 'Require monthly means of daily means, stream moda; do not infer conversion for a different product.')
        if 'expver' in a.dims:
            exigir(a.sizes['expver'] == 1 and int(a.expver.values[0]) == 1, 'ERA5T or mixed versions are not accepted.')
            a = a.isel(expver=0, drop=True)
        elif 'expver' in ds:
            exigir(np.asarray(ds.expver).size == 1 and int(np.asarray(ds.expver).ravel()[0]) == 1, 'Nonfinal expver.')
        else:
            ver = a.attrs.get('GRIB_experimentVersionNumber', ds.attrs.get('GRIB_experimentVersionNumber'))
            exigir(str(ver) in {'1', '0001'}, 'Raw file contains no evidence of final expver.')
        a = a.rename({x: y for x, y in [('valid_time', 'time'), ('latitude', 'lat'), ('longitude', 'lon')] if x in a.dims})
        exigir(set(a.dims) == {'time', 'lat', 'lon'} and a.sizes['time'] == 1, 'Unexpected raw target dimensions.')
        ts = pd.DatetimeIndex(a.time.values)
        exigir(ts[0].day == 1 and ts.strftime('%Y-%m').tolist() == [alvo], 'Raw month differs from the target.')
        a = a.assign_coords(time=[np.datetime64(alvo + '-01')], lon=(a.lon + 180) % 360 - 180).sortby('lat').sortby('lon')
        a = a.transpose('time', 'lat', 'lon')
        a, b = xr.align(a, n.tp, join='exact')
        exigir(a.shape == b.shape and np.allclose(a.values * 1000, b.values, rtol=1e-06, atol=1e-06), 'Target conversion does not match raw ERA5 multiplied by 1000.')
        return dict(stream='moda', expver=1, conversao='m * 1000 -> mm/day', mes_alvo=alvo)
