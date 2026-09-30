"""Reuse the archived input bytes and training calendar without neural dependencies."""
from pathlib import Path
from types import SimpleNamespace
import importlib.metadata
import numpy as np
import pandas as pd
import xarray as xr
from research.common.inputs import ROOT, library, require, sha256, write_json, find_unique, preflight

INDEX = 'competition/metadata/NOAA/indices_noaa.csv'
INDEX_SHA256 = 'b6f444fbec48681c1e5b46a5c22aa197ad91c7fbbdaff82c3ed3a9c63f7c15a4'


def calendar(plan):
    train = pd.date_range(plan['training_start'], plan['training_end'], freq='MS')
    targets = pd.date_range(plan['evaluation_start'], plan['evaluation_end'], freq='MS')
    require(len(train) == 360 and len(targets) == 24, 'Expected 360 fitting and 24 evaluation months.')
    require(train[-1] == targets[0] - pd.DateOffset(months=4), 'The four-month label gap changed.')
    return train, targets


def index_snapshot(m, train, targets):
    path = ROOT / INDEX
    require(sha256(path) == INDEX_SHA256, 'Archived NOAA snapshot changed.')
    full = pd.read_csv(path, parse_dates=['time_origem']).set_index('time_origem')
    old = pd.read_csv(ROOT / 'research/lagged_sources/ocean_indices.csv', parse_dates=['time_origem']).set_index('time_origem')
    shared = old.index.intersection(full.index)
    # The full archive starts five months later in 1976; none are needed by this 1990+ window.
    require(len(shared) == 526 and full.columns.equals(old.columns)
            and np.array_equal(full.loc[shared].values, old.loc[shared].values), 'NOAA overlap differs.')
    m.conferir_indices(full)
    required = pd.date_range(train[0] - pd.DateOffset(months=3), targets[-1] - pd.DateOffset(months=3), freq='MS')
    require(required.isin(full.index).all(), 'Missing NOAA source months; no interpolation allowed.')
    return full.loc[required].copy(), dict(sha256=INDEX_SHA256, shared_months=len(shared), overlap_exact=True,
             required_start=str(required[0].date()), required_end=str(required[-1].date()),
             older_unshared_months=old.index.difference(full.index).strftime('%Y-%m').tolist(),
             all_required_months_present=True)


def check(plan, official=None, seas5=None, cfsv2=None, sst=None):
    paths, audit = preflight(official, seas5, cfsv2, final=True)
    pca_versions = {name: importlib.metadata.version(name) for name in ['scikit-learn', 'scipy']}
    require(pca_versions == {'scikit-learn': '1.6.1', 'scipy': '1.16.3'},
            'Use the archived PCA environment: scikit-learn 1.6.1, scipy 1.16.3. Found ' + str(pca_versions))
    audit['pca_versions'] = pca_versions
    m = library()
    train, targets = calendar(plan)
    _, audit['indices'] = index_snapshot(m, train, targets)
    paths['sst'] = find_unique(m.SST_NOME, sst) / m.SST_NOME
    require(sha256(paths['sst']) == m.SST_HASH,
            'SST snapshot differs. Attach ersstv5_4graus_198201_202411.nc from the original SST preparation.')
    field, metadata = m.sst_auditar_arquivo(paths['sst'])
    needed = m.origem_mensal(train.append(targets), 2)
    require(needed.isin(pd.DatetimeIndex(field.time.values)).all(), 'Incomplete SST origins.')
    audit.update(sst=metadata, sst_used=True, new_download=False)
    print('Input files, versions, NOAA overlap and SST snapshot checked. No models fitted.', flush=True)
    return paths, audit


def load(paths, audit, output, plan):
    train, targets = calendar(plan)
    m = library()
    m.PASTA, m.PASTA_SEAS5_MANUAL, m.PASTA_CFSV2_MANUAL = paths['official'], paths['seas5'], paths['cfsv2']
    m.PASTA_SEAS5, m.MANIFESTO_SEAS5 = m.localizar_seas5()
    m.PASTA_CFSV2, m.MANIFESTO_CFSV2 = m.localizar_cfsv2()
    m.SAIDA = Path(output)
    # Decode only training rainfall. Evaluation labels are opened separately after freezing.
    with xr.open_dataset(paths['official'] / 'treino_tp.nc') as ds:
        require(ds.tp.attrs.get('units') == 'mm/day', 'Unexpected rainfall units.')
        rain = ds.tp.sel(time=train).transpose('time', 'lat', 'lon').load()
    require(pd.DatetimeIndex(rain.time.values).equals(train), 'Training rainfall dates changed.')
    require(np.array_equal(rain.lat, np.arange(-60, 15.25, .25))
            and np.array_equal(rain.lon, np.arange(-90, -24.75, .25)), 'Different rainfall grid.')
    require(np.isfinite(rain.values).all() and (rain.values >= 0).all(), 'Invalid rainfall.')
    atmosphere, _ = m.carregar_atmosfera(rain, train[0] - pd.DateOffset(months=4), targets[-1] - pd.DateOffset(months=4))
    m.INDICES_OC, indices = index_snapshot(m, train, targets)
    write_json(Path(output) / 'indices.json', indices)
    phases = ['desenvolvimento', 'somente_ajuste_final']
    seas = xr.concat([m.carregar_seas5(p, rain) for p in phases], dim='time').sel(time=slice(train[0], targets[-1]))
    cfs = xr.concat([m.carregar_cfsv2(p, rain) for p in phases], dim='time').sel(time=slice(train[0], targets[-1]))
    sst, _ = m.sst_auditar_arquivo(paths['sst'])
    sst = sst.sel(time=slice(None, targets[-1] - pd.DateOffset(months=2)))
    audit = dict(audit, rainfall_decoded_for_fit=[str(train[0].date()), str(train[-1].date())],
                 evaluation_rainfall_decoded=False, test_partition_read=False)
    return SimpleNamespace(lib=m, rain=rain, atmosphere=atmosphere, seas=seas, cfs=cfs, audit=audit), sst
