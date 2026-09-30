"""Load the existing audited inputs; no downloads, imputation or new sources."""
from pathlib import Path
import hashlib
import importlib.metadata
import importlib.util
import json
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT / 'research/lagged_sources/library.py'
REQUIRED = {'numpy': '2.0.2', 'pandas': '2.3.3', 'xarray': '2025.12.0', 'lightgbm': '4.6.0'}


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False, default=str) + '\n', encoding='utf-8')


def library(path=LIBRARY):
    # Separate module globals: callers cannot change another run's index table.
    spec = importlib.util.spec_from_file_location('fixed_lag_library', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_unique(filename, supplied=None):
    if supplied:
        p = Path(supplied)
        require((p / filename).is_file(), f'Missing {filename} in {p}')
        return p.resolve()
    candidates = sorted({p.parent.resolve() for p in Path('/kaggle/input').rglob(filename)})
    require(len(candidates) == 1, f'Attach exactly one input containing {filename}, or set its directory explicitly. Found {len(candidates)}.')
    return candidates[0]


def preflight(official=None, seas5=None, cfsv2=None, final=False):
    paths = dict(official=find_unique('treino_tp.nc', official),
                 seas5=find_unique('seas5_51_manifesto.json', seas5),
                 cfsv2=find_unique('cfsv2_manifesto.json', cfsv2))
    expected = json.loads((ROOT / 'research/lagged_sources/official_hashes.json').read_text())
    files = {}
    for row in expected:
        if not row['nome'].startswith('treino_'):
            continue
        p = paths['official'] / row['nome']
        require(p.is_file(), f'Missing official input: {p.name}')
        require(sha256(p) == row['sha256'], f'Official input hash mismatch: {p.name}')
        files[p.name] = row['sha256']
    for source, name in [('seas5', 'seas5_51_manifesto.json'), ('cfsv2', 'cfsv2_manifesto.json')]:
        manifest = paths[source] / name
        files[name] = sha256(manifest)
        m = json.loads(manifest.read_text(encoding='utf-8'))
        for phase in ['desenvolvimento'] + (['somente_ajuste_final'] if final else []):
            row = m['arquivos'][phase]
            p = paths[source] / row['arquivo']
            require(p.is_file() and sha256(p) == row['sha256'], f'Missing or altered {source}: {phase}')
            files[p.name] = row['sha256']
    versions = {n: importlib.metadata.version(n) for n in REQUIRED}
    require(versions == REQUIRED, f'The fixed baseline needs {REQUIRED}; found {versions}.')
    return paths, dict(files=files, versions=versions, final_training=final)


class Inputs:
    def __init__(self, official=None, seas5=None, cfsv2=None, output='outputs', final=False):
        self.paths, self.audit = preflight(official, seas5, cfsv2, final)
        self.lib = m = library()
        m.PASTA = self.paths['official']
        m.PASTA_SEAS5_MANUAL = self.paths['seas5']
        m.PASTA_CFSV2_MANUAL = self.paths['cfsv2']
        m.PASTA_SEAS5, m.MANIFESTO_SEAS5 = m.localizar_seas5()
        m.PASTA_CFSV2, m.MANIFESTO_CFSV2 = m.localizar_cfsv2()
        m.SAIDA = Path(output)
        m.SAIDA.mkdir(parents=True, exist_ok=True)
        end = '2022-12-01' if final else '2020-12-01'
        start = '1993-01-01' if final else '1976-10-01'
        with xr.open_dataset(m.PASTA / 'treino_tp.nc') as ds:
            require(ds.tp.attrs.get('units') == 'mm/day', 'Unexpected rainfall units.')
            self.rain = ds.tp.sel(time=slice(start, end)).transpose('time', 'lat', 'lon').load()
        require(pd.DatetimeIndex(self.rain.time.values).equals(pd.date_range(start, end, freq='MS')), 'Incomplete rainfall calendar.')
        require(np.array_equal(self.rain.lat, np.arange(-60, 15.25, .25)) and
                np.array_equal(self.rain.lon, np.arange(-90, -24.75, .25)), 'Different target grid.')
        require(np.isfinite(self.rain.values).all() and (self.rain.values >= 0).all(), 'Invalid rainfall.')
        atm_start = pd.Timestamp(start) - pd.DateOffset(months=4)
        atm_end = pd.Timestamp(end) - pd.DateOffset(months=4)
        self.atmosphere, _ = m.carregar_atmosfera(self.rain, atm_start, atm_end)
        index_path = ROOT / ('competition/metadata/NOAA/indices_noaa.csv' if final else 'research/lagged_sources/ocean_indices.csv')
        m.INDICES_OC = pd.read_csv(index_path, parse_dates=['time_origem']).set_index('time_origem')
        m.conferir_indices(m.INDICES_OC)
        self.audit['indices_sha256'] = sha256(index_path)
        self.audit['library_sha256'] = sha256(LIBRARY)
        phases = ['desenvolvimento'] + (['somente_ajuste_final'] if final else [])
        self.seas = xr.concat([m.carregar_seas5(p, self.rain) for p in phases], dim='time').sel(time=slice(start, end))
        self.cfs = xr.concat([m.carregar_cfsv2(p, self.rain) for p in phases], dim='time').sel(time=slice(start, end))
        self.audit.update(rainfall_read=[start, end], forecast_partitions=phases,
                          test_partition_read=False, sst_used=False)


class Context:
    """Training-only climatologies, with a bounded reference window for inner selection."""
    def __init__(self, data, reference, training):
        self.data = data
        m = data.lib
        self.reference = pd.DatetimeIndex(reference)
        self.training = pd.DatetimeIndex(training)
        require(24 <= len(self.training) <= 360 and len(self.reference) <= 360, 'Invalid training window.')
        require(self.training.isin(self.reference).all() and self.reference.is_monotonic_increasing and
                self.reference.is_unique and self.training.is_unique, 'Invalid training/reference dates.')
        self.cutoff = self.reference[-1]
        rain = data.rain.sel(time=self.reference).values
        self.counts = np.array([(self.reference.month == month).sum() for month in range(1, 13)])
        require((self.counts > 1).all(), 'At least two reference years per month are needed.')
        self.sums = np.stack([rain[self.reference.month == month].sum(0, dtype=np.float64) for month in range(1, 13)])
        self.climate = xr.DataArray((self.sums / self.counts[:, None, None]).astype('float32'),
            dims=('month', 'lat', 'lon'), coords=dict(month=np.arange(1, 13), lat=data.rain.lat, lon=data.rain.lon), attrs=dict(units='mm/day'))
        origins = m.origem_mensal(self.reference, 4)
        self.atm = m.ajustar_clima_atmosfera(data.atmosphere, origins)
        origins = m.origem_mensal(self.reference, 3)
        vals = m.INDICES_OC.loc[origins, m.INDICES_NOMES].to_numpy(dtype=np.float64)
        self.atm['_clima_indices'] = np.stack([vals[origins.month == month].mean(0) for month in range(1, 13)])
        self.seasonal = []
        for forecasts in [data.seas, data.cfs]:
            v = forecasts.sel(time=self.training)
            require(pd.DatetimeIndex(v.time_origem.values).equals(m.origem_mensal(self.training, 1)), 'Wrong seasonal initialization.')
            means = np.stack([v.values[self.training.month == month].mean(0, dtype=np.float64) for month in range(1, 13)]).astype('float32')
            require(np.isfinite(means).all(), 'Missing seasonal reference month.')
            self.seasonal.append(xr.DataArray(means, dims=self.climate.dims, coords=self.climate.coords))

    def features(self, target, training=False):
        d, m = self.data, self.data.lib
        t = pd.Timestamp(target)
        require(t in self.training if training else t > self.cutoff, 'Month crosses the training boundary.')
        x = np.column_stack([m.matriz_mes(d.atmosphere, self.atm, self.climate, t),
            m.valores_seas5(d.seas, t), m.valores_anomalia_seas5(d.seas, self.seasonal[0], t),
            m.valores_cfsv2(d.cfs, t), m.valores_anomalia_cfsv2(d.cfs, self.seasonal[1], t)]).astype('float32')
        if training:
            # Exclude the entire target map, including neighbors seen by convolutions.
            y = d.rain.sel(time=t).values
            month = t.month - 1
            x[:, 22] = ((self.sums[month] - y.astype('float64')) / (self.counts[month] - 1)).ravel().astype('float32')
        require(np.isfinite(x).all(), 'Nonfinite features.')
        return x.T.reshape(31, d.rain.sizes['lat'], d.rain.sizes['lon'])

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.climate.to_netcdf(folder / 'rainfall_climatology.nc')
        np.savez_compressed(folder / 'references.npz', sums=self.sums, counts=self.counts,
                            seas5=self.seasonal[0].values, cfsv2=self.seasonal[1].values, **self.atm)
        write_json(folder / 'context.json', dict(reference=self.reference.strftime('%Y-%m').tolist(),
                                                training=self.training.strftime('%Y-%m').tolist(), features=self.data.lib.FEATURES_MULTISSISTEMA))
