"""Small synthetic fields for tests; never operational inputs."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import xarray as xr
from .inputs import library


def fixture():
    m = library()
    dates = pd.date_range('1976-06-01', '2009-01-01', freq='MS')
    rng = np.random.default_rng(71)
    coords = dict(time=dates, lat=np.linspace(-60, 15, 9), lon=np.linspace(-90, -25, 13))
    def field():
        return xr.DataArray(rng.uniform(1, 5, (len(dates), 9, 13)).astype('float32'), dims=('time', 'lat', 'lon'), coords=coords)
    rain, seas, cfs = field(), field(), field()
    origins = m.origem_mensal(dates, 1)
    seas = seas.assign_coords(time_origem=('time', origins.values))
    cfs = cfs.assign_coords(time_origem=('time', origins.values), inicializacao_mais_recente=('time', origins.values))
    atmosphere = {name: field() for name in m.VARIAVEIS}
    m.INDICES_OC = pd.DataFrame(rng.normal(size=(len(dates), 4)), index=dates, columns=m.INDICES_NOMES)
    return SimpleNamespace(lib=m, rain=rain, seas=seas, cfs=cfs, atmosphere=atmosphere)
