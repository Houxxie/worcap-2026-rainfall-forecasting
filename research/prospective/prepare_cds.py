"""Validate final ERA5 or complete SEAS5 GRIBs and preserve their receipt lineage."""
from pathlib import Path
import argparse
import tempfile
import numpy as np
import pandas as pd
import xarray as xr
from registry import Registro, exigir, deslocar
from data_contracts import LAT, LON

FIELDS = {
    'era5_sl': {164: ('cloud_cover', {'(0 - 1)', '(0-1)', '1'}), 167: ('t2', {'K'}), 134: ('surface_pressure', {'Pa'})},
    'era5_pl': {129: ('geopotential_850', {'m**2 s**-2'}), 157: ('rel_hum_850', {'%'}),
                133: ('shum_850', {'kg kg**-1'}), 130: ('temperature_850', {'K'}),
                131: ('u_850', {'m s**-1'}), 132: ('v_850', {'m s**-1'})}}


def regular_grid(latitudes, longitudes, values):
    latitudes = np.asarray(latitudes, dtype='float64')
    longitudes = (np.asarray(longitudes, dtype='float64') + 180) % 360 - 180
    values = np.asarray(values, dtype='float64')
    lat, lon = np.unique(latitudes), np.unique(longitudes)
    exigir(values.size == len(lat) * len(lon) and np.isfinite(values).all(), 'Incomplete regular grid.')
    index = np.searchsorted(lat, latitudes) * len(lon) + np.searchsorted(lon, longitudes)
    exigir(len(np.unique(index)) == values.size, 'Duplicate geographic points.')
    out = np.empty(values.size)
    out[index] = values
    return lat, lon, out.reshape(len(lat), len(lon))


def convert(path, source, target):
    import eccodes as ec
    exigir(source in {*FIELDS, 'seas5'}, 'Unsupported CDS source.')
    origin = pd.Timestamp(deslocar(target, -1 if source == 'seas5' else -4) + '-01')
    target_date = pd.Timestamp(target + '-01')
    arrays, audit, headers = {}, [], set()
    coordinates = None
    with Path(path).open('rb') as f:
        while (g := ec.codes_grib_new_from_file(f)) is not None:
            try:
                get = lambda key: ec.codes_get(g, key)
                exigir(int(get('numberOfMissing')) == 0 and int(get('dataDate')) == int(origin.strftime('%Y%m%d')) and int(get('dataTime')) == 0,
                        'Missing pixels or wrong GRIB source month/time.')
                values = np.asarray(ec.codes_get_values(g), dtype='float64')
                if source == 'seas5':
                    exigir(str(get('system')) == '51' and str(get('origin')) == 'ecmf', 'Wrong SEAS5 system or centre.')
                    exigir(get('type') == 'fcmean' and get('stream') == 'msmm' and int(get('paramId')) == 172228, 'Wrong seasonal product.')
                    exigir(int(get('forecastMonth')) == 2 and int(get('verifyingMonth')) == int(target_date.strftime('%Y%m')), 'Wrong forecast month.')
                    exigir(str(get('units')).replace(' ', '') in {'ms**-1', 'ms^-1', 'ms-1', 'm/s'}, 'Unknown rainfall rate units.')
                    key = int(get('number'))
                    headers.add(int(get('numberOfForecastsInEnsemble')))
                    values *= 86400000.
                    exigir(np.isfinite(values).all() and values.min() >= -.01, 'Invalid precipitation or a negative value beyond the fixed tolerance.')
                    negative = np.flatnonzero(values < 0)
                    audit.append(dict(member=key, corrected=int(len(negative)), minimum_before=float(values.min()),
                                      indices_grib=negative.tolist(), values_before_mm_day=values[negative].tolist()))
                    values = np.maximum(values, 0)
                    units = 'mm/day'
                else:
                    exigir(str(get('class')) == 'ea' and str(get('expver')) in {'1', '0001'} and get('stream') == 'moda', 'Require final ERA5 monthly means: class ea, expver 1, stream moda.')
                    param = int(get('paramId'))
                    exigir(param in FIELDS[source], 'Unexpected atmospheric parameter.')
                    key, allowed_units = FIELDS[source][param]
                    units = get('units')
                    exigir(units in allowed_units, f'Unexpected units for {key}: {units}')
                    if source == 'era5_pl':
                        exigir(get('typeOfLevel') == 'isobaricInhPa' and int(get('level')) == 850, 'Pressure level must be 850 hPa.')
                    audit.append(dict(variable=key, paramId=param, units=units, expver=1))
                exigir(key not in arrays, 'Duplicate variable or member.')
                lat, lon, field = regular_grid(ec.codes_get_array(g, 'latitudes'), ec.codes_get_array(g, 'longitudes'), values)
                if coordinates is not None:
                    exigir(np.array_equal(lat, coordinates[0]) and np.array_equal(lon, coordinates[1]), 'Inconsistent GRIB grids.')
                coordinates = lat, lon
                arrays[key] = field
            finally:
                ec.codes_release(g)
    exigir(bool(arrays), 'Empty GRIB file.')
    if source == 'seas5':
        exigir(set(arrays) == set(range(51)) and headers <= {0, 51}, 'The future series requires every SEAS5 member 0–50.')
        exigir(np.all(np.diff(lat) == 1) and np.all(np.diff(lon) == 1) and lat.min() <= -60 and lat.max() >= 15 and lon.min() <= -90 and lon.max() >= -25,
                'Seasonal grid must cover the target domain at one degree.')
        mean = np.stack([arrays[i] for i in range(51)]).mean(0, dtype='float64').astype('float32')
        ds = xr.Dataset(dict(seas5_tp_media=(('time', 'lat', 'lon'), mean[None], dict(units='mm/day')),
            time_origem=('time', [origin]), n_membros=('time', [51])), coords=dict(time=[target_date], lat=lat, lon=lon),
            attrs=dict(schema='worcap_seas5_prospective_v1', system='51', forecastMonth=2, correction_tolerance_mm_day=.01))
    else:
        exigir(set(arrays) == {value[0] for value in FIELDS[source].values()}, 'Incomplete atmospheric variables.')
        exigir(np.array_equal(lat, LAT) and np.array_equal(lon, LON), 'ERA5 must match the quarter-degree official grid exactly.')
        ds = xr.Dataset({key: (('time', 'lat', 'lon'), a[None].astype('float32'), dict(units=next(r['units'] for r in audit if r['variable'] == key))) for key, a in arrays.items()},
            coords=dict(time=[origin], lat=lat, lon=lon), attrs=dict(schema='worcap_era5_prospective_v1', expver=1, stream='moda'))
    return ds, dict(validado=True, meses=[origin.strftime('%Y-%m')], meses_alvo=[target], auditoria=audit)


def prepare(registry, source, target, file, transport=None):
    r = registry if isinstance(registry, Registro) else Registro(registry)
    r.prontidao(target)
    raw = Path(file).read_bytes()
    receipt = r.recibo(source, raw, dict(validado=False, meses=[], formato='GRIB awaiting audit'),
                       transport or dict(metodo='local import', filename=Path(file).name, previous_download_time_unknown=True))
    with tempfile.TemporaryDirectory() as temp:
        raw_file = Path(temp) / 'original.grib'
        raw_file.write_bytes(r.ler_objeto(receipt['dados']['objeto']))
        try:
            ds, metadata = convert(raw_file, source, target)
        except Exception as error:
            r.adicionar('auditoria_falhou', dict(fonte=source, pais=[receipt['id']], erro_tipo=type(error).__name__, objetos=[]))
            raise
        result = Path(temp) / 'normalized.nc'
        ds.to_netcdf(result, engine='h5netcdf')
        return r.derivado(source, result.read_bytes(), metadata, [receipt['id']], Path(__file__))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['registry', 'source', 'target', 'file']:
        p.add_argument('--' + name, required=True)
    print(prepare(**vars(p.parse_args()))['id'])
