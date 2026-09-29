"""Download three narrowly scoped GRIB inputs with an existing CDS credential."""
from pathlib import Path
import argparse
import tempfile
from registry import Registro, deslocar, agora
from prepare_cds import prepare


def request_for(source, target):
    month = deslocar(target, -1 if source == 'seas5' else -4)
    year, month = month.split('-')
    if source == 'seas5':
        return 'seasonal-monthly-single-levels', dict(originating_centre='ecmwf', system='51',
            variable=['total_precipitation'], product_type=['monthly_mean'], year=[year], month=[month],
            leadtime_month=['2'], area=[16, -91, -61, -24], data_format='grib')
    variables = dict(era5_sl=['total_cloud_cover', '2m_temperature', 'surface_pressure'],
        era5_pl=['geopotential', 'relative_humidity', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind'])
    level = 'single' if source == 'era5_sl' else 'pressure'
    request = dict(product_type=['monthly_averaged_reanalysis'], variable=variables[source], year=[year], month=[month],
        time=['00:00'], area=[15, -90, -60, -25], grid=[.25, .25], data_format='grib', download_format='unarchived')
    if source == 'era5_pl':
        request['pressure_level'] = ['850']
    return f'reanalysis-era5-{level}-levels-monthly-means', request


def execute(registry, target):
    import cdsapi
    r = Registro(registry)
    r.prontidao(target)
    # cdsapi reads the existing environment/.cdsapirc. Never print or persist a key.
    client = cdsapi.Client(quiet=True, progress=False, debug=False)
    records = []
    for source in ['era5_sl', 'era5_pl', 'seas5']:
        if source in r.prontidao(target)['fontes']:
            print(source, 'already audited in this registry; reusing its recorded bytes', flush=True)
            continue
        dataset, request = request_for(source, target)
        started = agora().isoformat()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / (source + '.grib')
            print('Requesting', source, 'for target', target, flush=True)
            client.retrieve(dataset, request).download(str(path))
            records.append(prepare(r, source, target, path, transport=dict(metodo='CDS API',
                dataset=dataset, request=request, started_utc=started, completed_utc=agora().isoformat())))
    r.exportar_resumo(Path(registry) / 'resumo_registro.json')
    return records


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--registry', required=True)
    p.add_argument('--target', required=True)
    execute(**vars(p.parse_args()))
