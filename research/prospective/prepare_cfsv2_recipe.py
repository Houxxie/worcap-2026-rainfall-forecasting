"""Re-evaluate IRI's published CFSv2 recipe when its named product is incomplete.

This keeps the original provider, member selection, regridAverage operation and
unit conversion. It never uses CCSR values or joins a partial ensemble to another
product. All members are acquired together through the explicit IRI expression.
Every available cached member must match exactly before a derived input is valid.
"""
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import xarray as xr

if __package__:
    from . import cfsv2_adapter as adapter
else:
    import cfsv2_adapter as adapter


CATALOG_URL = ('https://raw.githubusercontent.com/iridl/dlentries/'
               '7b8bcdd8be1c2beec01e441bcc09e31d5854958a/'
               'entries/Models/NMME/NCEP-CFSv2/FORECAST/PENTAD_SAMPLES/MONTHLY/index.tex')


def recipe_url(origin):
    t = pd.Timestamp(origin)
    adapter.exigir(t == t.to_period('M').start_time and t >= pd.Timestamp('2011-04-01'),
                   'Explicit recipe is for operational forecasts from April 2011 onward.')
    month = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')[t.month - 1]
    s = f'(0000%201%20{month}%20{t.year})'
    # Follow the producer's expression in order; the first field supplies the
    # destination X/Y grid. Only S and L are restricted before the same regrid.
    return ('https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.NCEP-CFSv2/.MONTHLY/.prec/'
            'L/0.5/VALUE/M/1.0/VALUE/S/last/VALUE/'
            'SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.REALTIME_ENSEMBLE/.FLXF/.surface/.PRATE/'
            f'S/{s}/{s}/RANGE/L/1.5/VALUES/%5BX/Y%5D/regridAverage/nip/'
            'c:/0.001/(m3%20kg-1)/:c/mul/c:/1000/(mm%20m-1)/:c/mul/'
            'c:/86400/(s%20day-1)/:c/mul/Y/-61/16/RANGE/X/269/336/RANGE/data.nc')


def normalize_recipe(source, destination, origin):
    """Rename the expression output; values, coordinates and units stay intact."""
    with xr.open_dataset(source, decode_times=False, engine='netcdf4') as ds:
        ds = ds.load()
    adapter.exigir(set(ds.data_vars) == {'aprod'}, 'Unexpected IRI expression output.')
    adapter.exigir(ds.aprod.attrs.get('units') == 'mm/day', 'Recipe did not return mm/day.')
    adapter.exigir(ds.aprod.attrs.get('long_name') == 'Precipitation Rate', 'Unexpected source variable.')
    ds = ds.rename({'aprod': 'prec'})
    ds.prec.attrs['standard_name'] = 'lwe_precipitation_rate'
    ds.attrs.update(acquisition='explicit evaluation of the original IRI catalog expression',
                    catalog_definition=CATALOG_URL, acquisition_url=recipe_url(origin))
    ds.to_netcdf(destination, engine='netcdf4')
    # Apply the unchanged contract for lead, units, grid, origin and member IDs.
    return adapter.carregar_membros(destination, pd.DatetimeIndex([origin]))[0]


def verify_overlap(cached, explicit, expected):
    """Exact equality, without fitted tolerance, rescaling, interpolation or filling."""
    adapter.exigir(cached.shape == explicit.shape == (28, 78, 68), 'Wrong field shape.')
    expected = np.asarray(expected, dtype=bool)
    adapter.exigir(expected.shape == (28,) and expected.sum() in (24, 28), 'Wrong member mask.')
    adapter.exigir(not np.isinf(cached).any(), 'Infinite values in named product.')
    adapter.exigir(np.isnan(cached[~expected]).all() and np.isnan(explicit[~expected]).all(),
                   'Unexpected inactive members.')
    adapter.exigir(np.isfinite(explicit[expected]).all() and (explicit[expected] >= 0).all(),
                   'Explicit recipe still has missing, partial, infinite or negative members.')
    complete = np.isfinite(cached).all(axis=(1, 2)) & expected
    absent = np.isnan(cached).all(axis=(1, 2))
    adapter.exigir((complete | absent).all(), 'Partial named-product member; investigate separately.')
    adapter.exigir(complete.sum() >= expected.sum() - 4, 'Insufficient complete overlap with named product.')
    adapter.exigir(np.array_equal(cached[complete], explicit[complete]),
                   'Explicit recipe differs from named product; source compatibility not established.')
    return dict(complete_cached_members=(np.flatnonzero(complete) + 1).tolist(),
                recovered_members=(np.flatnonzero(expected & absent) + 1).tolist(),
                values_identical_on_overlap=True, maximum_absolute_difference=0.0,
                complete_expected_members=int(expected.sum()),
                ensemble_joined_or_imputed=False, alternate_provider_used=False)


def prepare(registry, target):
    # Existing modules are also used by the standalone Kaggle bundle.
    from registry import Registro, deslocar
    from collect_sources import baixar
    from cfsv2_compatibility import legacy_url

    registry = Path(registry)
    adapter.exigir((registry / 'eventos').is_dir() and any((registry / 'eventos').glob('*.json')),
                   'Restore the existing registry; do not start another history.')
    r = Registro(registry)
    r.eventos(conferir_objetos=True)
    origin = pd.Timestamp(deslocar(target, -1) + '-01')
    nominal_url = legacy_url(origin)
    table_url = ('https://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/'
                 '.REALTIME_ENSEMBLE/.FLXF/sampleS/' + nominal_url.split('/.prec/')[1].split('L/1.5')[0] + 'data.nc')
    parents = []
    for url in [nominal_url, table_url, recipe_url(origin)]:
        event = baixar(r, 'cfsv2', url, 'cfsv2_bruto', target)
        adapter.exigir(event is not None, 'Acquisition failed; no forecast input created.')
        parents.append(event)
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = []
            for i, event in enumerate(parents):
                p = root / f'parent_{i}.nc'
                p.write_bytes(r.ler_objeto(event['dados']['objeto']))
                paths.append(p)
            normalized = root / 'explicit_recipe_normalized.nc'
            explicit = normalize_recipe(paths[2], normalized, origin)
            cached = adapter.carregar_membros(paths[0], pd.DatetimeIndex([origin]))[0]
            table = adapter.tabela_inicializacoes(paths[1], pd.DatetimeIndex([origin]))
            equivalence = verify_overlap(cached, explicit, table[origin]['mascara'])
            ds, audit = adapter.processar_bloco(normalized, pd.DatetimeIndex([origin]), table)
            ds.attrs.update(acquisition_url=recipe_url(origin), catalog_definition=CATALOG_URL,
                            named_product_equivalence='exact equality on every complete cached member',
                            grid='IRI regridAverage on a one-degree grid; not the native Gaussian grid')
            out = root / 'cfsv2.nc'
            ds.to_netcdf(out, engine='netcdf4')
            # Keep every transformation source, including the unchanged adapter.
            code_ref = r.guardar(Path(adapter.__file__).read_bytes())
            proof = r.adicionar('equivalencia_fonte', dict(
                fonte='cfsv2', mes_alvo=target, pais=[e['id'] for e in parents],
                objects_description='unchanged member/date/mean adapter', objetos=[code_ref],
                catalog_definition=CATALOG_URL, comparison=equivalence,
                model_changed=False, source_product_changed=False))
            event = r.derivado('cfsv2', out.read_bytes(), dict(
                validado=True, meses=[origin.strftime('%Y-%m')], meses_alvo=[target],
                unidades='mm/day', lead=1.5, auditoria=audit, equivalencia=equivalence,
                equivalencia_evento=proof['id'], aquisicao='original IRI recipe, explicitly evaluated'),
                [e['id'] for e in parents], Path(__file__))
    except ValueError as error:
        r.adicionar('auditoria_falhou', dict(fonte='cfsv2', mes_alvo=target,
                    motivo=str(error), pais=[e['id'] for e in parents], objetos=[],
                    acao='explicit recipe rejected; no filling or alternate product used'))
        raise
    finally:
        r.exportar_resumo(registry / 'resumo_registro.json')
    print(f'CFSv2 validated: {audit[0]["membros"]} members; exact original-product overlap; {event["id"]}', flush=True)
    return event


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', required=True)
    parser.add_argument('--target', required=True)
    args = parser.parse_args()
    prepare(args.registry, args.target)
