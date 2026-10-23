"""Evaluate an issued forecast against final ERA5; never fit or issue a model.

Run this file directly. The legacy metric/registry modules intentionally keep
their original imports and bytes because standalone notebooks bundle them.
"""
from pathlib import Path
import argparse
import csv
import html
import importlib.metadata
import json
import platform
import sys
import tempfile

import numpy as np
import xarray as xr

from data_contracts import LAT, LON, conferir_observacao_bruta, metricas, validar_previsao
from evaluate import painel
from prepare_cds import regular_grid
from registry import Registro, agora, canonico, deslocar, digest, exigir, instante, mes

DATASET = 'reanalysis-era5-single-levels-monthly-means'
SOURCE_URL = 'https://cds.climate.copernicus.eu/datasets/' + DATASET
BASE = Path(__file__).resolve().parent


def existing_registry(path):
    path = Path(path).resolve()
    exigir((path / 'objetos').is_dir() and (path / 'eventos').is_dir()
           and any((path / 'eventos').glob('*.json')),
           'Restore a complete existing registry (eventos/ and objetos/) first.')
    return Registro(path)


def check(registry, target):
    """Read-only readiness: calendar eligibility does not assert data availability."""
    r = existing_registry(registry)
    events = r.eventos(conferir_objetos=True)
    protocol = r.protocolo()
    plan = protocol['dados']['plano']
    mes(target)
    exigir(plan['primeiro_mes'] <= target <= plan['ultimo_mes'], 'Target is outside the registered plan.')
    forecasts = [e for e in events if e['tipo'] == 'previsao_congelada'
                 and e['dados']['mes_alvo'] == target]
    exigir(len(forecasts) <= 1, 'Multiple forecasts for the same month.')
    eligible = mes(deslocar(target, 4))
    reasons = []
    if not forecasts:
        reasons.append('No issued forecast for this month; retrospective issuance is not allowed.')
    else:
        forecast = forecasts[0]
        exigir(instante(forecast['registrado_em_utc']) < mes(target), 'Forecast was recorded after its issue deadline.')
        exigir(forecast['dados']['protocolo'] == protocol['id'], 'Forecast belongs to another protocol.')
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'forecast.nc'
            path.write_bytes(r.ler_objeto(forecast['dados']['previsao']))
            validar_previsao(path, target)
    if agora() < eligible:
        reasons.append(f'Final ERA5 verification is eligible from {eligible.isoformat()} (T+4).')
    verified = [e for e in events if e['tipo'] == 'verificacao' and e['dados']['mes_alvo'] == target]
    return dict(target=target, ready=not reasons, blockers=reasons, eligible_from_utc=eligible.isoformat(),
                checked_at_utc=agora().isoformat(), forecast_id=forecasts[0]['id'] if forecasts else None,
                verification_ids=[e['id'] for e in verified], dataset=DATASET,
                note='Eligibility only. Final-product headers and every grid cell must also pass.')


def request_for(target):
    mes(target)
    year, month = target.split('-')
    return dict(product_type=['monthly_averaged_reanalysis'], variable=['total_precipitation'],
                year=[year], month=[month], time=['00:00'], area=[15, -90, -60, -25],
                grid=[0.25, 0.25], data_format='grib', download_format='unarchived')


def download(target, destination):
    import cdsapi
    # The standard CDS configuration supplies credentials; nothing is logged here.
    client = cdsapi.Client(quiet=True, progress=False, debug=False)
    client.retrieve(DATASET, request_for(target)).download(str(destination))


def decode_observation(path, target, folder):
    """Keep GRIB headers, require the full grid, convert moda metres to mm/day."""
    import eccodes as ec
    mes(target)
    encoded = Path(path).read_bytes()
    exigir(encoded.startswith(b'GRIB') and encoded.endswith(b'7777'), 'Require an original unarchived GRIB file.')
    # Decode in memory: ecCodes' file-stream cache can retain Windows file handles
    # during exception unwinding. One requested field must occupy the entire file.
    handle = None
    try:
        handle = ec.codes_new_from_message(encoded)
        exigir(handle is not None, 'Unable to decode the GRIB field.')
        exigir(ec.codes_get_message(handle) == encoded,
               'Require exactly one monthly precipitation field, without duplicates or trailing bytes.')
        get = lambda name: ec.codes_get(handle, name)
        exigir(str(get('class')) == 'ea' and str(get('expver')) in {'1', '0001'}
               and get('stream') == 'moda', 'Require final ERA5: class ea, expver 1, stream moda.')
        exigir(int(get('paramId')) == 228 and get('units') == 'm'
               and get('typeOfLevel') == 'surface', 'Require surface total precipitation in metres.')
        exigir(int(get('dataDate')) == int(target.replace('-', '') + '01')
               and int(get('dataTime')) == 0, 'Observation month/time differs from the target.')
        exigir(int(get('numberOfMissing')) == 0, 'Observation contains missing grid cells.')
        lat, lon, values = regular_grid(ec.codes_get_array(handle, 'latitudes'),
            ec.codes_get_array(handle, 'longitudes'), ec.codes_get_values(handle))
        exigir(np.array_equal(lat, LAT) and np.array_equal(lon, LON),
               'Observation must match the full 301 x 261 quarter-degree grid; no interpolation.')
        exigir(np.isfinite(values).all() and (values >= 0).all(),
               'Observation contains negative or nonfinite precipitation; no clipping.')
    except ec.CodesInternalError as error:
        raise ValueError('Unreadable or unsupported GRIB metadata; keep the original file for inspection.') from error
    finally:
        if handle is not None:
            ec.codes_release(handle)
    coords = dict(time=[np.datetime64(target + '-01')], lat=LAT, lon=LON)
    raw = xr.Dataset(dict(tp=(('time', 'lat', 'lon'), values[None],
        dict(units='m', GRIB_stream='moda', GRIB_experimentVersionNumber='0001'))), coords=coords)
    normalized = xr.Dataset(dict(tp=(('time', 'lat', 'lon'), values[None] * 1000,
                                    dict(units='mm/day'))), coords=coords)
    raw_path, normalized_path = Path(folder) / 'decoded_era5.nc', Path(folder) / 'observation.nc'
    raw.to_netcdf(raw_path, engine='h5netcdf')
    normalized.to_netcdf(normalized_path, engine='h5netcdf')
    audit = conferir_observacao_bruta(raw_path, normalized_path, target)
    return raw_path, normalized_path, audit


def require_new_output(output, registry):
    output = Path(output).resolve()
    exigir(not output.is_relative_to(Path(registry).resolve()), 'Report output must be outside the registry.')
    exigir(not output.exists(), 'Output already exists; use a new folder or the report command with another destination.')
    return output


def versions():
    return dict(python=platform.python_version(), **{name: importlib.metadata.version(name)
                for name in ['numpy', 'pandas', 'xarray', 'h5netcdf', 'eccodes']})


def evaluate(registry, target, output, grib=None, revision=False):
    """Append once under the registry lock. Identical GRIBs reuse the recorded result."""
    r = existing_registry(registry)
    ready = check(r.pasta, target)
    exigir(ready['ready'], ' '.join(ready['blockers']))
    output = require_new_output(output, r.pasta)
    # Avoid a paid/authenticated download if a primary result already exists.
    exigir(not ready['verification_ids'] or revision or grib is not None,
           'Already evaluated. Use report, or --revision for an explicitly labelled ERA5 revision.')
    received = agora().isoformat()
    with tempfile.TemporaryDirectory() as td:
        folder = Path(td)
        original = folder / 'original.grib'
        if grib is None:
            download(target, original)
            transport = dict(method='CDS API', dataset=DATASET, request=request_for(target),
                             started_utc=received, completed_utc=agora().isoformat())
        else:
            original.write_bytes(Path(grib).read_bytes())
            transport = dict(method='local GRIB import', filename=Path(grib).name,
                             imported_at_utc=received, previous_download_time_unknown=True)
        original_bytes = original.read_bytes()
        raw, observation, audit = decode_observation(original, target, folder)
        forecast = r.obter(ready['forecast_id'], 'previsao_congelada')
        fp = folder / 'forecast.nc'
        fp.write_bytes(r.ler_objeto(forecast['dados']['previsao']))
        rows = metricas(fp, observation)
        # This includes the exact converter and reused metric functions, not a mutable version label.
        code = canonico({name: (BASE / name).read_text(encoding='utf-8') for name in
                         ['verify_forecast.py', 'data_contracts.py', 'prepare_cds.py', 'evaluate.py', 'registry.py']})
        with r.transacao():
            current = check(r.pasta, target)
            exigir(current['ready'] and current['forecast_id'] == forecast['id'], 'Forecast eligibility changed during evaluation.')
            previous = [e for e in r.eventos() if e['tipo'] == 'verificacao' and e['dados']['mes_alvo'] == target]
            same = [e for e in previous if e['dados'].get('original_grib', {}).get('sha256') == digest(original_bytes)]
            if same:
                event = same[0]
                reused = True
            else:
                exigir(not previous or revision, 'Different observation already scored. Use --revision; the primary stays unchanged.')
                exigir(not revision or previous, 'No primary evaluation exists yet; omit --revision.')
                original_ref = r.guardar(original_bytes)
                raw_ref = r.guardar(raw.read_bytes())
                observation_ref = r.guardar(observation.read_bytes())
                code_ref = r.guardar(code)
                event = r._adicionar('verificacao', dict(
                    schema='forecast_verification_cli_v1', mes_alvo=target, protocolo=r.protocolo()['id'],
                    previsao=forecast['id'], revisao=len(previous), primaria=not previous, metricas=rows,
                    evidencia=dict(produto='ERA5 monthly_averaged_reanalysis', expver=1, mes_alvo=target,
                        url=SOURCE_URL, transporte=transport, conferencia_bytes=audit, versions=versions()),
                    observacao=observation_ref, bruto=raw_ref, original_grib=original_ref,
                    verification_code=code_ref, objetos=[original_ref, raw_ref, observation_ref, code_ref]))
                reused = False
    # Reports can be regenerated from the event if export is interrupted after registration.
    result = report(r.pasta, output, event['id'])
    return dict(**result, reused=reused, target=target, primary=event['dados']['primaria'])


MODEL_LABELS = {'precipitacao': 'Forecast', 'climatologia': 'Training climatology'}
REGION_LABELS = {'dominio': 'Full domain', 'sul_35': 'South of 35 S',
                 '35_a_15': '35 S to 15 S', 'norte_15': 'North of 15 S'}
STATE_LABELS = {'avaliado': 'Evaluated', 'emitido_aguardando_observacao': 'Issued; awaiting final observation',
                'nao_emitido_prazo_encerrado': 'Not issued; deadline passed', 'aguardando_emissao': 'Awaiting issuance'}


def write_csv(path, rows, columns):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def report(registry, output, event_id=None):
    """Export all primary months and explicitly labelled revisions, without scoring."""
    r = existing_registry(registry)
    output = require_new_output(output, r.pasta)
    with r.transacao():
        events = r.eventos(conferir_objetos=True)
        summary = painel(r)
        selected = r.obter(event_id, 'verificacao') if event_id else None
    rows = []
    for e in events:
        if e['tipo'] != 'verificacao':
            continue
        d = e['dados']
        for m in d['metricas']:
            rows.append(dict(target=d['mes_alvo'], revision=d['revisao'], primary=d['primaria'],
                event_id=e['id'], model=MODEL_LABELS[m['modelo']], region=REGION_LABELS[m['regiao']],
                n=m['n'], rmse=m['rmse'], mae=m['mae'], bias=m['vies'], area_weighted_rmse=m['rmse_area']))
    months = [dict(target=m['mes_alvo'], status=STATE_LABELS[m['estado']],
                   forecast_id=m['previsao'], primary_verification_id=m['verificacao']) for m in summary['meses']]
    aggregates = []
    for key, m in summary['metricas_primarias'].items():
        model, region = key.split('/')
        aggregates.append(dict(model=MODEL_LABELS[model], region=REGION_LABELS[region], n=m['n'],
                               rmse=m['rmse'], mae=m['mae'], bias=m['vies'], area_weighted_rmse=m['rmse_area']))
    snapshot = dict(schema='forecast_verification_report_v1', exported_at_utc=agora().isoformat(),
                    registry_tip=events[-1]['id'], selected_verification_id=selected['id'] if selected else None,
                    planned_months=summary['meses_planejados'], issued_months=summary['meses_emitidos'],
                    evaluated_months=summary['meses_avaliados'], months=months, primary_metrics=aggregates,
                    limitation='Verification against final ERA5, not independent rain gauges. Local timestamps are not independently certified.')
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'metrics.csv', rows, ['target', 'revision', 'primary', 'event_id', 'model', 'region',
                                           'n', 'rmse', 'mae', 'bias', 'area_weighted_rmse'])
    write_csv(output / 'months.csv', months, ['target', 'status', 'forecast_id', 'primary_verification_id'])
    write_csv(output / 'primary_metrics.csv', aggregates,
              ['model', 'region', 'n', 'rmse', 'mae', 'bias', 'area_weighted_rmse'])
    (output / 'summary.json').write_bytes(canonico(snapshot))
    # Hash-verified events include exact data/source/receipt references and conversion evidence.
    verification_events = [e for e in events if e['tipo'] == 'verificacao']
    (output / 'verification_events.json').write_bytes(canonico(verification_events))
    def table(data, columns):
        header = ''.join('<th>' + html.escape(label) + '</th>' for _, label in columns)
        body = ''.join('<tr>' + ''.join('<td>' + html.escape(
            f'{row[key]:.5f}' if isinstance(row[key], float) else str(row[key])) + '</td>'
            for key, _ in columns) + '</tr>' for row in data)
        return '<div class="scroll"><table><thead><tr>' + header + '</tr></thead><tbody>' + body + '</tbody></table></div>'
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Monthly forecast evaluation</title><style>body{font:16px/1.6 system-ui;margin:3rem auto;padding:0 1rem;max-width:1100px;color:#17313c;background:#f5f8f7}h1{line-height:1.2}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white}td,th{padding:.6rem;text-align:left;border-bottom:1px solid #d5e0de}th{white-space:nowrap}code{overflow-wrap:anywhere}a{color:#006b73}</style>
<h1>Monthly forecast evaluation</h1>'''
    page += f'<p>{snapshot["evaluated_months"]} evaluated / {snapshot["issued_months"]} issued / {snapshot["planned_months"]} planned months.</p>'
    page += '<p>Scores use monthly mean rainfall in mm/day. Primary results retain the first verified final ERA5 observation for each month. Revisions are listed separately in <a href="metrics.csv">metrics.csv</a>.</p>'
    page += '<h2>Primary results</h2>'
    page += table(aggregates, [('model', 'Model'), ('region', 'Region'), ('rmse', 'RMSE'), ('mae', 'MAE'),
                               ('bias', 'Bias'), ('area_weighted_rmse', 'Area-weighted RMSE')]) if aggregates else '<p>No observation has been evaluated yet.</p>'
    page += '<p>Global RMSE pools squared errors and grid-cell counts across evaluated months; it is not a mean of monthly RMSEs. Bias is forecast minus ERA5. Regional boundaries are 35°S and 15°S.</p>'
    page += '<h2>Every planned month</h2>' + table(months, [('target', 'Month'), ('status', 'Status')])
    page += '<p>' + html.escape(snapshot['limitation']) + '</p>'
    page += '<p>Registry snapshot: <code>' + snapshot['registry_tip'] + '</code>.</p>'
    page += '<p><a href="summary.json">Summary and pooled metrics</a> · <a href="verification_events.json">Recorded sources and verification events</a></p></html>'
    (output / 'report.html').write_text(page, encoding='utf-8', newline='\n')
    inventory = {p.name: dict(sha256=digest(p.read_bytes()), bytes=p.stat().st_size) for p in sorted(output.iterdir())}
    (output / 'complete.json').write_bytes(canonico(dict(schema=snapshot['schema'], files=inventory)))
    return dict(report=str(output / 'report.html'), selected_verification_id=snapshot['selected_verification_id'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ['check', 'evaluate', 'report']:
        p = sub.add_parser(name)
        p.add_argument('--registry', required=True)
        if name != 'report':
            p.add_argument('--target', required=True)
        if name != 'check':
            p.add_argument('--output', required=True)
        if name == 'evaluate':
            source = p.add_mutually_exclusive_group(required=True)
            source.add_argument('--grib', type=Path, help='Original final ERA5 GRIB, imported now.')
            source.add_argument('--download', action='store_true', help='Retrieve only after the T+4 check.')
            p.add_argument('--revision', action='store_true', help='Append a changed observation; preserve the primary.')
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    args.pop('download', None)
    try:
        value = {'check': check, 'evaluate': evaluate, 'report': report}[command](**args)
        print(json.dumps(value, indent=2))
        return 2 if command == 'check' and not value['ready'] else 0
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f'Verification stopped: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
