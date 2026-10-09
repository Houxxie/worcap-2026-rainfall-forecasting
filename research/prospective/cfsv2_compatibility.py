"""Compare CFSv2 services without fitting, filling gaps or authorizing issuance.

Successor files are local serializations of decoded OPeNDAP subsets, not raw
HTTP responses. Legacy NetCDF responses and HTTP metadata are kept separately.
Even numerical agreement on sampled months cannot certify a source migration.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

if __package__:
    from . import cfsv2_adapter as legacy
else:
    import cfsv2_adapter as legacy


SUCCESSOR = 'https://forecast.ccsr.columbia.edu/data/NMME/NOAA-NCEP/CFSv2/pentad_samples/forecast/pr'
NOTICE = 'https://iri.columbia.edu/resources/data-library/sunset/'
SOURCE_REGRID_NOTE = 'https://github.com/iridl/sigrid/blob/19d023912628a38a72435a335b68ba6f209bb178/server/test-regridding/compare_ingrid.py'
ATOL = 1e-5  # Diagnostic numerical comparison, not an operational acceptance rule.
RTOL = 1e-5


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(path):
    b = Path(path).read_bytes()
    return dict(sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def expected_mask(origin):
    return np.arange(28) < (28 if pd.Timestamp(origin).month == 11 else 24)


def successor_fields(ds, origin):
    """Validate coordinates and units; this does not establish initialization dates."""
    t = pd.Timestamp(origin)
    require(t == t.to_period('M').start_time, 'Origin must be the first day of its month.')
    require(set(ds.data_vars) == {'pr'}, 'Unexpected successor data variables.')
    require(set(ds.pr.dims) == {'S', 'M', 'L', 'Y', 'X'}, 'Unexpected successor dimensions.')
    require(ds.pr.attrs.get('units') == 'mm/day', 'Successor precipitation must be in mm/day.')
    require(ds.pr.attrs.get('standard_name') == 'lwe_precipitation_rate', 'Wrong precipitation variable.')
    require(pd.DatetimeIndex(ds.S.values).equals(pd.DatetimeIndex([t])), 'Wrong nominal origin.')
    for name, values in [('M', np.arange(1, 29)), ('L', [1]), ('Y', legacy.LAT), ('X', legacy.LON_360)]:
        require(np.array_equal(ds[name].values, values), f'Wrong successor {name} coordinates.')
    require(ds.target.dims == ('S', 'L') and ds.target_bnds.dims == ('S', 'L', 'nbound'),
            'Unexpected target coordinate dimensions.')
    want = (t + pd.DateOffset(months=1)).to_datetime64()
    end = (t + pd.DateOffset(months=2)).to_datetime64()
    require(ds.target.shape == (1, 1) and ds.target.values[0, 0] == want, 'Wrong target month.')
    require(ds.target_bnds.shape == (1, 1, 2) and np.array_equal(ds.target_bnds.values[0, 0], [want, end]),
            'Wrong target bounds; lead labels alone are insufficient.')
    a = ds.pr.transpose('S', 'M', 'L', 'Y', 'X').values[0, :, 0].astype('float64')
    require(not np.isinf(a).any() and not (a[np.isfinite(a)] < 0).any(), 'Invalid precipitation values.')
    require(np.isnan(a[~expected_mask(t)]).all(), 'Unexpected structurally inactive members.')
    return a


def compare_fields(old, new, origin):
    """Compare the identical complete member subset; never fill an ensemble."""
    shape = (28, len(legacy.LAT), len(legacy.LON_360))
    require(old.shape == new.shape == shape, 'Wrong member/grid shape.')
    mask = expected_mask(origin)
    old_counts = np.isfinite(old).sum(axis=(1, 2))
    new_counts = np.isfinite(new).sum(axis=(1, 2))
    size = len(legacy.LAT) * len(legacy.LON_360)
    common = mask & (old_counts == size) & (new_counts == size)
    result = dict(
        origin=str(pd.Timestamp(origin).date()),
        target=str((pd.Timestamp(origin) + pd.DateOffset(months=1)).date()),
        expected_members=(np.flatnonzero(mask) + 1).tolist(),
        legacy_finite_pixels=old_counts.tolist(), successor_finite_pixels=new_counts.tolist(),
        legacy_complete=bool((old_counts[mask] == size).all()),
        successor_complete=bool((new_counts[mask] == size).all()),
        common_complete_members=(np.flatnonzero(common) + 1).tolist(),
        numerical_agreement_on_overlap=None, numerical_tolerance=dict(atol=ATOL, rtol=RTOL),
        operationally_authorized=False,
        interpretation='Differences between forecast source fields, not errors against observed rainfall.',
    )
    if not common.any():
        result['status'] = 'no_complete_overlap'
        return result
    x, y = old[common], new[common]
    diff = y - x
    mean_diff = diff.mean(axis=0)
    result.update(
        status='compared',
        numerical_agreement_on_overlap=bool(np.allclose(x, y, atol=ATOL, rtol=RTOL)),
        ensemble_mean_difference_rmse=float(np.sqrt(np.mean(mean_diff ** 2))),
        member_field_difference_rmse=float(np.sqrt(np.mean(diff ** 2))),
        maximum_absolute_member_difference=float(np.abs(diff).max()),
        mean_signed_difference=float(diff.mean()),
        memberwise_difference_rmse=np.sqrt(np.mean(diff ** 2, axis=(1, 2))).tolist(),
    )
    return result


def legacy_url(origin):
    t = pd.Timestamp(origin)
    month = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')[t.month - 1]
    s = f'(0000%201%20{month}%20{t.year})'
    return legacy.FONTE + f'S/{s}/{s}/RANGE/L/1.5/VALUES/Y/-61/16/RANGE/X/269/336/RANGE/data.nc'


def download_raw(url, path):
    started = utc()
    with requests.get(url, stream=True, allow_redirects=False, timeout=(20, 90)) as r:
        require(r.status_code == 200, f'HTTP {r.status_code} from {url}')
        chunks, total = [], 0
        for chunk in r.iter_content(1024 * 1024):
            total += len(chunk)
            require(total <= 32 * 1024 ** 2, 'Diagnostic response exceeded 32 MiB.')
            chunks.append(chunk)
        with path.open('xb') as f:
            f.write(b''.join(chunks))
        receipt = dict(url=url, started_at_utc=started, received_at_utc=utc(),
                       kind='raw_http_response', **fingerprint(path),
                       headers={k: r.headers[k] for k in ['Date', 'Last-Modified', 'ETag', 'Content-Type'] if k in r.headers},
                       first_publication_time_unknown=True)
    write_json(path.with_suffix(path.suffix + '.receipt.json'), receipt)
    return receipt


def acquire_pair(folder, origin, engine='netcdf4', include_recipe=False):
    """Small regional requests only. No model or observation data are opened."""
    p = folder / str(pd.Timestamp(origin).date())
    p.mkdir()  # A new acquisition may not overwrite a previous receipt.
    print(f'{origin}: acquiring original and successor subsets', flush=True)
    old_path, new_path = p / 'legacy_raw.nc', p / 'successor_decoded.nc'
    old_receipt = download_raw(legacy_url(origin), old_path)
    started = utc()
    options = dict(timeout=45) if engine == 'pydap' else {}
    with xr.open_dataset(SUCCESSOR, engine=engine, backend_kwargs=options) as source:
        subset = source.sel(S=[pd.Timestamp(origin)], L=[1], Y=slice(-61, 16), X=slice(269, 336)).load()
    ended = utc()
    # Record the exact decoded subset before validation, including invalid data.
    subset.to_netcdf(new_path, engine='netcdf4')
    new_receipt = dict(url=SUCCESSOR, started_at_utc=started, received_at_utc=ended,
                       kind='local_serialization_of_decoded_opendap_subset_not_raw_http',
                       reader=engine,
                       selection=dict(S=str(pd.Timestamp(origin).date()), L=1, Y=[-61, 16], X=[269, 336]),
                       **fingerprint(new_path), first_publication_time_unknown=True,
                       initialization_dates_by_member_proven=False)
    write_json(p / 'successor_decoded.receipt.json', new_receipt)
    old = legacy.carregar_membros(old_path, pd.DatetimeIndex([pd.Timestamp(origin)]))[0]
    new = successor_fields(subset, origin)
    result = compare_fields(old, new, origin)
    result['files'] = dict(legacy=old_receipt, successor=new_receipt)
    if include_recipe:
        if __package__:
            from . import prepare_cfsv2_recipe as recipe
        else:
            import prepare_cfsv2_recipe as recipe
        recipe_path, normalized = p / 'recipe_raw.nc', p / 'recipe_normalized.nc'
        table_path = p / 'initializations_raw.nc'
        table_url = ('https://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/'
                     '.REALTIME_ENSEMBLE/.FLXF/sampleS/' + legacy_url(origin).split('/.prec/')[1].split('L/1.5')[0] + 'data.nc')
        result['files']['recipe'] = download_raw(recipe.recipe_url(origin), recipe_path)
        result['files']['initializations'] = download_raw(table_url, table_path)
        explicit = recipe.normalize_recipe(recipe_path, normalized, origin)
        dates = legacy.tabela_inicializacoes(table_path, pd.DatetimeIndex([origin]))
        result['original_recipe_overlap'] = recipe.verify_overlap(old, explicit, dates[origin]['mascara'])
        result['recipe_vs_successor'] = compare_fields(explicit, new, origin)
        result['files']['normalized_recipe'] = dict(kind='local_normalization', **fingerprint(normalized))
        result['catalog_definition'] = recipe.CATALOG_URL
    write_json(p / 'comparison.json', result)
    print(f'{origin}: {len(result["common_complete_members"])} common members; '
          f'mean-field difference RMSE={result.get("ensemble_mean_difference_rmse")}', flush=True)
    return result


def decision(months):
    # Missing downloads are inconclusive, never evidence of equivalent services.
    if any(m.get('numerical_agreement_on_overlap') is False for m in months):
        return 'not_numerically_interchangeable'
    return 'insufficient_evidence_for_operational_migration'


def run_diagnostics(output, origins, engine='netcdf4', include_recipe=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema='worcap_cfsv2_service_comparison_v1', started_at_utc=utc(),
                  provider_notice=NOTICE, provider_regridding_note=SOURCE_REGRID_NOTE,
                  source_policy_changed=False, model_changed=False, forecast_issued=False,
                  engine=engine, include_original_recipe=include_recipe, months=[])
    try:
        for extension in ['dds', 'das']:
            download_raw(SUCCESSOR + '.' + extension, output / ('successor.' + extension))
    except (OSError, ValueError, requests.RequestException) as error:
        report.update(finished_at_utc=utc(), decision='insufficient_evidence_for_operational_migration',
                      metadata_error=dict(error_type=type(error).__name__, message=str(error)[:500]))
        write_json(output / 'report.json', report)
        return report
    for origin in origins:
        try:
            report['months'].append(acquire_pair(output, origin, engine, include_recipe))
        except (OSError, ValueError, requests.RequestException, KeyError) as error:
            failure = dict(origin=str(pd.Timestamp(origin).date()), status='acquisition_or_contract_failed',
                           error_type=type(error).__name__, message=str(error)[:500], operationally_authorized=False,
                           numerical_agreement_on_overlap=None)
            report['months'].append(failure)
            print(f'{origin}: {failure["status"]} ({failure["error_type"]}); retained files are diagnostic only.', flush=True)
    report['finished_at_utc'] = utc()
    report['decision'] = decision(report['months'])
    write_json(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='A new diagnostic folder.')
    parser.add_argument('--origins', nargs='+', required=True, help='Nominal origins in YYYY-MM format.')
    parser.add_argument('--download', action='store_true', help='Explicitly permit public subset downloads.')
    parser.add_argument('--engine', choices=['netcdf4', 'pydap'], default='netcdf4',
                        help='Explicit OPeNDAP reader. Pydap uses Requests HTTPS; certificate validation stays enabled.')
    parser.add_argument('--include-recipe', action='store_true',
                        help='Also audit the complete original IRI expression and member initialization table.')
    args = parser.parse_args()
    require(args.download, 'Use --download for live acquisition. This command never issues a forecast.')
    origins = [pd.Timestamp(x + '-01') for x in args.origins]
    require(len(set(origins)) == len(origins), 'Duplicate origins.')
    report = run_diagnostics(args.output, origins, args.engine, args.include_recipe)
    print(report['decision'], flush=True)


if __name__ == '__main__':
    main()
