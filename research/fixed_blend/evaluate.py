"""Evaluate one fixed blend after the workflow verifies the archived experiment."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import xarray as xr

from research.diagnostics.analyze_errors import (
    BLOCKS, BIN_NAMES, MODELS as ORIGINAL_MODELS, SUMS, digest, exact_alignment,
    intensity_masks, pooled, require, save_json, stats, verify_file,
)

PROTOCOL = Path(__file__).with_name('protocol.json')
MODELS = ('hybrid', 'unet', 'blend', 'climatology')


def validate_protocol():
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    require(protocol['id'] == 'fixed_blend_v1' and protocol['weights'] == {'hybrid': .75, 'unet': .25},
            'This adapter only supports the prespecified 75/25 blend.')
    require(protocol['weight_search'] is False and protocol['conditional_switching'] is False
            and protocol['new_training'] is False, 'Fixed-blend protocol changed.')
    require(protocol['blocks'] == list(BLOCKS) and protocol['evaluation'] == ['2007-01', '2020-12'],
            'Fixed-blend evaluation period changed.')
    return protocol


def fixed_blend(hybrid, unet):
    hybrid, unet = np.asarray(hybrid, dtype='float64'), np.asarray(unet, dtype='float64')
    require(hybrid.shape == unet.shape, 'Blend maps have different shapes.')
    require(np.isfinite(hybrid).all() and np.isfinite(unet).all(), 'Blend inputs contain nonfinite values.')
    require((hybrid >= 0).all() and (unet >= 0).all(), 'Blend inputs must be reconstructed nonnegative rainfall.')
    return .75 * hybrid + .25 * unet


def error_products(observed, hybrid, unet):
    require(np.shape(observed) == np.shape(hybrid) == np.shape(unet), 'Unpaired error products.')
    h = np.asarray(hybrid, dtype='float64') - np.asarray(observed, dtype='float64')
    u = np.asarray(unet, dtype='float64') - np.asarray(observed, dtype='float64')
    return dict(n=h.size, hybrid_error_sum=float(h.sum()), unet_error_sum=float(u.sum()),
                hybrid_sse=float(np.square(h).sum()), unet_sse=float(np.square(u).sum()),
                error_product_sum=float((h * u).sum()), disagreement_sse=float(np.square(u - h).sum()))


def summarize_products(rows):
    result = {key: sum(row[key] for row in rows) for key in rows[0]}
    n = result['n']
    h2, u2, cross = result['hybrid_sse'], result['unet_sse'], result['error_product_sum']
    hvar = max(0., h2 - result['hybrid_error_sum'] ** 2 / n)
    uvar = max(0., u2 - result['unet_error_sum'] ** 2 / n)
    covariance_sum = cross - result['hybrid_error_sum'] * result['unet_error_sum'] / n
    result['centered_error_correlation'] = float(covariance_sum / np.sqrt(hvar * uvar)) if hvar * uvar > 0 else None
    result['blend_sse_from_products'] = .75 ** 2 * h2 + .25 ** 2 * u2 + 2 * .75 * .25 * cross
    # This identity tests the fixed combination, without estimating an optimal weight.
    result['blend_sse_from_disagreement'] = .75 * h2 + .25 * u2 - .75 * .25 * result['disagreement_sse']
    return result


def paired(table, keys):
    baseline = table[table.model == 'hybrid'].set_index(keys)
    candidate = table[table.model == 'blend'].set_index(keys).reindex(baseline.index)
    require(np.array_equal(candidate.n, baseline.n), 'Unpaired blend comparison.')
    result = baseline[['n', 'rmse', 'mae', 'bias']].rename(columns={k: 'hybrid_' + k for k in ['rmse', 'mae', 'bias']})
    for metric in ['rmse', 'mae', 'bias']:
        result['blend_' + metric] = candidate[metric]
        result['delta_' + metric] = candidate[metric] - baseline[metric]
    result['delta_absolute_bias'] = candidate.bias.abs() - baseline.bias.abs()
    result['delta_sse'] = candidate.sse - baseline.sse
    return result.reset_index()


def execute(predictions, observations, output, diagnostics, evidence):
    """Run only on maps already checked by the original diagnostic in this run."""
    predictions, observations, output, diagnostics, evidence = map(Path, (predictions, observations, output, diagnostics, evidence))
    require(not output.exists(), 'Choose a fresh fixed-blend output directory.')
    protocol = validate_protocol()
    verified = json.loads((diagnostics / 'verification.json').read_text(encoding='utf-8'))
    verify_file(observations, verified['inputs']['observations'])
    output.mkdir(parents=True)
    save_json(output / 'protocol.json', protocol)
    monthly, intensities, products = [], [], []
    pixel_sums = None
    for index, block in enumerate(BLOCKS):
        path = predictions / block / 'predictions.nc'
        verify_file(path, verified['inputs'][block + '/predictions.nc'])
        dates = pd.date_range(f'{2007 + 2 * index}-01-01', periods=24, freq='MS')
        with xr.open_dataset(observations) as source, xr.open_dataset(path) as dataset:
            require(source.tp.attrs.get('units') == 'mm/day', 'Unexpected rainfall units.')
            observed, dataset = exact_alignment(source.tp, dataset.load(), dates)
            observed = observed.load()
            lat, lon = observed.lat.values, observed.lon.values
            blend = fixed_blend(dataset.hybrid.values, dataset.unet.values)
            candidate = xr.Dataset({'blend': (('time', 'lat', 'lon'), blend, {'units': 'mm/day'})},
                                   coords=dataset.coords,
                                   attrs=dict(protocol='fixed_blend_v1', hybrid_weight=.75, unet_weight=.25,
                                              source_prediction_sha256=digest(path), independent_holdout='false',
                                              scope='historical development; not a new operational forecast'))
            target = output / 'predictions' / block / 'predictions.nc'
            target.parent.mkdir(parents=True)
            candidate.to_netcdf(target, engine='h5netcdf')
            with xr.open_dataset(target) as written:
                np.testing.assert_array_equal(written.blend.values, blend)
                require(written.blend.dtype == np.dtype('float64'), 'Blend precision changed on disk.')
            regions = {'full_domain': np.ones(len(lat), bool), 'south_of_35S': lat < -35,
                       '35S_to_15S': (lat >= -35) & (lat < -15), 'north_of_15S': lat >= -15}
            if pixel_sums is None:
                pixel_sums = {model: {key: np.zeros((len(lat), len(lon))) for key in ['sse', 'error', 'absolute_error']} for model in MODELS}
            for i, date in enumerate(dates):
                truth = observed.values[i].astype('float64')
                masks = intensity_masks(truth)
                product = error_products(truth, dataset.hybrid.values[i], dataset.unet.values[i])
                product.update(block=block, date=str(date.date()))
                products.append(product)
                for model in MODELS:
                    pred = blend[i] if model == 'blend' else dataset[model].values[i].astype('float64')
                    error = pred - truth
                    for metric, value in [('sse', error ** 2), ('error', error), ('absolute_error', np.abs(error))]:
                        pixel_sums[model][metric] += value
                    for region, mask in regions.items():
                        monthly.append(dict(block=block, date=str(date.date()), year=date.year,
                                            calendar_month=date.month, region=region, model=model,
                                            **stats(truth[mask], pred[mask])))
                    for label, mask in zip(BIN_NAMES, masks):
                        intensities.append(dict(block=block, model=model, intensity=label, **stats(truth[mask], pred[mask])))
        print(block, 'fixed 75/25 maps saved and read back; paired errors calculated', flush=True)

    monthly = pd.DataFrame(monthly)
    domain = monthly[monthly.region == 'full_domain']
    intensity_rows = pd.DataFrame(intensities)
    tables = {'global': pooled(domain, ['model']), 'blocks': pooled(domain, ['model', 'block']),
              'years': pooled(domain, ['model', 'year']), 'calendar': pooled(domain, ['model', 'calendar_month']),
              'region': pooled(monthly, ['model', 'region']), 'intensity': pooled(intensity_rows, ['model', 'intensity']),
              'block_intensity': pooled(intensity_rows, ['model', 'block', 'intensity'])}
    values = tables['global'].set_index('model')
    require((values.n == 168 * 301 * 261).all(), 'Incomplete comparison.')
    original = pd.read_csv(diagnostics / 'global.csv').set_index('model')
    np.testing.assert_allclose(values.loc[list(ORIGINAL_MODELS), SUMS], original.loc[list(ORIGINAL_MODELS), SUMS], rtol=1e-12, atol=1e-7)
    np.testing.assert_allclose(tables['intensity'].groupby('model')[SUMS].sum().sort_index(), values[SUMS].sort_index(), rtol=1e-12, atol=1e-7)
    for name, keys in [('blocks', ['block']), ('years', ['year']), ('calendar', ['calendar_month']),
                       ('region', ['region']), ('intensity', ['intensity']), ('block_intensity', ['block', 'intensity'])]:
        tables[name + '_comparison'] = paired(tables[name], keys)
    numerical_products = [{key: value for key, value in row.items() if key not in {'block', 'date'}} for row in products]
    complementarity = summarize_products(numerical_products)
    np.testing.assert_allclose([complementarity['blend_sse_from_products'], complementarity['blend_sse_from_disagreement']],
                               values.loc['blend', 'sse'], rtol=1e-12, atol=1e-7)
    maps = xr.Dataset({f'{model}_{metric}': (('lat', 'lon'), array, {'units': 'mm/day'})
                       for model in MODELS for metric, array in [
                           ('rmse', np.sqrt(pixel_sums[model]['sse'] / 168)),
                           ('mae', pixel_sums[model]['absolute_error'] / 168),
                           ('bias', pixel_sums[model]['error'] / 168)]}, coords=dict(lat=lat, lon=lon),
                      attrs=dict(months=168, protocol='fixed_blend_v1', independent_holdout='false'))
    summary = dict(protocol='fixed_blend_v1', months=168, gridpoint_months=int(values.loc['blend', 'n']),
                   hybrid_rmse=float(values.loc['hybrid', 'rmse']), unet_rmse=float(values.loc['unet', 'rmse']),
                   blend_rmse=float(values.loc['blend', 'rmse']),
                   delta_rmse_vs_hybrid=float(values.loc['blend', 'rmse'] - values.loc['hybrid', 'rmse']),
                   relative_rmse_reduction_percent=float(100 * (1 - values.loc['blend', 'rmse'] / values.loc['hybrid', 'rmse'])),
                   delta_mae_vs_hybrid=float(values.loc['blend', 'mae'] - values.loc['hybrid', 'mae']),
                   delta_absolute_bias_vs_hybrid=float(abs(values.loc['blend', 'bias']) - abs(values.loc['hybrid', 'bias'])),
                   blend_lower_rmse_grid_fraction=float((maps.blend_rmse < maps.hybrid_rmse).mean()),
                   input_hashes_verified=True, original_scores_reproduced=True, blend_maps_read_back_exactly=True,
                   mse_identity_verified=True, intensity_partition_verified=True, independent_holdout=False,
                   model_fitted=False, weight_search=False, model_promoted=False, forecast_issued=False)
    for name in ['blocks', 'years', 'calendar']:
        delta = tables[name + '_comparison'].delta_rmse
        summary[name] = dict(improved=int((delta < 0).sum()), worsened=int((delta > 0).sum()), tied=int((delta == 0).sum()))
    for name, table in tables.items():
        table.to_csv(output / (name + '.csv'), index=False, lineterminator='\n')
    monthly.to_csv(output / 'monthly_metrics.csv', index=False, lineterminator='\n')
    pd.DataFrame(products).to_csv(output / 'monthly_error_products.csv', index=False, lineterminator='\n')
    maps.to_netcdf(output / 'error_maps.nc', engine='h5netcdf')
    save_json(output / 'complementarity.json', complementarity)
    save_json(output / 'summary.json', summary)
    from .report import plot_results
    plot_results(tables, maps, output)
    print(json.dumps(summary, indent=2), flush=True)
    return summary
