"""Audit saved maps and describe errors without fitting or selecting a model.

Run from the repository root with --predictions, --observations and --output.
The observations must be the original treino_tp.nc, not the shifted tp_alvo.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'research/spatial_unet/evidence/seven_blocks'
BLOCKS = ('H1', 'H2', 'H3', 'H4', 'A', 'B', 'C')
MODELS = ('hybrid', 'unet', 'climatology')
BINS = np.array([0., 1., 3., 5., 10., np.inf])
BIN_NAMES = ('[0,1)', '[1,3)', '[3,5)', '[5,10)', '[10,infinity)')
COLORS = dict(hybrid='#087e8b', unet='#db654a', climatology='#9299a1')
SUMS = ['n', 'sse', 'error_sum', 'absolute_error_sum', 'observed_sum', 'predicted_sum']


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def stats(observed, predicted):
    observed, predicted = np.asarray(observed, dtype='float64'), np.asarray(predicted, dtype='float64')
    require(observed.shape == predicted.shape, 'Different observation/prediction shapes.')
    require(np.isfinite(observed).all() and np.isfinite(predicted).all(), 'Nonfinite rainfall.')
    error = predicted - observed
    return dict(n=error.size, sse=float(np.square(error).sum()), error_sum=float(error.sum()),
                absolute_error_sum=float(np.abs(error).sum()), observed_sum=float(observed.sum()),
                predicted_sum=float(predicted.sum()))


def pooled(frame, groups):
    result = frame.groupby(groups, as_index=False, sort=False, observed=True)[SUMS].sum()
    denominator = result.n.replace(0, np.nan)
    result['rmse'] = np.sqrt(result.sse / denominator)
    result['mae'] = result.absolute_error_sum / denominator
    result['bias'] = result.error_sum / denominator
    result['observed_mean'] = result.observed_sum / denominator
    result['predicted_mean'] = result.predicted_sum / denominator
    return result


def intensity_masks(observed):
    require(np.isfinite(observed).all() and (observed >= 0).all(), 'Invalid observed rainfall.')
    return [(observed >= low) & (observed < high) for low, high in zip(BINS[:-1], BINS[1:])]


def exact_alignment(observed, forecast, dates):
    require(set(forecast.data_vars) == set(MODELS), 'Unexpected forecast variables.')
    require(pd.DatetimeIndex(forecast.time.values).equals(dates), 'Wrong or reordered forecast dates.')
    for name in MODELS:
        require(forecast[name].dims == ('time', 'lat', 'lon'), 'Wrong forecast dimensions.')
        require(forecast[name].attrs.get('units') == 'mm/day', 'Wrong forecast units.')
    # Exact alignment rejects flipped, reordered or merely similar coordinates.
    obs = observed.sel(time=dates).transpose('time', 'lat', 'lon')
    obs, forecast = xr.align(obs, forecast, join='exact')
    require(np.isfinite(obs.values).all() and (obs.values >= 0).all(), 'Invalid observations.')
    for name in MODELS:
        require(np.isfinite(forecast[name]).all() and (forecast[name] >= 0).all(), 'Invalid predictions.')
    return obs, forecast


def verify_file(path, expected):
    actual = digest(path)
    require(actual == expected, f'Hash mismatch: {Path(path).name}')
    return actual


def comparison(table, keys, total_n):
    hybrid = table[table.model == 'hybrid'].set_index(keys)
    unet = table[table.model == 'unet'].set_index(keys).reindex(hybrid.index)
    require(np.array_equal(hybrid.n, unet.n), 'Unpaired diagnostic groups.')
    result = hybrid[['n', 'rmse', 'mae', 'bias']].rename(columns={c: 'hybrid_' + c for c in ['rmse', 'mae', 'bias']})
    for c in ['rmse', 'mae', 'bias']:
        result['unet_' + c] = unet[c]
        result['delta_' + c] = unet[c] - hybrid[c]
    result['delta_sse'] = unet.sse - hybrid.sse
    result['contribution_to_global_delta_mse'] = result.delta_sse / total_n
    return result.reset_index()


def figure_setup():
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})


def plot_results(tables, maps, histories, output):
    figure_setup()
    monthly = tables['calendar_comparison']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for model in MODELS:
        data = tables['calendar_month'].query('model == @model').sort_values('calendar_month')
        axes[0].plot(data.calendar_month, data.rmse, marker='o', label=model.title(), color=COLORS[model])
    axes[0].set(xlabel='Target calendar month', ylabel='Pooled RMSE (mm/day)', xticks=range(1, 13))
    axes[0].legend(frameon=False)
    axes[1].bar(monthly.calendar_month, monthly.delta_rmse,
                color=np.where(monthly.delta_rmse > 0, COLORS['unet'], COLORS['hybrid']))
    axes[1].axhline(0, color='#555', linewidth=.8)
    axes[1].set(xlabel='Target calendar month', ylabel='U-Net minus hybrid RMSE (mm/day)', xticks=range(1, 13))
    fig.suptitle('Calendar-month errors | 2007–2020 development years')
    fig.savefig(output / 'calendar_month.png', dpi=155); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for model in MODELS:
        data = tables['intensity'].query('model == @model').set_index('intensity').loc[list(BIN_NAMES)]
        axes[0].plot(range(5), data.rmse, marker='o', label=model.title(), color=COLORS[model])
        axes[1].plot(range(5), data.bias, marker='o', color=COLORS[model])
    for ax in axes:
        ax.set_xticks(range(5), ['0–<1', '1–<3', '3–<5', '5–<10', '≥10'])
        ax.set_xlabel('Observed monthly mean rainfall (mm/day)')
    axes[0].set_ylabel('Conditional RMSE (mm/day)'); axes[0].legend(frameon=False)
    axes[1].set_ylabel('Conditional bias: predicted − observed (mm/day)')
    axes[1].axhline(0, color='#555', linewidth=.8)
    fig.suptitle('Rainfall intensity | fixed descriptive bins, not daily extremes')
    fig.savefig(output / 'rainfall_intensity.png', dpi=155); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 6.7), constrained_layout=True)
    delta = maps.unet_rmse - maps.hybrid_rmse
    vmax = float(max(maps.hybrid_rmse.max(), maps.unet_rmse.max()))
    for ax, name in zip(axes[:2], ['hybrid', 'unet']):
        artist = ax.pcolormesh(maps.lon, maps.lat, maps[name + '_rmse'], shading='auto', cmap='viridis', vmin=0, vmax=vmax)
        ax.set_title(name.title() + ' RMSE')
    fig.colorbar(artist, ax=list(axes[:2]), label='RMSE (mm/day)', orientation='horizontal', shrink=.9)
    bound = float(np.abs(delta).max())
    artist = axes[2].pcolormesh(maps.lon, maps.lat, delta, shading='auto', cmap='RdBu_r', vmin=-bound, vmax=bound)
    axes[2].set_title('U-Net − hybrid RMSE')
    fig.colorbar(artist, ax=axes[2], label='Difference (mm/day)', orientation='horizontal')
    for ax in axes:
        ax.set(xlabel='Longitude (°)', ylabel='Latitude (°)', xlim=(-90, -25), ylim=(-60, 15), aspect='equal')
    fig.suptitle('Errors at each grid cell | all 168 months, full supplied domain')
    fig.savefig(output / 'spatial_errors.png', dpi=155); plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(12, 12), constrained_layout=True)
    for ax, (block, inner, outer, selected) in zip(axes.flat, histories):
        ax.plot(inner.epoch, np.sqrt(inner.training_mse), label='Inner training √MSE (online)', color='#9299a1')
        ax.plot(inner.epoch, inner.inner_rmse, label='Inner validation RMSE', color=COLORS['unet'])
        ax.plot(outer.epoch, np.sqrt(outer.training_mse), '--', label='Outer training √MSE (online)', color=COLORS['hybrid'])
        ax.axvline(selected, color='#333', linestyle=':', label='Selected epoch')
        ax.set(title=f'{block} | selected epoch {selected}', xlabel='Epoch', ylabel='mm/day')
    axes.flat[-1].axis('off')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc='upper left', frameon=False)
    axes.flat[-1].text(.05, .12, 'Training losses use evolving weights and unclipped\nanomalies. Validation uses a fixed end-of-epoch\nmodel and clipped rainfall; gaps are not directly\ncomparable generalization-error estimates.', transform=axes.flat[-1].transAxes, fontsize=9)
    fig.suptitle('Training histories | original epoch selection, no new fitting')
    fig.savefig(output / 'training_curves.png', dpi=150); plt.close(fig)


def execute(predictions, observations, output, evidence=EVIDENCE):
    predictions, observations, output, evidence = map(Path, (predictions, observations, output, evidence))
    require(not output.exists(), 'Choose a new output directory; existing reports are preserved.')
    signature = read_json(evidence / 'signature.json')
    signature_hash = digest(evidence / 'signature.json')
    inputs = {'observations': verify_file(observations, signature['inputs']['files']['treino_tp.nc']), 'signature': signature_hash}
    monthly_rows, intensity_rows, histories, training_rows = [], [], [], []
    pixel_sums = None
    for block_index, block in enumerate(BLOCKS):
        completed = read_json(evidence / block / 'complete.json')
        require(completed['signature'] == signature_hash, 'Run signature mismatch.')
        p = predictions / block / 'predictions.nc'
        inputs[block + '/predictions.nc'] = verify_file(p, completed['files']['predictions.nc'])
        for name in ['monthly_metrics.csv', 'inner/training_history.json', 'inner/training.json', 'unet/training_history.json', 'unet/training.json']:
            inputs[block + '/' + name] = verify_file(evidence / block / name, completed['files'][name])
        dates = pd.date_range(f'{2007 + 2 * block_index}-01-01', periods=24, freq='MS')
        archived = pd.read_csv(evidence / block / 'monthly_metrics.csv')
        with xr.open_dataset(observations) as source, xr.open_dataset(p) as ds:
            require(source.tp.attrs.get('units') == 'mm/day', 'Wrong observation units.')
            observed, ds = exact_alignment(source.tp, ds.load(), dates)
            observed = observed.load()
            lat, lon = observed.lat.values, observed.lon.values
            require(np.array_equal(lat, np.arange(-60, 15.25, .25)) and np.array_equal(lon, np.arange(-90, -24.75, .25)), 'Unexpected target grid.')
            regions = {'full_domain': np.ones(len(lat), bool), 'south_of_35S': lat < -35,
                       '35S_to_15S': (lat >= -35) & (lat < -15), 'north_of_15S': lat >= -15}
            original_names = dict(full_domain='dominio_inteiro', south_of_35S='lat_menor_que_menos35',
                                  **{'35S_to_15S': 'lat_menos35_a_menos15', 'north_of_15S': 'lat_maior_igual_menos15'})
            if pixel_sums is None:
                pixel_sums = {model: {metric: np.zeros((len(lat), len(lon))) for metric in ['sse', 'error', 'absolute_error']} for model in MODELS}
            for i, date in enumerate(dates):
                truth = observed.values[i].astype('float64')
                masks = intensity_masks(truth)
                for model in MODELS:
                    pred = ds[model].values[i].astype('float64')
                    error = pred - truth
                    pixel_sums[model]['sse'] += error ** 2
                    pixel_sums[model]['error'] += error
                    pixel_sums[model]['absolute_error'] += np.abs(error)
                    for region, mask in regions.items():
                        row = stats(truth[mask], pred[mask])
                        old = archived[(archived.modelo == model) & (archived.mes_alvo == str(date.date())) & (archived.regiao == original_names[region])]
                        require(len(old) == 1, 'Missing or duplicated archived monthly score.')
                        old = old.iloc[0]
                        for new_key, old_key in [('n', 'n'), ('sse', 'sse'), ('error_sum', 'soma_erro'), ('absolute_error_sum', 'soma_erro_absoluto')]:
                            require(np.isclose(row[new_key], old[old_key], rtol=1e-10, atol=1e-8), f'Recomputed score differs: {block}/{model}/{date}/{region}/{new_key}')
                        monthly_rows.append(dict(block=block, date=str(date.date()), calendar_month=date.month, region=region, model=model, **row))
                    for label, mask in zip(BIN_NAMES, masks):
                        intensity_rows.append(dict(block=block, model=model, intensity=label, **stats(truth[mask], pred[mask])))
        inner = pd.DataFrame(read_json(evidence / block / 'inner/training_history.json'))
        outer = pd.DataFrame(read_json(evidence / block / 'unet/training_history.json'))
        chosen = read_json(evidence / block / 'inner/training.json')['selected_epochs']
        require(chosen == int(inner.loc[inner.inner_rmse.idxmin(), 'epoch']) == len(outer), 'Epoch selection/history mismatch.')
        require(chosen == read_json(evidence / block / 'unet/training.json')['selected_epochs'], 'Outer epochs differ from inner selection.')
        histories.append((block, inner, outer, chosen))
        training_rows.append(dict(block=block, selected_epochs=chosen, inner_epochs_run=len(inner),
                                  first_inner_rmse=float(inner.inner_rmse.iloc[0]), best_inner_rmse=float(inner.inner_rmse.min()),
                                  last_inner_rmse=float(inner.inner_rmse.iloc[-1]),
                                  inner_training_mse_first=float(inner.training_mse.iloc[0]), inner_training_mse_last=float(inner.training_mse.iloc[-1])))
        print(block, 'maps, dates, hashes, archived scores and epoch selection verified', flush=True)
    monthly = pd.DataFrame(monthly_rows)
    domain = monthly.query("region == 'full_domain'")
    intensity_months = pd.DataFrame(intensity_rows)
    tables = {'global': pooled(domain, ['model']), 'calendar_month': pooled(domain, ['model', 'calendar_month']),
              'region': pooled(monthly, ['model', 'region']), 'region_calendar_month': pooled(monthly, ['model', 'region', 'calendar_month']),
              'intensity': pooled(intensity_months, ['model', 'intensity']), 'block_intensity': pooled(intensity_months, ['model', 'block', 'intensity']),
              'training': pd.DataFrame(training_rows)}
    total_n = int(tables['global'].query("model == 'hybrid'").n.iloc[0])
    require(total_n == 168 * 301 * 261, 'Incomplete evaluation.')
    np.testing.assert_allclose(tables['intensity'].groupby('model')[SUMS].sum().sort_index(), tables['global'].set_index('model')[SUMS].sort_index(), rtol=1e-12, atol=1e-7)
    archived_global = pd.read_csv(evidence / 'global.csv').set_index('modelo')
    for model in MODELS:
        now = tables['global'].set_index('model').loc[model]
        require(np.isclose(now.rmse, archived_global.loc[model, 'rmse'], rtol=0, atol=1e-12), 'Global score did not reproduce.')
    for name, keys in [('calendar', ['calendar_month']), ('region', ['region']), ('region_calendar', ['region', 'calendar_month']), ('intensity', ['intensity']), ('block_intensity', ['block', 'intensity'])]:
        base = {'calendar': 'calendar_month', 'region_calendar': 'region_calendar_month'}.get(name, name)
        tables[name + '_comparison'] = comparison(tables[base], keys, total_n)
    variables = {}
    for model in MODELS:
        for name, values in [('rmse', np.sqrt(pixel_sums[model]['sse'] / 168)), ('bias', pixel_sums[model]['error'] / 168), ('mae', pixel_sums[model]['absolute_error'] / 168)]:
            variables[model + '_' + name] = (('lat', 'lon'), values, dict(units='mm/day'))
    maps = xr.Dataset(variables, coords=dict(lat=lat, lon=lon), attrs=dict(months=168, independent_holdout='false', analysis='development diagnostics; no fitting'))
    better = maps.unet_rmse < maps.hybrid_rmse
    global_values = tables['global'].set_index('model')
    summary = dict(months=168, gridpoint_months=total_n, all_seven_prediction_hashes_verified=True,
                   monthly_region_scores_reproduced=len(monthly), intensity_partition_verified=True,
                   hybrid_rmse=float(global_values.loc['hybrid', 'rmse']), unet_rmse=float(global_values.loc['unet', 'rmse']),
                   global_delta_mse=float((global_values.loc['unet', 'sse'] - global_values.loc['hybrid', 'sse']) / total_n),
                   unet_lower_rmse_grid_fraction=float(better.mean()), unet_higher_rmse_grid_fraction=float((maps.unet_rmse > maps.hybrid_rmse).mean()),
                   calendar_months_unet_improved=int((tables['calendar_comparison'].delta_rmse < 0).sum()),
                   independent_holdout=False, model_fitted=False, model_promoted=False, forecast_issued=False)
    output.mkdir(parents=True)
    for name, table in tables.items():
        table.to_csv(output / (name + '.csv'), index=False, lineterminator='\n')
    maps.to_netcdf(output / 'error_maps.nc', engine='h5netcdf')
    save_json(output / 'summary.json', summary)
    plot_results(tables, maps, histories, output)
    save_json(output / 'verification.json', dict(inputs=inputs, code_sha256=digest(__file__),
        python=platform.python_version(), versions={name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'xarray', 'matplotlib', 'h5netcdf']},
        historical_run_environment=signature['inputs']['versions'], environment_scope='diagnostics only; original model was not rerun',
        bins_mm_day=['0', '1', '3', '5', '10', 'infinity'],
        files={p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True, help='Directory containing H1/.../C predictions.nc files')
    parser.add_argument('--observations', type=Path, required=True, help='Original official treino_tp.nc')
    parser.add_argument('--output', type=Path, required=True, help='New report directory')
    args = parser.parse_args()
    execute(args.predictions, args.observations, args.output)
