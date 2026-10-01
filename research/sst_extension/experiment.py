"""One paired, fixed SST experiment; freeze both predictions before opening labels."""
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import gc
import hashlib
import html
import importlib.metadata
import json
import platform
import shutil
import zipfile
import numpy as np
import pandas as pd
from research.common.presentation import display_frame
import xarray as xr
from research.common.inputs import ROOT, library, require, sha256, write_json
from .data import calendar, check, load, INDEX

PROTOCOL = Path(__file__).with_name('protocol.json')
REFERENCE = 'research/temporal_extension/evidence/kaggle_354026629/evaluation/global.csv'
MODELS = ('hybrid', 'hybrid_sst', 'climatology')


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def protocol():
    plan = json.loads(PROTOCOL.read_text())
    require(plan['id'] == 'sst_temporal_extension_2021_2022_v1', 'Unexpected SST protocol.')
    require([plan[k] for k in ['training_start', 'training_end', 'evaluation_start', 'evaluation_end']]
            == ['1990-10-01', '2020-09-01', '2021-01-01', '2022-12-01'], 'Experiment calendar changed.')
    require(plan['weights'] == [.375, .375, .25] and plan['residual_scales'] == [.9, .875]
            and plan['pca']['components'] == 8 and not plan['weight_search']
            and not plan['hyperparameter_search'], 'Keep the prespecified candidate fixed.')
    require([plan[k] for k in ['trees', 'leaves', 'learning_rate', 'seed', 'sampled_points_per_month', 'ridge_alpha']]
            == [300, 31, .05, 42, 5000, .1], 'Recorded training parameters changed.')
    require(plan['lags'] == dict(atmosphere=4, indices=3, sst=2, seasonal_initialization=1, training_rainfall=4)
            and plan['pca'] == dict(components=8, solver='full', area_weight='sqrt(cos(latitude))',
                                   standardize_variance=False, fit_mask_climatology_center_on_training_only=True),
            'Recorded source lags or PCA preprocessing changed.')
    m = library()
    require((m.ARVORES, m.SEMENTE, m.PONTOS_POR_MES, m.ANOS_TREINO, m.MOS_ALPHA, m.SST_COMPONENTES)
            == (300, 42, 5000, 30, .1, 8), 'Training configuration changed.')
    expected = {'objective': 'regression', 'metric': 'rmse', 'learning_rate': .05, 'num_leaves': 31,
                'min_data_in_leaf': 150, 'lambda_l2': 10., 'feature_fraction': .9,
                'bagging_fraction': .8, 'bagging_freq': 1, 'num_threads': 4, 'seed': 42,
                'deterministic': True, 'force_col_wise': True, 'verbosity': -1}
    require(m.PARAMETROS == expected and (m.LAG_ATMOSFERA, m.LAG_INDICES, m.LAG_SST, m.LAG_CHUVA_TREINO)
            == (4, 3, 2, 4), 'Fixed parameters or lags changed.')
    return plan


def source_files():
    names = [INDEX, REFERENCE, 'research/common/inputs.py', 'research/common/presentation.py', 'research/lagged_sources/library.py',
             'research/lagged_sources/ocean_indices.csv', 'research/lagged_sources/official_hashes.json',
             'research/sst_extension/protocol.json']
    names += [p.relative_to(ROOT).as_posix() for p in Path(__file__).parent.glob('*.py')]
    return sorted(set(names))


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def fit_pair(data, sst, cutoff, targets, folder):
    """Shared sampling and ridge; eight PCs are the only added predictors."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    m = data.lib
    train = data.rain.sel(time=slice(None, cutoff))
    X, y, ids, dates, climate, ca, _, _, cs, cc = m.amostrar_arvores_multissistema(
        train, data.atmosphere, data.seas, data.cfs, cutoff)
    require(dates[-1] == cutoff and dates.max() <= targets[0] - pd.DateOffset(months=4), 'Training crosses the label cutoff.')
    before = dict(X=array_hash(X), y=array_hash(y), ids=array_hash(ids))
    pca = m.sst_ajustar_pca(sst, dates)
    train_pc, forecast_pc = m.sst_transformar(sst, pca, dates), m.sst_transformar(sst, pca, targets)
    np.savez_compressed(folder / 'pca.npz', **pca)
    np.savez_compressed(folder / 'sample.npz', ids=ids, time=dates.values)
    np.savez_compressed(folder / 'climate_atmosphere_indices.npz', **ca)
    for name, field in [('rainfall', climate), ('seas5', cs), ('cfsv2', cc)]:
        field.to_netcdf(folder / ('climate_' + name + '.nc'))
    pd.DataFrame(forecast_pc, index=targets, columns=m.SST_FEATURES).to_csv(folder / 'forecast_pcs.csv', index_label='target')
    ridge = m.mos_fit(train, data.seas, data.cfs, cutoff)
    np.savez_compressed(folder / 'ridge.npz', **ridge)
    ridge_predictions = m.mos_predict(ridge, data.seas, data.cfs, targets)
    timings, predictions = [], {}
    count = 0
    for treatment in ['hybrid', 'hybrid_sst']:
        components = []
        for name, base, scale in [('arvores_seas5', m.FEATURES_SEAS5, .9),
                                  ('arvores_multissistema', m.FEATURES_MULTISSISTEMA, .875)]:
            count += 1
            begin = perf_counter()
            features = list(base)
            matrix = np.ascontiguousarray(X[:, :len(base)])
            if treatment == 'hybrid_sst':
                matrix = np.column_stack([matrix, np.repeat(train_pc, ids.shape[1], axis=0)]).astype('float32')
                features += m.SST_FEATURES
            require(np.array_equal(matrix[:, :len(base)], X[:, :len(base)]), 'Base features differ between treatments.')
            print(f'[{count}/4] {treatment}: {len(features)} features, {m.ARVORES} trees; fit and inference...', flush=True)
            dataset = m.lgb.Dataset(matrix, label=y, feature_name=features)
            model = m.lgb.train(m.PARAMETROS, dataset, num_boost_round=m.ARVORES)
            require(model.feature_name() == features, 'Feature order differs.')
            model.save_model(str(folder / f'{treatment}_{name}.txt'))
            del dataset, matrix
            gc.collect()
            args = (model, name, data.atmosphere, data.seas, data.cfs, ca, climate, cs, cc, targets)
            anomalies = m.pesquisa_prever_sst(*args, forecast_pc) if treatment == 'hybrid_sst' else m.prever_par(*args)
            components.append(m.reconstruir_chuva(climate, anomalies, scale).values.astype('float64'))
            timings.append(dict(model=treatment, component=name, seconds=perf_counter() - begin))
            print(f'[{count}/4] complete in {timings[-1]["seconds"]:.1f} s.', flush=True)
            del model, anomalies
        predictions[treatment] = .75 * (.5 * components[0] + .5 * components[1]) + .25 * ridge_predictions
    require(before == dict(X=array_hash(X), y=array_hash(y), ids=array_hash(ids)), 'Shared training arrays changed.')
    predictions['climatology'] = climate.values[targets.month - 1].astype('float64')
    coords = dict(time=targets, lat=data.rain.lat, lon=data.rain.lon)
    ds = xr.Dataset({name: xr.DataArray(value, dims=('time', 'lat', 'lon'), coords=coords,
                       attrs=dict(units='mm/day')) for name, value in predictions.items()},
                    attrs=dict(training_cutoff=str(cutoff.date()), nature='retrospective, historical vintages unverified'))
    require(all(np.isfinite(ds[n].values).all() and (ds[n].values >= 0).all() for n in MODELS), 'Invalid forecasts.')
    ds.to_netcdf(folder / 'predictions.nc', engine='h5netcdf', encoding={n: dict(zlib=True, complevel=4) for n in MODELS})
    with xr.open_dataset(folder / 'predictions.nc') as saved:
        for name in MODELS:
            np.testing.assert_array_equal(ds[name].values, saved[name].values)
    fit_audit = dict(training_start=str(dates[0].date()), training_end=str(dates[-1].date()),
        months=len(dates), examples=len(y), paired_array_hashes=before, shared_ridge=True,
        pca_training_origins=[str(pd.Timestamp(pca['origens_treino'][i]).date()) for i in [0, -1]],
        pca_components=8, pca_explained_variance_ratio=pca['variancia_explicada'].tolist(),
        parameters=m.PARAMETROS, trees=m.ARVORES, weights=[.375, .375, .25], scales=[.9, .875],
        evaluation_labels_read=False, timings=timings, seconds=perf_counter() - started)
    write_json(folder / 'fit.json', fit_audit)
    return ds, fit_audit


def seal(output):
    files = {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob('*'))
             if p.is_file() and p.name != 'frozen.json' and '__pycache__' not in p.parts}
    write_json(output / 'frozen.json', dict(frozen_at=utcnow(), files=files, forecast_issued=False,
               evaluation_labels_read=False))


def verify_frozen(output):
    output = Path(output)
    record = json.loads((output / 'frozen.json').read_text())
    require('models/predictions.nc' in record['files'] and 'signature.json' in record['files'], 'Incomplete frozen record.')
    for name, digest in record['files'].items():
        path = (output / name).resolve()
        require(path.is_relative_to(output.resolve()), 'Artifact path leaves output directory.')
        require(path.is_file() and sha256(path) == digest, 'Changed or missing frozen artifact: ' + name)
    return record


def freeze(output, official=None, seas5=None, cfsv2=None, sst=None):
    plan = protocol()
    paths, audit = check(plan, official, seas5, cfsv2, sst)
    output = Path(output).resolve()
    code = {name: sha256(ROOT / name) for name in source_files()}
    signature = dict(plan=plan, inputs=audit, code=code, python=platform.python_version(),
                     versions={n: importlib.metadata.version(n) for n in ['numpy', 'lightgbm', 'scikit-learn', 'scipy', 'xarray']})
    if output.exists():
        require((output / 'frozen.json').is_file(),
                'An unfinished run exists. Preserve it and select a new OUTPUT directory; it is not overwritten.')
        verify_frozen(output)
        require(json.loads((output / 'signature.json').read_text()) == signature, 'Code, configuration, inputs or environment changed.')
        print('Existing frozen predictions verified; no refitting.', flush=True)
        return output
    for path in [ROOT / 'research', ROOT / 'competition', paths['official'], paths['seas5'], paths['cfsv2'], paths['sst'].parent]:
        require(not output.is_relative_to(path.resolve()), 'Output must be separate from source code and inputs.')
    output.mkdir(parents=True)
    write_json(output / 'signature.json', signature)
    for name in code:
        dest = output / 'code' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
        require(sha256(dest) == code[name], 'Code changed while snapshotting.')
    write_json(output / 'plan.json', plan)
    data, field = load(paths, audit, output, plan)
    train, targets = calendar(plan)
    write_json(output / 'input_audit.json', data.audit)
    data.lib.calendario_emissoes(targets, train[-1]).to_csv(output / 'source_calendar.csv', index=False)
    fit_pair(data, field, train[-1], targets, output / 'models')
    seal(output)
    print('Both forecasts frozen. Evaluation rainfall has not been decoded.', flush=True)
    return output


def metrics(predictions, truth, m):
    require(set(predictions.data_vars) == set(MODELS), 'Incomplete model set.')
    truth, predictions = xr.align(truth, predictions, join='exact')
    require(truth.dims == ('time', 'lat', 'lon') and truth.attrs.get('units') == 'mm/day', 'Unexpected observations.')
    require(np.isfinite(truth.values).all() and (truth.values >= 0).all(), 'Invalid observations.')
    for name in MODELS:
        require(predictions[name].dims == truth.dims and predictions[name].attrs.get('units') == 'mm/day'
                and np.isfinite(predictions[name].values).all() and (predictions[name].values >= 0).all(), 'Invalid prediction field.')
    detail = m.pesquisa_diagnosticos(truth, {n: predictions[n] for n in MODELS}, 'D_2021_2022')
    whole = detail[detail.regiao == 'dominio_inteiro']
    tables = dict(monthly_metrics=detail)
    for name, frame, keys in [('global', whole, ['modelo']), ('years', whole, ['modelo', 'ano']),
                               ('regions', detail, ['modelo', 'regiao']), ('calendar', whole, ['modelo', 'mes'])]:
        tables[name] = m.pesquisa_resumir(frame, keys)
    return tables


def score(output, official=None):
    output = Path(output).resolve()
    verify_frozen(output)
    require(json.loads((output / 'plan.json').read_text()) == protocol(), 'Scoring protocol differs.')
    sig = json.loads((output / 'signature.json').read_text())
    require(sig['code'] == {name: sha256(ROOT / name) for name in source_files()}, 'Scoring code differs from frozen source.')
    from research.common.inputs import find_unique
    path = find_unique('treino_tp.nc', official) / 'treino_tp.nc'
    require(sha256(path) == sig['inputs']['files']['treino_tp.nc'], 'Observation file differs from frozen signature.')
    _, targets = calendar(sig['plan'])
    report = output / 'evaluation'
    report.mkdir(exist_ok=True)
    write_json(report / 'opened.json', dict(opened_at=utcnow(), frozen_sha256=sha256(output / 'frozen.json'),
                retrospective=True, independent_holdout=False))
    with xr.open_dataset(output / 'models/predictions.nc') as ds:
        predictions = ds.load()
    require(pd.DatetimeIndex(predictions.time.values).equals(targets), 'Prediction calendar changed.')
    with xr.open_dataset(path) as ds:
        truth = ds.tp.sel(time=targets).transpose('time', 'lat', 'lon').load()
    tables = metrics(predictions, truth, library())
    for name, frame in tables.items():
        frame.to_csv(report / (name + '.csv'), index=False)
    g = tables['global'].set_index('modelo')
    require((g.n == 1885464).all(), 'Incomplete full-grid evaluation.')
    archived = pd.read_csv(ROOT / REFERENCE).set_index('modelo').loc['hybrid']
    baseline_differences = {key: float(g.loc['hybrid', key] - archived[key]) for key in ['rmse', 'mae', 'vies', 'rmse_area']}
    reproduced = all(abs(v) < 1e-6 for v in baseline_differences.values())
    monthly = tables['monthly_metrics'].query("regiao == 'dominio_inteiro'")
    monthly = library().pesquisa_resumir(monthly, ['modelo', 'mes_alvo']).pivot(index='mes_alvo', columns='modelo', values='rmse')
    delta = monthly.hybrid_sst - monthly.hybrid
    summary = dict(status='complete' if reproduced else 'baseline_mismatch', complete_extension=True,
        baseline_reproduced=reproduced, baseline_differences=baseline_differences,
        delta={key: float(g.loc['hybrid_sst', key] - g.loc['hybrid', key]) for key in ['rmse', 'mae', 'vies', 'rmse_area']},
        months_better=int((delta < 0).sum()), months_worse=int((delta > 0).sum()), months_tied=int((delta == 0).sum()),
        evaluation=['2021-01', '2022-12'], independent_holdout=False, complete_seven_blocks=False,
        historical_publication_vintages_verified=False, forecast_issued=False, automatic_promotion=False,
        seconds_fit=json.loads((output / 'models/fit.json').read_text())['seconds'])
    write_json(report / 'summary.json', summary)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3.6), layout='constrained')
    ax.bar(range(len(delta)), delta, color=['#167d8d' if v < 0 else '#bd6745' for v in delta])
    ax.axhline(0, color='#263945', lw=.8)
    ax.set(xticks=list(range(0, 24, 3)), xticklabels=[t[:7] for t in delta.index[::3]],
           ylabel='SST minus reference model RMSE (mm/day)', title='Additional 2021–2022 block: negative values favor SST')
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(report / 'monthly_comparison.png', dpi=150)
    plt.close(fig)
    text = '<!doctype html><meta charset="utf-8"><title>SST temporal extension</title>'
    text += '<style>body{font:16px system-ui;max-width:1050px;margin:36px auto;padding:0 20px;color:#213442}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}</style>'
    text += '<h1>Does global SST add information?</h1><p>Fixed 8-PC candidate, evaluated on 2021–2022 with the same samples and base inputs as the reference model. Four tree fits and one shared ridge; no neural training or parameter search.</p>'
    text += '<p><strong>Retrospective extension; both years have been consulted before.</strong> Monthly lags do not verify historical product vintages. These results do not authorize operational promotion.</p>'
    text += '<h2>Pooled errors, mm/day</h2>' + display_frame(tables['global'][['modelo', 'rmse', 'mae', 'vies', 'rmse_area']]).to_html(index=False, float_format=lambda x: f'{x:.6f}')
    text += '<h2>Annual errors</h2>' + display_frame(tables['years'][['modelo', 'ano', 'rmse', 'mae', 'vies']]).to_html(index=False, float_format=lambda x: f'{x:.6f}')
    text += '<h2>Audit and decision inputs</h2><pre>' + html.escape(json.dumps(summary, indent=2)) + '</pre>'
    (report / 'report.html').write_text(text, encoding='utf-8')
    verify_frozen(output)
    write_json(report / 'complete.json', dict(files={p.name: sha256(p) for p in report.iterdir()
               if p.is_file() and p.name != 'complete.json'}, summary=summary))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def export_reports(output):
    output = Path(output).resolve()
    verify_frozen(output)
    complete = json.loads((output / 'evaluation/complete.json').read_text())
    for name, digest in complete['files'].items():
        path = (output / 'evaluation' / name).resolve()
        require(path.is_relative_to(output / 'evaluation') and sha256(path) == digest, 'Changed evaluation artifact.')
    destination = output.parent / 'sst_temporal_extension_reports.zip'
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.suffix in {'.json', '.csv', '.html', '.png', '.py'} and '__pycache__' not in path.parts:
                archive.write(path, path.relative_to(output).as_posix())
    return destination
