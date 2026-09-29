"""Orchestrate the existing fixed experiment and diagnostic without retuning it."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import shutil
import sys

from .config import ROOT, BLOCKS, PROTOCOL, WorkflowError, require, read_json, load

REPORT_FILES = ('monthly_metrics.csv', 'split.json', 'source_calendar.csv', 'baseline_check.json',
                'inner/training_history.json', 'inner/training.json', 'unet/training_history.json', 'unet/training.json')
PACKAGES = ('numpy', 'pandas', 'xarray', 'matplotlib', 'h5netcdf', 'lightgbm', 'torch')


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def code_files():
    names = []
    for folder in ['common', 'spatial_unet', 'diagnostics', 'workflow']:
        names.extend(p for p in (ROOT / 'research' / folder).glob('*.py'))
    names += [PROTOCOL, ROOT / 'research/lagged_sources/library.py', ROOT / 'research/lagged_sources/official_hashes.json',
              ROOT / 'research/lagged_sources/ocean_indices.csv', ROOT / 'research/lagged_sources/evidence/metricas_blocos.csv']
    return sorted(set(names))


def verify_hash(path, expected):
    path = Path(path)
    require(path.is_file(), f'Missing file: {path}. Attach or restore the saved output; reports alone do not contain forecast maps.')
    actual = sha256(path)
    require(actual == expected, f'File hash differs: {path}. Use the matching snapshot; do not edit the expected hash to bypass this check.')
    return actual


def inspect_saved(inputs):
    evidence, predictions = Path(inputs['evidence']), Path(inputs['predictions'])
    require((evidence / 'signature.json').is_file(), f'No signature.json in {evidence}. Set inputs.evidence to the complete saved-run reports.')
    signature = read_json(evidence / 'signature.json')
    require(signature.get('protocol') == read_json(PROTOCOL), 'The saved run uses a different scientific protocol. This adapter supports spatial_unet_v1 only.')
    signature_hash = sha256(evidence / 'signature.json')
    expected_rain = next(row['sha256'] for row in read_json(ROOT / 'research/lagged_sources/official_hashes.json') if row['nome'] == 'treino_tp.nc')
    require(signature.get('inputs', {}).get('files', {}).get('treino_tp.nc') == expected_rain, 'Source signature names a different observation snapshot.')
    inventory = {'signature.json': signature_hash,
                 'observations': verify_hash(inputs['observations'], expected_rain)}
    for block in BLOCKS:
        path = evidence / block / 'complete.json'
        require(path.is_file(), f'Missing completed block {block}: {path}. This workflow requires all seven blocks.')
        completion = read_json(path)
        require(completion.get('signature') == signature_hash, f'{block}: source signature does not match the completed block.')
        inventory[block + '/complete.json'] = sha256(path)
        for name in ('predictions.nc',) + REPORT_FILES:
            require(name in completion.get('files', {}), f'{block}: completion record has no {name}.')
            source = predictions / block / name if name == 'predictions.nc' else evidence / block / name
            inventory[block + '/' + name] = verify_hash(source, completion['files'][name])
    # The diagnostics also cross-check this aggregate against the verified maps.
    require((evidence / 'global.csv').is_file(), 'Missing global.csv in source evidence.')
    inventory['global.csv'] = sha256(evidence / 'global.csv')
    return inventory, signature_hash


def preflight(config):
    for name in ('numpy', 'pandas', 'xarray', 'matplotlib', 'h5netcdf'):
        require(importlib.util.find_spec(name) is not None, f'Missing {name}. Install research/workflow/requirements_evaluation.txt in an evaluation environment.')
    require(sys.version_info >= (3, 12), 'Use Python 3.12 or later for this workflow.')
    if config['mode'] == 'saved':
        inventory, source_signature = inspect_saved(config['inputs'])
    else:
        require(importlib.util.find_spec('torch') is not None, 'Training needs PyTorch. Use the separate spatial_unet training environment, preferably on a Kaggle GPU.')
        from research.common.inputs import preflight as training_preflight
        _, audit = training_preflight(**config['inputs'])
        import torch
        require(config['device'] != 'cuda' or torch.cuda.is_available(), 'CUDA was requested but is unavailable. Enable a Kaggle GPU, or explicitly choose cpu for a slower run.')
        inventory, source_signature = audit['files'], None
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    code = {p.relative_to(ROOT).as_posix(): sha256(p) for p in code_files()}
    facts = dict(mode=config['mode'], device=config.get('device'), scientific_protocol=read_json(PROTOCOL),
                 inputs=inventory, code=code, environment=dict(python=platform.python_version(), packages=versions))
    return dict(**facts, source_signature_sha256=source_signature, fingerprint=identity_hash(facts),
                scope='development_2007_2020', independent_holdout=False, model_promoted=False, forecast_issued=False)


def copy_evidence(source, destination):
    source, destination = Path(source), Path(destination)
    names = ['signature.json', 'global.csv'] + [f'{b}/{n}' for b in BLOCKS for n in ('complete.json',) + REPORT_FILES]
    for name in names:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)


def comparisons(evidence, destination):
    import numpy as np
    import pandas as pd
    destination.mkdir()
    months = pd.concat([pd.read_csv(evidence / b / 'monthly_metrics.csv') for b in BLOCKS], ignore_index=True)
    months = months[months.regiao == 'dominio_inteiro']
    require(not months.duplicated(['modelo', 'mes_alvo']).any(), 'Duplicate months in paired comparison.')
    for filename, key in [('blocks', 'bloco'), ('years', 'ano')]:
        values = months.groupby([key, 'modelo'], as_index=False)[['n', 'sse', 'soma_erro_absoluto', 'soma_erro']].sum()
        values['rmse'] = np.sqrt(values.sse / values.n)
        values['mae'] = values.soma_erro_absoluto / values.n
        values['bias'] = values.soma_erro / values.n
        baseline = values[values.modelo == 'hybrid'].set_index(key)
        values['delta_rmse_vs_hybrid'] = values.rmse - values[key].map(baseline.rmse)
        values['_period'] = values[key].map({b: i for i, b in enumerate(BLOCKS)}) if key == 'bloco' else values[key]
        values['_model'] = values.modelo.map({'hybrid': 0, 'unet': 1, 'climatology': 2})
        values = values.sort_values(['_period', '_model'])
        values.rename(columns={'modelo': 'model', 'bloco': 'block', 'ano': 'year'})[[
            'block' if key == 'bloco' else 'year', 'model', 'n', 'rmse', 'mae', 'bias', 'delta_rmse_vs_hybrid'
        ]].to_csv(destination / (filename + '.csv'), index=False, lineterminator='\n')


def write_manifest(run):
    files = {p.relative_to(run).as_posix(): dict(sha256=sha256(p), bytes=p.stat().st_size)
             for p in sorted(run.rglob('*')) if p.is_file() and p != run / 'manifest.json'}
    write_json(run / 'manifest.json', dict(schema='rainfall_run_manifest_v1', files=files))


def verify_run(run):
    run = Path(run).resolve()
    require((run / 'state.json').is_file(), f'No workflow state in {run}.')
    require(read_json(run / 'state.json').get('status') == 'completed', 'This run is incomplete or failed. Inspect state.json and execution.log; it is not a valid comparison.')
    require((run / 'manifest.json').is_file(), 'Completed run has no manifest.json.')
    manifest = read_json(run / 'manifest.json')
    require(manifest.get('schema') == 'rainfall_run_manifest_v1' and isinstance(manifest.get('files'), dict), 'Invalid run manifest.')
    expected = manifest['files']
    require({'state.json', 'config.json', 'identity.json', 'report.html', 'diagnostics/summary.json'} <= set(expected), 'Run manifest omits required artifacts.')
    for name, item in expected.items():
        p = (run / name).resolve()
        require(p.is_relative_to(run) and p != run and name != 'manifest.json', 'Manifest path leaves the run or references itself.')
        verify_hash(p, item['sha256'])
        require(p.stat().st_size == item['bytes'], f'Size differs: {name}.')
    actual = {p.relative_to(run).as_posix() for p in run.rglob('*') if p.is_file() and p != run / 'manifest.json'}
    require(actual == set(expected), 'Files were added or removed after completion. Keep extra notes outside the frozen run directory.')
    return dict(status='verified', run_id=read_json(run / 'identity.json')['run_id'], files=len(expected))


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log
    def write(self, text):
        self.stream.write(text); self.log.write(text)
        return len(text)
    def flush(self):
        self.stream.flush(); self.log.flush()


def run(config_path):
    config = load(config_path)
    audit = preflight(config)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run_id = f'{config["name"]}-{timestamp}-{audit["fingerprint"][:10]}'
    folder = Path(config['output_root']) / run_id
    folder.mkdir(parents=True, exist_ok=False)
    record = dict(**audit, run_id=run_id, started_at=utcnow())
    stage = 'snapshot'
    write_json(folder / 'config.json', config)
    write_json(folder / 'identity.json', record)
    write_json(folder / 'state.json', dict(status='running', stage=stage, started_at=record['started_at']))
    try:
        # Code and configuration are captured before any fitting takes place.
        for name, expected in audit['code'].items():
            target = folder / 'code' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
            verify_hash(target, expected)
        with (folder / 'execution.log').open('w', encoding='utf-8', newline='\n') as log, redirect_stdout(Tee(sys.stdout, log)):
            print('Run:', run_id, '| mode:', config['mode'], '| development years 2007–2020', flush=True)
            if config['mode'] == 'train':
                stage = 'training'
                write_json(folder / 'state.json', dict(status='running', stage=stage, started_at=record['started_at']))
                from research.spatial_unet.run_experiment import execute as fit
                fit(folder / 'training', blocks=BLOCKS, **config['inputs'], device=config['device'])
                source = dict(predictions=str(folder / 'training'), evidence=str(folder / 'training'),
                              observations=str(Path(config['inputs']['official']) / 'treino_tp.nc'))
                record['source_artifacts'], record['source_signature_sha256'] = inspect_saved(source)
            else:
                source = config['inputs']
            stage = 'diagnostics'
            write_json(folder / 'state.json', dict(status='running', stage=stage, started_at=record['started_at']))
            copy_evidence(source['evidence'], folder / 'source_evidence')
            from research.diagnostics.analyze_errors import execute as diagnose
            summary = diagnose(source['predictions'], source['observations'], folder / 'diagnostics', evidence=folder / 'source_evidence')
            record['models_fitted_in_this_run'] = config['mode'] == 'train'
            record['evaluation_summary'] = summary
            write_json(folder / 'identity.json', record)
            stage = 'report'
            write_json(folder / 'state.json', dict(status='running', stage=stage, started_at=record['started_at']))
            comparisons(folder / 'source_evidence', folder / 'comparison')
            from .report import render
            render(folder, record)
            print('Report:', folder / 'report.html', flush=True)
        for name, expected in audit['code'].items():
            verify_hash(ROOT / name, expected)
        if config['mode'] == 'saved':
            current, _ = inspect_saved(source)
            require(current == audit['inputs'], 'Inputs changed during evaluation. This run cannot be marked complete.')
        write_json(folder / 'state.json', dict(status='completed', started_at=record['started_at'], finished_at=utcnow(),
                                             model_promoted=False, forecast_issued=False))
        write_manifest(folder)
        verify_run(folder)
    except (Exception, KeyboardInterrupt) as error:
        write_json(folder / 'state.json', dict(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', stage=stage, started_at=record['started_at'], failed_at=utcnow(),
                                             error_type=type(error).__name__, error=str(error), model_promoted=False, forecast_issued=False))
        raise WorkflowError(f'Run failed during {stage}: {error}\nPartial outputs preserved in {folder}. Fix the cause and start a new run; no model was promoted.') from error
    return folder


def list_runs(root):
    root = Path(root)
    require(root.is_dir(), f'Output directory not found: {root}. Complete a run first or correct --root.')
    rows = []
    for path in sorted(root.glob('*/state.json')):
        state = read_json(path)
        record = read_json(path.parent / 'identity.json')
        summary = record.get('evaluation_summary', {})
        rows.append(dict(run_id=record['run_id'], status=state['status'], mode=record['mode'],
                         hybrid_rmse=summary.get('hybrid_rmse'), unet_rmse=summary.get('unet_rmse')))
    return rows
