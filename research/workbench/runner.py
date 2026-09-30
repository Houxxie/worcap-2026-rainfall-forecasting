"""Preserve existing protocols, reuse verified work, and separate reports from fitting."""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path
import importlib.metadata
import json
import os
import shutil
import sys
import zipfile
from .config import ROOT, load, require, read_json, WorkflowError
from .inputs import discover, legacy_config, registry_snapshot
from .results import inspect_result
from research.workflow.runner import sha256, identity_hash, write_json, Tee


def now():
    return datetime.now(timezone.utc).isoformat()


def files_at(path):
    path = Path(path)
    if path.is_file():
        return {'file': sha256(path)}
    return {p.relative_to(path).as_posix(): sha256(p) for p in sorted(path.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}


def sources():
    from research.workflow.runner import code_files
    from research.sst_unet_blend.experiment import source_files
    files = set(code_files()) | {ROOT/n for n in source_files()}
    files |= set((ROOT/'research/workbench').glob('*.py'))
    files.add(ROOT/'research/prospective/registry.py')
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(files)}


def check(config):
    discovered = discover(config)
    result = dict(**discovered, task=config['task'], checked_at=now(), model_fitting=config['task']=='train',
                  note='Input discovery only; no training, downloads or forecast issuance.')
    if not result['ready']:
        return result
    try:
        inputs = result['inputs']
        if config['task'] == 'prepare':
            audit = registry_snapshot(inputs['registry'], config['target'])
            identities = dict(registry_tip=audit['tip'], target=config['target'], checked_at=audit['checked_at'])
        elif config['task'] == 'review':
            source = inspect_result(inputs['results'])
            audit = dict(title=source['title'], period=source['period'], missing=source['missing'])
            identities = files_at(inputs['results'])
        elif config['experiment'] == 'sst_unet_blend_v1':
            from research.sst_unet_blend.experiment import protocol
            audit = dict(protocol=protocol(), models_to_fit=0)
            identities = {k: sha256(v) for k,v in inputs.items()}
        else:
            from research.workflow.runner import preflight
            audit = preflight(legacy_config(config, inputs, config['output_root']))
            identities = audit['inputs']
        packages = {}
        for name in ['numpy', 'pandas', 'xarray', 'matplotlib', 'h5netcdf']:
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                require(False, f'Missing {name}. Install research/workbench/requirements.txt for evaluation.')
        scientific = dict(task=config['task'], experiment=config['experiment'], target=config['target'],
                          map_month=config['map_month'], inputs=identities, code=sources(), environment=packages,
                          adapter=audit.get('fingerprint'))
        result.update(audit=audit, identity=scientific, fingerprint=identity_hash(scientific),
                      note=('Will fit the unchanged seven-block models. This can take a long time.' if config['task']=='train' else
                            'No model fitting or provider downloads. Existing forecast rules remain unchanged.'))
    except (ValueError, OSError, ImportError, KeyError, TypeError) as error:
        result.update(ready=False, errors=[f'{type(error).__name__}: {error}'])
    return result


def inventory(folder):
    return {p.relative_to(folder).as_posix(): dict(sha256=sha256(p), bytes=p.stat().st_size)
            for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'completion.json' and '__pycache__' not in p.parts}


def verify(folder):
    folder = Path(folder).resolve()
    require((folder/'state.json').is_file(), 'No workbench state found. Use the exact run directory.')
    require(read_json(folder/'state.json')['status'] == 'completed', 'Run incomplete. Use recover to inspect preserved outputs.')
    record = read_json(folder/'completion.json')
    require(record['schema'] == 'rainfall_workbench_completion_v1', 'Unknown completion format.')
    require({'state.json','settings.json','identity.json','report.html','summary.json','payload.json','report_bundle.json'} <= set(record['files']), 'Completion is missing required files.')
    require(inventory(folder) == record['files'], 'Completed output changed. Restore the original output; do not reuse it for fitting.')
    return dict(status='verified', path=str(folder), files=len(record['files']), fingerprint=record['fingerprint'])


def summary_report(folder, config, source, audit):
    from .report import render, forecast_map
    info = dict(run_id=folder.name, task=config['task'], status='completed', figures=[],
                kind='prepare' if config['task']=='prepare' else 'comparison')
    if info['kind'] == 'prepare':
        info.update(title='Monthly forecast preparation', period=config['target'], registry=source,
                    status='already issued' if source['status']['previsao_emitida'] else 'ready for issuance' if source['status']['pronto'] else 'blocked',
                    limitations='Read-only readiness at the recorded time. No source download, fitting or forecast issuance. This check cannot prove when providers first published their data.',
                    provenance=dict(registry_tip=source['tip'], model_id=source['model_id'], sources=source['status']['fontes']))
    else:
        data = inspect_result(source)
        data['table'].to_csv(folder/'metrics.csv', index=False)
        data['years'].to_csv(folder/'years.csv', index=False)
        info.update(title=data['title'], period=data['period'], primary=data['primary'], limitations=data['limitations'],
                    provenance=data['provenance'])
        for i, name in enumerate(data['figures']):
            target = f'figure_{i+1}.png'
            (folder/target).write_bytes(data['artifacts'].read(name))
            info['figures'].append(dict(file=target, title='Saved diagnostic · '+Path(name).stem.replace('_',' '),
                                        caption='Previously calculated diagnostic. The full grid includes ocean cells.'))
        # Preview only verified saved fields. ZIP reports deliberately omit them.
        artifact = data['artifacts']
        map_names = [n for n in ['predictions.nc', 'models/predictions.nc', 'fixed_blend/predictions/C/predictions.nc'] if n in artifact.names]
        if not artifact.zip and map_names:
            month = forecast_map(artifact.source/map_names[0], folder/'forecast_preview.png', config['map_month'])
            info['figures'].insert(0, dict(file='forecast_preview.png', title=f'Monthly rainfall maps · {month}',
                                          caption='Retrospective forecasts, not a newly issued operational forecast. All panels share the same scale.'))
        elif config['map_month']:
            require(False, 'A map month was requested but the result contains no complete prediction file. Attach the full output or clear map_month.')
    info['provenance']['workbench_fingerprint'] = audit['fingerprint']
    write_json(folder/'summary.json', info)
    render(folder, info)
    report_names = ['summary.json','report.html','settings.json','identity.json']
    report_names += [p.name for p in folder.iterdir() if p.is_file() and p.suffix in {'.png','.csv'}]
    write_json(folder/'report_bundle.json', dict(schema='rainfall_workbench_report_v1',
               files={n:dict(sha256=sha256(folder/n), bytes=(folder/n).stat().st_size) for n in sorted(set(report_names))}))


def execute_backend(folder, config, audit):
    task = config['task']
    if task == 'prepare':
        data = registry_snapshot(audit['inputs']['registry'], config['target'])
        require(data['tip'] == audit['audit']['tip'], 'Registry changed during preparation. Run again to obtain current readiness.')
        write_json(folder/'readiness.json', data)
        return data
    if task == 'review':
        return audit['inputs']['results']
    if config['experiment'] == 'sst_unet_blend_v1':
        from research.sst_unet_blend.experiment import execute
        output = folder/'comparison'
        execute(output, **audit['inputs'])
        return str(output)
    from research.workflow.runner import run as legacy_run
    settings = legacy_config(config, audit['inputs'], folder/'engine')
    write_json(folder/'engine_config.json', settings)
    return str(legacy_run(folder/'engine_config.json'))


def run(config_path):
    config = load(config_path)
    audit = check(config)
    require(audit['ready'], 'Preflight stopped:\n'+'\n'.join(audit['errors']))
    output = Path(config['output_root']); output.mkdir(parents=True, exist_ok=True)
    lock = output/('.'+audit['fingerprint']+'.lock')
    try:
        fd = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise WorkflowError(f'A matching job is active or was interrupted. Inspect {lock} and status before removing a stale lock; no duplicate fit was started.') from None
    with os.fdopen(fd, 'w') as f:
        f.write(json.dumps(dict(pid=os.getpid(), started_at=now())))
    folder = None
    try:
        for candidate in sorted(output.glob('*/completion.json')):
            if read_json(candidate).get('fingerprint') == audit['fingerprint']:
                verify(candidate.parent)
                print('Reusing verified completed run:', candidate.parent)
                return candidate.parent
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        folder = output/f'{config["name"]}-{stamp}-{audit["fingerprint"][:8]}'
        folder.mkdir()
        write_json(folder/'settings.json', config)
        write_json(folder/'identity.json', audit)
        write_json(folder/'state.json', dict(status='running', stage='snapshot', started_at=now()))
        for name, digest in audit['identity']['code'].items():
            destination=folder/'code'/name; destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,destination)
            require(sha256(destination)==digest,'Code changed during snapshot.')
        stage='execution'
        write_json(folder/'state.json',dict(status='running',stage=stage))
        with (folder/'execution.log').open('w',encoding='utf-8') as log:
            with redirect_stdout(Tee(sys.stdout,log)),redirect_stderr(Tee(sys.stderr,log)):
                print('Task:',config['task'],'| Experiment:',config['experiment'],'| Run:',folder.name,flush=True)
                source=execute_backend(folder,config,audit)
                write_json(folder/'payload.json',dict(source=source))
                stage='report'
                write_json(folder/'state.json',dict(status='running',stage=stage))
                summary_report(folder,config,source,audit)
                print('Report:',folder/'report.html',flush=True)
        require(sources()==audit['identity']['code'],'Source code changed during execution.')
        if config['task']=='review':
            require(files_at(audit['inputs']['results'])==audit['identity']['inputs'],'Review input changed during execution.')
        write_json(folder/'state.json',dict(status='completed',finished_at=now(),forecast_issued=False,model_promoted=False))
        write_json(folder/'completion.json',dict(schema='rainfall_workbench_completion_v1',fingerprint=audit['fingerprint'],files=inventory(folder)))
        verify(folder)
        return folder
    except (Exception,KeyboardInterrupt) as error:
        if folder is not None:
            write_json(folder/'state.json',dict(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                stage=locals().get('stage','snapshot'),error=str(error),failed_at=now(),forecast_issued=False))
            raise WorkflowError(f'{error}\nOutputs preserved in {folder}. Use: python -m research.workbench recover --run "{folder}"') from error
        raise
    finally:
        lock.unlink()


def recover(folder):
    """Propose a report-only continuation; never silently restart an expensive fit."""
    folder=Path(folder).resolve()
    require((folder/'state.json').is_file(),'Choose a workbench run directory.')
    state=read_json(folder/'state.json')
    if state['status']=='completed':
        return dict(**verify(folder),next_action='Open report.html or export the report ZIP. No rerun needed.')
    candidates=[]
    if (folder/'payload.json').is_file():
        source=read_json(folder/'payload.json')['source']
        if isinstance(source,str): candidates.append(Path(source))
    candidates += [p.parent for p in folder.glob('engine/*/manifest.json')]
    for candidate in candidates:
        try:
            inspect_result(candidate)
        except (ValueError,OSError,KeyError):
            continue
        return dict(status=state['status'],stage=state.get('stage'),reusable=str(candidate),
                    next_action='Use task=review with inputs.results pointing to reusable. This regenerates the report without fitting.',
                    settings=dict(schema='rainfall_workbench_v1',name='recovered-report',task='review',
                                  inputs=dict(results=str(candidate)),output_root=str(folder.parent)))
    # A failed report after training can still leave all completed model blocks.
    for signature in folder.glob('engine/*/training/signature.json'):
        source=signature.parent
        try:
            from research.workflow.runner import inspect_saved
            identity=read_json(folder/'identity.json')
            inputs=dict(observations=str(Path(identity['inputs']['official'])/'treino_tp.nc'),predictions=str(source),evidence=str(source))
            inspect_saved(inputs)
        except (ValueError,OSError,KeyError,StopIteration):
            continue
        return dict(status=state['status'],reusable=str(source),next_action='All seven fitted blocks are intact. Evaluate these saved maps; do not train again.',
                    settings=dict(schema='rainfall_workbench_v1',name='recovered-evaluation',task='evaluate',experiment='hybrid_unet_v1',inputs=inputs,output_root=str(folder.parent)))
    return dict(status=state['status'],stage=state.get('stage'),error=state.get('error'),
                next_action='Read execution.log and correct the failing stage. No complete reusable result was found. Partial files remain preserved; training is never restarted automatically.')


def status(root):
    rows=[]
    for p in sorted(Path(root).glob('*/state.json')):
        try:
            s=read_json(p); c=read_json(p.parent/'settings.json')
            rows.append(dict(run=p.parent.name,task=c['task'],experiment=c['experiment'],status=s['status'],
                             stage=s.get('stage'),report=str(p.parent/'report.html') if s['status']=='completed' else None))
        except (ValueError,OSError,KeyError):
            continue
    return rows


def export(folder):
    folder=Path(folder).resolve(); verify(folder)
    manifest=read_json(folder/'report_bundle.json')
    output=folder.parent/(folder.name+'-reports.zip')
    require(not output.exists(),'Report ZIP already exists: '+str(output))
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED) as z:
        for name in [*manifest['files'],'report_bundle.json']:
            z.write(folder/name,name)
    return output
