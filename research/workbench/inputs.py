"""Locate artifacts and report all missing inputs before a costly operation."""
from pathlib import Path
from research.workflow.config import BLOCKS, DEFAULT_EVIDENCE, require, read_json
from research.workflow.runner import sha256


def discover(config):
    paths = dict(config['inputs'])
    roots = [Path(p) for p in config['search_roots'] if Path(p).is_dir()]
    def find(filename):
        return sorted({p.resolve() for root in roots for p in root.rglob(filename) if p.is_file()})
    errors = []
    checks = []
    def choose(key, choices, explanation, file=False, digest=None):
        if key in paths:
            choices = [Path(paths[key])]
        valid = [p for p in choices if (p.is_file() if file else p.is_dir())]
        if digest:
            valid = [p for p in valid if sha256(p) == digest]
        if len(valid) == 1:
            paths[key] = str(valid[0])
            checks.append(dict(input=key, status='found', path=str(valid[0]), message=explanation))
        else:
            message = (f'{len(valid)} matching artifacts. ' + explanation +
                       f' Set inputs.{key} explicitly if several copies are attached.')
            errors.append(key + ': ' + message)
            checks.append(dict(input=key, status='missing' if not valid else 'ambiguous', path='', message=message,
                               candidates=[str(p) for p in choices[:12]]))
    task = config['task']
    if task == 'prepare':
        registries = {p.parent.parent for p in find('*.json') if p.parent.name == 'eventos' and (p.parent.parent/'objetos').is_dir()}
        choose('registry', sorted(registries), 'Attach the complete latest registry, including eventos/ and objetos/.')
    elif task == 'review':
        candidates = find('sst_unet_blend_reports.zip') + find('sst_temporal_extension_reports.zip')
        candidates += [p.parent for p in find('complete.json') if (p.parent/'global_metrics.csv').is_file()]
        candidates += [p.parent.parent for p in find('complete.json') if p.parent.name == 'evaluation'
                       and (p.parent.parent/'frozen.json').is_file()]
        candidates += [p.parent for p in find('manifest.json') if (p.parent/'state.json').is_file() and (p.parent/'identity.json').is_file()]
        # A review can be a file or directory, unlike the other input selectors.
        if 'results' not in paths and len(candidates) == 1:
            paths['results'] = str(candidates[0])
        choose('results', candidates, 'Select one completed result folder or report ZIP. Full maps are optional for review.',
               file=bool(paths.get('results', '').lower().endswith('.zip')))
    elif task == 'train':
        for key, filename in [('official', 'treino_tp.nc'), ('seas5', 'seas5_51_manifesto.json'), ('cfsv2', 'cfsv2_manifesto.json')]:
            choose(key, sorted({p.parent for p in find(filename)}), f'Attach the data folder containing {filename}.')
    elif config['experiment'] == 'sst_unet_blend_v1':
        from research.sst_unet_blend.experiment import protocol
        for key, spec in protocol()['inputs'].items():
            choose(key, find(spec['filename']), f'Attach the correct saved {key} file; a reports ZIP has no maps. Expected SHA-256 {spec["sha256"]}.',
                   file=True, digest=spec['sha256'])
    else:
        choose('observations', find('treino_tp.nc'), 'Attach the original competition rainfall file treino_tp.nc.', file=True)
        candidates = sorted({p.parent.parent for p in find('predictions.nc') if p.parent.name in BLOCKS
                             and all((p.parent.parent/b/'predictions.nc').is_file() for b in BLOCKS)})
        choose('predictions', candidates, 'Attach the complete seven-block prediction output (H1 through C).')
        if 'evidence' not in paths and 'predictions' in paths:
            source = Path(paths['predictions'])
            paths['evidence'] = str(source if (source/'signature.json').is_file() else DEFAULT_EVIDENCE)
        choose('evidence', [], 'Use matching source reports; archived evidence is available for the original run.')
    output = Path(config['output_root'])
    for key, value in paths.items():
        p = Path(value)
        boundary = p.parent if p.is_file() else p
        if output.is_relative_to(boundary):
            errors.append(f'output_root is inside inputs.{key}. Choose a separate destination.')
    return dict(ready=not errors, inputs=paths, checks=checks, errors=errors)


def legacy_config(config, inputs, output_root):
    from research.workflow.config import normalize
    raw = dict(schema='rainfall_workflow_v1', name=config['name'],
               mode='train' if config['task'] == 'train' else 'saved', experiment=config['experiment'],
               inputs=inputs, output_root=str(output_root))
    if config['task'] == 'train':
        raw['device'] = config['device']
    return normalize(raw, Path.cwd())


def registry_snapshot(folder, target):
    """Read an existing registry only; never create a new empty history."""
    from research.prospective.registry import Registro, deslocar, agora, mes
    folder = Path(folder)
    require((folder/'eventos').is_dir() and (folder/'objetos').is_dir(), 'Missing registry folders. Restore the complete latest snapshot.')
    require(any((folder/'eventos').glob('*.json')), 'An empty registry cannot establish source arrivals.')
    registry = Registro(folder)
    events = registry.eventos(conferir_objetos=True)
    plan = registry.protocolo()['dados']['plano']
    status = registry.prontidao(target)
    rows = []
    for source, rule in plan['fontes'].items():
        if not rule['obrigatoria']:
            continue
        expected = deslocar(target, -rule['lag'])
        selected = next((e for e in events if e['id'] == status['fontes'].get(source)), None)
        candidates = [e for e in events if e['tipo'] == 'fonte_recebida' and e['dados']['fonte'] == source
                      and expected in e['dados']['inspecao'].get('meses', [])]
        latest = candidates[-1] if candidates else None
        rows.append(dict(source=source, required_month=expected, status='validated' if selected else 'missing or unvalidated',
                         selected_event=selected['id'] if selected else None,
                         received_at=selected['registrado_em_utc'] if selected else None,
                         latest_receipt=latest['registrado_em_utc'] if latest else None,
                         first_publication='unknown'))
    return dict(checked_at=agora().isoformat(), target=target, deadline_utc=mes(target).isoformat(),
                status=status, sources=rows, events=len(events), tip=events[-1]['id'],
                model_id=status['modelo'], source_check='Recorded bytes only; providers were not contacted.',
                next_action=('A forecast is already registered; do not overwrite it.' if status['previsao_emitida'] else
                             'Use the existing issuance workflow before the deadline.' if status['pronto'] else
                             'Resolve the listed source/model/deadline blockers; do not substitute incomplete inputs.'))
