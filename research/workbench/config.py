"""User choices are separate from frozen scientific settings."""
from pathlib import Path
import re
from research.workflow.config import read_json, require, resolve_path, WorkflowError

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ('hybrid_unet_v1', 'fixed_blend_v1', 'sst_unet_blend_v1')


def normalize(raw, base):
    require(isinstance(raw, dict), 'Settings must be a JSON object.')
    require(set(raw) <= {'schema', 'name', 'task', 'experiment', 'inputs', 'search_roots',
                        'output_root', 'device', 'target', 'map_month'}, 'Unknown setting. Start with a workbench example.')
    require(raw.get('schema') == 'rainfall_workbench_v1', 'Use schema rainfall_workbench_v1.')
    task = raw.get('task')
    require(task in {'evaluate', 'train', 'prepare', 'review'}, 'Choose task: evaluate, train, prepare or review.')
    name = raw.get('name', task)
    require(isinstance(name, str) and re.fullmatch(r'[a-z][a-z0-9_-]{0,63}', name), 'Use a short lowercase name with hyphens.')
    inputs = raw.get('inputs', {})
    require(isinstance(inputs, dict), 'inputs must contain named paths or null for discovery.')
    experiment = raw.get('experiment', 'hybrid_unet_v1')
    if task in {'evaluate', 'train'}:
        require(experiment in EXPERIMENTS, 'Unknown experiment. See the workbench catalogue.')
        require(task != 'train' or experiment == 'hybrid_unet_v1', 'Training supports the existing seven-block hybrid/U-Net protocol only.')
    else:
        require('experiment' not in raw, 'prepare/review do not select or fit an experiment.')
    keys = ({'official', 'seas5', 'cfsv2'} if task == 'train' else
            {'registry'} if task == 'prepare' else {'results'} if task == 'review' else
            {'observations', 'sst_predictions', 'unet_predictions'} if experiment == 'sst_unet_blend_v1' else
            {'observations', 'predictions', 'evidence'})
    require(set(inputs) <= keys, f'Expected input keys: {sorted(keys)}.')
    resolved = {k: str(resolve_path(v, base, k)) for k, v in inputs.items() if v is not None}
    search = raw.get('search_roots', ['/kaggle/input', '/kaggle/working'])
    require(isinstance(search, list) and all(isinstance(v, str) for v in search), 'search_roots must be a list of folders.')
    output = resolve_path(raw.get('output_root', 'outputs/workbench'), base, 'output_root')
    require(not output.is_relative_to(ROOT/'research') and not output.is_relative_to(ROOT/'competition'), 'Keep outputs outside source folders.')
    if task != 'train':
        require('device' not in raw, 'Only task=train uses device. Evaluation and preparation use CPU.')
    target = raw.get('target')
    require(task == 'prepare' or target is None, 'target belongs to task=prepare.')
    if task == 'prepare':
        from research.prospective.registry import mes
        mes(target)
    month = raw.get('map_month')
    if month is not None:
        from research.prospective.registry import mes
        mes(month)
    return dict(schema=raw['schema'], name=name, task=task, experiment=experiment if task in {'evaluate', 'train'} else None,
                inputs=resolved, search_roots=[str(resolve_path(p, base, 'search_roots')) for p in search],
                output_root=str(output), device=raw.get('device', 'cuda') if task == 'train' else None,
                target=target, map_month=month)


def load(path):
    path = Path(path).resolve()
    require(path.is_file(), f'Settings not found: {path}. Copy an example first.')
    return normalize(read_json(path), path.parent)
