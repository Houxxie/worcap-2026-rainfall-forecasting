"""Validate user settings before reading large arrays or fitting models."""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[2]
BLOCKS = ('H1', 'H2', 'H3', 'H4', 'A', 'B', 'C')
DEFAULT_EVIDENCE = ROOT / 'research/spatial_unet/evidence/seven_blocks'
PROTOCOL = ROOT / 'research/spatial_unet/protocol.json'


class WorkflowError(ValueError):
    """An actionable configuration, provenance or execution error."""


def require(ok, message):
    if not bool(ok):
        raise WorkflowError(message)


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'Duplicate JSON key: {key}. Keep one value.')
        result[key] = value
    return result


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=_unique_keys)
    except json.JSONDecodeError as error:
        raise WorkflowError(f'Invalid JSON in {path}, line {error.lineno}: {error.msg}') from error


def resolve_path(value, base, label):
    require(isinstance(value, str) and value.strip(), f'Set {label} to a nonempty file or directory path.')
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def normalize(raw, base):
    require(isinstance(raw, dict), 'Configuration must be a JSON object.')
    require(set(raw) <= {'schema', 'name', 'mode', 'output_root', 'inputs', 'device', 'experiment'},
            'Unknown configuration field. Use saved.example.json or train.example.json; scientific settings live in protocol.json.')
    require(raw.get('schema') == 'rainfall_workflow_v1', 'Set schema to rainfall_workflow_v1.')
    require(isinstance(raw.get('name'), str) and re.fullmatch(r'[a-z][a-z0-9_-]{0,63}', raw['name']),
            'Name must start with a lowercase letter and use up to 64 letters, digits, underscores or hyphens.')
    mode = raw.get('mode')
    require(mode in {'saved', 'train'}, 'Mode must be saved (no fitting) or train (all seven blocks).')
    experiment = raw.get('experiment', 'hybrid_unet_v1')
    require(experiment in {'hybrid_unet_v1', 'fixed_blend_v1'}, 'Experiment must be hybrid_unet_v1 or fixed_blend_v1.')
    require(experiment != 'fixed_blend_v1' or mode == 'saved', 'fixed_blend_v1 evaluates saved maps only; it does not train models.')
    supplied = raw.get('inputs')
    require(isinstance(supplied, dict), 'inputs must be an object containing the dataset paths.')
    keys = {'observations', 'predictions', 'evidence'} if mode == 'saved' else {'official', 'seas5', 'cfsv2'}
    require(set(supplied) <= keys, f'Unexpected inputs for {mode}. Expected: {sorted(keys)}.')
    required = keys - ({'evidence'} if mode == 'saved' else set())
    require(required <= set(supplied), f'Missing inputs: {sorted(required - set(supplied))}.')
    paths = {k: str(resolve_path(supplied[k], base, 'inputs.' + k)) for k in required}
    if mode == 'saved':
        paths['evidence'] = str(resolve_path(supplied['evidence'], base, 'inputs.evidence') if supplied.get('evidence') else DEFAULT_EVIDENCE)
        require('device' not in raw, 'saved mode does not use a GPU; remove device.')
    else:
        require(raw.get('device', 'cuda') in {'cpu', 'cuda'}, 'Device must be cpu or cuda.')
    output = resolve_path(raw.get('output_root'), base, 'output_root')
    # Never create an output inside a source dataset or code bundle.
    for key, value in paths.items():
        parent = Path(value).parent if key == 'observations' else Path(value)
        require(not output.is_relative_to(parent), f'output_root is inside inputs.{key}. Choose a separate output folder.')
    require(not output.is_relative_to(ROOT / 'research') and not output.is_relative_to(ROOT / 'competition'),
            'Use an outputs directory outside the source folders.')
    result = dict(schema=raw['schema'], name=raw['name'], mode=mode, output_root=str(output), inputs=paths)
    result['experiment'] = experiment
    if mode == 'train':
        result['device'] = raw.get('device', 'cuda')
    return result


def load(path):
    path = Path(path).resolve()
    require(path.is_file(), f'Configuration not found: {path}. Copy an example and edit its paths first.')
    return normalize(read_json(path), path.parent)


def discover_saved(root='/kaggle/input'):
    """Locate one complete saved run; never choose silently among duplicates."""
    root = Path(root)
    observations = sorted(root.rglob('treino_tp.nc'))
    runs = sorted({p.parent for p in root.rglob('signature.json')
                   if all((p.parent / b / 'predictions.nc').is_file() for b in BLOCKS)})
    require(len(observations) == 1, f'Found {len(observations)} treino_tp.nc files under {root}. Attach one official dataset or set its path manually.')
    require(len(runs) == 1, f'Found {len(runs)} complete seven-block outputs under {root}. Attach the saved U-Net notebook OUTPUT (including predictions.nc), or set paths manually.')
    return dict(observations=str(observations[0]), predictions=str(runs[0]), evidence=str(runs[0]))
