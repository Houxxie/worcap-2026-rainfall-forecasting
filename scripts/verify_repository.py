"""Check the curated repository without downloading data or fitting models."""
from pathlib import Path
from urllib.parse import unquote, urlsplit
import ast
import hashlib
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def local_links(path, text, listed):
    # Markdown links in prose; code blocks may contain illustrative paths.
    prose = re.sub(r'```.*?```', '', text, flags=re.S)
    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', prose):
        target = target.strip().split(' "',1)[0].strip('<>')
        if not target or target.startswith('#') or urlsplit(target).scheme:
            continue
        target = unquote(target.split('#',1)[0])
        resolved = (path.parent/target).resolve()
        require(resolved.is_relative_to(ROOT), f'Link leaves repository: {path.name}: {target}')
        relative = resolved.relative_to(ROOT).as_posix()
        require(relative in listed or any(n.startswith(relative.rstrip('/')+'/') for n in listed),
                f'Broken link: {path.relative_to(ROOT)} → {target}')


def main():
    manifest = json.loads((ROOT/'repository_manifest.json').read_text(encoding='utf-8'))
    if '--refresh-inventory' in sys.argv:
        names = subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode('utf-8').split('\0')
        names = sorted(n for n in names if n and n!='repository_manifest.json')
        manifest['files'] = {n:dict(sha256=hashlib.sha256((ROOT/n).read_bytes()).hexdigest(),bytes=(ROOT/n).stat().st_size) for n in names}
        (ROOT/'repository_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    files = manifest['files']
    listed = set(files) | {'repository_manifest.json'}
    count = {'files':0, 'python_files':0, 'notebooks':0, 'code_cells':0}
    for name,item in files.items():
        path = ROOT/name
        require(path.is_file(), f'Missing file: {name}')
        data = path.read_bytes()
        require(len(data)==item['bytes'], f'Size mismatch: {name}')
        require(hashlib.sha256(data).hexdigest()==item['sha256'], f'Hash mismatch: {name}')
        count['files']+=1
        if path.suffix=='.py':
            compile(data,name,'exec');count['python_files']+=1
        elif path.suffix=='.ipynb':
            nb=json.loads(data)
            require(nb['nbformat']==4, f'Unexpected notebook format: {name}')
            for i,c in enumerate(nb['cells']):
                source=''.join(c['source'])
                if c['cell_type']=='code':
                    compile(source,f'{name}:cell-{i}','exec')
                    require(not c.get('outputs') and c.get('execution_count') is None, f'Unreviewed output: {name}:{i}')
                    count['code_cells']+=1
                else:
                    require(not any(ord(s)<32 and s not in '\n\t\r' for s in source), f'Invalid Markdown control character: {name}:{i}')
                    local_links(path,source,listed)
            count['notebooks']+=1
        elif path.suffix=='.md':local_links(path,data.decode('utf-8'),listed)

    nb=json.loads((ROOT/'competition/hybrid_forecast.ipynb').read_text(encoding='utf-8'))
    notebook_tree=ast.parse('\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code'))
    script_tree=ast.parse((ROOT/'competition/hybrid_forecast.py').read_text(encoding='utf-8'))
    script_tree.body=script_tree.body[1:]  # The script has one leading module docstring.
    require(ast.dump(script_tree)==ast.dump(notebook_tree),'Competition script and notebook disagree.')
    for folder,filename in [('lagged_sources','lagged_sources.ipynb'),('prospective','prospective_registry.ipynb')]:
        path=ROOT/'research'/folder/filename
        nb=json.loads(path.read_text(encoding='utf-8'))
        tree=ast.parse(''.join(nb['cells'][1]['source']))
        bundle=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BUNDLE' for t in n.targets))
        for name,content in bundle.items():
            require((path.parent/name).read_text(encoding='utf-8')==content,f'Outdated notebook bundle: {folder}/{name}')
    for p in [ROOT/'README.md',ROOT/'competition/README.md']:
        text=p.read_text(encoding='utf-8')
        require('1.65354' not in text and '1.65183' not in text, 'Legacy leaderboard score in presentation.')
    for name in ['research/spatial_unet/spatial_unet.ipynb', 'research/prospective/baseline_inference.ipynb', 'research/workflow/experiment_workflow.ipynb', 'research/temporal_extension/temporal_extension.ipynb', 'research/bias_calibration/unet_bias_calibration.ipynb']:
        nb = json.loads((ROOT/name).read_text(encoding='utf-8'))
        tree = ast.parse(''.join(nb['cells'][1]['source']))
        bundle = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BUNDLE' for t in n.targets))
        for path, content in bundle.items():
            require((ROOT/path).read_text(encoding='utf-8') == content, f'Outdated standalone bundle: {name}: {path}')
    print(json.dumps(dict(**count, hashes=True, syntax=True, local_links=True,
        notebook_script_consistency=True, research_bundles=True, real_data_training_performed=False),indent=2))


if __name__=='__main__':
    try:main()
    except (ValueError, OSError, SyntaxError, StopIteration) as error:
        print('Repository check failed:',error,file=sys.stderr)
        raise SystemExit(1)
