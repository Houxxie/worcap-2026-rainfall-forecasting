"""Build self-contained research notebooks from the reviewed source modules."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    compile(text, '<notebook-cell>', 'exec')
    return {"cell_type": "code", "execution_count": None, "outputs": [], "metadata": {}, "source": text.splitlines(keepends=True)}


def build(folder, name, files, following):
    package = ROOT / 'research' / folder
    bundle = {n: (package/n).read_text(encoding='utf-8') for n in files}
    hashes = {n: hashlib.sha256(v.encode('utf-8')).hexdigest() for n,v in bundle.items()}
    bootstrap = '''from pathlib import Path
import hashlib, json, sys, runpy, shutil
BASE = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path.cwd()/'outputs'
'''+f"RAIZ = BASE/'rainfall_{folder}'\n"+'''CODIGO = RAIZ/'code'
CODIGO.mkdir(parents=True, exist_ok=True)
'''+f'BUNDLE = {bundle!r}\nBUNDLE_SHA256 = {hashes!r}\n'+'''for name, content in BUNDLE.items():
    assert hashlib.sha256(content.encode('utf-8')).hexdigest() == BUNDLE_SHA256[name]
    path = CODIGO/name
    if path.exists() and path.read_text(encoding='utf-8') != content:
        raise RuntimeError('This directory contains a different source version. Preserve it and choose a new output directory.')
    if not path.exists():
        path.write_text(content, encoding='utf-8', newline='\n')
sys.path.insert(0, str(CODIGO))
print('Source bundle verified:', CODIGO)
'''.replace("newline='\n'", "newline='\\n'")
    cells = [markdown(bundle['README.md']), code(bootstrap)]+following
    for i,c in enumerate(cells):c['id']=f'{folder}-{i:02d}'
    nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.12')),nbformat=4,nbformat_minor=5)
    (package/name).write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8',newline='\n')
    print('Built', (package/name).relative_to(ROOT))


def main():
    build('lagged_sources','lagged_sources.ipynb',
          ['README.md','library.py','run_comparison.py','test_temporal_contract.py','verify_results.py',
           'protocol.json','official_hashes.json','source_metadata.json','ocean_indices.csv','requirements.txt'], [
        markdown('## 1. Check the temporal contract\nSynthetic tests verify month offsets and invariance to future inputs. They do not train on real rainfall.'),
        code("runpy.run_path(str(CODIGO/'test_temporal_contract.py'), run_name='__main__')\n"),
        markdown('## 2. Run the paired comparison\nAttach the official training data, prepared SEAS5/CFSv2 and the recorded ERSSTv5 snapshot before running this cell. Seven chronological blocks are fitted. The configuration is fixed; this is not a new independent holdout.'),
        code("runpy.run_path(str(CODIGO/'run_comparison.py'), run_name='__main__')\n"),
        markdown('## 3. Verify and inspect the reports\nGlobal errors are recomputed from monthly sums. Source-lag calendars and archived invariance checks are also verified.'),
        code("checks = runpy.run_path(str(CODIGO/'verify_results.py'))\nchecks['conferir'](RAIZ)\nfrom IPython.display import display, Markdown, Image, FileLink\ndisplay(Markdown((RAIZ/'resultado.md').read_text(encoding='utf-8')))\ndisplay(Image(filename=str(RAIZ/'comparacao.png')))\ndisplay(FileLink(str(RAIZ/'relatorios.zip')))\n")])
    restore = '''from registry import Registro
REGISTRO = RAIZ/'registry'
# Resume only a verified event chain and its content-addressed objects.
if not (REGISTRO/'eventos').exists() and Path('/kaggle/input').is_dir():
    previous = [p.parent for p in Path('/kaggle/input').rglob('resumo_registro.json')
                if (p.parent/'eventos').is_dir() and (p.parent/'objetos').is_dir()]
    if len(previous) > 1:
        raise RuntimeError('Attach exactly one previous registry to resume.')
    if previous:
        Registro(previous[0]).eventos(conferir_objetos=True)
        REGISTRO.mkdir(parents=True, exist_ok=True)
        for folder in ['eventos','objetos']:
            shutil.copytree(previous[0]/folder, REGISTRO/folder)
        print('Previous registry verified and restored. Receipt times are unchanged.')
print('Registry prepared. No forecast has been issued.')
'''
    build('prospective','prospective_registry.ipynb',
          ['README.md','registry.py','data_contracts.py','collect_sources.py','prepare_cfsv2.py',
           'cfsv2_adapter.py','evaluate.py','test_registry.py','plan.json','requirements.txt'], [
        markdown('## 1. Restore the previous registry\nAttach the last saved output to preserve the chain across Kaggle sessions. Never replace receipt timestamps with earlier dates.'),code(restore),
        markdown('## 2. Run isolated tests\nSynthetic inputs are created in temporary directories. They never enter the real registry.'),
        code("runpy.run_path(str(CODIGO/'test_registry.py'),run_name='__main__')\n"),
        markdown('## 3. Collect public sources and audit CFSv2\nEnable Internet. These downloads require no token. Responses are recorded at the actual receipt time. Missing ensemble members block preparation; the code does not substitute another month or product. Review the target month before running.'),
        code("from collect_sources import executar\nfrom prepare_cfsv2 import preparar\nALVO = '2026-10'\nESTADO = executar(REGISTRO, CODIGO/'plan.json', ALVO, incluir_cfsv2=True)\nCFS_RECEIPT = preparar(REGISTRO, ALVO)\n"),
        markdown('## 4. Inspect readiness and preserve outputs\nThis notebook records arrivals and validates readiness. Fitting and emission are implemented in the separate baseline_inference.ipynb workflow. No forecast is emitted by this source-collection notebook.'),
        code("from evaluate import painel\nfrom IPython.display import display, Markdown, FileLink\nr = Registro(REGISTRO)\nr.exportar_resumo(REGISTRO/'resumo_registro.json')\nstate = r.prontidao(ALVO)\n(REGISTRO/'prontidao.json').write_text(json.dumps(state,indent=2,ensure_ascii=False),encoding='utf-8')\n(REGISTRO/'painel.json').write_text(json.dumps(painel(r),indent=2,ensure_ascii=False),encoding='utf-8')\nprint(json.dumps(state,indent=2,ensure_ascii=False))\ndisplay(Markdown('**Save the version with outputs.** The registry records when each input arrived. This run does not fit a model or issue a forecast.'))\ndisplay(FileLink(str(REGISTRO/'resumo_registro.json')))\n")])


if __name__ == '__main__':
    main()
