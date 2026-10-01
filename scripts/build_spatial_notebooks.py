"""Build standalone Kaggle notebooks with a repository-shaped source bundle."""
from pathlib import Path
import hashlib
import json
from build_notebooks import markdown, code

ROOT = Path(__file__).resolve().parents[1]


def build(output, introduction, names, following):
    bundle = {n: (ROOT / n).read_text(encoding='utf-8') for n in sorted(set(names))}
    hashes = {n: hashlib.sha256(v.encode('utf-8')).hexdigest() for n, v in bundle.items()}
    bootstrap = '''from pathlib import Path
import hashlib, json, sys, runpy, shutil
BASE = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path.cwd()/'outputs'
'''+f"RUN_ROOT = BASE/{Path(output).stem!r}\n"+'''CODE_ROOT = RUN_ROOT/'code'
CODE_ROOT.mkdir(parents=True, exist_ok=True)
'''+f'BUNDLE = {bundle!r}\nBUNDLE_SHA256 = {hashes!r}\n'+'''for name, content in BUNDLE.items():
    assert hashlib.sha256(content.encode('utf-8')).hexdigest() == BUNDLE_SHA256[name]
    path = CODE_ROOT/name
    if path.exists() and path.read_text(encoding='utf-8') != content:
        raise RuntimeError('Different source code already exists. Preserve this run and choose a new RUN_ROOT.')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content, encoding='utf-8', newline='\\n')
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT/'research/prospective'))
print('Reviewed source bundle:', CODE_ROOT)
'''
    if Path(output).stem == 'baseline_inference':
        # Install readers before importing xarray, which caches optional engines.
        bootstrap = '''import importlib.util, subprocess, sys
READERS = {'netCDF4': 'netCDF4>=1.6,<2', 'cftime': 'cftime>=1.6,<2',
           'h5netcdf': 'h5netcdf>=1.3,<2', 'cdsapi': 'cdsapi>=0.7,<1',
           'eccodes': 'eccodes>=2.38,<3'}
missing = [requirement for name, requirement in READERS.items() if importlib.util.find_spec(name) is None]
if missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *missing], check=True)
''' + bootstrap
    cells = [markdown((ROOT / introduction).read_text(encoding='utf-8')), code(bootstrap)] + following
    for i, cell in enumerate(cells):
        cell['id'] = f'cell-{i:02d}'
    notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python', version='3.12')), nbformat=4, nbformat_minor=5)
    (ROOT / output).write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    print('Built', output)


def main():
    common = [p.relative_to(ROOT).as_posix() for p in (ROOT / 'research/common').glob('*.py')]
    common += ['research/lagged_sources/library.py', 'research/lagged_sources/official_hashes.json',
               'research/lagged_sources/ocean_indices.csv', 'research/lagged_sources/evidence/metricas_blocos.csv']
    spatial = [p.relative_to(ROOT).as_posix() for p in (ROOT / 'research/spatial_unet').glob('*.py')]
    build('research/spatial_unet/spatial_unet.ipynb', 'research/spatial_unet/README.md',
          common + spatial + ['research/spatial_unet/protocol.json'], [
        markdown('## 1. Environment and isolated checks\nUse a Kaggle GPU for training. These tests use synthetic arrays only. They do not measure rainfall skill or create records in the prospective registry.'),
        code("import torch, unittest\nprint('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())\nsuite = unittest.defaultTestLoader.discover(str(CODE_ROOT/'research/spatial_unet'), pattern='test_spatial.py')\nchecks = unittest.TextTestRunner(verbosity=2).run(suite)\nassert checks.wasSuccessful(), 'Synthetic checks failed; stop before training.'\n"),
        markdown('## 2. Locate and verify the same existing inputs\nLeave a directory as `None` for automatic discovery. If multiple copies are attached, enter the exact folder containing its manifest. There is no download or automatic package replacement.'),
        code("OFFICIAL = None\nSEAS5 = None\nCFSV2 = None\nfrom research.common.inputs import preflight\nPATHS, INPUT_AUDIT = preflight(OFFICIAL, SEAS5, CFSV2)\nprint({k: str(v) for k,v in PATHS.items()})\nprint('All input hashes and baseline package versions verified.')\n"),
        markdown('## 3. Run the prespecified pilot or the complete comparison\nThe default H1 pilot measures feasibility. For the seven-block comparison, set `BLOCKS = ["H1", "H2", "H3", "H4", "A", "B", "C"]`. Do not tune the architecture against one pilot score and then describe the same years as untouched validation. Model fitting prints each epoch; keep the resulting outputs.'),
        code("BLOCKS = ['H1']\nOUTPUT = RUN_ROOT/'results'\nfrom research.spatial_unet.run_experiment import execute\nRESULT = execute(OUTPUT, blocks=BLOCKS, official=PATHS['official'], seas5=PATHS['seas5'], cfsv2=PATHS['cfsv2'])\n"),
        markdown('## 4. Inspect and preserve the results\nA negative RMSE difference favors the U-Net in this development comparison. Also inspect MAE, bias, individual years and regions. No result here changes the existing prospective baseline or emits a neural forecast.'),
        code("import pandas as pd\nfrom research.common.presentation import display_frame\nfrom IPython.display import display, FileLink\nfor name in ['global','blocks','years','regions']:\n    print(name)\n    display(display_frame(pd.read_csv(OUTPUT/(name+'.csv'))))\nprint(json.dumps(RESULT,indent=2))\nREPORT = RUN_ROOT/'spatial_reports.zip'\nimport zipfile\nwith zipfile.ZipFile(REPORT,'w',zipfile.ZIP_DEFLATED) as z:\n    for p in OUTPUT.rglob('*'):\n        if p.is_file() and p.suffix in {'.csv','.json'}:\n            z.write(p,p.relative_to(OUTPUT))\ndisplay(FileLink(str(REPORT.relative_to(BASE))))\nprint('Save the notebook version with all outputs to retain models and prediction maps as well.')\n")])

    prospective = [p.relative_to(ROOT).as_posix() for p in (ROOT / 'research/prospective').glob('*.py')]
    restore = '''from registry import Registro
REGISTRY = RUN_ROOT/'registry'
previous = [p.parent for p in Path('/kaggle/input').rglob('resumo_registro.json')
            if (p.parent/'eventos').is_dir() and (p.parent/'objetos').is_dir()]
if not (REGISTRY/'eventos').exists():
    if len(previous) != 1:
        raise RuntimeError('Attach exactly one complete existing registry. This continuation will not invent a new chain.')
    Registro(previous[0]).eventos(conferir_objetos=True)
    REGISTRY.mkdir(parents=True, exist_ok=True)
    for folder in ['eventos','objetos']:
        shutil.copytree(previous[0]/folder,REGISTRY/folder)
r = Registro(REGISTRY)
r.eventos(conferir_objetos=True)
TARGET = '2026-11'  # October is already frozen in the latest registry.
print(json.dumps(r.prontidao(TARGET),indent=2))
'''
    build('research/prospective/baseline_inference.ipynb', 'research/prospective/BASELINE_RUN.md',
          common + prospective + ['research/prospective/plan.json', 'competition/metadata/NOAA/indices_noaa.csv'], [
        markdown('## 1. Restore the actual event chain\nAttach the previous registry output with its events and objects. An absent or ambiguous chain stops this continuation. Review the target month; a passed deadline cannot be backdated.'), code(restore),
        markdown('## 2. Verify inputs and fit the fixed final baseline\nThe fit uses 1993–2022 with research lags. An already registered model is reused. No competition model is substituted.'),
        code("OFFICIAL = None\nSEAS5 = None\nCFSV2 = None\nfrom fit_baseline import execute as fit_final\nif not any(e['tipo']=='modelo_congelado' for e in r.eventos()):\n    MODEL = fit_final(RUN_ROOT/'model', REGISTRY, OFFICIAL, SEAS5, CFSV2)\nelse:\n    print('The registry already contains its immutable model package.')\n"),
        markdown('## 3. Collect and normalize sources\nPublic inputs need Internet. CFSv2 uses the original IRI catalog recipe, requiring all expected members and exact overlap with the named product. It does not substitute CCSR. To request ERA5/SEAS5, configure your existing CDS credential privately and set `COLLECT_CDS=True`. Future sources may not be available yet; missing data continue to block issuance.'),
        code("from collect_sources import executar\nfrom prepare_cfsv2_recipe import prepare\nCOLLECT_CDS = False\nassert not any(e['tipo']=='previsao_congelada' and e['dados']['mes_alvo']==TARGET for e in r.eventos()), 'This month already has a frozen forecast.'\nexecutar(REGISTRY, CODE_ROOT/'research/prospective/plan.json', TARGET, incluir_cfsv2=False)\nprepare(REGISTRY, TARGET)\nif COLLECT_CDS:\n    from collect_cds import execute as acquire_cds\n    acquire_cds(REGISTRY, TARGET)\nelse:\n    print('CDS acquisition disabled. Audited ERA5/SEAS5 receipts are still required.')\n"),
        markdown('## 4. Freeze only a complete, timely forecast\nThe file is generated from exactly the model/source bytes selected by the registry. Incomplete inputs or an expired deadline block emission, and an existing issued forecast is not replaced.'),
        code("from issue_forecast import execute as issue\nstate = r.prontidao(TARGET)\nprint(json.dumps(state,indent=2))\nalready_issued = any(e['tipo']=='previsao_congelada' and e['dados']['mes_alvo']==TARGET for e in r.eventos())\nif state['pronto'] and not already_issued:\n    FORECAST = issue(REGISTRY,TARGET,RUN_ROOT/('forecast_'+TARGET+'.nc'))\nelse:\n    print('No new forecast issued:', 'already frozen' if already_issued else state['faltantes'])\nr.exportar_resumo(REGISTRY/'resumo_registro.json')\nprint('Save all registry events and objects with the notebook output.')\n")])


if __name__ == '__main__':
    main()
