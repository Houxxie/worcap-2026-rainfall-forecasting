"""Fit the fixed 1993–2022 reference model with longer input lags and register its complete model package."""
from pathlib import Path
import argparse
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from research.common.inputs import Inputs, LIBRARY, require, write_json, sha256
from research.common.baseline import fit
from registry import Registro


def execute(output, registry, official=None, seas5=None, cfsv2=None):
    r = Registro(registry)
    plan = json.loads(Path(__file__).with_name('plan.json').read_text())
    event = r.iniciar(plan)
    require(not any(e['tipo'] == 'modelo_congelado' for e in r.eventos()), 'The existing model is immutable; no refit is allowed in this series.')
    output = Path(output)
    require(not output.exists(), 'Preserve existing model files; choose a new package directory.')
    output.mkdir(parents=True)
    data = Inputs(official, seas5, cfsv2, output=output, final=True)
    state = fit(data, '2022-12-01', output)
    require(state['audit']['training_start'] == '1993-01-01' and state['audit']['training_months'] == 360, 'Training differs from the frozen plan.')
    shutil.copyfile(LIBRARY, output / 'biblioteca.py')
    shutil.copyfile(Path(__file__).with_name('infer_baseline.py'), output / 'inferir.py')
    write_json(output / 'ambiente.json', data.audit['versions'])
    audit = dict(state['audit'], treino_inicio='1993-01', treino_fim='2022-12', protocolo_id=event['id'],
        lags=dict(atmosphere=4, indices=3, seasonal=1, rainfall_training=4),
        inference_sha256=sha256(output / 'inferir.py'), library_sha256=sha256(output / 'biblioteca.py'),
        fitting_code_sha256=sha256(__file__), model_package_is_competition_model=False,
        historical_vintages_verified=False)
    write_json(output / 'auditoria_treino.json', audit)
    result = r.modelo({name: output / name for name in plan['arquivos_modelo']}, audit)
    r.exportar_resumo(Path(registry) / 'resumo_registro.json')
    print('Reference model with longer input lags package frozen:', result['id'], flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--registry', required=True)
    for name in ['official', 'seas5', 'cfsv2']:
        parser.add_argument('--' + name)
    execute(**vars(parser.parse_args()))
