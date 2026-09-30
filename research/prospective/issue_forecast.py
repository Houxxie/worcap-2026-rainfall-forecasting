"""Materialize verified registry objects, infer once and freeze before the deadline."""
from pathlib import Path
import argparse
import tempfile
from registry import Registro, exigir, digest
from infer_baseline import predict


def execute(registry, target, output):
    r = Registro(registry)
    status = r.prontidao(target)
    exigir(status['pronto'], 'Forecast blocked: ' + '; '.join(status['faltantes']))
    model = r.obter(status['modelo'], 'modelo_congelado')
    source = Path(__file__).with_name('infer_baseline.py').read_bytes()
    exigir(model['dados']['arquivos']['inferir.py']['sha256'] == digest(source), 'Frozen inference code differs; use its reviewed version.')
    output = Path(output)
    exigir(not output.exists(), 'Never overwrite an existing forecast file.')
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        package = temp / 'model'
        package.mkdir()
        for name, ref in model['dados']['arquivos'].items():
            exigir(Path(name).name == name, 'Invalid model filename.')
            (package / name).write_bytes(r.ler_objeto(ref))
        inputs = {}
        for source_name, ident in status['fontes'].items():
            event = r.obter(ident, 'fonte_recebida')
            path = temp / (source_name + '.input')
            path.write_bytes(r.ler_objeto(event['dados']['objeto']))
            inputs[source_name] = path
        result = predict(package, inputs, target)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_netcdf(temp / 'forecast.nc', engine='h5netcdf')
        content = (temp / 'forecast.nc').read_bytes()
        with output.open('xb') as f:
            f.write(content)
    inference = {k: status[k] for k in ['modelo', 'fontes', 'protocolo']}
    inference.update(previsao_sha256=digest(content), inference_code_sha256=digest(source))
    event = r.congelar(target, output, inference)
    r.exportar_resumo(Path(registry) / 'resumo_registro.json')
    print('Forecast frozen:', event['id'])
    return event


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--registry', required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--output', required=True)
    execute(**vars(p.parse_args()))
