"""Export and verify an existing registry without changing its event chain."""
from pathlib import Path
import argparse
import json
import tempfile
import zipfile

if __package__:
    from .registry import Registro, exigir, digest, agora
else:
    from registry import Registro, exigir, digest, agora


def backup(registry, output):
    registry, output = Path(registry).resolve(), Path(output).resolve()
    exigir((registry/'eventos').is_dir() and any((registry/'eventos').glob('*.json')),
           'Restore the existing registry first; an empty registry cannot be backed up.')
    exigir(not output.is_relative_to(registry), 'Keep backups outside the live registry.')
    exigir(not output.exists(), 'Never overwrite a previous registry backup.')
    exigir(not output.with_suffix(output.suffix+'.verification.json').exists(), 'Backup verification already exists.')
    r = Registro(registry)
    output.parent.mkdir(parents=True, exist_ok=True)
    with r.transacao(), tempfile.TemporaryDirectory() as tmp:
        events = r.eventos(conferir_objetos=True)
        paths = sorted((registry/'eventos').glob('*.json'))
        objects = {ref['sha256']: ref for e in events for ref in e['dados'].get('objetos', [])}
        paths += [registry/'objetos'/(h+'.bin') for h in sorted(objects)]
        inventory = dict(schema='rainfall_registry_backup_v1', created_at_utc=agora().isoformat(),
                         event_count=len(events), registry_tip=events[-1]['id'], files={},
                         independent_timestamp=False)
        archive = Path(tmp)/'registry.zip'
        summary = Path(tmp)/'resumo_registro.json'
        r.exportar_resumo(summary)
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
            for path in paths + [summary]:
                name = 'registry/' + (path.relative_to(registry).as_posix() if path != summary else path.name)
                data = path.read_bytes()
                inventory['files'][name] = dict(sha256=digest(data), bytes=len(data))
                z.writestr(name, data)
            z.writestr('backup_inventory.json', json.dumps(inventory, indent=2))
        with zipfile.ZipFile(archive) as z:
            exigir(z.testzip() is None, 'Archive CRC failed.')
            for name, spec in inventory['files'].items():
                data = z.read(name)
                exigir(len(data) == spec['bytes'] and digest(data) == spec['sha256'], 'Archive verification failed.')
        exigir(r.eventos(conferir_objetos=True) == events, 'Registry changed during backup.')
        data = archive.read_bytes()
        with output.open('xb') as f:
            f.write(data)
        proof = dict(archive=output.name, bytes=len(data), sha256=digest(data),
                     event_count=len(events), registry_tip=events[-1]['id'],
                     files_verified=len(inventory['files']), referenced_objects_verified=len(objects),
                     registry_unchanged=True)
        with output.with_suffix(output.suffix+'.verification.json').open('x', encoding='utf-8', newline='\n') as f:
            json.dump(proof, f, indent=2)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', required=True)
    parser.add_argument('--output', required=True)
    print(json.dumps(backup(**vars(parser.parse_args())), indent=2))
