# Preserva os novos dados, modelos e resultados; não executa novos ajustes.
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from IPython.display import FileLink, display
import hashlib, json

_base_backup=Path('/kaggle/working/nimbus_avaliacao_2025')
if (_base_backup/'resumo_2025.json').is_file():
    _arquivo_backup=_base_backup.parent/'nimbus_avaliacao_2025_preservada.zip'
    _inventario_backup=[]
    with ZipFile(_arquivo_backup,'w',ZIP_DEFLATED,compresslevel=4) as _z:
        for _p in sorted(_base_backup.rglob('*')):
            if not _p.is_file() or _p.suffix in {'.zip','.pyc'} or '.partial' in _p.name:
                continue
            _rel=_p.relative_to(_base_backup).as_posix()
            _z.write(_p,_rel)
            _h=hashlib.sha256()
            with _p.open('rb') as _f:
                for _b in iter(lambda:_f.read(1024*1024),b''):_h.update(_b)
            _inventario_backup.append(dict(arquivo=_rel,bytes=_p.stat().st_size,sha256=_h.hexdigest()))
        _z.writestr('inventario_preservacao.json',json.dumps(_inventario_backup,indent=2))
    print('Preservados:',len(_inventario_backup),'arquivos; ZIP:',round(_arquivo_backup.stat().st_size/1024**2,1),'MiB.')
    display(FileLink('nimbus_avaliacao_2025_preservada.zip'))
else:
    print('Preservação final aguardando relatório concluído; saídas parciais mantidas.')
