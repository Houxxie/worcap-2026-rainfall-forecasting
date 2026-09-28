# Executar após a comparação completa; não treina nem altera modelos.
from zipfile import ZipFile, ZIP_DEFLATED
from IPython.display import FileLink, display
import os

exigir(RESULTADO_PESQUISA_SST['status'] == 'comparacao_completa', 'Comparação incompleta.')
_pasta_relatorios = Path(RESULTADO_PESQUISA_SST['saida'])
_arquivo_relatorios = _pasta_relatorios.parent / 'nimbus_sst_relatorios.zip'
_inventario_relatorios = []
with ZipFile(_arquivo_relatorios, 'w', ZIP_DEFLATED) as _zip:
    for _p in sorted(_pasta_relatorios.rglob('*')):
        if _p.is_file() and _p.suffix in {'.json', '.csv', '.png', '.npz'}:
            _rel = str(_p.relative_to(_pasta_relatorios))
            _zip.write(_p, _rel)
            _inventario_relatorios.append(dict(arquivo=_rel, sha256=sha256(_p), bytes=_p.stat().st_size))
    _zip.writestr('inventario_download.json', json.dumps(_inventario_relatorios, indent=2))

print('RESUMO DA COMPARAÇÃO')
print(json.dumps({k:v for k,v in RESULTADO_PESQUISA_SST.items() if k not in {'protocolo','auditorias'}}, ensure_ascii=False, indent=2))
print('Relatórios, coeficientes PCA e auditorias:', _arquivo_relatorios)
print('Modelos e previsões completas continuam na pasta de saída salva no Kaggle.')
display(FileLink(os.path.relpath(_arquivo_relatorios, Path.cwd())))
