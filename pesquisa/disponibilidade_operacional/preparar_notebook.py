"""Gera uma biblioteca autocontida; não executa treino nem modifica a entrega."""
from pathlib import Path
import ast
import base64
import gzip
import hashlib
import io
import json
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import nbformat

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
source = ROOT/'entregas/banca_8G/WorCAP_modelo_hibrido_reproducao.py'
text = source.read_text(encoding='utf-8-sig')
tree = ast.parse(text)
functions = {n.name: ast.get_source_segment(text, n) for n in tree.body if isinstance(n, ast.FunctionDef)}
assignments = {t.id: ast.get_source_segment(text, n) for n in tree.body if isinstance(n, ast.Assign)
               for t in n.targets if isinstance(t, ast.Name)}
constants = ['PONTOS_POR_MES', 'ANOS_TREINO', 'ARVORES', 'SEMENTE', 'CORTE_FINAL',
    'FIM_DESENVOLVIMENTO', 'PARAMETROS', 'BLOCOS', 'VARIAVEIS', 'INDICES_NOMES',
    'FONTES_INDICES', 'INDICES_CSV_GZIP_BASE64', 'INDICES_CSV_SHA256',
    'SCHEMA_CFSV2', 'POLITICA_CFSV2', 'FONTE_CFSV2', 'FEATURE_SEAS5', 'FEATURE_SEAS5_ANOM',
    'FEATURE_CFSV2', 'FEATURE_CFSV2_ANOM', 'NOMES_MODELOS', 'FASES_CFSV2', 'MOS_ALPHA', 'MOS_FRACAO']
names = ['exigir', 'sha256', 'salvar_json', 'localizar_dados', 'conferir_campo', 'ajustar_clima_atmosfera',
    'reconstruir_chuva', 'auditar_chuva_desenv', 'carregar_atmosfera', 'carregar_indices_incorporados',
    'conferir_indices', 'auditar_dataset', 'localizar_seas5', 'carregar_seas5', 'valores_seas5',
    'amostrar_m5_documentado', 'valores_anomalia_seas5', 'amostrar_arvores_seas5',
    'validar_manifesto_cfsv2', 'localizar_cfsv2', 'auditar_cfsv2', 'carregar_cfsv2',
    'ajustar_clima_previsao', 'ajustar_clima_seas5', 'valores_cfsv2', 'valores_anomalia_cfsv2',
    'amostrar_arvores_multissistema', 'prever_par', 'mos_fit', 'mos_predict']
imports = '''from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter
import base64, gzip, io, json, hashlib, gc, platform, importlib.metadata
import numpy as np
import pandas as pd
import xarray as xr
import lightgbm as lgb
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from IPython.display import display, Markdown, Image, FileLink
DATASET, SYSTEM, SCHEMA = 'seasonal-monthly-single-levels', '51', 'worcap_seas5_v1'
PASTA_SEAS5_MANUAL = PASTA_CFSV2_MANUAL = None
'''
library = imports + '\n\n' + '\n'.join(assignments[n] for n in constants)
library += '\n\n' + '\n\n'.join(functions[n] for n in names)
# Alteração explícita somente na função de montagem; datas reais nunca são renomeadas.
matriz = functions['matriz_mes']
assert matriz.count("(destino.to_period('M') - 1).to_timestamp()") == 1
matriz = matriz.replace("(destino.to_period('M') - 1).to_timestamp()", 'origem_mensal([destino], LAG_ATMOSFERA)[0]')
matriz = matriz.replace('Atmosfera do mês anterior ao alvo.', 'Atmosfera T−4; índices T−3; meses reais preservados.')
library += '\n\n' + matriz
sst_source = ROOT/'pesquisa/sst_pca/extensao_sst.py'
sst_text = sst_source.read_text(encoding='utf-8')
st = ast.parse(sst_text)
sf = {n.name: ast.get_source_segment(sst_text, n) for n in st.body if isinstance(n, ast.FunctionDef)}
sa = {t.id: ast.get_source_segment(sst_text, n) for n in st.body if isinstance(n, ast.Assign)
      for t in n.targets if isinstance(t, ast.Name)}
library += '\n\n' + '\n'.join(sa[n] for n in ['SST_COMPONENTES', 'SST_NOME', 'SST_HASH'])
for name in ['sst_auditar_arquivo', 'sst_ajustar_pca', 'sst_transformar',
             'pesquisa_prever_sst', 'pesquisa_diagnosticos', 'pesquisa_resumir']:
    code = sf[name]
    if name in ['sst_ajustar_pca', 'sst_transformar']:
        assert code.count("(alvos_treino.to_period('M')-1).to_timestamp()" if name == 'sst_ajustar_pca'
                          else "(alvos.to_period('M')-1).to_timestamp()") == 1
        code = code.replace("(alvos_treino.to_period('M')-1).to_timestamp()", 'origem_mensal(alvos_treino, LAG_SST)')
        code = code.replace("(alvos.to_period('M')-1).to_timestamp()", 'origem_mensal(alvos, LAG_SST)')
    library += '\n\n' + code
library += '\n\n' + (OUT/'defasagens.py').read_text(encoding='utf-8') + '\n'
compile(library, 'biblioteca.py', 'exec')
(OUT/'biblioteca.py').write_text(library, encoding='utf-8')

# Os cinco meses anteriores ao snapshot incorporado já existem no bruto preservado.
# Somente esses meses são anexados; o trecho sobreposto precisa reproduzir exatamente o antigo.
ns = {}
exec(imports + '\n' + '\n'.join(assignments[n] for n in ['INDICES_NOMES', 'FONTES_INDICES',
    'INDICES_CSV_GZIP_BASE64', 'INDICES_CSV_SHA256']), ns)
old = pd.read_csv(io.BytesIO(gzip.decompress(base64.b64decode(ns['INDICES_CSV_GZIP_BASE64']))),
                  parse_dates=['time_origem']).set_index('time_origem')
early = pd.DataFrame(index=pd.date_range('1976-07-01', '1976-11-01', freq='MS'))
provenance = []
for f in ns['FONTES_INDICES']['fontes']:
    p = ROOT/'entregas/etapa6A_indices_oceanicos/fontes_noaa'/f['arquivo_bruto']
    assert hashlib.sha256(p.read_bytes()).hexdigest() == f['sha256_bruto']
    vals = {}
    for line in p.read_text().splitlines():
        toks = line.split()
        if len(toks) != 13 or not toks[0].isdigit(): continue
        year = int(toks[0])
        if not 1800 <= year <= 2100: continue
        for month, v in enumerate(toks[1:], 1): vals[pd.Timestamp(year, month, 1)] = float(v)
    series = pd.Series(vals)
    assert np.array_equal(series.loc[old.index].values, old[f['indice']].values)
    early[f['indice']] = series.reindex(early.index)
    provenance.append(dict(indice=f['indice'], url=f['url'], sha256_bruto=f['sha256_bruto']))
extended = pd.concat([early[ns['INDICES_NOMES']], old]).loc[:'2020-09-01']
assert np.isfinite(extended.values).all() and extended.values.min() > -90
extended.index.name = 'time_origem'
extended.to_csv(OUT/'indices_noaa_197607_202009.csv', float_format='%.9f', lineterminator='\n')
(OUT/'fontes_noaa.json').write_text(json.dumps(dict(fontes=provenance,
    captura=ns['FONTES_INDICES']['captura_utc'], extensao_197607_197611=True,
    sobreposicao_exata=True, vintages=False), indent=2, ensure_ascii=False), encoding='utf-8')

ref_node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
               and t.id == 'BANCA_REF' for t in n.targets))
ref = json.loads(ast.literal_eval(ref_node.value.args[0]))
(OUT/'hashes_oficiais.json').write_text(json.dumps(ref['entradas'], indent=2), encoding='utf-8')
protocol_path = OUT/'protocolo.json'
if not protocol_path.exists():
    protocol = dict(id='nimbus_defasagens_4_3_2_v1', registrado_em_utc=datetime.now(timezone.utc).isoformat(),
        objetivo='Contribuição pareada de oito PCs SST com entradas e chuva de treino defasadas',
        natureza='retrospectivo com produtos consolidados; disponibilidade e vintages não comprovados',
        lags_meses=dict(atmosfera=4, indices=3, sst=2, chuva_treino=4, inicializacao_sazonal=1),
        avaliacao=['2007-01-01','2020-12-01'], holdout_independente=False, consultar_2025=False,
        inicio_comum='1982-03-01', meses_maximos=360,
        corte_treino='primeiro alvo de cada bloco menos quatro meses; setembro do ano anterior',
        frequencia_ajuste='uma vez a cada bloco de 24 meses; previsões mensais com entradas atualizadas',
        climatologias='chuva/atmosfera/índices: 360 meses por fonte; sazonais e PCA: meses dos exemplos',
        pca=dict(componentes=8, solver='full', mascara_clima_centro='somente treino',
                 peso_area='sqrt(cos(latitude))', padronizar_desvio=False),
        arvores=300, folhas=31, taxa=.05, semente=42, pontos_por_mes=5000,
        ridge_alpha=.1, pesos=[.375,.375,.25], escalas_residuais=[.9,.875],
        busca_parametros=False, busca_lags=False, busca_pesos=False,
        metrica_primaria='sqrt(SSE/n) sem ponderação, todos os pontos; diferença SST menos controle',
        secundarios=['MAE','vies','RMSE coslat','blocos','anos','meses','3 faixas de latitude'],
        fontes_em_producao='data de publicação e recebimento comprovadas antes da emissão; ausência bloqueia',
        lags_nao_certificam_publicacao=True, cfs_publicacao_iri='desconhecida',
        arquivo_sst='ersstv5_4graus_198201_202411.nc',
        hash_sst='908034f6418833302ea432f25850ad2982d1114b747b7b298e26a350eaf2891d')
    protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding='utf-8')

sources = [source, sst_source, OUT/'defasagens.py', OUT/'executar_comparacao.py', OUT/'testar_contrato.py']
(OUT/'proveniencia.json').write_text(json.dumps([dict(arquivo=str(p.relative_to(ROOT)),
    sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources], indent=2), encoding='utf-8')
filenames = ['biblioteca.py','executar_comparacao.py','testar_contrato.py','protocolo.json',
    'indices_noaa_197607_202009.csv','fontes_noaa.json','hashes_oficiais.json','proveniencia.json','LEIA_ME.md']
bundle = {n:(OUT/n).read_text(encoding='utf-8') for n in filenames}
hashes = {n:hashlib.sha256(v.encode()).hexdigest() for n,v in bundle.items()}
bootstrap = '''from pathlib import Path
import json, hashlib, runpy
RAIZ = Path('/kaggle/working/nimbus_defasagens_4_3_2')
CODIGO = RAIZ/'codigo'
CODIGO.mkdir(parents=True, exist_ok=True)
''' + 'ARQUIVOS = ' + repr(bundle) + '\nHASHES_CODIGO = ' + repr(hashes) + '''
for nome, conteudo in ARQUIVOS.items():
    assert hashlib.sha256(conteudo.encode()).hexdigest() == HASHES_CODIGO[nome]
    caminho = CODIGO/nome
    if caminho.exists():
        assert caminho.read_text(encoding='utf-8') == conteudo, 'Código diferente; preservar execução: ' + nome
    else:
        caminho.write_text(conteudo, encoding='utf-8')
print('Código e protocolo registrados. Nenhum modelo ajustado nesta célula.')
print('Protocolo SHA-256:', HASHES_CODIGO['protocolo.json'])
'''
cells = [nbformat.v4.new_markdown_cell((OUT/'LEIA_ME.md').read_text(encoding='utf-8')),
    nbformat.v4.new_code_cell(bootstrap),
    nbformat.v4.new_markdown_cell('## 1. Testes temporais antes dos dados reais\nTesta virada do ano, corte da chuva, meses das fontes, PCA e bloqueio quando a chegada é desconhecida.'),
    nbformat.v4.new_code_cell("_TESTES = runpy.run_path(str(CODIGO/'testar_contrato.py'), run_name='__main__')"),
    nbformat.v4.new_markdown_cell('## 2. Comparação pareada\nQuatro LightGBMs e uma regressão local por bloco. Somente arquivos de desenvolvimento são carregados. Pesos, escalas e defasagens ficam fixos. Relatórios parciais são preservados por bloco.'),
    nbformat.v4.new_code_cell("RESULTADO = runpy.run_path(str(CODIGO/'executar_comparacao.py'), run_name='__main__')"),
    nbformat.v4.new_markdown_cell('## 3. Resultado e arquivos\nAs tabelas discriminam o efeito de SST sob a mesma política temporal. Não comparam isoladamente o custo de cada defasagem nem certificam operação real.'),
    nbformat.v4.new_code_cell("from IPython.display import display, Markdown, Image, FileLink\nassert (RAIZ/'resultado.md').is_file(), 'Comparação ainda não concluída.'\ndisplay(Markdown((RAIZ/'resultado.md').read_text(encoding='utf-8')))\ndisplay(Image(filename=str(RAIZ/'comparacao.png')))\ndisplay(FileLink('nimbus_defasagens_4_3_2/relatorios.zip'))")]
verificacao = (OUT/'conferir_resultados.py').read_text(encoding='utf-8').split("if __name__ == '__main__':")[0]
cells.append(nbformat.v4.new_code_cell(verificacao + '\nconferir(RAIZ)\n'))
doc = nbformat.v4.new_notebook(cells=cells)
doc.metadata.kernelspec = dict(display_name='Python 3', language='python', name='python3')
nbformat.validate(doc)
for c in doc.cells:
    if c.cell_type == 'code': compile(c.source, '<cell>', 'exec')
nbformat.write(doc, OUT/'Nimbus_PA_defasagens_e_SST.ipynb')
print('Notebook gerado:', OUT/'Nimbus_PA_defasagens_e_SST.ipynb')
