"""Gera extensão auditável; não executa treinamentos nem modifica a entrega original."""
import ast
import hashlib
import json
import nbformat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
old = ROOT / 'entregas/etapa8F_M10_SST_PCA/06_M10_SST_global_PCA_8_componentes.py'
source = old.read_text(encoding='utf-8-sig')
tree = ast.parse(source)
names = ['sst_auditar_arquivo', 'sst_ajustar_pca', 'sst_transformar', 'sst_auditar_causalidade']
functions = [ast.get_source_segment(source, n) for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name in names]
assert len(functions) == len(names)
header = '''# Pesquisa Nimbus PA — extensão SST, protocolo fixo de 28/09/2026.
# Executar depois da conferência histórica e numérica do notebook híbrido.
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
import importlib.metadata

SST_COMPONENTES = 8
SST_FEATURES = [f'sst_lag1_pc_{i:02d}_treino_fold' for i in range(1, 9)]
SST_NOME = 'ersstv5_4graus_198201_202411.nc'
SST_HASH = '908034f6418833302ea432f25850ad2982d1114b747b7b298e26a350eaf2891d'
'''
runner = (OUT / 'comparacao_sst.py').read_text(encoding='utf-8')
code = header + '\n\n'.join(functions) + '\n\n' + runner
compile(code, 'extensao_sst.py', 'exec')
(OUT / 'extensao_sst.py').write_text(code, encoding='utf-8', newline='\n')
provenance = dict(fonte_funcoes=str(old.relative_to(ROOT)),
    fonte_sha256=hashlib.sha256(old.read_bytes()).hexdigest(), funcoes_preservadas=names,
    extensao_sha256=hashlib.sha256(code.encode()).hexdigest(),
    escopo='Uma comparação retrospectiva fixa; sem busca, seleção por placar ou emissão operacional.')
(OUT / 'proveniencia_extensao.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding='utf-8')
doc = nbformat.read(ROOT/'entregas/banca_8G/WorCAP_modelo_hibrido_apresentacao.ipynb',as_version=4)
for cell in doc.cells:
    if cell.cell_type == 'code':
        cell.source = cell.source.replace('REEXECUTAR_VALIDACAO = False','REEXECUTAR_VALIDACAO = True')
        if cell.source.startswith('ec_exigir = exigir'):
            cell.source = (OUT/'exportar_referencia.py').read_text(encoding='utf-8')
        cell.outputs = []
        cell.execution_count = None
intro = '''# Nimbus PA — pesquisa retrospectiva com SST

Esta cópia reproduz a solução híbrida e acrescenta uma comparação controlada.
Os registros apresentados na parte original são históricos; a reprodução é confirmada pelos hashes e
pelas métricas reexecutadas, antes de iniciar o experimento.

**Inputs:** competição WorCAP, `worcap-seas5-dados`, `worcap-cfsv2-dados` e
`nimbus-pa-sst-ersstv5-snapshot`. Execute em CPU no ambiente exigido pela referência.
Anexe também `nimbus-pa-referencia-csv-auditoria` para a conferência do CSV arquivado.
Na execução de 28/09, os componentes de árvores foram idênticos byte a byte; o híbrido
apresentou 532 diferenças de uma unidade na sexta casa decimal. A auditoria registra
essa diferença e verifica todos os valores antes de aceitar a reprodução numérica.
O CSV arquivado é usado exclusivamente para conferência, sem entrar no treinamento.
Os arquivos locais da competição permanecem intactos. Nenhuma submissão é enviada.

**Hipótese:** oito componentes da temperatura da superfície do mar acrescentam informação aos
dois componentes LightGBM. A regressão ridge, as amostras, os períodos, os pesos, os
hiperparâmetros e a semente são mantidos. Não há busca de pesos após os resultados.

**Limites:** 2007–2020 já foi consultado durante o desenvolvimento. Os resultados são exploratórios,
não uma estimativa independente da habilidade futura. SST de T−1 em um arquivo consolidado é
uma referência retrospectiva, não prova de disponibilidade até o fim de T−1.
'''
doc.cells.insert(0,nbformat.v4.new_markdown_cell(intro))
doc.cells.extend([
    nbformat.v4.new_markdown_cell('''## Experimento fixo: oito PCs de SST nos dois componentes de árvores

O snapshot ERSSTv5 usa 60°S–60°N em grade subamostrada de 4°. Os dados de T−1 são transformados
por uma PCA com oito componentes, ajustada apenas no treino de cada bloco. Máscara oceânica,
médias mensais e centragem também usam só o treino. A ponderação é a raiz de cosseno da latitude.
Não se divide cada pixel por seu desvio padrão.

Reutilizamos os modelos de controle recém-reproduzidos. As matrizes e os alvos são conferidos por
hash. As duas árvores candidatas recebem os mesmos dados, acrescidos dos oito PCs.
Ambas as soluções mantêm pesos 37,5% + 37,5% + 25%. A comparação inclui RMSE, MAE, viés,
RMSE ponderado por área, anos, meses do calendário e faixas de latitude explicitamente definidas.
As previsões completas e os mapas de diferença do erro quadrático são preservados em NetCDF.
'''),nbformat.v4.new_code_cell(code),
    nbformat.v4.new_code_cell((OUT/'exportar_relatorios.py').read_text(encoding='utf-8')),
    nbformat.v4.new_code_cell((OUT/'apresentar_resultado.py').read_text(encoding='utf-8'))])
nbformat.validate(doc)
nbformat.write(doc,OUT/'Nimbus_PA_pesquisa_SST.ipynb')
print(json.dumps(provenance, indent=2, ensure_ascii=False))
