"""Gera notebook de coleta/registro, sem treinamento ou previsão fictícia."""
from pathlib import Path
import hashlib
import json
import nbformat

RAIZ=Path(__file__).resolve().parent
nomes=['registro.py','dados.py','coletar.py','preparar_cfsv2.py','adaptador_cfsv2.py',
       'avaliar.py','testar_registro.py','plano.json','README.md','requirements.txt']
bundle={n:(RAIZ/n).read_text(encoding='utf-8') for n in nomes}
bootstrap='''from pathlib import Path
import json, sys, hashlib, runpy, shutil
BASE=Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path.cwd()/'outputs'
RAIZ=BASE/'nimbus_acompanhamento'
CODIGO=RAIZ/'codigo'
REGISTRO=RAIZ/'registro'
CODIGO.mkdir(parents=True,exist_ok=True)
'''+ 'ARQUIVOS='+repr(bundle)+'''
for nome,conteudo in ARQUIVOS.items():
    p=CODIGO/nome
    if p.exists() and p.read_text(encoding='utf-8') != conteudo:
        raise RuntimeError('Código diferente na pasta existente; preservar execução e usar outra pasta.')
    if not p.exists():p.write_text(conteudo,encoding='utf-8')
sys.path.insert(0,str(CODIGO))
from registro import Registro
# Restaurar comprovantes de uma sessão anterior. Copiar somente a cadeia e seus objetos.
if not (REGISTRO/'eventos').exists() and Path('/kaggle/input').is_dir():
    anteriores=[p.parent for p in Path('/kaggle/input').rglob('resumo_registro.json')
        if (p.parent/'eventos').is_dir() and (p.parent/'objetos').is_dir()]
    if len(anteriores)>1:raise RuntimeError('Mais de um registro anterior: anexe somente a cadeia escolhida.')
    if anteriores:
        Registro(anteriores[0]).eventos(conferir_objetos=True)
        REGISTRO.mkdir(parents=True,exist_ok=True)
        for pasta in ['eventos','objetos']:shutil.copytree(anteriores[0]/pasta,REGISTRO/pasta)
        print('Registro anterior conferido e restaurado; horários preservados.')
print('Código preparado. Ainda não foi emitida uma previsão.')
'''
coleta='''from coletar import executar
from preparar_cfsv2 import preparar
ALVO='2026-10'
ESTADO=executar(REGISTRO,CODIGO/'plano.json',ALVO,incluir_cfsv2=True)
_CFS=preparar(REGISTRO,ALVO)
'''
relatorio='''from avaliar import painel
from IPython.display import display, Markdown, FileLink
r=Registro(REGISTRO)
r.exportar_resumo(REGISTRO/'resumo_registro.json')
estado=r.prontidao(ALVO)
(REGISTRO/'prontidao.json').write_text(json.dumps(estado,indent=2,ensure_ascii=False),encoding='utf-8')
(REGISTRO/'painel.json').write_text(json.dumps(painel(r),indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(estado,indent=2,ensure_ascii=False))
display(Markdown('**Salve a versão com outputs.** O registro é a evidência de chegada dos arquivos. '
    'Este notebook não reajusta o modelo, não lê chuva do alvo e não produz uma previsão enquanto houver pendências.'))
display(FileLink(str((REGISTRO/'resumo_registro.json').relative_to(BASE))))
'''
cells=[nbformat.v4.new_markdown_cell(bundle['README.md']),nbformat.v4.new_code_cell(bootstrap),
    nbformat.v4.new_markdown_cell('## Testes isolados\nDados sintéticos são criados somente em pastas temporárias. Nenhuma previsão de teste entra no registro real.'),
    nbformat.v4.new_code_cell("_TESTES=runpy.run_path(str(CODIGO/'testar_registro.py'),run_name='__main__')"),
    nbformat.v4.new_markdown_cell('## Coleta pública e auditoria\nInternet ligada; nenhum token. Cada resposta recebe horário real. Ausência de membros CFSv2 bloqueia a entrada. Não instalar nem migrar versões de fontes silenciosamente.'),
    nbformat.v4.new_code_cell(coleta),
    nbformat.v4.new_markdown_cell('## Prontidão e preservação\nOs arquivos de saída contêm os bytes recebidos e a cadeia de eventos. Restaurar a saída anterior é necessário para continuidade entre sessões.'),
    nbformat.v4.new_code_cell(relatorio)]
doc=nbformat.v4.new_notebook(cells=cells)
doc.metadata.kernelspec=dict(display_name='Python 3',language='python',name='python3')
for c in cells:
    if c.cell_type=='code':compile(c.source,'<cell>','exec')
nbformat.validate(doc)
nbformat.write(doc,RAIZ/'Nimbus_PA_registro_prospectivo.ipynb')
origem=RAIZ.parent/'avaliacao_2025/adquirir_cfsv2_2025.py'
proveniencia=dict(origem_cfsv2=str(origem.relative_to(RAIZ.parent.parent)),
    sha256_origem=hashlib.sha256(origem.read_bytes()).hexdigest(),
    alteracoes='sete funções preservadas; nenhuma exceção de membro permitida no acompanhamento novo',
    arquivos={n:hashlib.sha256((RAIZ/n).read_bytes()).hexdigest() for n in nomes})
(RAIZ/'proveniencia.json').write_text(json.dumps(proveniencia,indent=2,ensure_ascii=False),encoding='utf-8')
print('Notebook autossuficiente gerado, oito células; sem execução de modelos.')
