"""Constrói uma etapa privada; nunca executa o código original da competição."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import shutil
import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
OUT.mkdir(exist_ok=True)

def salvar(nome, texto):
    (OUT / nome).write_text(texto, encoding='utf-8', newline='\n')

protocolo = {
    'id': 'nimbus_avaliacao_retrospectiva_2025_v1',
    'registrado_em_utc': datetime.now(timezone.utc).isoformat(),
    'status': 'registrado_antes_de_adquirir_alvos_2025',
    'treino_alvos': ['1993-01-01', '2022-12-01'],
    'teste_alvos': ['2025-01-01', '2025-12-01'],
    'origens_entradas': ['2024-12-01', '2025-11-01'],
    'anos_ja_consultados': {'2007-2020': 'desenvolvimento',
        '2021-2022': 'reserva consultada em 17/09/2026; também ajuste final',
        '2023-2024': 'placares e escolha das submissões'},
    'controle': 'hibrido fixo: duas arvores e ridge local',
    'candidato': 'mesmo hibrido; oito PCs ERSSTv5 nas duas arvores',
    'arvores': 300, 'folhas': 31, 'taxa': 0.05, 'semente': 42,
    'pontos_por_mes': 5000, 'anos_treino': 30, 'ridge_alpha': 0.1,
    'pesos': [0.375, 0.375, 0.25], 'escalas_residuais': [0.9, 0.875],
    'pca': {'n': 8, 'solver': 'full', 'lag': 1,
        'mascara_clima_centragem_fit': 'somente treino 1992-12 a 2022-11',
        'peso': 'sqrt(cos(latitude))', 'padronizar_desvio': False},
    'sem_reajuste_em_2023_2024_2025': True,
    'alvo': 'ERA5 monthly_averaged_reanalysis total_precipitation * 1000; mm/day',
    'grade': {'lat': [-60,15,0.25], 'lon': [-90,-25,0.25]},
    'metrica_primaria': 'sqrt(SSE/n) em todos os 942732 pontos-meses',
    'diagnosticos': ['MAE', 'vies', 'RMSE_area_coslat', '12 meses',
        'lat<-35', '-35<=lat<-15', 'lat>=-15'],
    'selecao_ou_busca_no_teste': False,
    'liberar_alvos_somente_apos': 'previsoes dos dois modelos gravadas, hashes e compatibilidade ERA5 aprovados',
    'piloto_compatibilidade': ['2022-01-01', '2022-07-01'],
    'tolerancia_piloto': 'np.allclose rtol=1e-5 atol=1e-6 em escala distribuida; se falhar, parar e investigar sem consultar 2025',
    'natureza': 'retrospectiva com produtos consolidados; nao simulacao operacional por vintages',
    'independencia': '2025 nao consta entre os periodos consultados nos registros do projeto; torna-se consultado na primeira avaliacao',
    'limite': 'um ano nao comprova generalizacao a todos os anos futuros',
}
p = OUT/'protocolo_congelado.json'
if not p.exists():
    salvar(p.name, json.dumps(protocolo, ensure_ascii=False, indent=2))
else:
    protocolo = json.loads(p.read_text(encoding='utf-8'))
sha = hashlib.sha256(p.read_bytes()).hexdigest()
salvar('protocolo.sha256', sha+'\n')

# Mantém exatamente os auditores dos produtos sazonais, em namespaces separados.
cfs_path = ROOT/'entregas/etapa8A_CFSv2_dados/01_CFSv2_baixar_e_auditar.py'
cfs = cfs_path.read_text(encoding='utf-8-sig').split('# %% Consolidar desenvolvimento')[0]
cfs = cfs.replace("SAIDA = BASE/'worcap_CFSv2_dados'", "SAIDA = BASE/'nimbus_avaliacao_2025'/'CFSv2'")
cfs = cfs.replace("'1982-02-01', '2024-12-01'", "'2025-01-01', '2025-12-01'")
cfs = cfs.replace("exigir(len(ALVOS) == 515, 'Este protocolo prepara 515 meses-alvo: fev/1982 a dez/2024.')", "exigir(len(ALVOS) == 12, 'Exigidos os 12 alvos de 2025.')")
cfs = cfs.replace("    url = FONTES_DATAS[fase]+intervalo_url(datas)+'data.nc'", "    if not len(datas):\n        continue\n    url = FONTES_DATAS[fase]+intervalo_url(datas)+'data.nc'")
cfs += '''
partes=[]
for path in BLOCOS:
    with xr.open_dataset(path) as ds: partes.append(ds.load())
completo=xr.concat(partes,dim='time').sortby('time')
exigir(pd.DatetimeIndex(completo.time.values).equals(ALVOS),'CFSv2 incompleto.')
salvar_nc(completo,SAIDA/'cfsv2_2025.nc')
pd.DataFrame(AUDITORIA).to_csv(SAIDA/'auditoria_2025.csv',index=False)
json_salvar(SAIDA/'manifesto_2025.json',dict(fonte=FONTE,pedidos=PEDIDOS,
    pilotos=AUDITORIA_PILOTOS,sha256=sha256(SAIDA/'cfsv2_2025.nc'),
    publicacao_historica_comprovada=False,alvos=['2025-01','2025-12']))
print('CFSv2 2025 pronto. Datas de inicialização não comprovam data de publicação.')
'''
salvar('adquirir_cfsv2_2025.py', cfs)

seas_path = ROOT/'entregas/etapa7A_SEAS5/01_SEAS5_baixar_e_auditar.py'
seas = seas_path.read_text(encoding='utf-8-sig').split('# %% Artefatos separados:')[0]
seas = seas.replace("SAIDA = BASE/'worcap_SEAS5_dados'", "SAIDA = BASE/'nimbus_avaliacao_2025'/'SEAS5'")
seas = seas.replace("INICIO_ALVOS = '1982-01-01'", "INICIO_ALVOS = '2025-01-01'")
seas = seas.replace("FIM_ALVOS = '2024-12-01'", "FIM_ALVOS = '2025-12-01'")
seas += '''
completo=xr.concat(partes,dim='time').sortby('time')
auditar_dataset(completo,ALVOS)
completo.to_netcdf(SAIDA/'seas5_2025.nc',engine='netcdf4')
pd.DataFrame(linhas).to_csv(SAIDA/'auditoria_2025.csv',index=False)
json_salvar(SAIDA/'manifesto_2025.json',dict(fonte=FONTE,system=SYSTEM,
    pedidos=fontes,sha256=sha256(SAIDA/'seas5_2025.nc'),
    publicacao_historica_comprovada=False,alvos=['2025-01','2025-12']))
print('SEAS5 2025 pronto: 12 meses auditados; nenhuma chuva observada lida.')
'''
salvar('adquirir_seas5_2025.py', seas)

registros=[]
for src in [cfs_path, seas_path, ROOT/'entregas/banca_8G/WorCAP_modelo_hibrido_reproducao.py',
            ROOT/'pesquisa/sst_pca/extensao_sst.py']:
    registros.append(dict(arquivo=str(src.relative_to(ROOT)),sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
salvar('proveniencia_codigo.json',json.dumps(registros,indent=2,ensure_ascii=False))
for nome in ['adquirir_cfsv2_2025.py','adquirir_seas5_2025.py']:
    compile((OUT/nome).read_text(encoding='utf-8'),nome,'exec')
print('Protocolo registrado:',sha)

# Extensão NOAA extraída do mesmo snapshot bruto que gerou os índices históricos.
fontes=json.loads((ROOT/'entregas/etapa6A_indices_oceanicos/fontes_indices.json').read_text(encoding='utf-8-sig'))
origens=pd.date_range('2024-12-01','2025-11-01',freq='MS')
tabela=pd.DataFrame(index=origens)
reg=[]
for nome in ['nino34','nino12','tna','tsa']:
    raw=ROOT/'entregas/etapa6A_indices_oceanicos/fontes_noaa'/f'{nome}.data'
    esperado=next(f for f in fontes['fontes'] if f['indice']==nome)
    digest=hashlib.sha256(raw.read_bytes()).hexdigest()
    assert digest==esperado['sha256_bruto']
    vals={}
    for linha in raw.read_text().splitlines():
        tokens=linha.split()
        if len(tokens)!=13 or not tokens[0].isdigit(): continue
        ano=int(tokens[0])
        if not 1800<=ano<=2100: continue
        for mes,v in enumerate(tokens[1:],1): vals[pd.Timestamp(ano,mes,1)]=float(v)
    tabela[nome]=pd.Series(vals).reindex(origens)
    reg.append(dict(indice=nome,fonte=esperado['url'],sha256_bruto=digest,
        captura_original=fontes['captura_utc']))
assert np.isfinite(tabela.values).all() and (tabela.values>-90).all()
tabela.index.name='time_origem'
tabela.to_csv(OUT/'indices_2025.csv',float_format='%.9f')
salvar('fontes_indices_2025.json',json.dumps(dict(fontes=reg,
    treino_original_preservado=True,vintages_operacionais=False,
    sha256_csv=hashlib.sha256((OUT/'indices_2025.csv').read_bytes()).hexdigest()),indent=2,ensure_ascii=False))

# A etapa de inferência incorpora funções originais verificadas, sem executar validação antiga ou submissões.
base=(ROOT/'entregas/banca_8G/WorCAP_modelo_hibrido_reproducao.py').read_text(encoding='utf-8-sig').split('def banca_validar():')[0]
base=base.replace("BASE / 'worcap_banca_hibrida' / EXECUCAO", "BASE / 'nimbus_avaliacao_2025' / 'treino_metadata'")
pca=(ROOT/'pesquisa/sst_pca/extensao_sst.py').read_text(encoding='utf-8').split('def pesquisa_sst_executar():')[0]
predict=base+'\n'+pca+'\nPROTOCOLO_2025_SHA = '+repr(sha)+'\n'+(OUT/'prever_fixo_runner.py').read_text(encoding='utf-8')
salvar('prever_modelos_2025.py',predict)

arquivos=['protocolo_congelado.json','auditoria_disponibilidade.md',
    'adquirir_cfsv2_2025.py','adquirir_sst_2025.py','adquirir_seas5_2025.py','adquirir_era5_2025.py',
    'indices_2025.csv','fontes_indices_2025.json','prever_modelos_2025.py','avaliar_reserva.py',
    'relatorio_resultados.py']
bundle={n:(OUT/n).read_text(encoding='utf-8') for n in arquivos}
for nome,conteudo in bundle.items():
    if nome.endswith('.py'): compile(conteudo,nome,'exec')
bootstrap='''# Preparação do teste independente retrospectivo — não altera a entrega original.
from pathlib import Path
import hashlib, json, runpy
from IPython.display import display, Markdown
RAIZ_2025=Path('/kaggle/working/nimbus_avaliacao_2025')
CODIGO_2025=RAIZ_2025/'codigo'
CODIGO_2025.mkdir(parents=True,exist_ok=True)
'''+ 'ARQUIVOS_2025 = '+repr(bundle)+'''
for nome,conteudo in ARQUIVOS_2025.items():
    path=CODIGO_2025/nome
    if path.exists() and path.read_text(encoding='utf-8')!=conteudo and nome=='adquirir_era5_2025.py':
        antigo=path.read_bytes()
        assert hashlib.sha256(antigo).hexdigest()=='72af1e2a97ff6bec5dd53f71f8db7a47fbcdcce29fe023e0417fde48d4377402'
        assert not (RAIZ_2025/'reserva_consultada.json').exists(), 'Não reparar após consultar a reserva.'
        assert not (RAIZ_2025/'previsoes_congeladas/registro_previsoes.json').exists()
        historico=CODIGO_2025/'historico';historico.mkdir(exist_ok=True)
        (historico/'adquirir_era5_2025_v1.py').write_bytes(antigo)
        path.write_text(conteudo,encoding='utf-8')
        (historico/'reparo_leitura_mensal.json').write_text(json.dumps(dict(
            motivo='Piloto real: tp mensal 06 UTC e atmosfera 00 UTC; normalizar cada arquivo antes de unir por mês, sem alterar os campos.',
            sha256_antes=hashlib.sha256(antigo).hexdigest(),
            sha256_depois=hashlib.sha256(path.read_bytes()).hexdigest(),
            alvos_2025_consultados=False,protocolo_estatistico_inalterado=True),indent=2,ensure_ascii=False),encoding='utf-8')
    if path.exists():
        assert path.read_text(encoding='utf-8')==conteudo, 'Código/protocolo alterado: '+nome
    else: path.write_text(conteudo,encoding='utf-8')
protocolo=RAIZ_2025/'protocolo_congelado.json'
texto=ARQUIVOS_2025['protocolo_congelado.json']
if protocolo.exists(): assert protocolo.read_text(encoding='utf-8')==texto
else: protocolo.write_text(texto,encoding='utf-8')
noaa=RAIZ_2025/'NOAA';noaa.mkdir(exist_ok=True)
for nome in ['indices_2025.csv','fontes_indices_2025.json']:
    (noaa/nome).write_text(ARQUIVOS_2025[nome],encoding='utf-8')
display(Markdown('''+repr('''### Avaliação de 2025 — protocolo registrado

Os anos até 2024 já foram consultados. A nova comparação mantém os dois modelos fixos,
treinados até dezembro/2022. Os alvos de 2025 só serão lidos depois de gravar ambas as previsões.

**Limite:** avaliação retrospectiva. ERA5 mensal, índices HadISST e SST de T−1 não têm
disponibilidade comprovada no fim de T−1. O relatório registra as defasagens e as fontes.
''')+'''))
print('Protocolo SHA-256:',hashlib.sha256(protocolo.read_bytes()).hexdigest())
'''
publicos="""for arquivo in ['adquirir_cfsv2_2025.py','adquirir_sst_2025.py']:
    runpy.run_path(str(CODIGO_2025/arquivo),run_name='__main__')
print('CFSv2, SST e índices preparados. Nenhum alvo de 2025 foi consultado.')
"""
cds="""# O token só é usado pelo cliente CDS; nunca é exibido ou salvo.
from kaggle_secrets import UserSecretsClient
try:
    _token_2025=UserSecretsClient().get_secret('CDS_API_KEY')
    _cds_ok_2025=bool(_token_2025)
    del _token_2025
except Exception:
    _cds_ok_2025=False
if _cds_ok_2025:
    for arquivo in ['adquirir_era5_2025.py','adquirir_seas5_2025.py']:
        runpy.run_path(str(CODIGO_2025/arquivo),run_name='__main__')
else:
    print('PENDENTE: habilite CDS_API_KEY em Add-ons → Secrets e reexecute somente esta célula.')
    print('Os dados públicos e o protocolo estão preservados. Nenhum teste de 2025 foi avaliado.')
"""
pred="""necessarios=['ERA5/atmosfera_origens_2025.nc','ERA5/compatibilidade_piloto.json',
    'SEAS5/seas5_2025.nc','CFSv2/cfsv2_2025.nc','SST/ersstv5_4graus_202411_202511.nc']
faltam=[p for p in necessarios if not (RAIZ_2025/p).is_file()]
if faltam: print('Inferência aguardando:',faltam)
elif (RAIZ_2025/'previsoes_congeladas/registro_previsoes.json').is_file():
    print('Previsões já congeladas: sem novo ajuste.')
else:
    _espaco_2025=runpy.run_path(str(CODIGO_2025/'prever_modelos_2025.py'),run_name='__main__')
    del _espaco_2025
"""
aval="""if (RAIZ_2025/'previsoes_congeladas/registro_previsoes.json').is_file():
    runpy.run_path(str(CODIGO_2025/'avaliar_reserva.py'),run_name='__main__')
else: print('Avaliação bloqueada: primeiro congelar as duas previsões completas.')
"""
relatorio="""# Apenas apresenta resultados já calculados; não ajusta modelos.
from pathlib import Path
from IPython.display import display, Markdown, Image, FileLink
import runpy
RAIZ_2025=Path('/kaggle/working/nimbus_avaliacao_2025')
CODIGO_2025=RAIZ_2025/'codigo'
"""+"_codigo_relatorio="+repr(bundle['relatorio_resultados.py'])+"\n"+"""if (RAIZ_2025/'metricas_2025.csv').is_file():
    (CODIGO_2025/'relatorio_resultados.py').write_text(_codigo_relatorio,encoding='utf-8')
    _modulo_relatorio=runpy.run_path(str(CODIGO_2025/'relatorio_resultados.py'))
    _resumo_relatorio=_modulo_relatorio['executar'](RAIZ_2025)
    display(Markdown((RAIZ_2025/'resultado_2025.md').read_text(encoding='utf-8')))
    display(Image(filename=str(RAIZ_2025/'comparacao_2025.png')))
    display(FileLink('nimbus_avaliacao_2025/nimbus_relatorio_2025.zip'))
else:
    print('Relatório aguardando a conclusão da avaliação. Nenhum modelo será reajustado.')
"""
doc=nbformat.v4.new_notebook(cells=[
    nbformat.v4.new_markdown_cell('''# Nimbus PA — avaliação retrospectiva em 2025

Comparação fixada antes dos resultados: híbrido de referência versus o mesmo híbrido
com oito PCs de SST. Inputs: competição WorCAP, SEAS5, CFSv2 e snapshot ERSSTv5 de treino.
Use o ambiente da referência (Python 3.12, NumPy 2.0.2, pandas 2.3.3, xarray 2025.12.0,
LightGBM 4.6.0), Internet e o segredo `CDS_API_KEY` da sua conta. O CDS exige as licenças
dos produtos ERA5 mensal de superfície/pressão e SEAS5 já aceitas nessa conta.

O piloto usa janeiro/julho de 2022 para conferir compatibilidade dos campos. Se falhar,
o processo para antes de consultar as respostas de 2025. Não há busca de pesos ou hiperparâmetros.
Os arquivos são salvos em `nimbus_avaliacao_2025`, separados da versão da competição.
'''),nbformat.v4.new_code_cell(bootstrap),
    nbformat.v4.new_markdown_cell('## 1. Preditores públicos; nenhuma precipitação-alvo'),
    nbformat.v4.new_code_cell(publicos),
    nbformat.v4.new_markdown_cell('## 2. Piloto ERA5, atmosfera e SEAS5 — acesso Copernicus'),
    nbformat.v4.new_code_cell(cds),
    nbformat.v4.new_markdown_cell('## 3. Ajuste fixo até 2022 e congelamento das previsões de 2025'),
    nbformat.v4.new_code_cell(pred),
    nbformat.v4.new_markdown_cell('## 4. Primeira abertura da reserva; depois disso 2025 estará consultado'),
    nbformat.v4.new_code_cell(aval),
    nbformat.v4.new_code_cell((OUT/'celula_status_kaggle.py').read_text(encoding='utf-8')),
    nbformat.v4.new_code_cell(relatorio),
    nbformat.v4.new_markdown_cell((OUT/'auditoria_disponibilidade.md').read_text(encoding='utf-8'))])
doc.metadata.kernelspec=dict(display_name='Python 3',language='python',name='python3')
nbformat.validate(doc)
nbformat.write(doc,OUT/'Nimbus_PA_avaliacao_2025.ipynb')
salvar('celula_preparacao_kaggle.py',bootstrap+'\n'+publicos+'\n'+cds)
salvar('celula_inferencia_kaggle.py',pred)
salvar('celula_avaliacao_kaggle.py',aval)
salvar('celula_relatorio_kaggle.py',relatorio)
print('Notebook e scripts compilados. Nenhum treinamento ou avaliação executado pelo construtor.')
