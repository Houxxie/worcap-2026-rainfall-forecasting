from pathlib import Path
from IPython.display import display, Markdown, FileLink
import json, hashlib
_raiz = Path('/kaggle/working/nimbus_avaliacao_2025')
_itens = {
    'CFSv2 (12 meses; membros e médias auditados)': 'CFSv2/cfsv2_2025.nc',
    'SST (sobreposição com snapshot de treino conferida)': 'SST/ersstv5_4graus_202411_202511.nc',
    'Índices NOAA do snapshot documentado': 'NOAA/indices_2025.csv',
    'ERA5 (entradas e piloto de compatibilidade)': 'ERA5/atmosfera_origens_2025.nc',
    'SEAS5 (previsões para 2025)': 'SEAS5/seas5_2025.nc',
    'Previsões dos dois modelos congeladas': 'previsoes_congeladas/registro_previsoes.json',
    'Métricas da reserva': 'metricas_2025.csv',
}
_linhas = '\n'.join(f'| {nome} | {"Arquivo presente" if (_raiz/rel).is_file() else "Pendente"} |'
                    for nome,rel in _itens.items())
_consultada = (_raiz/'reserva_consultada.json').exists()
display(Markdown(f'''## Avaliação de 2025 — estado da execução

Referência híbrida versus a mesma arquitetura com oito PCs de SST.
Treino fixo: **1993–2022**. Avaliação: **janeiro–dezembro de 2025**.

| Etapa | Estado |
|---|---|
{_linhas}

**Chuva-alvo de 2025 consultada: {"sim" if _consultada else "não"}.**

A auditoria de disponibilidade está concluída. ERA5 mensal e índices/SST
do mês anterior exigem revisão das defasagens para uso antes do início do mês.
Este experimento é retrospectivo; a comparação operacional será outra etapa.

Os arquivos presentes são conferidos por hash nas etapas de aquisição e inferência.
As métricas só são calculadas depois do congelamento das duas previsões.
'''))
_estado = dict(protocolo_sha256=hashlib.sha256((_raiz/'protocolo_congelado.json').read_bytes()).hexdigest(),
    arquivos={nome:(_raiz/rel).is_file() for nome,rel in _itens.items()},
    alvos_2025_consultados=_consultada,natureza='retrospectiva')
(_raiz/'estado_execucao.json').write_text(json.dumps(_estado,indent=2,ensure_ascii=False),encoding='utf-8')
display(FileLink('nimbus_avaliacao_2025/codigo/auditoria_disponibilidade.md'))
