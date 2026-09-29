"""Independently verify exported reports without fitting models."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

def conferir(raiz):
    raiz = Path(raiz)
    inventario = json.loads((raiz / 'inventario.json').read_text(encoding='utf-8'))
    verificados = 0
    for item in inventario:
        p = raiz / item['arquivo']
        if not p.is_file():
            assert p.suffix in {'.txt', '.nc', '.npz'}, item['arquivo']
            continue
        assert p.stat().st_size == item['bytes'], p
        assert hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256'], p
        verificados += 1
    d = pd.read_csv(raiz / 'metricas_mensais_regioes.csv')
    assert not d.duplicated(['modelo', 'mes_alvo', 'regiao']).any()
    assert set(d.modelo) == {'controle_defasado', 'sst_defasada', 'climatologia'}
    assert len(d) == 3 * 168 * 4
    assert set(pd.to_datetime(d.mes_alvo)) == set(pd.date_range('2007-01-01', '2020-12-01', freq='MS'))
    cols = ['n', 'sse', 'soma_erro', 'soma_erro_absoluto', 'sse_area', 'peso_area']
    reg = d[d.regiao != 'dominio_inteiro'].groupby(['modelo', 'mes_alvo'])[cols].sum().sort_index()
    dom = d[d.regiao == 'dominio_inteiro']
    assert np.allclose(reg, dom.set_index(['modelo', 'mes_alvo'])[cols].sort_index(), rtol=1e-12, atol=1e-07)
    for nome, eixo, dados in [('globais', ['modelo'], dom), ('blocos', ['modelo', 'bloco'], dom), ('anos', ['modelo', 'ano'], dom), ('regioes', ['modelo', 'regiao'], d)]:
        soma = dados.groupby(eixo)[cols].sum().sort_index()
        salvo = pd.read_csv(raiz / f'metricas_{nome}.csv').set_index(eixo).sort_index()
        assert soma.index.equals(salvo.index)
        assert np.allclose(soma, salvo[cols], rtol=1e-12, atol=1e-07), nome
        for coluna, esperado in [('rmse', np.sqrt(soma.sse / soma.n)), ('mae', soma.soma_erro_absoluto / soma.n), ('vies', soma.soma_erro / soma.n), ('rmse_area', np.sqrt(soma.sse_area / soma.peso_area))]:
            assert np.allclose(esperado, salvo[coluna], rtol=1e-12, atol=1e-12), (nome, coluna)
    globais = pd.read_csv(raiz / 'metricas_globais.csv').set_index('modelo')
    assert (globais.n == 13198248).all()
    for bloco in ['H1', 'H2', 'H3', 'H4', 'A', 'B', 'C']:
        pasta = raiz / bloco
        cal = pd.read_csv(pasta / 'calendario_emissoes.csv')
        t = pd.to_datetime(cal.time_alvo).dt.to_period('M')
        for coluna, lag in [('atmosfera', 4), ('indices', 3), ('sst', 2), ('inicializacao_sazonal', 1)]:
            assert (pd.to_datetime(cal[coluna]).dt.to_period('M') == t - lag).all()
        corte = pd.to_datetime(cal.ultimo_alvo_treino).dt.to_period('M')
        assert (corte <= t.iloc[0] - 4).all()
        aud = json.loads((pasta / 'auditoria.json').read_text())
        assert aud['meses'] <= 360
        for chave in ['chuva_pos_corte_invariante', 'atmosfera_apos_Tmenos4_invariante', 'indices_apos_Tmenos3_invariantes', 'sst_apos_Tmenos2_invariante', 'climas_sazonais_somente_treino']:
            assert aud[chave] is True, (bloco, chave)
    resumo = json.loads((raiz / 'resumo.json').read_text())
    assert resumo['alvos_2025_lidos'] is False and resumo['producao_liberada'] is False
    delta = globais.loc['sst_defasada', 'rmse'] - globais.loc['controle_defasado', 'rmse']
    assert np.isclose(delta, resumo['delta_rmse'], rtol=1e-12, atol=1e-12)
    resultado = dict(arquivos_sha256_conferidos=verificados, linhas_mensais=len(d), agregacoes_conferidas=True, calendarios_conferidos=True, auditorias_conferidas=True, delta_rmse=float(delta), sem_novo_ajuste=True)
    print(json.dumps(resultado, indent=2))
    print(globais[['rmse', 'mae', 'vies', 'rmse_area']].to_string())
if __name__ == '__main__':
    conferir(sys.argv[1])
