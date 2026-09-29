"""Fixed lag experiment over seven historical development blocks."""
from pathlib import Path
import runpy
CODIGO = Path(__file__).resolve().parent
RAIZ = CODIGO.parent
exec(compile((CODIGO / 'library.py').read_text(encoding='utf-8'), str(CODIGO / 'library.py'), 'exec'), globals())
import zipfile
import matplotlib.pyplot as plt

def registrar(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding='utf-8')

def arr_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def conferir_configuracao(protocolo):
    exigir((LAG_ATMOSFERA, LAG_INDICES, LAG_SST, LAG_CHUVA_TREINO) == (4, 3, 2, 4), 'Lags changed.')
    exigir((ARVORES, SEMENTE, PONTOS_POR_MES, ANOS_TREINO, MOS_ALPHA) == (300, 42, 5000, 30, 0.1), 'Configuration changed.')
    exigir(protocolo['pesos'] == [0.375, 0.375, 0.25] and protocolo['escalas_residuais'] == [0.9, 0.875], 'Weights changed.')
    exigir(protocolo['consultar_2025'] is False and protocolo['avaliacao'] == ['2007-01-01', '2020-12-01'], 'Evaluation changed.')
    esperado = {'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.05, 'num_leaves': 31, 'min_data_in_leaf': 150, 'lambda_l2': 10.0, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'num_threads': 4, 'seed': 42, 'deterministic': True, 'force_col_wise': True, 'verbosity': -1}
    exigir(PARAMETROS == esperado, 'Parameters differ from the fixed comparison.')

def resultado_final(detalhes):
    exigir(not detalhes.duplicated(['modelo', 'mes_alvo', 'regiao']).any(), 'Duplicate months.')
    exigir(set(detalhes.modelo) == {'controle_defasado', 'sst_defasada', 'climatologia'}, 'Incomplete treatments.')
    datas = pd.DatetimeIndex(pd.to_datetime(detalhes.mes_alvo.unique())).sort_values()
    exigir(datas.equals(pd.date_range('2007-01-01', '2020-12-01', freq='MS')), 'Incomplete period.')
    dominio = detalhes[detalhes.regiao == 'dominio_inteiro']
    globais = pesquisa_resumir(dominio, ['modelo'])
    exigir((globais.n == 13198248).all(), 'Unexpected prediction count.')
    blocos = pesquisa_resumir(dominio, ['modelo', 'bloco'])
    anos = pesquisa_resumir(dominio, ['modelo', 'ano'])
    regioes = pesquisa_resumir(detalhes, ['modelo', 'regiao'])
    for nome, df in [('mensais_regioes', detalhes), ('globais', globais), ('blocos', blocos), ('anos', anos), ('regioes', regioes)]:
        df.to_csv(RAIZ / f'metricas_{nome}.csv', index=False, lineterminator='\n')
    consistencia = {}
    for nome, df, eixo in [('blocos', blocos, 'bloco'), ('anos', anos, 'ano')]:
        p = df.pivot(index=eixo, columns='modelo', values='rmse')
        delta = p.sst_defasada - p.controle_defasado
        consistencia[nome] = dict(melhores=int((delta < 0).sum()), piores=int((delta > 0).sum()), empates=int((delta == 0).sum()), pior_delta=float(delta.max()), melhor_delta=float(delta.min()))
    g = globais.set_index('modelo')
    delta = float(g.loc['sst_defasada', 'rmse'] - g.loc['controle_defasado', 'rmse'])
    resumo = dict(status='completo', protocolo_sha256=sha256(CODIGO / 'protocol.json'), rmse_controle=float(g.loc['controle_defasado', 'rmse']), rmse_sst=float(g.loc['sst_defasada', 'rmse']), delta_rmse=delta, reducao_percentual=-100 * delta / float(g.loc['controle_defasado', 'rmse']), consistencia=consistencia, disponibilidade_historica_comprovada=False, holdout_independente=False, producao_liberada=False, alvos_2025_lidos=False)
    registrar(RAIZ / 'resumo.json', resumo)
    linhas = ['| Model | RMSE | MAE | Bias | Area-weighted RMSE |', '|---|---:|---:|---:|---:|']
    labels = {'controle_defasado': 'Hybrid with lagged sources', 'sst_defasada': 'Hybrid with lagged sources and 8 SST PCs', 'climatologia': 'Climatology'}
    for n in labels:
        r = g.loc[n]
        linhas.append(f'| {labels[n]} | {r.rmse:.6f} | {r.mae:.6f} | {r.vies:+.6f} | {r.rmse_area:.6f} |')
    rel = f"# Lagged-source comparison\n\nEvaluation: 2007–2020; 13,198,248 pixel-months per model across seven historical blocks.\nBoth hybrids were refitted with the same samples and temporal boundaries.\n\n{chr(10).join(linhas)}\n\nUnits: mm/day. **SST minus baseline RMSE: {delta:+.6f}**.\nSST improves in **{consistencia['blocos']['melhores']}/7 blocks** and **{consistencia['anos']['melhores']}/14 years**.\nGlobal metrics aggregate error sums rather than averaging individual RMSE values.\n\nERA5 atmosphere uses T−4, ocean indices T−3, SST T−2 and seasonal initializations T−1.\nTraining ends in the September preceding each block's first January, with at most 360 months.\nTrees, ridge, weights and residual scales stay fixed. PCA is fitted only on SST associated\nwith training months and receives no rainfall target.\n\nThis is retrospective sensitivity analysis on years already used during development.\nConsolidated products and hindcasts do not reconstruct each historically published vintage.\nThe historical publication calendar of the IRI CFSv2 aggregate remains unverified.\nThese results do not establish operational readiness or future improvement.\nThe 2025 target was not loaded or evaluated in this experiment.\n\nEach block saves models, PCA, climatologies, regression, predictions and audits.\nThe scientific configuration was fixed before the original comparison; this English\nedition retains it without a new search over lags, weights or hyperparameters.\n"
    (RAIZ / 'resultado.md').write_text(rel, encoding='utf-8')
    anual = anos.pivot(index='ano', columns='modelo', values='rmse')
    dp = anual.sst_defasada - anual.controle_defasado
    with plt.rc_context({'axes.spines.top': False, 'axes.spines.right': False}):
        fig, ax = plt.subplots(2, 1, figsize=(11, 7), layout='constrained')
        for n, col in [('controle_defasado', '#566980'), ('sst_defasada', '#087f79')]:
            ax[0].plot(anual.index, anual[n], marker='o', label=labels[n], color=col)
        ax[0].set(title='Identical lagged sources, with and without global SST', ylabel='RMSE (mm/day)')
        ax[0].legend(frameon=False)
        ax[1].bar(dp.index, dp, color=['#087f79' if v < 0 else '#bc654b' for v in dp])
        ax[1].axhline(0, color='#253a47', lw=0.8)
        ax[1].set(title='Annual SST contribution - negative values indicate improvement', ylabel='RMSE difference (mm/day)', xticks=dp.index, xlabel='Year')
        for a in ax:
            a.grid(axis='y', alpha=0.15)
        fig.savefig(RAIZ / 'comparacao.png', dpi=160)
        plt.close(fig)
    itens = []
    for p in sorted(RAIZ.rglob('*')):
        if p.is_file() and p.suffix not in {'.zip', '.pyc'} and ('__pycache__' not in p.parts):
            itens.append(dict(arquivo=p.relative_to(RAIZ).as_posix(), sha256=sha256(p), bytes=p.stat().st_size))
    registrar(RAIZ / 'inventario.json', itens)
    with zipfile.ZipFile(RAIZ / 'relatorios.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for r in itens:
            p = RAIZ / r['arquivo']
            if p.suffix in {'.json', '.csv', '.md', '.png', '.py'}:
                z.write(p, r['arquivo'])
        z.write(RAIZ / 'inventario.json', 'inventario.json')
    print('COMPARISON COMPLETE', json.dumps(resumo, ensure_ascii=False), flush=True)
    display(globais[['modelo', 'rmse', 'mae', 'vies', 'rmse_area']])
    return resumo

def executar():
    global PASTA, PASTA_SEAS5, MANIFESTO_SEAS5, PASTA_CFSV2, MANIFESTO_CFSV2, INDICES_OC, SAIDA
    SAIDA = RAIZ
    protocolo = json.loads((CODIGO / 'protocol.json').read_text(encoding='utf-8'))
    conferir_configuracao(protocolo)
    versoes = {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'xarray', 'lightgbm', 'scikit-learn', 'scipy']}
    exigir(all((versoes[k] == v for k, v in {'numpy': '2.0.2', 'pandas': '2.3.3', 'xarray': '2025.12.0', 'lightgbm': '4.6.0'}.items())), 'Use the reference environment; current versions: ' + str(versoes))
    PASTA = localizar_dados(None)
    PASTA_SEAS5, MANIFESTO_SEAS5 = localizar_seas5()
    PASTA_CFSV2, MANIFESTO_CFSV2 = localizar_cfsv2()
    entradas = []
    for r in json.loads((CODIGO / 'official_hashes.json').read_text()):
        if not r['nome'].startswith('treino_'):
            continue
        p = PASTA / r['nome']
        exigir(p.is_file() and sha256(p) == r['sha256'], 'Official file changed: ' + r['nome'])
        entradas.append(r)
    ssts = sorted(Path('/kaggle/input').rglob(SST_NOME))
    exigir(ssts and all((sha256(p) == SST_HASH for p in ssts)), 'Missing or altered SST snapshot.')
    sst, meta_sst = sst_auditar_arquivo(ssts[0])
    sst = sst.sel(time=slice(None, '2020-10-01'))
    assinatura = dict(protocolo_sha256=sha256(CODIGO / 'protocol.json'), codigo={p.name: sha256(p) for p in sorted(CODIGO.iterdir()) if p.is_file()}, entradas=entradas, versoes=versoes, sst_sha256=SST_HASH, manifesto_seas5=sha256(PASTA_SEAS5 / 'seas5_51_manifesto.json'), manifesto_cfsv2=sha256(PASTA_CFSV2 / 'cfsv2_manifesto.json'))
    if (RAIZ / 'assinatura_execucao.json').exists():
        exigir(json.loads((RAIZ / 'assinatura_execucao.json').read_text()) == assinatura, 'Code, inputs or environment changed; use a separate output directory.')
    else:
        registrar(RAIZ / 'assinatura_execucao.json', assinatura)
    registrar(RAIZ / 'sst_fonte.json', meta_sst)
    print('Lags 4/3/2; training ends at T-4. Inputs and protocol verified.', flush=True)
    chuva = auditar_chuva_desenv()
    atmosfera, meta_at = carregar_atmosfera(chuva, pd.Timestamp('1976-06-01'), pd.Timestamp('2020-08-01'))
    INDICES_OC = pd.read_csv(CODIGO / 'ocean_indices.csv', parse_dates=['time_origem']).set_index('time_origem')
    conferir_indices(INDICES_OC)
    seas = carregar_seas5('desenvolvimento', chuva)
    cfs = carregar_cfsv2('desenvolvimento', chuva)
    registrar(RAIZ / 'atmosfera_fontes.json', meta_at)
    todos = []
    for b in BLOCOS:
        inicio = pd.Timestamp(b['inicio'])
        corte = origem_mensal([inicio], LAG_CHUVA_TREINO)[0]
        datas = pd.date_range(inicio, periods=24, freq='MS')
        pasta = RAIZ / b['nome']
        pasta.mkdir(exist_ok=True)
        if (pasta / 'concluido.json').exists():
            check = json.loads((pasta / 'concluido.json').read_text())
            exigir(check['assinatura'] == sha256(RAIZ / 'assinatura_execucao.json'), 'Block signature changed.')
            for nome, h in check['arquivos'].items():
                exigir(sha256(pasta / nome) == h, 'Cache changed: ' + nome)
            todos.append(pd.read_csv(pasta / 'metricas.csv'))
            print(b['nome'], 'cache verified.', flush=True)
            continue
        calendario_emissoes(datas, corte).to_csv(pasta / 'calendario_emissoes.csv', index=False)
        chuva_treino = chuva.sel(time=slice(None, corte))
        X, y, ids, alvos, clima, ca, somas, contagens, cs, cc = amostrar_arvores_multissistema(chuva_treino, atmosfera, seas, cfs, corte)
        exigir(alvos[-1] == corte and alvos.max() <= origem_mensal([datas[0]], 4)[0], 'Labels extend beyond the cutoff.')
        pca = sst_ajustar_pca(sst, alvos)
        pcs = sst_transformar(sst, pca, alvos)
        pcv = sst_transformar(sst, pca, datas)
        auditoria = auditar_invariancia_defasada(chuva, atmosfera, seas, cfs, sst, corte, datas, clima, ca, cs, cc, pca)
        auditoria.update(inicio=str(alvos[0].date()), corte=str(corte.date()), meses=len(alvos), n_exemplos=len(y), sha256_X=arr_hash(X), sha256_y=arr_hash(y), sha256_ids=arr_hash(ids), identidade_pareada='identical base features, targets, points, references and ridge; only 8 PCs are appended', referencia_chuva=[str(janela_referencia(corte)[0].date()), str(corte.date())], fontes_clima=dict(atmosfera=[str(origem_mensal(janela_referencia(corte), 4)[0].date()), str(origem_mensal(janela_referencia(corte), 4)[-1].date())], indices=[str(origem_mensal(janela_referencia(corte), 3)[0].date()), str(origem_mensal(janela_referencia(corte), 3)[-1].date())]))
        registrar(pasta / 'auditoria.json', auditoria)
        np.savez_compressed(pasta / 'pca.npz', **pca)
        np.savez_compressed(pasta / 'amostras.npz', pontos=ids, time_alvo=alvos.values)
        np.savez_compressed(pasta / 'clima_atmosfera_indices.npz', **ca)
        for nome, cl in [('chuva', clima), ('seas5', cs), ('cfsv2', cc)]:
            cl.to_netcdf(pasta / f'clima_{nome}.nc')
        fit = mos_fit(chuva_treino, seas, cfs, corte)
        np.savez_compressed(pasta / 'ridge.npz', **fit)
        ridge = mos_predict(fit, seas, cfs, datas)
        modelos, previsoes = ({}, {})
        print(f'{b['nome']} | training {alvos[0].date()} to {corte.date()} | {len(y):,} examples; four tree models and one shared ridge', flush=True)
        for tratamento in ['controle_defasado', 'sst_defasada']:
            componentes = []
            for nome, base_features, scale in [('arvores_seas5', FEATURES_SEAS5, 0.9), ('arvores_multissistema', FEATURES_MULTISSISTEMA, 0.875)]:
                t0 = perf_counter()
                nbase = len(base_features)
                entrada = np.ascontiguousarray(X[:, :nbase])
                features = list(base_features)
                if tratamento == 'sst_defasada':
                    entrada = np.column_stack([entrada, np.repeat(pcs, ids.shape[1], axis=0)]).astype(np.float32)
                    features += SST_FEATURES
                exigir(np.array_equal(entrada[:, :nbase], X[:, :nbase]), 'Base feature matrix changed.')
                data = lgb.Dataset(entrada, label=y, feature_name=features)
                model = lgb.train(PARAMETROS, data, num_boost_round=ARVORES)
                exigir(model.feature_name() == features, 'Feature order changed.')
                model.save_model(str(pasta / f'{tratamento}_{nome}.txt'))
                del data, entrada
                gc.collect()
                if tratamento == 'sst_defasada':
                    anom = pesquisa_prever_sst(model, nome, atmosfera, seas, cfs, ca, clima, cs, cc, datas, pcv)
                else:
                    anom = prever_par(model, nome, atmosfera, seas, cfs, ca, clima, cs, cc, datas)
                componentes.append(reconstruir_chuva(clima, anom, scale).values.astype(np.float64))
                print(b['nome'], tratamento, nome, f'fit and prediction: {perf_counter() - t0:.1f}s', flush=True)
                del model, anom
            previsoes[tratamento] = 0.75 * (0.5 * componentes[0] + 0.5 * componentes[1]) + 0.25 * ridge
            del componentes
        exigir(arr_hash(X) == auditoria['sha256_X'] and arr_hash(y) == auditoria['sha256_y'], 'Training data changed.')
        previsoes['climatologia'] = clima.values[datas.month - 1].astype(np.float64)
        coords = dict(time=datas, lat=chuva.lat, lon=chuva.lon)
        preds = {n: xr.DataArray(v, dims=('time', 'lat', 'lon'), coords=coords, attrs=dict(units='mm/day')) for n, v in previsoes.items()}
        ds = xr.Dataset(preds, attrs=dict(natureza='retrospective; vintages unverified', corte_treino=str(corte.date())))
        ds.to_netcdf(pasta / 'previsoes.nc', engine='h5netcdf', encoding={n: dict(zlib=True, complevel=4) for n in ds.data_vars})
        metricas = pesquisa_diagnosticos(chuva.sel(time=datas), preds, b['nome'])
        metricas.to_csv(pasta / 'metricas.csv', index=False, lineterminator='\n')
        arquivos = {p.name: sha256(p) for p in sorted(pasta.iterdir()) if p.is_file() and p.name != 'concluido.json'}
        registrar(pasta / 'concluido.json', dict(assinatura=sha256(RAIZ / 'assinatura_execucao.json'), arquivos=arquivos))
        todos.append(metricas)
        bloco = pesquisa_resumir(metricas[metricas.regiao == 'dominio_inteiro'], ['modelo'])
        print(b['nome'], 'completed:', bloco[['modelo', 'rmse']].to_dict('records'), flush=True)
        del X, y, ids, clima, ca, somas, contagens, cs, cc, pca, pcs, pcv, fit, ridge, preds, ds, previsoes
        gc.collect()
    return resultado_final(pd.concat(todos, ignore_index=True))
if __name__ == '__main__':
    RESULTADO = executar()
