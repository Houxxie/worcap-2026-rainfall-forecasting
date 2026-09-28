"""Extensão da reprodução: mantém amostras, referências, árvores e mistura fixas.

Os modelos históricos do controle são carregados da execução recém-verificada.
Somente os dois componentes de árvores com SST são treinados novamente.
O componente ridge permanece exatamente igual nos dois tratamentos.
"""

def pesquisa_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def pesquisa_prever_sst(modelo, nome, atmosfera, seas, cfs, ca, clima, cs, cc, datas, scores):
    datas = pd.DatetimeIndex(datas)
    exigir((datas > pd.Timestamp(cs.attrs['corte'])).all(), 'Inferência invade treino.')
    exigir(scores.shape == (len(datas), 8), 'Scores SST desalinhados.')
    out = np.empty((len(datas), clima.sizes['lat'], clima.sizes['lon']), dtype=np.float32)
    expected = (FEATURES_SEAS5 if nome == 'arvores_seas5' else FEATURES_MULTISSISTEMA) + SST_FEATURES
    exigir(modelo.feature_name() == expected, 'Ordem de features SST incorreta.')
    for i, t in enumerate(datas):
        x = np.column_stack([matriz_mes(atmosfera, ca, clima, t),
            valores_seas5(seas, t), valores_anomalia_seas5(seas, cs, t)]).astype(np.float32)
        if nome == 'arvores_multissistema':
            x = np.column_stack([x, valores_cfsv2(cfs, t), valores_anomalia_cfsv2(cfs, cc, t)]).astype(np.float32)
        x = np.column_stack([x, np.broadcast_to(scores[i], (len(x), 8))]).astype(np.float32)
        exigir(x.shape[1] == len(expected) and np.isfinite(x).all(), 'Features SST inválidas.')
        out[i] = modelo.predict(x).reshape(out.shape[1:])
    return xr.DataArray(out, dims=('time', 'lat', 'lon'),
        coords=dict(time=datas, lat=clima.lat, lon=clima.lon), attrs=dict(units='mm/day'))


def pesquisa_diagnosticos(obs, previsoes, bloco):
    """Dados completos, sem amostragem de pixels e sem mascarar erros."""
    linhas = []
    lat = obs.lat.values
    regioes = {'dominio_inteiro': np.ones(lat.size, dtype=bool),
        'lat_menor_que_menos35': lat < -35,
        'lat_menos35_a_menos15': (lat >= -35) & (lat < -15),
        'lat_maior_igual_menos15': lat >= -15}
    for modelo, pred in previsoes.items():
        obs, pred = xr.align(obs, pred, join='exact')
        for i, t in enumerate(pd.DatetimeIndex(obs.time.values)):
            erro = pred.values[i].astype(np.float64) - obs.values[i].astype(np.float64)
            exigir(np.isfinite(erro).all(), 'Erro não finito nos diagnósticos.')
            for regiao, mask in regioes.items():
                if not mask.any():
                    continue
                e = erro[mask]
                w = np.broadcast_to(np.cos(np.deg2rad(lat[mask]))[:, None], e.shape)
                linhas.append(dict(modelo=modelo, bloco=bloco, mes_alvo=str(t.date()),
                    ano=t.year, mes=t.month, regiao=regiao, n=e.size,
                    sse=float(np.square(e).sum()), soma_erro=float(e.sum()),
                    soma_erro_absoluto=float(np.abs(e).sum()),
                    sse_area=float((w * np.square(e)).sum()), peso_area=float(w.sum())))
    return pd.DataFrame(linhas)


def pesquisa_resumir(mensais, grupos):
    cols = ['n', 'sse', 'soma_erro', 'soma_erro_absoluto', 'sse_area', 'peso_area']
    r = mensais.groupby(grupos, as_index=False)[cols].sum()
    r['rmse'] = np.sqrt(r.sse / r.n)
    r['mae'] = r.soma_erro_absoluto / r.n
    r['vies'] = r.soma_erro / r.n
    r['rmse_area'] = np.sqrt(r.sse_area / r.peso_area)
    return r


def pesquisa_sst_executar():
    global INDICES_OC
    conferir_protocolo()
    exigir('h5netcdf' in xr.backends.list_engines(), 'Backend h5netcdf necessário para salvar as previsões.')
    exigir(REEXECUTAR_VALIDACAO and BANCA_NUMERICAMENTE_CONFIRMADO, 'Primeiro confirme a referência histórica e numérica.')
    ref_path = BANCA_SAIDA / 'validacao_reexecutada_mensal.csv'
    exigir(ref_path.is_file(), 'Métricas reexecutadas do controle ausentes.')
    ref_mensais = pd.read_csv(ref_path)
    ref_global = agregar_metricas(ref_mensais, ['modelo'])
    exigir(len(ref_mensais) == 840, 'Validação de referência incompleta.')
    exigir(abs(float(ref_global.loc[ref_global.modelo == 'solucao_hibrida', 'rmse'].iloc[0])
        - 1.7489511692458355) < 1e-7, 'A referência híbrida não foi reproduzida.')
    candidatos = sorted(Path('/kaggle/input').rglob(SST_NOME))
    exigir(candidatos and all(sha256(p) == SST_HASH for p in candidatos), 'Snapshot SST ausente ou diferente.')
    sst, meta = sst_auditar_arquivo(candidatos[0])
    sst = sst.sel(time=slice('1982-01-01', '2020-11-01'))
    stamp = pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%SZ')
    retomar = globals().get('PESQUISA_RETOMAR')
    saida = Path(retomar) if retomar else Path('/kaggle/working') / 'nimbus_pesquisa_sst' / stamp
    if retomar:
        exigir((saida/'protocolo.json').is_file(), 'Retomada sem protocolo registrado.')
    else:
        saida.mkdir(parents=True, exist_ok=False)
    protocolo = dict(experimento='Hibrido com oito PCs SST nos dois componentes de arvores',
        controle=str(BANCA_SAIDA), hash_csv_controle=BANCA_AUDITORIA['submissao_sha256'],
        auditoria_reproducao=AUDITORIA_NUMERICA,
        snapshot_sst=meta, periodo_avaliacao=['2007-01-01', '2020-12-01'], blocos=BLOCOS,
        anos_maximos=30, pontos_por_mes=5000, semente=42,
        features=[37,39], parametros_arvores=PARAMETROS, arvores=300,
        pesos=[0.375,0.375,0.25], escalas_residuais=BANCA_PESOS,
        pca=dict(componentes=8, solver='full', ponderacao='sqrt(cos(latitude))',
            anomalias='mensais; mascara, medias e PCA somente no treino',
            padronizacao_por_desvio=False, lag_meses=1),
        controle_ridge='mesmos coeficientes e previsoes nos dois tratamentos',
        busca_hiperparametros=False, busca_pesos=False, holdout_independente=False,
        aviso_temporal='Retrospectivo: disponibilidade operacional e revisoes historicas nao comprovadas.',
        ambiente={**VERSOES, 'scikit-learn': importlib.metadata.version('scikit-learn'),
            'scipy': importlib.metadata.version('scipy')})
    if retomar:
        exigir(json.loads((saida/'protocolo.json').read_text()) == json.loads(json.dumps(protocolo,default=str)),
            'Protocolo diferente: não reutilizar modelos desta pasta.')
    else:
        pesquisa_json(saida / 'protocolo.json', protocolo)
    print('PESQUISA SST: protocolo salvo antes dos resultados; oito PCs e mistura fixa.', flush=True)
    tp = auditar_chuva_desenv()
    campos, _ = carregar_atmosfera(tp, pd.Timestamp('1976-12-01'), pd.Timestamp('2020-11-01'))
    indices, _ = carregar_indices_incorporados()
    INDICES_OC = indices.loc['1976-12-01':'2020-11-01'].copy()
    del indices
    seas = carregar_seas5('desenvolvimento', tp)
    cfs = carregar_cfsv2('desenvolvimento', tp)
    metricas, diagnos, auditorias = [], [], []
    for b in BLOCOS:
        nome, corte = b['nome'], b['corte']
        datas = pd.date_range(b['inicio'], periods=24, freq='MS')
        pasta = saida / nome
        pasta.mkdir(exist_ok=bool(retomar))
        ref_pasta = BANCA_SAIDA / 'modelos' / nome
        X, y, ids, alvos, clima, ca, somas, contagens, cs, cc = amostrar_arvores_multissistema(tp, campos, seas, cfs, corte)
        hashes = dict(sha256_29_features=hashlib.sha256(np.ascontiguousarray(X[:,:29]).tobytes()).hexdigest(),
            sha256_y=hashlib.sha256(y.tobytes()).hexdigest(), sha256_pontos=hashlib.sha256(ids.tobytes()).hexdigest())
        ref_audit = json.loads((ref_pasta / f'auditoria_par_{nome}.json').read_text())
        exigir(all(ref_audit[k] == v for k,v in hashes.items()), 'Amostras diferentes da referência: ' + nome)
        estado = sst_ajustar_pca(sst, alvos)
        causal = sst_auditar_causalidade(sst, alvos, datas, estado)
        pcs_treino = sst_transformar(sst, estado, alvos)
        pcs_valid = sst_transformar(sst, estado, datas)
        np.savez_compressed(pasta / 'pca.npz', **estado)
        for rotulo, ts, pcs in [('treino',alvos,pcs_treino),('validacao',datas,pcs_valid)]:
            tab = pd.DataFrame(pcs, columns=SST_FEATURES)
            tab.insert(0,'time_alvo',ts)
            tab.insert(1,'time_origem',(pd.DatetimeIndex(ts).to_period('M')-1).to_timestamp())
            tab.to_csv(pasta / f'pca_scores_{rotulo}.csv',index=False)
        registro = dict(bloco=nome, meses=len(alvos), exemplos=len(X), **hashes,
            variancia_explicada=estado['variancia_explicada'].tolist(), auditoria_temporal=causal)
        pesquisa_json(pasta / 'auditoria.json', registro)
        auditorias.append(registro)
        controles, candidatos_prev = [], []
        for comp, fs in [('arvores_seas5',FEATURES_SEAS5),('arvores_multissistema',FEATURES_MULTISSISTEMA)]:
            base = lgb.Booster(model_file=str(ref_pasta / f'modelo_{comp}_{nome}.txt'))
            anom = prever_par(base, comp, campos, seas, cfs, ca, clima, cs, cc, datas)
            controles.append(reconstruir_chuva(clima, anom, BANCA_PESOS[comp]))
            del base, anom
            inicio = perf_counter()
            arquivo_modelo = pasta / f'{comp}_sst.txt'
            if retomar and arquivo_modelo.is_file():
                modelo = lgb.Booster(model_file=str(arquivo_modelo))
                exigir(modelo.feature_name() == fs+SST_FEATURES and modelo.current_iteration() == 300,
                    'Modelo parcial incompatível com a retomada.')
                print(f'{nome}: {comp} recuperado da execução interrompida.',flush=True)
            else:
                matriz = np.empty((len(X), len(fs)+8), dtype=np.float32)
                matriz[:,:len(fs)] = X[:,:len(fs)]
                matriz[:,len(fs):] = np.repeat(pcs_treino, ids.shape[1], axis=0)
                exigir(np.array_equal(matriz[:,:len(fs)],X[:,:len(fs)]) and np.isfinite(matriz).all(), 'Matriz SST inválida.')
                treino = lgb.Dataset(matriz, label=y, feature_name=fs+SST_FEATURES)
                modelo = lgb.train(dict(PARAMETROS), treino, num_boost_round=300)
                modelo.save_model(str(arquivo_modelo))
                del treino, matriz
            pd.DataFrame(dict(feature=modelo.feature_name(), gain=modelo.feature_importance('gain'))).to_csv(pasta / f'{comp}_importancia.csv',index=False)
            gc.collect()
            anom = pesquisa_prever_sst(modelo, comp, campos, seas, cfs, ca, clima, cs, cc, datas, pcs_valid)
            candidatos_prev.append(reconstruir_chuva(clima, anom, BANCA_PESOS[comp]))
            print(f'{nome}: {comp} + SST concluído em {perf_counter()-inicio:.1f} s.', flush=True)
            del modelo, anom
        with np.load(ref_pasta / 'coeficientes_ridge.npz', allow_pickle=False) as z:
            ajuste = {k:z[k] for k in z.files}
        ajuste['corte'] = str(ajuste['corte'].item())
        ridge = controles[0].copy(data=mos_predict(ajuste, seas, cfs, datas))
        controle = .75*(.5*controles[0].astype(np.float64)+.5*controles[1].astype(np.float64))+.25*ridge
        candidato = .75*(.5*candidatos_prev[0].astype(np.float64)+.5*candidatos_prev[1].astype(np.float64))+.25*ridge
        ref_mes = ref_mensais[(ref_mensais.modelo=='solucao_hibrida') & (ref_mensais.bloco==nome)].sort_values('mes_alvo')
        m_controle = metricas_mensais(tp.sel(time=datas), controle, 'controle', nome, corte)
        exigir(np.allclose(m_controle.rmse,ref_mes.rmse,rtol=0,atol=1e-7), 'Predições de controle divergentes: '+nome)
        m_sst = metricas_mensais(tp.sel(time=datas), candidato, 'sst_8pcs', nome, corte)
        metricas.extend([m_controle,m_sst])
        todas = pd.concat(metricas, ignore_index=True)
        todas.to_csv(saida / 'metricas_mensais.csv',index=False)
        climatologia = xr.concat([clima.sel(month=t.month).drop_vars('month') for t in datas],dim=pd.Index(datas,name='time'))
        dp = pesquisa_diagnosticos(tp.sel(time=datas),dict(controle=controle,sst_8pcs=candidato,climatologia=climatologia),nome)
        diagnos.append(dp)
        dp.to_csv(pasta/'diagnosticos_mensais.csv',index=False)
        # Preservar as previsões permite reavaliar áreas e extremos sem retreinar.
        ds_out = xr.Dataset(dict(controle=controle, sst_8pcs=candidato))
        ds_out.to_netcdf(pasta/'previsoes_validacao.nc',engine='h5netcdf',encoding={v:dict(zlib=True,complevel=4) for v in ds_out.data_vars})
        erro0=controle.values-tp.sel(time=datas).values.astype(np.float64)
        erro1=candidato.values-tp.sel(time=datas).values.astype(np.float64)
        xr.Dataset(dict(delta_mse=(('lat','lon'),(erro1**2-erro0**2).mean(axis=0))),
            coords=dict(lat=tp.lat,lon=tp.lon)).to_netcdf(pasta/'mapa_delta_mse.nc',engine='h5netcdf')
        r=agregar_metricas(pd.concat([m_controle,m_sst]),['modelo']).set_index('modelo')
        print(f'{nome}: controle={r.loc["controle","rmse"]:.9f}; SST={r.loc["sst_8pcs","rmse"]:.9f}; delta={r.loc["sst_8pcs","rmse"]-r.loc["controle","rmse"]:+.9f}',flush=True)
        del X,y,ids,alvos,clima,ca,somas,contagens,cs,cc,controles,candidatos_prev,ridge,controle,candidato,ds_out,erro0,erro1,climatologia,ajuste
        gc.collect()
    mensais = pd.concat(metricas,ignore_index=True)
    exigir(len(mensais)==336 and not mensais.duplicated(['modelo','mes_alvo']).any(),'Comparação incompleta.')
    diag = pd.concat(diagnos,ignore_index=True)
    diag.to_csv(saida/'diagnosticos_mensais.csv',index=False)
    resultados={}
    for rotulo,chaves in [('globais',['modelo']),('blocos',['modelo','bloco']),('anos',['modelo','ano']),('meses_calendario',['modelo','mes'])]:
        tab=pesquisa_resumir(diag[diag.regiao=='dominio_inteiro'],chaves)
        tab.to_csv(saida/f'metricas_{rotulo}.csv',index=False)
        resultados[rotulo]=tab
    pesquisa_resumir(diag,['modelo','regiao']).to_csv(saida/'metricas_faixas_latitude.csv',index=False)
    g=resultados['globais'].set_index('modelo')
    ganhos={}
    for escala in ['blocos','anos']:
        key='bloco' if escala=='blocos' else 'ano'
        p=resultados[escala].pivot(index=key,columns='modelo',values='rmse')
        d=p.sst_8pcs-p.controle
        ganhos[escala]=dict(melhores=int((d<0).sum()),piores=int((d>0).sum()),media_delta=float(d.mean()))
    resumo=dict(status='comparacao_completa',saida=str(saida),
        rmse_controle=float(g.loc['controle','rmse']),rmse_sst=float(g.loc['sst_8pcs','rmse']),
        delta_rmse=float(g.loc['sst_8pcs','rmse']-g.loc['controle','rmse']),
        consistencia=ganhos,holdout_independente=False,
        conclusao='Resultado exploratorio nos mesmos anos de desenvolvimento; nao comprova ganho em anos futuros.',
        protocolo=protocolo,auditorias=auditorias)
    pesquisa_json(saida/'resultado.json',resumo)
    # Figuras de diagnóstico; nenhum resultado retroalimenta os pesos.
    fig, axes = plt.subplots(1, 2, figsize=(13,4), constrained_layout=True)
    for ax,key,col,title in [(axes[0],'anos','ano','Diferença por ano'),
                             (axes[1],'meses_calendario','mes','Diferença por mês do calendário')]:
        p=resultados[key].pivot(index=col,columns='modelo',values='rmse')
        delta=p.sst_8pcs-p.controle
        ax.bar(delta.index.astype(str),delta.values,color=np.where(delta.values<0,'#218c74','#c44536'))
        ax.axhline(0,color='black',linewidth=.8)
        ax.set(title=title,ylabel='RMSE SST − RMSE controle (mm/dia)')
        ax.tick_params(axis='x',rotation=45)
    fig.suptitle('Pesquisa retrospectiva: valores negativos favorecem SST')
    fig.savefig(saida/'comparacao_temporal.png',dpi=160,bbox_inches='tight')
    plt.show()
    display(resultados['globais'][['modelo','rmse','mae','vies','rmse_area']])
    display(resultados['blocos'][['modelo','bloco','rmse','mae','vies']])
    print('PESQUISA SST CONCLUÍDA:',saida,flush=True)
    print('Delta RMSE:',resumo['delta_rmse'],'| Consistência:',ganhos)
    print('Nenhum peso foi otimizado; nenhum CSV foi submetido; avaliação retrospectiva em anos já consultados.')
    return resumo


# Libera grandes objetos do ajuste final que não participam desta comparação.
for _nome in ['tp_final','campos_final','campos_teste','SEAS_FINAL','CFS_FINAL','SEAS_TESTE','CFS_TESTE',
              'MODELOS','CLIMA','CA','CS','CC','PREVISAO_FINAL','PREVISAO_RIDGE','GRADE_ARVORES',
              'BANCA_PREVISOES','anom','previsto','mapas']:
    globals().pop(_nome,None)
gc.collect()
RESULTADO_PESQUISA_SST = pesquisa_sst_executar()
