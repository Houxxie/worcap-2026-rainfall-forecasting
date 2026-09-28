"""Política temporal candidata; não certifica vintages nem datas de publicação."""
LAG_ATMOSFERA = 4
LAG_INDICES = 3
LAG_SST = 2
LAG_CHUVA_TREINO = 4
INICIO_COMUM = pd.Timestamp('1982-03-01')
FEATURES = ([v + '_lag4' for v in VARIAVEIS]
    + [v + '_lag4_anomalia' for v in VARIAVEIS]
    + ['latitude', 'longitude', 'mes_alvo_sin', 'mes_alvo_cos', 'clima_tp_alvo']
    + [n + '_lag3_anomalia_fold' for n in INDICES_NOMES])
FEATURES_SEAS5 = FEATURES + [FEATURE_SEAS5, FEATURE_SEAS5_ANOM]
FEATURES_MULTISSISTEMA = FEATURES_SEAS5 + [FEATURE_CFSV2, FEATURE_CFSV2_ANOM]
SST_FEATURES = [f'sst_lag2_pc_{i:02d}_treino_fold' for i in range(1, 9)]


def origem_mensal(alvos, lag):
    datas = pd.DatetimeIndex(alvos)
    exigir(lag >= 1 and datas.equals(datas.to_period('M').to_timestamp()), 'Datas/lag inválidos.')
    return (datas.to_period('M') - lag).to_timestamp()


def janela_referencia(corte, anos=30):
    corte = pd.Timestamp(corte)
    exigir(corte.day == 1, 'Corte precisa identificar um mês.')
    return pd.date_range(end=corte, periods=12 * anos, freq='MS')


def calendario_pareado(corte):
    datas = janela_referencia(corte)
    datas = datas[datas >= INICIO_COMUM]
    exigir(24 <= len(datas) <= 360, 'Janela comum inválida.')
    return datas


def calendario_pares(datas_disponiveis, corte, anos=30):
    alvos = janela_referencia(corte, anos)
    origens = origem_mensal(alvos, LAG_ATMOSFERA)
    disponiveis = pd.DatetimeIndex(datas_disponiveis)
    exigir(disponiveis.is_unique and origens.isin(disponiveis).all(), 'Faltam origens atmosféricas.')
    return origens, alvos


def estatisticas_climatologia(tp, corte, anos=30):
    datas = janela_referencia(corte, anos)
    treino = tp.sel(time=datas).transpose('time', 'lat', 'lon')
    exigir(np.isfinite(treino.values).all(), 'Chuva de treino incompleta.')
    somas = np.stack([treino.values[datas.month == m].sum(axis=0, dtype=np.float64)
                      for m in range(1, 13)])
    contagens = np.array([(datas.month == m).sum() for m in range(1, 13)], dtype=np.int64)
    exigir((contagens == anos).all(), 'Climatologia precisa de todos os meses.')
    clima = xr.DataArray((somas / contagens[:, None, None]).astype(np.float32),
        dims=('month', 'lat', 'lon'), coords=dict(month=np.arange(1, 13), lat=tp.lat, lon=tp.lon),
        attrs=dict(units='mm/day', ajuste_inicio=str(datas[0].date()), ajuste_fim=str(datas[-1].date())))
    return clima, somas, contagens


def ajustar_clima_indices(tabela, origens_atmosfera, corte):
    # Calendário próprio: índices T−3, enquanto atmosfera utiliza T−4.
    alvos = janela_referencia(corte)
    exigir(pd.DatetimeIndex(origens_atmosfera).equals(origem_mensal(alvos, LAG_ATMOSFERA)),
           'Referência atmosférica fora do protocolo.')
    origens = origem_mensal(alvos, LAG_INDICES)
    valores = tabela.loc[origens, INDICES_NOMES].to_numpy(dtype=np.float64)
    exigir(np.isfinite(valores).all(), 'Índices de treino incompletos.')
    return np.stack([valores[origens.month == m].mean(axis=0, dtype=np.float64) for m in range(1, 13)])


def valores_indices_mes(tabela, clima_indices, destino):
    origem = origem_mensal([destino], LAG_INDICES)[0]
    valores = tabela.loc[origem, INDICES_NOMES].to_numpy(dtype=np.float64)
    exigir(np.isfinite(valores).all(), 'Índice indisponível; não preencher.')
    return (valores - clima_indices[origem.month - 1]).astype(np.float32)


def exigir_disponibilidade_real(registros, emitido_em, fontes_necessarias):
    """Porta para futura produção; não é usada para certificar este retrospectivo.

    O registro precisa se referir aos bytes recebidos, não à data nominal da previsão.
    Valida metadados fornecidos; sua autenticidade exige um coletor auditável.
    """
    emissao = pd.Timestamp(emitido_em)
    exigir(emissao.tzinfo is not None, 'Emissão exige timezone.')
    exigir(len(registros) == len(fontes_necessarias), 'Quantidade de fontes incorreta.')
    exigir({r.get('fonte') for r in registros} == set(fontes_necessarias), 'Fonte faltante/duplicada.')
    for r in registros:
        for chave in ['disponivel_em', 'recebido_em', 'url', 'sha256', 'versao']:
            exigir(bool(r.get(chave)), 'Disponibilidade desconhecida: ' + r.get('fonte', '?') + '/' + chave)
        recebido, publicado = pd.Timestamp(r['recebido_em']), pd.Timestamp(r['disponivel_em'])
        exigir(recebido.tzinfo is not None and publicado.tzinfo is not None, 'Datas exigem timezone.')
        exigir(publicado <= recebido <= emissao, 'Fonte chegou após a emissão ou cronologia inválida.')
        exigir(len(r['sha256']) == 64 and set(r['sha256']) <= set('0123456789abcdef'), 'SHA-256 inválido.')
    return True


def calendario_emissoes(datas, corte):
    datas = pd.DatetimeIndex(datas)
    limite = origem_mensal([datas[0]], LAG_CHUVA_TREINO)[0]
    exigir(pd.Timestamp(corte) <= limite, 'Chuva de treino recente demais para a primeira emissão.')
    return pd.DataFrame(dict(time_alvo=datas, limite_emissao=datas - pd.Timedelta(nanoseconds=1),
        atmosfera=origem_mensal(datas, LAG_ATMOSFERA), indices=origem_mensal(datas, LAG_INDICES),
        sst=origem_mensal(datas, LAG_SST), inicializacao_sazonal=origem_mensal(datas, 1),
        ultimo_alvo_treino=pd.Timestamp(corte), disponibilidade_historica_comprovada=False))


def auditar_invariancia_defasada(chuva, atmosfera, seas, cfs, sst, corte, datas, clima, ca, cs, cc, pca):
    """Muta cópias pequenas reais; dados após cada limite não podem afetar T."""
    primeiro = pd.Timestamp(datas[0])
    pontos = np.array([0, 1, chuva.sizes['lon']], dtype=np.int32)
    # Recorte espacial pequeno mantém todas as datas para testar os ajustes.
    ch = chuva.isel(lat=slice(0, 2), lon=slice(0, 2)).copy(deep=True)
    at = {k: v.isel(lat=slice(0, 2), lon=slice(0, 2)).copy(deep=True) for k, v in atmosfera.items()}
    alvos_ref = janela_referencia(corte)
    orig_at = origem_mensal(alvos_ref, LAG_ATMOSFERA)
    c0, _, _ = estatisticas_climatologia(ch, corte)
    a0 = ajustar_clima_atmosfera(at, orig_at)
    a0['_clima_indices'] = ajustar_clima_indices(INDICES_OC, orig_at, corte)
    x0 = matriz_mes(at, a0, c0, primeiro)
    ch.values[pd.DatetimeIndex(ch.time.values) > pd.Timestamp(corte)] = 123456
    c1, _, _ = estatisticas_climatologia(ch, corte)
    exigir(np.array_equal(c0.values, c1.values), 'Chuva após corte alterou climatologia.')
    for v in at.values():
        v.values[pd.DatetimeIndex(v.time.values) > origem_mensal([primeiro], LAG_ATMOSFERA)[0]] = 7654321
    a1 = ajustar_clima_atmosfera(at, orig_at)
    a1['_clima_indices'] = a0['_clima_indices']
    exigir(all(np.array_equal(a0[k], a1[k]) for k in a0), 'Futuro alterou climatologia atmosférica.')
    exigir(np.array_equal(x0, matriz_mes(at, a1, c1, primeiro)), 'Atmosfera recente/futura alterou T.')
    ni = INDICES_OC.copy(deep=True)
    ni.loc[ni.index > origem_mensal([primeiro], LAG_INDICES)[0], :] = 99999
    ci = ajustar_clima_indices(ni, orig_at, corte)
    exigir(np.array_equal(ci, a0['_clima_indices']), 'Índices futuros alteraram ajuste.')
    exigir(np.array_equal(valores_indices_mes(ni, ci, primeiro), x0[0, 23:]), 'Índice futuro alterou feature.')
    for pre, nome in [(seas, 'SEAS5'), (cfs, 'CFSv2')]:
        pequeno = pre.isel(lat=slice(0, 2), lon=slice(0, 2)).copy(deep=True)
        antes = ajustar_clima_previsao(pequeno, corte, nome)
        pequeno.values[pd.DatetimeIndex(pequeno.time.values) > pd.Timestamp(corte)] = 12345
        exigir(np.array_equal(antes, ajustar_clima_previsao(pequeno, corte, nome)), 'Futuro alterou clima sazonal.')
    copia = sst.copy(deep=True)
    copia.values[pd.DatetimeIndex(copia.time.values) > origem_mensal([primeiro], LAG_SST)[0]] = 12345
    novo = sst_ajustar_pca(copia, calendario_pareado(corte))
    for k in ['mascara', 'clima_mensal', 'centro', 'componentes', 'origens_treino']:
        exigir(np.array_equal(novo[k], pca[k]), 'SST futura alterou PCA: ' + k)
    exigir(np.array_equal(sst_transformar(copia, novo, [primeiro]), sst_transformar(sst, pca, [primeiro])),
           'SST mais recente que T−2 alterou previsão de T.')
    return dict(chuva_pos_corte_invariante=True, atmosfera_apos_Tmenos4_invariante=True,
        indices_apos_Tmenos3_invariantes=True, sst_apos_Tmenos2_invariante=True,
        climas_sazonais_somente_treino=True, publicacao_historica_comprovada=False)
