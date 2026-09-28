# Colar em uma célula ANTES da célula de downloads que falhou.
# Executar esta célula; depois a célula de downloads e as duas células seguintes.
# Mantém os downloads e as verificações de hash existentes.
necessarios = ['np','pd','xr','exigir','baixar','carregar_membros','sha256','SAIDA','json_salvar']
faltantes = [nome for nome in necessarios if nome not in globals()]
if faltantes:
    raise RuntimeError('Execute antes as células iniciais do notebook: ' + ', '.join(faltantes))
POLITICA_MEMBROS = 'ensemble_completo_ou_excecao_201908_m17_auditada_v2'
EXCECOES_MEMBROS = {'2019-08-01': (17,)}

def membros_utilizaveis(membros, origem, info):
    esperados = info['mascara']
    exigir(np.isnan(membros[~esperados]).all(), f'Membro fora da tabela com dados em {origem}.')
    completos = np.isfinite(membros).all(axis=(-1,-2))
    faltantes = tuple(int(i+1) for i in np.flatnonzero(esperados & ~completos))
    if faltantes:
        exigir(faltantes == EXCECOES_MEMBROS.get(str(origem.date())),
               f'Membros ausentes/parciais não previstos em {origem.date()}: {faltantes}.')
        exigir(np.isnan(membros[np.array(faltantes)-1]).all(),
               'A exceção exige membro inteiramente ausente; lacunas parciais ou infinitos não são aceitos.')
    usados = esperados & completos
    exigir((membros[usados] >= 0).all(), f'Precipitação negativa em {origem.date()}. Investigar.')
    datas = info['datas'][usados[esperados]]
    return usados, datas, faltantes


def processar_bloco(path, origens, tabela):
    origens = pd.DatetimeIndex(origens)
    membros = carregar_membros(path, origens)
    hash_bruto = sha256(path)
    medias, linhas, datas_min, datas_max, ns, fases = [], [], [], [], [], []
    for k, origem in enumerate(origens):
        info = tabela[origem]
        validos, datas, faltantes = membros_utilizaveis(membros[k], origem, info)
        v = membros[k,validos]
        exigir(np.isfinite(v).all(), f'Membro previsto ausente/parcial em {origem}. Não usar média parcial.')
        exigir((v >= 0).all(), f'Precipitação negativa em {origem}: mínimo {v.min():.12g} mm/day. Investigar.')
        media = v.mean(axis=0, dtype=np.float64).astype(np.float32)
        alvo = origem+pd.offsets.MonthBegin(1)
        medias.append(media); ns.append(int(validos.sum()))
        datas_min.append(datas.min()); datas_max.append(datas.max())
        fases.append(0 if info['fase_fonte'] == 'hindcast' else 1)
        linhas.append(dict(time_alvo=str(alvo.date()), time_origem=str(origem.date()),
            lead=LEAD, membros=ns[-1], membro_ids=','.join(map(str,np.flatnonzero(validos)+1)),
            membros_esperados=int(info['mascara'].sum()), membros_indisponiveis_ids=list(faltantes),
            politica_membros=POLITICA_MEMBROS,
            inicializacao_mais_antiga=str(datas.min().date()),
            inicializacao_mais_recente=str(datas.max().date()),
            inicializacoes_anteriores_ao_alvo=True, fase_fonte=info['fase_fonte'],
            minimo_membros_mm_day=float(v.min()), maximo_membros_mm_day=float(v.max()),
            minimo_media_mm_day=float(media.min()), maximo_media_mm_day=float(media.max()),
            media_espacial_mm_day=float(media.mean(dtype=np.float64)),
            membros_ausentes_estruturais=28-int(info['mascara'].sum()), pixels_ausentes_membros_validos=0,
            negativos_corrigidos=0, arquivo_bruto=path.name, sha256_bruto=hash_bruto))
    ds = xr.Dataset({
        'cfsv2_tp_media': (('time','lat','lon'), np.stack(medias)),
        'n_membros': ('time', np.array(ns,dtype=np.int16)),
        'time_origem': ('time', origens.values),
        'inicializacao_mais_antiga': ('time', pd.DatetimeIndex(datas_min).values),
        'inicializacao_mais_recente': ('time', pd.DatetimeIndex(datas_max).values),
        'fase_fonte': ('time',np.array(fases,dtype=np.int8))},
        coords={'time': (origens.to_period('M')+1).to_timestamp(), 'lat': LAT, 'lon': LON_360-360},
        attrs=dict(schema=SCHEMA, fonte=FONTE, modelo='NCEP-CFSv2',
            amostragem='PENTAD_SAMPLES_FULL; 24 membros, 28 em novembro',
            produto='previsao mensal; media aritmetica local dos membros completos disponiveis',
            politica_membros=POLITICA_MEMBROS,
            lead_iri=LEAD, referencia_temporal='S nominal no mes anterior ao alvo; datas reais auditadas',
            observacoes_de_chuva_utilizadas='nenhuma', climatologia_aplicada='nenhuma',
            grade='nativa de 1 grau; interpolacao fica para o notebook de modelagem'))
    ds.cfsv2_tp_media.attrs = dict(units='mm/day', long_name='Precipitacao prevista: media dos membros CFSv2')
    ds.lat.attrs['units'], ds.lon.attrs['units'] = 'degrees_north', 'degrees_east'
    ds.fase_fonte.attrs = dict(flag_values=np.array([0,1],dtype=np.int8), flag_meanings='hindcast operacional')
    return ds, linhas


def processar_agregados(path_media, path_cobertura, origens, tabela):
    origens = pd.DatetimeIndex(origens)
    with xr.open_dataset(path_media, decode_times=False, engine='netcdf4') as d:
        media_ds = d.load()
    with xr.open_dataset(path_cobertura, decode_times=False, engine='netcdf4') as d:
        cobertura = d.load()
    for d in (media_ds, cobertura):
        exigir(set(d.data_vars) == {'prec'}, 'Variável de resposta inesperada.')
        exigir(meses_de_S(d.S).equals(origens), 'Datas da média ou cobertura divergentes.')
        exigir(d.sizes['L'] == 1 and float(d.L.values[0]) == LEAD, 'Lead da resposta divergente.')
        exigir(d.L.attrs.get('units') == 'months' and float(d.L.attrs.get('pointwidth',-1)) == 1,
               'Produto não é uma previsão mensal.')
    exigir(set(media_ds.prec.dims) == {'S','L','Y','X'}, 'Dimensões da média divergentes.')
    exigir(media_ds.prec.attrs.get('units') == 'mm/day', 'Unidade da média divergente.')
    exigir(np.array_equal(media_ds.Y.values,LAT) and np.array_equal(media_ds.X.values,LON_360),
           'Grade da média divergente.')
    exigir(set(cobertura.prec.dims) == {'S','L','M'} and
           np.array_equal(cobertura.M.values,np.arange(1,29)), 'Cobertura por membro divergente.')
    # O atributo de unidade herdado pela expressão de cobertura não é interpretado como chuva.
    # 0*prec+1 vale 1 em cada pixel válido; a soma espacial conta pixels, não precipitação.
    contagens = cobertura.prec.transpose('S','L','M').values[:,0].astype(np.float64)
    medias = media_ds.prec.transpose('S','L','Y','X').values[:,0].astype(np.float64)
    exigir(np.isfinite(medias).all() and (medias >= 0).all(), 'Média negativa, ausente ou não finita.')
    linhas, ns, mins, maxs, fases = [], [], [], [], []
    hm, hc = sha256(path_media), sha256(path_cobertura)
    for k, origem in enumerate(origens):
        info = tabela[origem]; esperados = info['mascara']; validos = esperados.copy(); datas = info['datas']
        ruins = tuple(int(i+1) for i in np.flatnonzero(esperados & (contagens[k] != len(LAT)*len(LON_360))))
        excecao = None
        if ruins:
            exigir(ruins == EXCECOES_MEMBROS.get(str(origem.date())),
                   f'Cobertura incompleta não prevista em {origem.date()}: membros {ruins}.')
            print(f'{origem:%Y-%m}: conferindo exceção do membro 17 nos campos individuais...', flush=True)
            path_raw, meta_raw = baixar(url_campos(pd.DatetimeIndex([origem])),
                                       f'cfsv2_excecao_membros_{origem:%Y%m}.nc')
            raw = carregar_membros(path_raw, pd.DatetimeIndex([origem]))[0]
            validos, datas, faltantes = membros_utilizaveis(raw, origem, info)
            exigir(faltantes == ruins, 'Campos individuais e contagens de cobertura discordam.')
            media_local = raw[validos].mean(axis=0, dtype=np.float64)
            exigir(np.allclose(media_local, medias[k], rtol=2e-7, atol=2e-6),
                   'Média da IRI diverge da média dos 23 membros completos. Parar e investigar.')
            excecao = dict(membros_indisponiveis_ids=list(faltantes),
                arquivo=str(path_raw.relative_to(SAIDA)), **meta_raw,
                maximo_delta_media_mm_day=float(np.max(np.abs(media_local-medias[k]))))
        vazios = contagens[k,~esperados]
        exigir(np.all(np.isnan(vazios) | (vazios == 0)), 'Membro inesperado contém pixels válidos.')
        ns.append(int(validos.sum())); mins.append(datas.min()); maxs.append(datas.max())
        fases.append(0 if info['fase_fonte']=='hindcast' else 1)
        linhas.append(dict(time_alvo=str((origem+pd.offsets.MonthBegin(1)).date()),
            time_origem=str(origem.date()), lead=LEAD, membros=ns[-1],
            membros_esperados=int(esperados.sum()), membros_indisponiveis_ids=list(ruins),
            politica_membros=POLITICA_MEMBROS, auditoria_excecao=excecao,
            membro_ids=','.join(map(str,np.flatnonzero(validos)+1)),
            inicializacao_mais_antiga=str(datas.min().date()), inicializacao_mais_recente=str(datas.max().date()),
            inicializacoes_anteriores_ao_alvo=True, fase_fonte=info['fase_fonte'],
            pixels_validos_por_membro=len(LAT)*len(LON_360), membros_com_cobertura_completa=ns[-1],
            minimo_media_mm_day=float(medias[k].min()), maximo_media_mm_day=float(medias[k].max()),
            media_espacial_mm_day=float(medias[k].mean()), negativos_corrigidos=0,
            arquivo_media=path_media.name, sha256_media=hm,
            arquivo_cobertura=path_cobertura.name, sha256_cobertura=hc))
    ds=xr.Dataset({
        'cfsv2_tp_media': (('time','lat','lon'),medias.astype(np.float32)),
        'n_membros': ('time',np.array(ns,dtype=np.int16)),
        'time_origem': ('time',origens.values),
        'inicializacao_mais_antiga': ('time',pd.DatetimeIndex(mins).values),
        'inicializacao_mais_recente': ('time',pd.DatetimeIndex(maxs).values),
        'fase_fonte': ('time',np.array(fases,dtype=np.int8))},
        coords={'time':(origens.to_period('M')+1).to_timestamp(),'lat':LAT,'lon':LON_360-360},
        attrs=dict(schema=SCHEMA,fonte=FONTE,modelo='NCEP-CFSv2',lead_iri=LEAD,
            produto='media aritmetica dos membros calculada pela IRI; cobertura completa auditada por membro',
            politica_membros=POLITICA_MEMBROS,
            amostragem='PENTAD_SAMPLES_FULL; 24 membros, 28 em novembro',
            observacoes_de_chuva_utilizadas='nenhuma',climatologia_aplicada='nenhuma',
            grade='nativa de 1 grau; sem interpolacao'))
    ds.cfsv2_tp_media.attrs=dict(units='mm/day',long_name='Precipitacao prevista: media dos membros CFSv2')
    ds.lat.attrs['units'],ds.lon.attrs['units']='degrees_north','degrees_east'
    ds.fase_fonte.attrs=dict(flag_values=np.array([0,1],dtype=np.int8),flag_meanings='hindcast operacional')
    return ds,linhas

# Inclui a política também no manifesto gerado pela célula final da versão anterior.
if '_json_salvar_base_cfsv2_2019' not in globals():
    _json_salvar_base_cfsv2_2019 = json_salvar

def json_salvar(path, obj):
    if Path(path).name == 'cfsv2_manifesto.json':
        obj = dict(obj, politica_membros=POLITICA_MEMBROS,
            excecoes_permitidas=EXCECOES_MEMBROS,
            excecoes_aplicadas=[r for r in AUDITORIA.to_dict('records')
                               if r.get('membros_indisponiveis_ids')])
    return _json_salvar_base_cfsv2_2019(path, obj)

print('Correção aplicada. Reexecute a célula de downloads e depois as duas células seguintes.')
print('O cache será reaproveitado. Agosto/2019 terá uma conferência adicional dos membros individuais.')
