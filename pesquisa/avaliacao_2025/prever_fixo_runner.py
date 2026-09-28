"""Usa as funções congeladas incorporadas pelo construtor do notebook."""
def prever_2025():
    global INDICES_OC
    raiz=BASE/'nimbus_avaliacao_2025'
    saida=raiz/'previsoes_congeladas'
    exigir(not (saida/'registro_previsoes.json').exists(),
        'Previsões já congeladas. Não sobrescrever ou selecionar outra execução após avaliar.')
    saida.mkdir(parents=True,exist_ok=True)
    protocolo_path=raiz/'protocolo_congelado.json'
    exigir(sha256(protocolo_path)==PROTOCOLO_2025_SHA,'Protocolo modificado.')
    piloto=json.loads((raiz/'ERA5/compatibilidade_piloto.json').read_text())
    exigir(piloto['aprovado'],'Piloto ERA5 não aprovado.')
    datas=pd.date_range('2025-01-01','2025-12-01',freq='MS')
    origens=(datas.to_period('M')-1).to_timestamp()
    entradas={}
    for nome,rel in [('atmosfera','ERA5/atmosfera_origens_2025.nc'),
        ('seas','SEAS5/seas5_2025.nc'),('cfs','CFSv2/cfsv2_2025.nc'),
        ('sst','SST/ersstv5_4graus_202411_202511.nc'),('indices','NOAA/indices_2025.csv')]:
        path=raiz/rel
        exigir(path.is_file(),'Entrada ausente: '+str(path))
        entradas[nome]=dict(arquivo=str(path),sha256=sha256(path))
    with xr.open_dataset(PASTA/'treino_tp.nc') as ds:
        tp=ds.tp.sel(time=slice('1993-01-01','2022-12-01')).transpose('time','lat','lon').load()
    exigir(tp.sizes['time']==360 and tp.attrs.get('units')=='mm/day','Treino oficial incompatível.')
    campos,_=carregar_atmosfera(tp,pd.Timestamp('1992-12-01'),pd.Timestamp('2022-11-01'))
    indices,_=carregar_indices_incorporados()
    extra=pd.read_csv(raiz/'NOAA/indices_2025.csv',parse_dates=['time_origem']).set_index('time_origem')
    exigir(extra.index.equals(origens) and extra.columns.tolist()==INDICES_NOMES,'Índices novos desalinhados.')
    INDICES_OC=pd.concat([indices,extra]); conferir_indices(INDICES_OC)
    seas=xr.concat([carregar_seas5('desenvolvimento',tp).sel(time=slice('1993-01-01',None)),
                   carregar_seas5('somente_ajuste_final',tp)],dim='time')
    cfs=xr.concat([carregar_cfsv2('desenvolvimento',tp).sel(time=slice('1993-01-01',None)),
                  carregar_cfsv2('somente_ajuste_final',tp)],dim='time')
    X,y,ids,alvos,clima,ca,somas,contagens,cs,cc=amostrar_arvores_multissistema(tp,campos,seas,cfs,CORTE_FINAL)
    exigir(len(alvos)==360 and X.shape==(1800000,31),'Calendário ou tamanho do treino divergente.')
    ajuste=mos_fit(tp,seas,cfs,CORTE_FINAL)
    np.savez_compressed(saida/'ridge.npz',**ajuste)
    snapshots=list(Path('/kaggle/input').rglob(SST_NOME))
    exigir(len(snapshots)>0 and all(sha256(p)==SST_HASH for p in snapshots),'Snapshot SST do treino alterado.')
    sst,_=sst_auditar_arquivo(snapshots[0])
    estado=sst_ajustar_pca(sst,alvos)
    pcs=sst_transformar(sst,estado,alvos)
    np.savez_compressed(saida/'pca.npz',**estado)
    with xr.open_dataset(raiz/'SST/ersstv5_4graus_202411_202511.nc') as ds:
        novo_sst=ds.sst.load()
    pcs_pred=sst_transformar(novo_sst,estado,datas)
    with xr.open_dataset(raiz/'ERA5/atmosfera_origens_2025.nc') as ds:
        atm={v:ds[v].transpose('time','lat','lon').astype('float32').load() for v in VARIAVEIS}
    for campo in atm.values(): conferir_campo(campo,tp,origens)
    def sazonal(rel,var):
        with xr.open_dataset(raiz/rel) as ds: bruto=ds.load()
        exigir(pd.DatetimeIndex(bruto.time.values).equals(datas),'Alvos sazonais incorretos.')
        exigir(pd.DatetimeIndex(bruto.time_origem.values).equals(origens),'Origens sazonais incorretas.')
        a=bruto[var].interp(lat=tp.lat,lon=tp.lon,method='linear').astype('float32')
        exigir(np.isfinite(a).all() and (a>=0).all(),'Interpolação inválida.')
        for coord in ['time_origem','n_membros','inicializacao_mais_antiga','inicializacao_mais_recente','fase_fonte']:
            if coord in bruto: a=a.assign_coords({coord:('time',bruto[coord].values)})
        return a
    sp=sazonal('SEAS5/seas5_2025.nc','seas5_tp_media')
    cp=sazonal('CFSv2/cfsv2_2025.nc','cfsv2_tp_media')
    controles=[]; candidatos=[]
    for comp,features in [('arvores_seas5',FEATURES_SEAS5),('arvores_multissistema',FEATURES_MULTISSISTEMA)]:
        matriz=np.ascontiguousarray(X[:,:len(features)])
        controle=lgb.train(dict(PARAMETROS),lgb.Dataset(matriz,label=y,feature_name=features),num_boost_round=300)
        controle.save_model(str(saida/f'{comp}_controle.txt'))
        a=prever_par(controle,comp,atm,sp,cp,ca,clima,cs,cc,datas)
        controles.append(reconstruir_chuva(clima,a,BANCA_PESOS[comp]))
        ampliada=np.column_stack([matriz,np.repeat(pcs,ids.shape[1],axis=0)]).astype('float32')
        exigir(np.array_equal(ampliada[:,:len(features)],matriz),'Amostras do candidato alteradas.')
        candidato=lgb.train(dict(PARAMETROS),lgb.Dataset(ampliada,label=y,feature_name=features+SST_FEATURES),num_boost_round=300)
        candidato.save_model(str(saida/f'{comp}_sst.txt'))
        a=pesquisa_prever_sst(candidato,comp,atm,sp,cp,ca,clima,cs,cc,datas,pcs_pred)
        candidatos.append(reconstruir_chuva(clima,a,BANCA_PESOS[comp]))
        del matriz,ampliada,controle,candidato,a; gc.collect()
        print(comp,': referência e candidato concluídos.',flush=True)
    ridge=controles[0].copy(data=mos_predict(ajuste,sp,cp,datas))
    controle=.375*controles[0].astype('float64')+.375*controles[1].astype('float64')+.25*ridge
    candidato=.375*candidatos[0].astype('float64')+.375*candidatos[1].astype('float64')+.25*ridge
    cli=xr.concat([clima.sel(month=t.month).drop_vars('month') for t in datas],dim=pd.Index(datas,name='time'))
    resultado=xr.Dataset(dict(controle=controle,sst_8pcs=candidato,climatologia=cli))
    resultado.attrs.update(natureza='retrospectiva',treino_fim='2022-12-01',protocolo_sha256=PROTOCOLO_2025_SHA)
    arquivo=saida/'previsoes_2025.nc'
    resultado.to_netcdf(arquivo,engine='h5netcdf')
    registro=dict(protocolo_sha256=PROTOCOLO_2025_SHA,arquivo=str(arquivo),sha256=sha256(arquivo),
        criado_em_utc=datetime.now(timezone.utc).isoformat(),alvos_2025_lidos=False,
        entradas=entradas,sha256_X=hashlib.sha256(X.tobytes()).hexdigest(),
        sha256_y=hashlib.sha256(y.tobytes()).hexdigest(),
        sha256_pontos=hashlib.sha256(ids.tobytes()).hexdigest())
    pesquisa_json(saida/'registro_previsoes.json',registro)
    print('Duas previsões de 2025 congeladas. A chuva de 2025 ainda não foi carregada.')

if __name__=='__main__': prever_2025()
