necessarios=['ERA5/atmosfera_origens_2025.nc','ERA5/compatibilidade_piloto.json',
    'SEAS5/seas5_2025.nc','CFSv2/cfsv2_2025.nc','SST/ersstv5_4graus_202411_202511.nc']
faltam=[p for p in necessarios if not (RAIZ_2025/p).is_file()]
if faltam: print('Inferência aguardando:',faltam)
elif (RAIZ_2025/'previsoes_congeladas/registro_previsoes.json').is_file():
    print('Previsões já congeladas: sem novo ajuste.')
else:
    _espaco_2025=runpy.run_path(str(CODIGO_2025/'prever_modelos_2025.py'),run_name='__main__')
    del _espaco_2025
