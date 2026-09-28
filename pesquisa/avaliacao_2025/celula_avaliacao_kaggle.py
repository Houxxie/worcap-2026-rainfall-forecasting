if (RAIZ_2025/'previsoes_congeladas/registro_previsoes.json').is_file():
    runpy.run_path(str(CODIGO_2025/'avaliar_reserva.py'),run_name='__main__')
else: print('Avaliação bloqueada: primeiro congelar as duas previsões completas.')
