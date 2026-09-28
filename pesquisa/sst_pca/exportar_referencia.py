# Auditoria de pesquisa: não modifica nem substitui o CSV arquivado da competição.
ec_exigir = exigir
ec_hash = sha256
EC_LINHAS_TESTE = 1885464
CSV_ARVORES = BANCA_SAIDA / 'componentes' / BANCA_REF['nomes_csv']['ensemble_arvores']
AUDITORIA_ARVORES = ec_csv_media(BANCA_COMPONENTES['arvores_seas5'], BANCA_COMPONENTES['arvores_multissistema'], PASTA / 'sample_submission.csv', CSV_ARVORES)
exigir(AUDITORIA_ARVORES['submissao_sha256'] == BANCA_REF['csv_hashes']['ensemble_arvores'], 'A média das árvores divergiu da referência.')
GRADE_ARVORES = ler_grade_csv(CSV_ARVORES, PASTA / 'sample_submission.csv', PREVISAO_RIDGE)
PREVISAO_FINAL = PREVISAO_RIDGE.copy(data=0.75 * GRADE_ARVORES + 0.25 * PREVISAO_RIDGE.values)
BANCA_CSV = BANCA_SAIDA / 'submission_hibrida.csv'
BANCA_AUDITORIA = salvar_submissao(PASTA / 'sample_submission.csv', PREVISAO_FINAL, BANCA_CSV)
BANCA_IDENTICO = BANCA_AUDITORIA['submissao_sha256'] == BANCA_REF['hash_solucao']
BANCA_NUMERICAMENTE_CONFIRMADO = BANCA_IDENTICO
AUDITORIA_NUMERICA = dict(identico_byte_a_byte=BANCA_IDENTICO,
    sha256_esperado=BANCA_REF['hash_solucao'], sha256_obtido=BANCA_AUDITORIA['submissao_sha256'])
if not BANCA_IDENTICO:
    referencias = list(Path('/kaggle/input').rglob('submission_hibrida_documentada.csv'))
    exigir(len(referencias) == 1, 'Anexe a referência arquivada para investigar a diferença de hash.')
    ref_csv = referencias[0]
    exigir(sha256(ref_csv) == BANCA_REF['hash_solucao'], 'Referência arquivada com hash incorreto.')
    ref_df, novo_df = pd.read_csv(ref_csv), pd.read_csv(BANCA_CSV)
    exigir(ref_df.columns.equals(novo_df.columns) and ref_df.id.equals(novo_df.id), 'IDs/colunas/ordem divergentes.')
    delta = novo_df.tp_mm_day.to_numpy() - ref_df.tp_mm_day.to_numpy()
    exigir(len(delta) == EC_LINHAS_TESTE and np.isfinite(delta).all(), 'Comparação incompleta.')
    AUDITORIA_NUMERICA.update(n=len(delta), diferentes=int((delta != 0).sum()),
        max_abs=float(np.abs(delta).max()), media_abs=float(np.abs(delta).mean()),
        rmse_entre_previsoes=float(np.sqrt(np.square(delta).mean())),
        tolerancias=dict(max_abs=1.01e-6, media_abs=1e-9, rmse=1e-7),
        criterio='Conferência integral após diagnóstico de 28/09/2026; até uma unidade na sexta casa decimal.',
        causa='Diferença numérica residual; causa exata ainda não isolada. Não constitui identidade de arquivo.',
        uso_referencia='Somente auditoria; valores arquivados não entram em ajuste nem substituem previsões.')
    BANCA_NUMERICAMENTE_CONFIRMADO = (AUDITORIA_NUMERICA['max_abs'] <= 1.01e-6
        and AUDITORIA_NUMERICA['media_abs'] <= 1e-9 and AUDITORIA_NUMERICA['rmse_entre_previsoes'] <= 1e-7)
    del ref_df, novo_df, delta
AUDITORIA_NUMERICA['numericamente_confirmado'] = BANCA_NUMERICAMENTE_CONFIRMADO
salvar_json('identidade_csv.json', AUDITORIA_NUMERICA)
exigir(BANCA_NUMERICAMENTE_CONFIRMADO, 'A reprodução ultrapassou os limites numéricos documentados.')
salvar_json('manifesto_reproducao.json', dict(protocolo=protocolo_banca(), ambiente=VERSOES,
    contrato=CONTRATO, auditoria_componentes=BANCA_AUDITORIAS, auditoria_final=BANCA_AUDITORIA,
    identidade=AUDITORIA_NUMERICA, envio_automatico=False))
print('REFERÊNCIA CONFIRMADA NUMERICAMENTE; identidade byte a byte:', BANCA_IDENTICO)
print(json.dumps(AUDITORIA_NUMERICA, indent=2, ensure_ascii=False))
display(FileLink(os.path.relpath(BANCA_CSV, Path.cwd())))
