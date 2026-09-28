from IPython.display import Markdown, display

_r = RESULTADO_PESQUISA_SST
_g = pd.read_csv(Path(_r['saida'])/'metricas_globais.csv').set_index('modelo')
_ganho = 100*(_r['rmse_controle']-_r['rmse_sst'])/_r['rmse_controle']
display(Markdown(f"""## Nimbus PA — pesquisa SST concluída

Comparação temporal de 2007 a 2020. Mesmos exemplos, parâmetros e pesos.
Oito componentes de SST adicionados às duas árvores; ridge preservado.

| Métrica (mm/dia) | Referência híbrida | Com SST |
|---|---:|---:|
| RMSE | {_g.loc['controle','rmse']:.6f} | {_g.loc['sst_8pcs','rmse']:.6f} |
| MAE | {_g.loc['controle','mae']:.6f} | {_g.loc['sst_8pcs','mae']:.6f} |
| Viés | {_g.loc['controle','vies']:+.6f} | {_g.loc['sst_8pcs','vies']:+.6f} |

**Redução de RMSE: {_ganho:.3f}%**. Melhora em
**{_r['consistencia']['blocos']['melhores']}/7 blocos** e
**{_r['consistencia']['anos']['melhores']}/14 anos**. O viés médio aumentou.

**Leitura:** ganho agregado pequeno. Manter o candidato separado da referência;
validar em período independente antes de adotá-lo.
Estes anos já foram usados no desenvolvimento, e a disponibilidade operacional
da SST mensal consolidada ainda precisa ser conferida.

Referência histórica reproduzida. O CSV final difere do arquivado em apenas
532 valores de 0,000001 mm/dia; equivalência numérica auditada, sem identidade de bytes.

Modelos, previsões por ponto, relatórios e auditorias estão na saída desta execução.
"""))
