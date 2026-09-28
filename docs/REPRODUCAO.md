# Reprodução, dados e estado das entregas

## Solução da competição

Comece pelo [notebook explicado](../competicao/Nimbus_PA_WorCAP_2026.ipynb) e pelas
[instruções originais](../entregas/Nimbus_PA_homologacao/LEIA_ME.md). O notebook é
autossuficiente em código, mas precisa dos Inputs. Use Python 3.12 e o
[ambiente de referência](../entregas/Nimbus_PA_homologacao/requirements.txt).

| Entrada | Como identificar |
|---|---|
| Arquivos oficiais WorCAP | Nomes e hashes em `referencia_auditada.json` da entrega; acesso depende da plataforma/organização |
| SEAS5 preparado | `seas5_51_manifesto.json` e três NetCDFs: desenvolvimento, somente ajuste final e teste |
| CFSv2 preparado | `cfsv2_manifesto.json` e três NetCDFs com as mesmas fases |
| Índices NOAA | Snapshot incorporado ao código; cópia legível na entrega |

O pacote local de homologação preserva os snapshots sazonais, mas dados binários
ficam fora do Git. Ainda não existe um endereço público do projeto para baixar esse
pacote. As [fontes originais](../entregas/Nimbus_PA_homologacao/DADOS_EXTERNOS.md)
permitem localizar os produtos; baixar hoje uma série revisada não garante os
mesmos bytes da competição. A reprodução exata exige os snapshots cujos hashes
estão documentados. A publicação de dados preparados será tratada separadamente,
considerando as condições das fontes.

Com `REEXECUTAR_VALIDACAO=False`, a execução ajusta os componentes finais e gera
os dois CSVs. Com `True`, repete também os blocos históricos. O script verifica
os hashes esperados; não reduz a exigência se ambiente ou dados forem diferentes.

## Pesquisa atual

O [notebook de fontes defasadas](../pesquisa/disponibilidade_operacional/Nimbus_PA_defasagens_e_SST.ipynb)
requer dados oficiais, SEAS5, CFSv2 e o snapshot ERSSTv5 indicado em seu
[protocolo](../pesquisa/disponibilidade_operacional/protocolo.json). Consulte o
[guia da etapa](../pesquisa/disponibilidade_operacional/LEIA_ME.md). Ele compara
controle e candidato com SST; não exporta os CSVs da competição.

O [notebook prospectivo](../pesquisa/acompanhamento/Nimbus_PA_registro_prospectivo.ipynb)
registra coletas e prontidão. Ainda não fecha o ciclo de treino final e emissão
real. Seus comandos usam o [ambiente de coleta](../pesquisa/acompanhamento/requirements.txt),
separado do ambiente estrito de reprodução dos modelos.

## Conferir a organização sem treinar

Na raiz do projeto, com Python 3.12:

```bash
python scripts/conferir_versoes.py
```

O comando confere catálogo, integridade dos artefatos arquivados disponíveis,
células do notebook preparado e links locais dos novos guias. Dados binários
ausentes são informados como pendências; não são considerados verificados.
Para exigir também todos os arquivos do pacote local:

```bash
python scripts/conferir_versoes.py --exigir-dados
```

Isso não executa treinamento, não faz downloads e não comprova o envio do e-mail
original. Os hashes demonstram identidade com o inventário arquivado. Os testes
e métricas dos modelos têm relatórios próprios, com seus limites de reprodução.
