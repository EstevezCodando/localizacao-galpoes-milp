# Execução corrigida da pesquisa

Atualização de 03/10/2026, após a auditoria de implementação.

## O que mudou

- Tempos das trajetórias passam a ter um único deslocamento por estágio. A integral primal preserva o melhor incumbente acumulado, e resultados posteriores ao horizonte não contam como sucesso até o prazo.
- SCIP recebe somente o orçamento remanescente após montagem, com uma reserva de 5% para saída. A PL recebe limite próprio; se interrompida, seus custos reduzidos não são usados como certificado. Não há novo solver quando o orçamento já acabou.
- `t_total` inclui extração/validação e há `estouro_s`. O limite é cooperativo: montagem, rotinas nativas e coleta de resultados podem ultrapassá-lo. Isso é registrado, não truncado nos tempos reportados.
- As fases são registradas quando aplicáveis: `t_atributos`, `t_inferencia` (inclui normalização/agregação tabular), `t_pl`, `t_reparo`, `t_montagem`, `t_solver`, `t_validacao` e `t_heuristica`. `t_score` agrega etapas e não deve ser somado novamente às fases.
- O teste identifica as sementes de treino e solver. Os modelos legados contêm três GNNs e apenas uma versão de cada tabular/GNN+PL; o executor não apresenta cópias do mesmo modelo como treinamentos independentes. Métodos sem modelo usam `seed_treino=-1`.
- O baseline `warm_classico` gera uma solução com LNS e usa o restante do orçamento no solver completo. Esse custo inicial entra nas métricas.
- Os resultados da Etapa 1 ficam em diretórios identificados por hash em `results/pesquisa/etapa1/v2`. O manifesto registra hashes de fontes, dados, modelos, configuração, versões e ambiente. SQLite grava cada resultado em transação; o CSV é exportado ao terminar ou sair pelo bloco de finalização. Para retomar, repetir o mesmo comando com as mesmas entradas.
- Ordenações dos atributos são reutilizadas, inclusive para obter ranks por inversão de permutações. Em três instâncias positivas das famílias uniforme/clusters/corredor, os atributos foram exatamente iguais aos do código anterior. Custos/capacidades zero têm tratamento explícito e saída finita. O atributo quadrático de concorrência foi preservado: removê-lo exigiria uma ablação e retreino.
- A seleção de rho reutiliza scores durante a grade de validação, em vez de repetir PL/inferência por fração.
- CVRP exige demanda, capacidade do veículo e custo fixo em unidades inteiras. Frações são rejeitadas com orientação para escalar unidades, em vez de truncadas silenciosamente. Rotas são reconstruídas e cobertura, carga e custo são recalculados. A métrica de distância permanece `round(100*distância)` por arco.
- A avaliação CLRP aceita orçamento total de rotas compartilhado entre depósitos; o executor CLRP v2 usa esse orçamento, inclui tempo de rotas e desconta a inicialização do orçamento neural. A busca por surrogate mantém estado incremental após mover um cliente.

## Piloto de engenharia

Comando executado com o Python do ambiente existente:

```powershell
.venv/Scripts/python.exe -m alocacao_capacitada.pesquisa.etapa1 testar --split validacao --tempo 3 --workers 1 --limite 4 --seeds 0 1 2 --solver-seeds 0 --metodos completo warm_classico lns reducao_classica topk:classico topk:pl topk:lgbm topk:gnn adaptativa:gnn warm:gnn
```

São quatro instâncias 30×150, uma por combinação de família (uniforme/clusters) e razão de capacidade (1,5/3,0), escolhidas sem usar seus resultados. Dez métodos produzem 64 execuções ao considerar três sementes para cada política GNN. Foram mantidos os modelos e rho previamente treinados/escolhidos; não houve novo treinamento nem nova seleção baseada nos resultados deste piloto.

Este é um teste de funcionamento e instrumentação na validação, com horizonte curto. Não é um novo teste fechado nem uma demonstração de superioridade estatística. Outros processos Python estavam presentes no computador e não foram interrompidos; a carga concorrente não foi quantificada. Para comparar desempenho publicável, usar máquina isolada, mais instâncias, sementes do solver e horizonte pré-especificado após o piloto.

### Resultado observado

Execução `f5ed8b753d947b30c0aa`: 64/64 soluções finais válidas, **62/64 com incumbente até 3 segundos**. Nenhuma provou otimalidade nem atingiu desvio de 1% em relação à referência até o horizonte. Houve dois estouros acima de 50 ms; o maior foi de 0,487 s. Repetir o comando devolveu **zero execuções pendentes**, confirmando a retomada sem duplicação.

| Método | Execuções com incumbente no prazo | Integral primal média (menor é melhor) | Desvio mediano* |
|---|---:|---:|---:|
| GNN adaptativa | 12/12 | 0,245 | 8,73% |
| LightGBM top-k | 4/4 | 0,490 | 19,09% |
| GNN top-k | 12/12 | 0,547 | 18,92% |
| Warm start clássico | 4/4 | 0,569 | 46,90% |
| Redução clássica | 4/4 | 0,585 | 46,83% |
| Warm start GNN | 12/12 | 0,599 | 43,91% |
| PL top-k | 4/4 | 0,615 | 32,03% |
| Solver completo | 4/4 | 0,652 | 32,50% |
| Ranking clássico top-k | 4/4 | 0,671 | 31,99% |
| LNS | 2/4 | 0,720 | 9,18% |

*Primeiro média entre sementes por instância, depois mediana entre instâncias, condicionada à existência de solução no prazo. Para LNS, isso usa apenas duas instâncias; não interpretar 9,18% como desempenho nas quatro. A integral inclui o período sem incumbente. As referências são melhores custos conhecidos dos rótulos/resultados, não necessariamente ótimos provados.

Neste pequeno piloto, a expansão adaptativa teve menor integral primal, enquanto top-k neural não superou LightGBM nessa métrica. Isso é uma observação exploratória. A diferença forte entre custo final e integral do LNS mostra por que o instante da primeira solução importa. Não se calculou significância estatística com quatro instâncias.

Artefatos: `results/pesquisa/etapa1/v2/f5ed8b753d947b30c0aa/{manifesto.json,resultados.csv,metricas.csv,resumo.csv,checkpoint.sqlite}`. Para refazer somente a análise:

```powershell
.venv/Scripts/python.exe scripts/resumir_piloto_pesquisa.py results/pesquisa/etapa1/v2/f5ed8b753d947b30c0aa
```

Um microbenchmark exploratório das ordenações foi executado enquanto o piloto terminava; portanto houve também carga introduzida pela própria revisão. Seus tempos não são usados como evidência de aceleração final. Uma avaliação de desempenho definitiva deve rodar sem essa interferência.

### Verificação da implementação

- Suíte completa: **120 testes aprovados em 47,88 s**, incluindo 11 novas regressões para métricas, censura, custos/capacidade zero, orçamento esgotado, sementes, checkpoint, rotas e isolamento das medições.
- Equivalência dos cinco blocos/índices do grafo com o código anterior, nas três famílias positivas verificadas.
- Smoke test CLRP com seis clientes: MIP, busca por surrogate, capacidade e custo das rotas validados; não houve benchmark CLRP completo nem retreino do surrogate.
- `git diff --check` sem problemas. Ruff passou no conjunto verificado com exclusão de N806, N812 e B905 (convenções e ocorrências legadas); isso não representa uma execução sem exceções de todo o lint/mypy do repositório.

## Repetir uma avaliação mais ampla

O mesmo comando aceita `--tempo 60`, remoção de `--limite`, `--split teste` e `--solver-seeds 0 1 2`. Executar isso apenas após congelar o desenho: o split histórico já foi usado anteriormente, portanto não o apresentar como um teste nunca inspecionado. Para generalização nova, gerar/selecionar outro conjunto independente antes de observar resultados.

Repetir o comando sem alterações retoma o checkpoint e não repete linhas prontas. Qualquer mudança de conteúdo em fontes/modelos/dados/configuração produz outro identificador. Os CSVs históricos não são sobrescritos. A saída CLRP v2 é separada da histórica, mas ainda não possui o mesmo mecanismo de manifesto/checkpoint da Etapa 1.

Os modelos tabulares de múltiplas sementes, treinamento novo, ablação do atributo de concorrência, troca por PyG/DuckDB/Polars/PyVRP e benchmarks extensos ficam para experimentos posteriores. Não foram instaladas bibliotecas adicionais sem evidência de que o custo de migração se paga.
