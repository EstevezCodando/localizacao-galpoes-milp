# Auditoria da implementação — 03/10/2026

**Veredito: implementação parcial, com núcleo matemático validado em casos pequenos, mas ainda inadequada para sustentar uma conclusão de aceleração pela GNN.** Há problemas reproduzidos nas métricas e contratos e lacunas na execução experimental. Trocar bibliotecas antes de corrigir a medição não resolve isso.

O diretório `Learning_the_Search_Space_Projeto` é um plano documental. O código correspondente está em `src/alocacao_capacitada/pesquisa`. Esta revisão examinou principalmente a Etapa 1, seus dados, atributos, solver, treinamento, políticas híbridas e análise; examinou também o caminho CLRP/Deep Sets e pontos do processamento territorial. Não constitui auditoria integral de todas as reconstruções bibliográficas, provas dos artigos ou dados externos.

Não foram alterados algoritmos, modelos treinados ou resultados existentes. Arquivos de resultados já estavam modificados no início; o CSV lido é um retrato parcial, não um experimento congelado. Não foram refeitos treinamentos ou benchmarks longos.

## Evidência executada

- `tests/test_pesquisa_g0.py`: **18 testes aprovados**, incluindo 13 microinstâncias com comparação SCIP versus enumeração, validação, relaxação linear, escopo de bound e warm start.
- Suíte completa: **109 testes aprovados em 37,86 s**, usando `-p no:cacheprovider --basetemp <diretório novo no workspace>`. A primeira tentativa teve 14 erros de preparação por permissão no diretório temporário padrão e 95 aprovações; o novo diretório resolveu os erros ambientais.
- Casos adicionais executados diretamente: integral primal com reinício pior; atributos para custos zero; limite online muito curto; CVRP com demanda fracionária; microbenchmark de seleção top-k; perfil de construção do grafo.
- Os testes atuais não cobrem suficientemente relógio global, trajetórias combinadas, sementes do experimento, contratos numéricos dos atributos ou validação independente das rotas da implementação de pesquisa.

## Achados prioritários

### 1. P1 — Trajetórias contam a montagem duas vezes

Em `exato.py:115`, `traj.t0` recebe o instante anterior à montagem. Os tempos dos eventos já incluem essa montagem. Entretanto `hibrido.py:78,125,149,152,165,172,196` soma `t_montagem` novamente. Isso afeta tempo até alvo e integral primal, com distorção diferente por tamanho do modelo e número de estágios. Nos caminhos top-k/warm, o deslocamento também não representa precisamente todo o reparo anterior à chamada.

**Correção:** adotar um único relógio online; deslocar cada trajetória apenas pelo tempo efetivamente decorrido antes de chamar o solver. Testar com relógio controlado e estágios de durações conhecidas.

### 2. P1 — Integral primal esquece o melhor incumbente global

`analise.py:43` substitui o gap anterior pelo gap de cada evento. Ao juntar estágios, uma solução pior de uma nova resolução pode elevar artificialmente o gap, apesar de uma solução melhor continuar disponível.

Reprodução: `integral_primal([(1.,100.),(2.,150.)],100.,10.)` devolveu **0,5**. O resultado correto é **0,1**: após t=1 já se conhece custo 100. A métrica deve integrar o mínimo acumulado dos custos válidos. Também tratar BKS zero explicitamente.

Em `analise.py:98`, sucesso até alvo é contado com `isfinite`, mesmo quando o evento ocorre depois do horizonte. Aplicar `t_alvo <= T`. As médias/IC de desvio descartam falhas; rotular como condicionais ao sucesso e manter métricas que penalizem falhas para a comparação principal.

### 3. P1 — Orçamento online e custo total incompletos

`exato.py:82` passa o orçamento ao SCIP sem uma deadline comum ao pipeline; `exato.py:117` encerra `t_total` antes de extrair e validar a solução. `relaxacao_linear` não recebe limite de tempo. Políticas usam pisos de 0,05 s mesmo quando o orçamento restante acabou. A construção do modelo tampouco é interrompida quando o prazo se esgota.

Uma chamada 30×150 com orçamento de **0,01 s** levou **0,276 s**, dos quais **0,216 s** eram montagem. É um teste de contrato, não uma estimativa de desempenho típico. No CSV parcial T60 havia execuções chegando a aproximadamente **62 s**.

**Correção:** deadline baseada em `perf_counter`, propagada a atributos, PL, reparo e solver; registrar estouros inevitáveis separadamente; reservar extração/validação; não iniciar novo estágio sem orçamento. O status da PL limitada deve distinguir ausência de solução, inviabilidade e bound utilizável.

### 4. P1 — Três GNNs treinadas, somente semente zero no teste

`etapa1.py:114` treina três GNNs, mas `_rodar`, em `etapa1.py:215`, chama `_RK.score(nome, p)` com `seed=0` implícita. Os jobs, CSV e chave de retomada não identificam semente de treino/solver. Portanto o teste atual não mede variabilidade das três GNNs. MLP e LightGBM também são treinados uma única vez.

**Correção:** identificar `(instância, método, seed_treino, seed_solver, configuração)` em cada execução e checkpoint. Agregar repetições por instância antes da inferência estatística, conforme o protocolo.

### 5. P1 para dados fracionários — CVRP pode aceitar sobrecarga real

`lrp.py:91` converte demanda para inteiro por truncamento; capacidade e custo fixo também são convertidos. Duas demandas de **1,9**, veículo com capacidade **3**, foram atendidas por **uma rota** no caso executado, embora a carga real seja **3,8**. O gerador sintético atual usa demandas inteiras, portanto este caso não demonstra erro nos resultados sintéticos existentes; demonstra falha ao generalizar o contrato para dados fracionários.

**Correção:** exigir inteiros com validação explícita ou escalar unidades de forma controlada. Retornar rotas e validar visitas, cargas e custos sobre os dados originais, em vez de retornar somente objetivo e quantidade de rotas.

### 6. P2 — Atributos não suportam valores aceitos pelo domínio

`Problema` aceita custos e capacidades zero. `atributos.py:52–112` divide por mediana de custo, média de custo fixo e capacidade sem tratamento consistente. Uma instância viável com custos fixos/atendimento zero produziu atributos não finitos nos três blocos.

**Correção:** definir normalização para instâncias degeneradas, eliminar pares incapazes de atender a demanda quando apropriado e validar finitude antes do treino/inferência. Não esconder capacidade nula apenas adicionando epsilon: seu significado operacional precisa ser preservado.

### 7. P2 — Comparação CLRP não usa orçamento total comum

`exp_lrp.py:96` contabiliza a inicialização por aproximação contínua, mas concede ao NEO outro orçamento integral. `lrp.py:109–118` concede tempo por depósito; abrir mais depósitos dá mais tempo total de roteamento. O CSV registra `t_mip`, mas não o tempo completo das rotas.

Isso permite estudar qualidade sob esses orçamentos, mas não sustenta comparação a tempo total igual. Registrar `t_rotas`, `t_total`, prazo comum e orçamento efetivamente consumido; distinguir claramente BKS heurístico de ótimo CLRP. Enumerar conjuntos de depósitos com atribuição heurística não enumera todas as soluções CLRP.

### 8. P2 — Retomada pode misturar versões de experimentos

`etapa1.py:246–263` usa apenas `(instância, método)` para reconhecer trabalho concluído. Trocar modelo, fração de poda ou implementação reutiliza linhas antigas. Além disso, regrava todo o CSV a cada resultado, aumentando I/O cumulativo e deixando risco de arquivo incompleto se houver interrupção.

**Correção:** manifesto com hashes de dados/modelo/configuração/commit, identificador de execução, escrita transacional ou por partições e exportação final. Há ainda um erro pequeno em `etapa1.py:148`: `r.gap or np.nan` exclui gaps zero da média; testar `None` explicitamente.

## O que os artefatos existentes mostram

O retrato consultado continha **418 linhas, 25 instâncias, 17 métodos e 2 soluções marcadas inválidas**; os métodos tinham 24 ou 25 observações. Não usar essa tabela incompleta como comparação final pareada.

O registro offline informa 160 instâncias de treino: 159 com gap e 1 ótima. Rótulos aproximados não invalidam o método, mas devem ser tratados como tais. A geração de rótulos somou aproximadamente 4 horas registradas por instância; isso não equivale necessariamente a quatro horas de relógio nem a CPU-horas medidas. Os treinos registrados foram: LightGBM 4,64 s, MLP 19,22 s e GNN 304–376 s por semente, fora outros custos offline.

A validação escolheu rho=0,8 para GNN, MLP e LightGBM; PL também ficou em 0,8. O reparo pode aumentar a fração real. Logo, a hipótese de acelerar removendo a maior parte das instalações não está demonstrada. No ranking de validação, recall@k do PL foi 0,8722, GNN sem PL 0,8277–0,8340 e GNN+PL 0,8727. Esses números justificam priorizar o baseline PL; não demonstram superioridade final de nenhum método de otimização.

## Eficiência: mudanças em ordem de retorno provável

| Prioridade | Mudança | Evidência e limite |
|---|---|---|
| Alta | Reutilizar ordenações em `grafo` e construir ranks por inversão da permutação | Há várias ordenações repetidas por eixo; preserva atributos se o desempate for mantido |
| Alta | Usar seleção parcial quando só são necessários k vizinhos | NumPy já instalado; não substituir ranks completos por ranks parciais |
| Alta | Armazenar scores/atributos por instância na escolha de rho | A grade recalcula a mesma inferência e PL para cada rho; cache offline reduz trabalho redundante |
| Alta | Checkpoints incrementais e versionados | Evita regravar todo o histórico; SQLite ou partições Parquet são opções |
| Média | Remover ou aproximar o atributo de concorrência após ablação | `cdist(u,u)` custa O(m²n); muda a entrada do modelo, exige retreino e validação |
| Média | Atualizar estados de busca local incrementalmente | `busca_surrogate` reconstrói one-hot e soma latente a cada iteração, embora só mova um cliente |
| Condicional | GNN em minibatches via PyTorch Geometric | Pode reduzir overhead por grafo; preservar offsets bipartidos e normalização por grafo |
| Condicional | DuckDB ou Polars para ingestão/joins grandes | Útil quando I/O/memória dominarem; não acelera branch-and-bound |

**Medições locais exploratórias:** seleção de 10 menores por coluna em matriz aleatória 500×5000, mediana de cinco execuções: `argsort` **111,43 ms**, `argpartition` **87,71 ms**, aproximadamente **21% menos tempo nessa operação**. A saída parcial não é ordenada; empates precisam de política consistente. Não foi medido ganho ponta a ponta dessa substituição.

Perfil de uma construção de grafo 500×1000: **0,897 s** total, **0,515 s** em `cdist` e **0,247 s** em `argsort`. É uma instância artificial e uma execução com profiler, não benchmark publicável. Ainda assim identifica um gargalo concreto. O grafo final é esparso, mas construir matrizes densas e comparar todos os perfis continua custoso.

O pipeline territorial já usa seleção de colunas CSV, STRtree e consultas OSRM em blocos com cache. Melhorias simples: persistir centroides de CEP/município compartilhados por `build_panel` e `seller_hubs`; normalizar nomes só para linhas sem correspondência espacial, preferencialmente por valores distintos. Não há justificativa atual para introduzir Spark/Dask.

## Bibliotecas e algoritmos a comparar

1. **Manter SCIP/PySCIPOpt, HiGHS, NumPy e LightGBM como base.** Já atendem ao problema; o maior impedimento imediato é a validade experimental. Comparar solver completo com warm start clássico, redução por custo reduzido, LNS e ranking PL com tempo total igual antes de ampliar a GNN. Redução clássica/LNS já existem; falta uma comparação confiável e um baseline explícito de completo com warm start clássico no conjunto de métodos.
2. **PyTorch Geometric:** batching de grafos bipartidos, quando o perfil de treinamento justificar. A atual agregação usa `cnt.max()` global; num batch isso mistura a escala entre instâncias, então a normalização deve ser por grafo. [Documentação de batching](https://pytorch-geometric.readthedocs.io/en/2.6.1/advanced/batching.html).
3. **DuckDB ou Polars, escolher um conforme o fluxo:** DuckDB para consultas SQL/Parquet; Polars lazy para transformação de DataFrames. Ambos podem antecipar filtros e leitura de colunas. Medir contra pandas com as mesmas saídas e memória máxima. [DuckDB/Parquet](https://duckdb.org/docs/current/guides/file_formats/query_parquet), [Polars lazy](https://docs.pola.rs/user-guide/lazy/optimizations/).
4. **PyVRP:** candidato a baseline de roteamento, com origem em Hybrid Genetic Search e suporte a múltiplos depósitos. Não assumir que substitui o CLRP com decisão de abertura/capacidade de depósito; o encaixe mais direto é na avaliação CVRP após a alocação. Comparar custo real sob prazo igual, sem promessa prévia de ganho. [Documentação oficial](https://pyvrp.org/about/about.html).
5. **Numba:** candidato apenas para loops numéricos identificados por perfil, como avaliação de movimentos/reparo; não para chamadas ao solver ou objetos pandas. Contabilizar compilação ou declarar aquecimento. [Documentação de JIT](https://numba.readthedocs.io/en/stable/user/jit.html).
6. **Branch-and-price/Benders especializado:** extensão de maior custo de engenharia. Na fonte única, fixar aberturas ainda deixa uma atribuição inteira; uma decomposição que simplesmente relaxe essa atribuição muda o problema. Só investigar após medir a limitação das alternativas existentes; não é troca direta de biblioteca.

Referência para seleção parcial: [NumPy argpartition](https://numpy.org/doc/2.1/reference/generated/numpy.argpartition.html).

## Critério para continuar

Primeiro corrigir métricas, contratos e manifesto; acrescentar regressões para os casos reproduzidos; depois repetir um piloto pareado com limite global, todas as sementes e resultados completos. Medir separadamente atributos, PL, inferência, montagem, solver, extração/validação e rotas.

Continuar com poda neural somente se houver ganho líquido, sob o mesmo critério de qualidade, frente ao solver e ao melhor baseline simples, com falhas e fallback incluídos. Caso rho continue próximo de 1 ou o overhead elimine o ganho, priorizar warm start/LNS ou reportar o resultado negativo. A evidência desta auditoria sustenta correções e um novo piloto, não uma alegação de aceleração operacional.
