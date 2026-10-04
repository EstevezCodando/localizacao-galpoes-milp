# Síntese — Aprendizado de máquina para localização e roteamento: o que funcionou, o que não funcionou e uma linha de pesquisa

*Rascunho de 03/10/2026. A seção da Etapa 1 (poda, warm start e prioridade aprendidos no SSCFLP)
aguarda a reexecução pedida pela auditoria; os números dela não estão aqui.*

## 1. Os artigos, um a um

| ID | Trabalho | Mecanismo | O que testamos | Resultado nesta reconstrução |
|---|---|---|---|---|
| P01 | Bengio, Lodi & Prouvost (EJOR 2021) | Taxonomia: ML de ponta a ponta, ML para configurar e ML junto do solver | Usada para classificar cada frente | Mostra que todas as frentes testadas são "ML dentro do algoritmo" e que o risco comum é a generalização |
| P02 | Gasse et al. (NeurIPS 2019) | GCNN imita strong branching | R3 em SCIP 10, ranqueador por candidato | O SCIP 10 resolve `facilities` quase na raiz; o aprendido empata com "mais fracionário" (acc@1 0,36 contra 0,39). Teste no solver em andamento |
| P03 | Gupta et al. (NeurIPS 2020) | Híbrido GNN/MLP para baratear a inferência | Tempo de inferência medido em todas as frentes | Confirmado como fator decisivo: nas trocas, o aprendido é mais lento que o clássico por causa da inferência |
| P04 | Guo, Xu & Jin (2023) | Política de trocas por RL | Imitação supervisionada (sem PPO) em p-mediana sobre redes | Ganho do aprendizado pequeno e só na escala de treino |
| P05 | Su et al. (SIGSPATIAL 2024) | GNN com conhecimento de domínio guia trocas | Filtro de trocas aprendido contra clássico | O filtro clássico sozinho dá 47× com 6,6% de perda; a maior parte da aceleração vem do filtro, não do aprendizado |
| P06 | Qian et al. (ICML 2026) | MPNN não supervisionada para UniFL | Reconstrução completa com perda de custo esperado | Generaliza (1,03 em n = 1000 contra 1,19 do MP), mas é instável (3 de 4 sementes colapsaram antes da correção) e empata com MP + busca local |
| P07 | Kaleem et al. (2024/2026) | Deep Sets embutido no MIP do LRP | Embutimento por big-M em SCIP e busca guiada | O MIP neural não otimiza em solver aberto (gap de 591%); a aproximação contínua clássica empata ou vence |
| P08 | Wang, Cao & Huang (CEUS 2026) | Graph RL multiobjetivo, localização urbana | Não testado | Texto completo não acessado; fica como leitura complementar, sem tomar números como evidência |

## 2. Achados transversais

1. **O baseline que importa é o clássico equivalente, não o solver cru.** Em todas as frentes,
   uma heurística clássica com a mesma função capturou a maior parte do ganho: o ranking por PL
   na Etapa 1, o filtro ganho de adição × perda de remoção nas trocas, Mettu–Plaxton + busca
   local no UniFL e a aproximação contínua no LRP. Uma comparação só contra "SCIP completo" ou
   "FLP→VRP" infla o mérito do aprendizado.
2. **Erro de previsão baixo não implica decisão melhor.** No LRP, MAPE de 7,5% e Kendall τ de
   0,34 entre configurações. Na Etapa 1, AUC alta não basta, porque o reparo de viabilidade e a
   capacidade apertada exigem manter cerca de 80% dos centros para preservar a melhor solução.
3. **O solver evoluiu.** Ganhos relatados contra o SCIP 6 (2019) não se transferem para o
   SCIP 10, que fecha na raiz instâncias antes exploradas por branching.
4. **Inferência e engenharia decidem o tempo.** Atributos recalculados, chamadas por nó e
   Python no laço interno anulam a economia de avaliações. A própria auditoria encontrou um erro
   de relógio que enviesaria a comparação de tempo da Etapa 1.
5. **A generalização é o calcanhar de Aquiles e também o melhor resultado.** A MPNN do UniFL
   generalizou bem na escala. O filtro de trocas por imitação, não. A diferença é que a perda do
   UniFL é o próprio objetivo, enquanto a imitação aprende um alvo míope.

## 3. Proposta de linha de pesquisa

**Título de trabalho:** *Aprendizado como correção de heurísticas clássicas em localização
capacitada: avaliação a tempo total igual, com certificados.*

**Tese:** o aprendizado deve corrigir o sinal clássico, e não substituí-lo, e ser avaliado
contra ele a tempo total igual, com certificado de qualidade quando possível.

| Eixo | Recombinação proposta | Ganho esperado | Por que pode funcionar |
|---|---|---|---|
| A | Ranking residual: GNN aprende `score − score_PL` ou o resíduo do clássico | precisão de poda com menos dados | o PL já tem recall 0,87; aprender só o erro dele |
| B | Poda **certificada**: ranking aprendido + certificado de custo reduzido do PL original + expansão guiada pelo certificado | ótimo provado em menos tempo quando o gap do PL é pequeno | une ML (onde olhar) com dualidade (prova de que o resto não importa); já implementado em `hibrido.adaptativa` |
| C | LNS com vizinhanças aprendidas: o score escolhe quais centros e regiões destruir no LNS do projeto | melhor integral primal | o LNS já é o método mais forte no Olist; aprender a vizinhança é o mecanismo de P04/P05 aplicado onde há espaço |
| D | Pré-treino não supervisionado pela perda de custo esperado (P06) estendida com penalidade de capacidade | dispensa as 4 CPU-h de rótulos SCIP | a perda de P06 generalizou na escala, ao contrário da imitação |
| E | Surrogate residual sobre a aproximação contínua, com treino por ranking e correção iterativa (LRP) | decisões melhores que a CA | ataca os dois fracassos medidos: erro alto relativo às diferenças e exploração dos erros |
| F | Prioridade de branching estática aprendida no SSCFLP | ganho sem custo por nó | aproveita a parte de P02 que sobrevive ao solver moderno |

**Protocolo mínimo para qualquer alegação:** tempo total igual com prazo global; integral
primal com mínimo acumulado; 3 ou mais sementes de treino; teste fechado; famílias e escalas
não vistas; benchmarks com ótimo publicado (Holmberg); resultados negativos reportados.

## 4. Estado do estado da arte, nesta evidência

- Para **decidir onde abrir instalações** com capacidade, o estado da arte prático continua
  sendo MILP + matheurística (LNS) com boas relaxações. O aprendizado ainda não demonstrou ganho
  líquido robusto sobre esse conjunto nesta reconstrução.
- O aprendizado mostrou valor mais claro como **gerador de soluções iniciais que generaliza na
  escala** (P06) e como **ranking** (AUC 0,95). Converter ranking em tempo de solução é o
  problema aberto, e a Etapa 1 mede exatamente isso.
- Em **localização-roteamento**, o embutimento neural depende de solver comercial. Em solver
  aberto, a aproximação contínua clássica é o patamar a superar.

## 5. Pendências

- Etapa 1 reexecutada com as correções da auditoria (teste, Holmberg, corredor, escala, 3
  sementes de GNN, baseline completo com warm start clássico).
- Resultado do learn2branch no solver (em execução).
- Reprodução R1 de P02 e P07 em ambiente histórico ou com licença acadêmica do Gurobi.
