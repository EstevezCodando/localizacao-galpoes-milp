# Frente P07 — NEO-LRP: Deep Sets embutido no MIP para localização-roteamento (Kaleem, Lee, Kwon & Subramanyam)

**Nível:** R3, reconstrução aberta. O código oficial usa Gurobi e Gurobi-ML, e não havia
licença disponível. Aqui o mesmo princípio roda em SCIP, com ReLU codificada por big-M.
Dados: `results/pesquisa/lrp/`. Código: `src/alocacao_capacitada/pesquisa/lrp.py`, `exp_lrp.py`.

## 1. O que o artigo propõe

O custo de roteamento de cada depósito é aproximado por uma rede Deep Sets
`rho(sum_j phi(i, j))`. Como `phi(i, j)` depende só do par, a soma é linear nas variáveis de
atribuição, e `rho` é uma MLP ReLU pequena que entra no MIP. Depois de decidir a localização e
a atribuição, as rotas são construídas de fato. O treino usa custos de CVRP obtidos por solver
heurístico.

**Classificação (P01):** ML como componente do modelo de otimização. O surrogate substitui uma
parte cara do objetivo, e o solver exato otimiza sobre ele.

## 2. Como foi implementado

| Componente | Implementação |
|---|---|
| CLRP | estilo Prodhon: grade 50×50, demanda U{11..20}, veículo Q = 70 e F = 1000, rota = 100 × distância, capacidade do depósito 2,5× a demanda média por depósito, abertura U(4000, 8000) × sqrt(n/50) |
| Rótulos | 2.400 subconjuntos de treino e 480 de validação (aleatórios, vizinhança do depósito e espalhados), CVRP no OR-Tools (GLS, 1 s): 23 min com 2 processos |
| Surrogate | Deep Sets com phi de 5 entradas → 32 → 8 e rho 8 → 16 → 16 → 1; treino em 198 s; **MAPE de validação de 7,5%** |
| `neo_ds` | rho embutida por big-M com limites de ativação por propagação de intervalos; 160 binárias neurais (m = 5); partida pela solução da aproximação contínua |
| `flp_vrp` | SSCFLP com custo radial `2 × 100 × d_ij + F d_j / Q`, depois rotas |
| `aprox_cont` | aproximação contínua linear: linha-haul `2 r d / Q` + percurso local `0,75 ×` distância típica entre vizinhos (Daganzo, 1984; BHH) + veículo rateado |
| `neo_multi` | MIP neural partindo da melhor solução entre CA e FLP, segundo o surrogate (desvio, ver §4) |
| `neo_busca` | busca local de realocação com custo pelo surrogate, todas as realocações num lote (Caminho B do doc 09; desvio) |
| Avaliação | sempre por rotas reais (OR-Tools, 2 s por depósito). Referência = melhor custo final entre os métodos e, com m = 5, a enumeração dos 31 conjuntos de depósitos |

20 instâncias de teste: n ∈ {20, 50, 100} clientes, m ∈ {5, 10} depósitos, uniforme e clusters.

## 3. Resultados

Desvio contra a referência, calculado sobre o custo final com rotas reais:

| Método | Desvio mediano | Desvio médio | É o melhor | Tempo do MIP (mediana) |
|---|---:|---:|---:|---:|
| **aprox_cont** (clássico) | **0,03%** | **0,67%** | 45% | 1,5 s |
| neo_ds (SCIP) | 0,05% | 0,69% | 45% | 61 s |
| neo_multi | 0,08% | 1,33% | 40% | 61 s |
| neo_busca | 0,30% | 1,48% | 45% | 2,0 s |
| flp_vrp | 4,54% | 5,60% | 15% | 0,1 s |

Por tamanho (desvio mediano): em n = 100 e m = 10, `neo_busca` teve 0,14% contra 0,34% da
aproximação contínua. Em n = 50 e m = 5, 0,70% contra 1,08%. Em n = 20 a CA é ótima em todas.
Pareado contra a CA, `neo_busca` vence 35% e perde 40%; `neo_multi` vence 15% e perde 20%.

Erro do surrogate **nas configurações escolhidas**: erro absoluto mediano de 2,1% a 4,6%.
**Kendall τ = 0,34** entre a ordem prevista e a ordem real das configurações de cada instância.
O ruído da avaliação de rotas (mesma configuração reavaliada) é de 0,017% em média, com máximo
de 0,28%, então diferenças acima de cerca de 0,3% são sinal.

### Diagnóstico do MIP neural (o achado principal)

O `neo_ds` devolveu **exatamente a solução de partida em 20 de 20 instâncias**. Numa instância
com n = 50 e m = 5, após 120 s e 2.657 nós, o bound dual estava em 9.222 contra 63.746 do
incumbente (gap de 591%). O SCIP não encontrou nem a solução FLP, que pelo próprio surrogate é
melhor (60.916). Sem solução inicial, o SCIP não acha nenhuma solução viável em 30 s.

A codificação big-M da ReLU, mesmo com limites por intervalos, produz uma relaxação muito fraca,
e as heurísticas primais do SCIP não exploram o espaço. Fica separada uma pergunta que o artigo
não isola: **o benefício do NEO-LRP depende do solver comercial e da formulação do Gurobi-ML**.
Num solver aberto, o embutimento direto não otimiza.

### Limites desta comparação (auditoria de 03/10/2026, itens 5 e 7)

- **É um estudo de qualidade sob orçamentos próprios, não a tempo total igual.** O `neo_ds`
  recebe 60 s de MIP além da inicialização pela CA. O roteamento recebe 2 s **por depósito**, então
  configurações com mais depósitos ganham mais tempo de rotas. O CSV registra `t_mip`, mas não o
  tempo das rotas nem o total. As conclusões sobre tempo valem só para o MIP.
- **A referência é heurística.** A enumeração dos 31 conjuntos de depósitos usa uma atribuição
  por aproximação contínua para cada conjunto, sem enumerar todas as soluções do CLRP. O "desvio"
  é medido contra o melhor encontrado (BKS heurístico), não contra o ótimo.
- **Contrato do CVRP:** `cvrp()` converte demanda e capacidade para inteiro por truncamento. As
  instâncias deste estudo têm demandas inteiras (U{11..20}) e Q = 70, então os resultados acima
  não são afetados. Com dados fracionários, porém, a função aceitaria sobrecarga real, e a
  correção (exigir inteiros ou escalar e validar as rotas devolvidas) fica pendente antes de
  qualquer uso com dados reais.

## 4. Desvios do pré-registro

`neo_multi` e `neo_busca` foram criados **depois** de observar o travamento do MIP, para separar
"o surrogate é útil?" de "o embutimento funciona?". Usam o mesmo modelo, os mesmos rótulos e as
mesmas instâncias.

## 5. Leitura

- **H6 (o surrogate melhora a decisão final) não se sustenta neste cenário.** A aproximação
  contínua clássica, linear e sem treino, empata ou vence todas as variantes neurais com custo
  online de 1,5 s e custo offline zero.
- **O surrogate tem erro baixo e discrimina mal.** Um MAPE de 7,5% parece bom, mas as
  configurações concorrentes diferem em 1% a 5%, e um τ de 0,34 mostra que a ordem prevista
  pouco acompanha a real. Há indício de exploração dos erros: o surrogate superestima em geral
  (viés mediano de +3,3% nas configurações da CA), mas só +0,6% nas configurações que a busca
  neural escolheu. A busca vai justamente para onde o modelo é relativamente otimista.
- **Sinal de escala:** em n = 100, a busca guiada pelo surrogate teve mediana melhor que a CA.
  Com 4 instâncias, é só uma hipótese para testar em escala maior.
- **FLP→VRP é um baseline fraco** (+4,5%). Superá-lo não demonstra valor; a comparação
  relevante é com a aproximação contínua.

## 6. Pontos fortes e fracos

| Fortes | Fracos |
|---|---|
| Princípio elegante: agregação linear torna o surrogate embutível | Embutimento big-M não otimiza no SCIP (gap de 591%, 0 melhorias) |
| Treino barato (23 min de rótulos + 3 min) | Erro de previsão comparável às diferenças entre decisões |
| Busca guiada é rápida (2 s) | Não supera uma aproximação clássica sem treino |
| Avaliação final por rotas reais evita autoengano | Rótulos heurísticos (OR-Tools 1 s), não ótimos |

## 7. Melhorias e recombinações sugeridas

1. **Modelo residual sobre a aproximação contínua.** Aprender só a correção
   `custo_real − CA` (doc 09). A CA já é forte, e o resíduo exige menos dados e preserva a
   ordem.
2. **Treino orientado à decisão (decision-focused / ranking loss)**: otimizar a ordem entre
   configurações concorrentes de cada instância, em vez do MSE do custo.
3. **Correção iterativa (Caminho C do doc 09):** reavaliar com rotas reais as configurações
   exploradas pela busca e retreinar com elas. Isso ataca a exploração dos erros do surrogate.
4. **Embutimento mais forte:** formulações de ReLU com envoltória convexa ou partições
   (Anderson et al., 2020), ou decomposição por depósito com geração de colunas. Uma alternativa
   é obter acesso acadêmico ao Gurobi e reproduzir no nível R1.
5. **Reprodução R1** com os pesos e benchmarks oficiais (Prodhon, Tuzun–Burke, Schneider)
   assim que houver licença. É a única forma de comparar com os números do artigo.
