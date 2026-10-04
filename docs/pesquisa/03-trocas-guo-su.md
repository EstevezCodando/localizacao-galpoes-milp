# Frentes P04 e P05 — Trocas guiadas por aprendizado em localização sobre redes (Guo, Xu & Jin 2023; Su et al., SIGSPATIAL 2024)

**Nível:** R3, reconstrução metodológica. P04 não tem repositório confirmado; P05 tem só uma
demonstração. Dados: `results/pesquisa/swap_v2/` (v1 em `swap/`). Código:
`src/alocacao_capacitada/pesquisa/swap.py`, `exp_swap.py`.

## 1. O que os artigos propõem

Os dois usam busca por trocas (fechar um centro e abrir outro) em que uma política aprendida
propõe as trocas. P04 treina a política por reforço (PPO) para p-mediana em grafos. P05 usa uma
GNN guiada por conhecimento de domínio e relata até 1000× de aceleração com menos de 5% de perda
de acessibilidade em quatro cidades. O mecanismo comum de ganho é **avaliar muito menos trocas**
do que as `p (n − p)` de uma iteração completa.

**Classificação (P01):** ML aprende uma decisão interna de uma heurística, a escolha da
vizinhança, e o algoritmo continua avaliando os movimentos exatamente.

## 2. Como foi implementado

| Componente | Implementação |
|---|---|
| Problema | p-mediana ponderada em grafo viário sintético (6 vizinhos, desvio de 0% a 40% por aresta, caminhos mínimos por Dijkstra), p = round(sqrt(n)) |
| Famílias | `uniforme` (pesos U(1, 10)) e `cidade` (pesos concentrados em polos) |
| Busca completa | melhor melhora em todas as trocas, Δ exato vetorizado com 1ª e 2ª menor distância (Teitz & Bart) |
| Filtro | avalia exatamente só as `4 × k_in` trocas mais promissoras, k_in ∈ {8, 16, 32} |
| Filtro clássico | entrada pelo ganho de adição `sum w max(0, d1 − d_k)`, saída pela perda de remoção. É o "conhecimento" que P05 injeta |
| Filtro aprendido | 2 LightGBM (entrada e saída) imitando o melhor ganho exato de cada candidato em 1.054 estados de 30 redes com n = 200. **Substitui o PPO de P04** (desvio documentado); coleta em 4 s e treino em 8 s |
| Regimes | **com fallback** (quando o filtro não melhora, faz uma varredura completa e só para em ótimo local verdadeiro) e **sem fallback** (para quando o filtro não melhora, como nos artigos) |
| Teste | n = 200 (10 redes, ótimo SCIP provado), 500 (8) e 1000 (5); 3 inícios aleatórios por rede, iguais para todos os filtros |

A v1 recalculava a cada iteração um atributo estático, uma ordenação O(n² log n) da matriz de
distâncias, o que penalizava injustamente o aprendido. A v2 guarda esse atributo em cache, com
valores verificados como idênticos, e acrescenta o regime sem fallback.

## 3. Resultados (k_in = 16)

Desvio mediano contra o ótimo (n = 200) ou contra o melhor encontrado (n ≥ 500). A aceleração é
o tempo da busca completa dividido pelo tempo do método, pareada pelo mesmo início.

| Filtro | Fallback | n=200 desvio / acel. | n=500 desvio / acel. | n=1000 desvio / acel. |
|---|---|---|---|---|
| completo | — | 0,56% / 1,0× | 0,30% / 1,0× | 0,41% / 1,0× |
| clássico | sim | 0,55% / 1,14× | 0,32% / **1,55×** | 0,23% / **1,57×** |
| aprendido | sim | 0,48% / 0,17× | 0,25% / 0,91× | 0,42% / 1,21× |
| aleatório | sim | 0,47% / 1,03× | 0,28% / 1,16× | 0,20% / 1,10× |
| clássico | não | 5,1% / 4,1× | 6,6% / 23× | 6,6% / **47×** |
| aprendido | não | **3,5%** / 0,31× | 8,6% / 4,9× | 8,8% / 14× |
| aleatório | não | 11,7% / 6,6× | 17,7% / 164× | 19,9% / 622× |

Avaliações exatas economizadas, com fallback: 1,0 a 2,5×. O fallback dispara de 7 a 38 vezes por
execução. Sem fallback, o filtro avalia de 60 a 1.500 vezes menos trocas.

## 4. Leitura

- **Boa parte da aceleração relatada nesse tipo de trabalho vem do filtro, não do
  aprendizado.** Um filtro clássico de duas linhas, sem treino, dá 47× com 6,6% de perda em
  n = 1000, a mesma ordem de grandeza de "grande aceleração com menos de 5% a 10% de perda". Um
  filtro aleatório chega a 622× com 20% de perda. A comparação que importa é aprendido contra
  clássico **com o mesmo orçamento de avaliações**.
- **O aprendido só vence o clássico na escala de treino** (3,5% contra 5,1% em n = 200). Em
  n = 500 e 1000 perde (8,6% a 8,8% contra 6,6%). A imitação feita em n = 200 não generaliza para
  redes maiores, onde a estrutura de ganhos muda.
- **Custo de inferência:** o aprendido é sempre mais lento que o clássico (14× contra 47× sem
  fallback em n = 1000), porque calcula mais atributos e chama o modelo a cada iteração (o ponto
  de P03 sobre o custo de inferência).
- **Com fallback, não há o que aprender:** a qualidade é a da busca completa e o ganho de tempo
  fica limitado (1,1 a 1,6×) pelas varreduras completas de verificação.

## 5. Pontos fortes e fracos

| Fortes | Fracos |
|---|---|
| Mecanismo geral e simples: ranquear e avaliar só o topo | Ganho atribuível ao aprendizado é pequeno e não generaliza na escala |
| Treino baratíssimo (12 s) | Imitação do melhor ganho é míope (1 passo); P04 usa RL justamente por isso |
| Com fallback, a qualidade é garantida igual à da busca completa | Sem fallback, perde de 3% a 9% contra 0,3% a 0,6% da completa |

## 6. Melhorias e recombinações sugeridas

1. **Aprendizado contra o clássico, não contra o completo:** treinar o resíduo sobre o score
   clássico (`ganho_add − perda_rem`) com rótulos de várias escalas.
2. **Filtro adaptativo:** começar com k_in pequeno e dobrar quando o filtro falha, em vez da
   varredura completa. É um meio-termo contínuo entre os dois regimes testados.
3. **RL ou imitação com horizonte maior** (P04), com rótulos pelo ganho após h passos e não
   só pelo ganho imediato.
4. **Combinar com a fast interchange de Whitaker / Resende–Werneck**, que já reduz o custo da
   varredura completa com estruturas incrementais. Esse é o baseline clássico forte que um
   artigo da área precisa vencer.
