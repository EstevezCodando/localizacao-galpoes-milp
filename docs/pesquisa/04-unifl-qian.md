# Frente P06 — Aprender a aproximar o UniFL com GNN (Qian, Morris, Jegelka & Sohler, ICML 2026)

**Nível:** R3, reconstrução metodológica (o código oficial não foi confirmado). Dados:
`results/pesquisa/unifl/`. Código: `src/alocacao_capacitada/pesquisa/unifl.py`, `exp_unifl.py`.

## 1. O que o artigo propõe

Uniform Facility Location: todo ponto é cliente e candidato, custo de abertura uniforme `f`,
sem capacidade. O artigo usa uma MPNN cujo desenho segue o algoritmo de Mettu–Plaxton (MP,
3-aproximação). O MP estima um raio por ponto e abre pontos em ordem de raio. A rede é treinada
sem rótulos, minimizando o custo esperado sob aberturas independentes, com garantias de
aproximação sob as hipóteses do artigo.

**Classificação (Bengio, Lodi & Prouvost, P01):** aprendizado de ponta a ponta de uma heurística
construtiva, sem solver no laço, com função de perda derivada do objetivo.

## 2. Como foi implementado

| Componente | Implementação |
|---|---|
| Instâncias | n pontos no quadrado unitário (uniforme ou 4 a 8 clusters); `f = 0,19 n / k^1,5` com `k = sqrt(n)` |
| Ótimo | SCIP, formulação forte (`x_ij <= y_i`), partida pela busca local; provado em n = 100, 200 e 8 de 10 em 500 (300 s) |
| Mettu–Plaxton | raio exato pela equação `sum_j max(0, r_i - d_ij) = f` (verificada numericamente); abre se nenhum aberto estiver a menos de `2 r_i` |
| Busca local | add/drop/swap vetorizada, primeira melhora (Arya et al., 2004) |
| MPNN | 4 camadas, dim 64, grafo de 16 vizinhos, agregação soma + máximo; atributos de distância em unidades de `f`; variante com o raio de MP como atributo |
| Perda | `E[custo] = f sum p_i + sum_j sum_k c_jk p_k prod_{l<k}(1 - p_l) + M prod(1 - p)` sobre os 24 candidatos mais próximos de cada cliente, com `M = 2 d_24` |
| Treino | 200 instâncias (n = 100 e 200), 30 épocas, 2 sementes por variante; cerca de 5 a 6 min por modelo em CPU |
| Decodificação | limiar escolhido na validação (20 instâncias) e fechamento guloso dos centros que não pagam o próprio custo fixo |

## 3. Resultados

Razão de custo contra o ótimo provado (n ≤ 500) ou contra o melhor método (n = 1000).
Mediana sobre 20, 20, 10 e 6 instâncias. As variantes agregam as 2 sementes.

| Método | n=100 | n=200 | n=500 | n=1000 | tempo n=1000 |
|---|---:|---:|---:|---:|---:|
| Mettu–Plaxton | 1,120 | 1,123 | 1,132 | 1,186 | 0,07 s |
| MP + busca local | 1,0005 | 1,0019 | 1,0060 | 1,0005 | 1,7 s |
| MPNN original (pré-registrada) | 2,29 | 2,06 | 3,01 | 3,95 | 0,17 s |
| MPNN original + busca local | 1,0004 | 1,0022 | 1,0039 | 1,0022 | 2,4 s |
| **MPNN estável** | 1,109 | 1,054 | **1,027** | **1,026** | 0,27 s |
| MPNN estável + busca local | 1,0000 | 1,0015 | 1,0032 | 1,0027 | 2,7 s |
| MPNN estável com raio de MP | 1,083 | 1,039 | 1,027 | 1,027 | 0,29 s |

p90 da MPNN estável: 1,24 (n=100), 1,12 (n=200), 1,045 (n=500), 1,04 (n=1000). O ótimo exato
levou 1,4 s (n=100), 5,9 s (n=200) e 92 s (n=500, mediana).

### Resultado negativo e desvio do pré-registro

Na versão pré-registrada, **3 de 4 modelos colapsaram**: abriam 1 centro e ficavam 3 a 6 vezes
acima do ótimo. Só a semente 0 sem raio funcionou (1,09 / 1,05 / 1,03). O diagnóstico é a
saturação do sigmoide. O gradiente em `p = 0` aponta para abrir, mas depois de uma oscilação
inicial os logits ficam muito negativos e o gradiente desaparece. Não é um mínimo da perda.

A correção, aplicada **depois** de ver o teste e por isso registrada como desvio, foi
`p = 0,01 + 0,98 sigmoide`, clipping de gradiente em 1,0 e lr de 5e-4 em vez de 1e-3. Com ela,
**4 de 4 sementes ficaram estáveis**. O teste é o mesmo; a escolha do limiar continua na
validação.

## 4. Leitura

- **A tese de generalização do artigo se sustenta nesta reconstrução.** A MPNN estável,
  treinada só em n ≤ 200, supera o MP a partir de n = 200 e melhora com a escala (1,027 em
  n = 1000 contra 1,186 do MP). O MP piora com n; a rede não.
- **O raio de MP como atributo não ajudou** de forma consistente (1,083 contra 1,109 em
  n = 100; empate a partir de n = 500). A rede recupera essa informação sozinha.
- **O ganho prático é pequeno quando há busca local.** MP + busca local e MPNN + busca local
  ficam entre 1,000 e 1,006 em todos os tamanhos. A inicialização aprendida não muda a
  qualidade final e custa cerca de 0,2 s a mais de inferência em n = 1000.
- **Fragilidade de treino:** a perda não supervisionada é sensível a saturação. Em uso real
  isso exige monitorar o colapso e ter um fallback (o próprio MP).

## 5. Pontos fortes e fracos

| Fortes | Fracos |
|---|---|
| Sem rótulos: treino barato (cerca de 5 min em CPU) | Instável sem regularização da saída |
| Generaliza de n ≤ 200 para n = 1000 | Não supera a busca local clássica, só a substitui como partida |
| Inferência rápida (0,27 s em n = 1000) | Garantias valem só para UniFL (sem capacidade, custo uniforme) |
| Perda diferenciável derivada do objetivo | Busca local de primeira melhora em Python domina o tempo em n = 1000 |

## 6. Melhorias e recombinações sugeridas

1. **Transferir a perda de custo esperado para o SSCFLP** como pré-treino não supervisionado
   do ranqueador da Etapa 1. Ela dispensa as 4 CPU-h de rótulos do SCIP. A capacidade entraria
   como penalidade esperada de sobrecarga.
2. **Usar a MPNN como gerador de vizinhanças** (quais centros abrir ou fechar) dentro da busca
   local, em vez de só como partida. Em n = 1000 a busca local leva 2,4 s e a inferência 0,27 s.
3. **Amostragem múltipla:** decodificar várias soluções das probabilidades e polir só a
   melhor. É paralelizável e explora a incerteza da saída.
4. **Verificar o artigo na fonte:** comparar a arquitetura exata da v3 com esta reconstrução
   antes de qualquer alegação de reprodução.
