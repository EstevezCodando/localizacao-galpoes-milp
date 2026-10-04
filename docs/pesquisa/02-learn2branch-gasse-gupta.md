# Frentes P02 e P03 — Aprender a ramificar imitando strong branching (Gasse et al., NeurIPS 2019; Gupta et al., NeurIPS 2020)

**Nível:** R3, reconstrução. O repositório oficial (`ds4dm/learn2branch`) depende do Ecole e de
PySCIPOpt/SCIP antigos, sem suporte a Python 3.12 no Windows, e não foi executado. Dados:
`results/pesquisa/l2b/`. Código: `src/alocacao_capacitada/pesquisa/l2b.py`, `exp_l2b.py`.

## 1. O que os artigos propõem

**P02:** uma GCNN sobre o grafo bipartido variável–restrição do PL do nó imita o strong branching
(SB), a regra cara que escolhe a variável testando os dois ramos. Ela é avaliada em quatro
famílias, entre elas `facilities` (CFLP de Cornuéjols, atribuição contínua). No SCIP 6, a GCNN
supera o *reliability pseudocost branching* (RPB, padrão do SCIP) em tempo.

**P03:** a GCNN é cara em CPU, então um modelo híbrido usa a GNN só na raiz e MLPs baratas nos
nós. O artigo relata até 26% menos tempo. O ponto central é que o **custo de inferência por nó**
decide o ganho líquido.

**Classificação (P01):** ML aprende uma decisão algorítmica interna do solver exato, por
imitação de um especialista. O solver preserva a otimalidade, e o aprendizado só muda a ordem
da busca.

## 2. Como foi implementado

| Componente | Implementação | Diferença em relação a P02 |
|---|---|---|
| Instâncias | `facilities` reimplementado a partir do gerador de Cornuéjols et al.: 100 clientes × 100 centros, razão 5, atribuição contínua, cobertura de capacidade e `x_ij <= y_j` | mesma família, gerador reconstruído e não copiado |
| Solver | SCIP 10.0 via PySCIPOpt 6.2.1, 1 thread, relógio de parede | P02 usou SCIP 6.0.1 |
| Professor | SB completo via `getVarStrongbranch` (score = produto dos ganhos) em 30% dos nós; nos demais, relpscost | P02 usou o vanillafullstrong do SCIP com exploração |
| Estado | 11 atributos por candidato (fração, LP, custo fixo, capacidade, custo reduzido, pseudocustos, profundidade, número de candidatos), no estilo Khalil et al. (2016) | P02 usa o grafo variável–restrição completo; extraí-lo em Python a cada nó custaria mais que o nó |
| Modelo | LightGBM LambdaRank por nó, o equivalente ao baseline LambdaMART de P02 | a GCNN não foi treinada (ver §4) |
| Coleta | 36 instâncias, 150 s cada, 0,81 h | **953 amostras de treino** e 139 de validação; P02 usou cerca de 100 mil |

## 3. Resultados de imitação (validação, 119 nós com ≥ 2 candidatos)

| Regra | acc@1 | acc@5 |
|---|---:|---:|
| aleatória | 0,235 | 0,824 |
| produto de pseudocustos | 0,218 | 0,807 |
| custo reduzido | 0,210 | 0,790 |
| **mais fracionária** | **0,387** | **0,933** |
| LightGBM (aprendida) | 0,361 | 0,924 |

Mediana de **6 candidatos por nó**: com tão poucos candidatos, acc@5 é quase trivial (a regra
aleatória já dá 0,82). A regra aprendida **empata com "mais fracionária"**, uma regra que a
literatura considera fraca em tempo de solução (Achterberg, Koch & Martin, 2005).

## 4. Achado estrutural: o SCIP 10 deixou pouco para aprender

No piloto, 3 de 4 instâncias 60×60 fecharam **na raiz (1 nó)** depois de 10 a 40 s de cortes.
Na 100×100, o relpscost precisou de 291 nós em 84 s. Na coleta (100×100, SB em 30% dos nós),
**31 de 36 instâncias terminaram antes de 150 s** (mediana de 68 s, já incluindo o custo do SB),
com mediana de apenas 18 amostras por instância (mínimo 2, máximo 150). Desde o SCIP 6 de P02, os separadores e o
presolve do SCIP passaram a resolver a maior parte de `facilities` antes do branching. Com
poucos nós e poucos candidatos, há pouco sinal para imitação, e o teto de ganho de qualquer
regra de branching é limitado pela fração do tempo gasta em B&B.

## 5. Resultados no solver

20 instâncias de teste, limite de 300 s, um núcleo. Fonte: `results/pesquisa/l2b/teste.csv`.
Médias geométricas deslocadas (deslocamento de 1 s no tempo e de 100 nós), como em Gasse et al.

| Regra | Tempo (s) | Nós | Resolvidas | Tempo em inferência |
|---|---:|---:|---:|---:|
| Aprendida (LightGBM por candidato) | 91,7 | 222 | 19/20 | 0,37% |
| `pscost` | 93,2 | 226 | 19/20 | – |
| `relpscost` (padrão do SCIP) | 100,1 | 90 | 19/20 | – |
| `fullstrong` | 107,4 | 40 | 18/20 | – |

Leitura:

- A regra aprendida foi 8% mais rápida que o padrão do SCIP (mais rápida em 14 de 20
  instâncias; Wilcoxon pareado, p = 0,021), **mas é indistinguível do `pscost`** (p = 0,93), que
  não usa aprendizado nenhum. Ela usa 2,5 vezes mais nós que o `relpscost` e compensa com nós
  mais baratos, exatamente o perfil do `pscost`.
- A acurácia de imitação (36% no topo-1, seção 3) está no nível da heurística "variável mais
  fracionária" (39%). O modelo não aprendeu o *strong branching*; aprendeu uma regra barata.
- O custo de inferência é desprezível (0,37% do tempo), então o resultado não é limitado pela
  inferência em Python, e sim pela qualidade da decisão.
- Com 20 instâncias e uma semente, o resultado é indicativo. A conclusão que se sustenta é a
  negativa: neste solver e nesta família, o ganho atribuído ao aprendizado é reproduzido por uma
  regra clássica de mesma função.

## 6. Pontos fortes e fracos

| Fortes | Fracos |
|---|---|
| Preserva a otimalidade: só reordena a busca | Dados escassos com o solver moderno (953 amostras) |
| Coleta e treino baratos (0,81 h + 0,7 s) | Atributos por candidato perdem a estrutura do PL que a GCNN usa |
| Mede diretamente o custo de inferência (P03) | Ganho limitado pela fração de tempo em B&B, pequena no SCIP 10 |

## 7. Melhorias e recombinações sugeridas

1. **Reproduzir R1 num ambiente histórico** (Linux, Python 3.6 a 3.8, Ecole e SCIP 7), para
   separar "o método não transfere" de "o solver evoluiu".
2. **Famílias mais difíceis para o SCIP 10**, como o próprio SSCFLP desta pesquisa, onde o
   SCIP passa centenas a milhares de nós em 60 s. É ali que uma regra de branching tem espaço.
3. **Extração de estado em C/Cython** (a API de observação do Ecole) antes de usar a GCNN; em
   Python, o custo por nó domina.
4. **Prioridades de branching estáticas aprendidas**, que custam zero por nó: o método
   `prioridade:gnn` da Etapa 1 testa exatamente isso no SSCFLP.
