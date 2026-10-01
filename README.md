<div align="center">

# Localização de Galpões com MILP

**Onde abrir centros de distribuição e qual deles atende cada região, ao menor custo.**

Localização capacitada com fonte única (SSCFLP) sobre pedidos reais do Brasil, resolvida por heurística,
matheurística (LNS) e programação inteira mista, com frete oficial da ANTT, aluguel de galpões coletado
e validação contra ótimos publicados.

`Python 3.12` · `OR-Tools (SCIP, GLOP)` · `scikit-learn` · `uv` · `mypy --strict` · `ruff` · `pytest`

[Mapa e resultados](docs/index.html) · [Estudo integrado](docs/estudo_integrado/index.html)

</div>

---

## O problema

Uma empresa precisa decidir **quais centros de distribuição abrir** entre vários candidatos e **a qual centro cada região de demanda será atendida**. Cada centro tem capacidade limitada e um custo fixo para operar, e cada região é atendida por um único centro (fonte única). Abrir poucos centros barateia a estrutura, mas encarece o frete; abrir muitos faz o contrário. O objetivo é o equilíbrio de menor custo total.

```
min   Σ f_i y_i  +  Σ d_j c_ij x_ij  +  p Σ d_j (1 − Σ_i x_ij)
s.a.  Σ_i x_ij ≤ 1                       para cada região j
      Σ_j d_j x_ij ≤ Q_i y_i             capacidade, ligada à abertura
      x_ij ≤ y_i                         reforço: relaxação linear mais justa
      x, y binários
```

`y_i` abre o centro *i*; `x_ij` atribui a região *j* ao centro *i*. A penalidade `p` por pedido não atendido mantém toda instância viável e torna explícito o custo de adiar pedidos, em vez de esconder inviabilidade.

## Como o trabalho foi conduzido

O projeto avançou em etapas, cada uma respondendo a uma dúvida da anterior.

### 1. Modelo e referências de comparação

Um modelo único ([`solvers/model.py`](src/alocacao_capacitada/solvers/model.py)) serve ao MILP e à relaxação linear, que fornece um **limite inferior** para medir a distância de qualquer solução ao ótimo. Como comparação, uma heurística gulosa e uma busca local. Toda solução passa por uma **avaliação única** ([`domain/evaluation.py`](src/alocacao_capacitada/domain/evaluation.py)): nenhum método avalia a si mesmo, e o domínio (`Instance`, `Solution`) não depende de nenhum solver.

### 2. Dados reais e custos reais

Os dados são 96.476 pedidos entregues do [Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), agrupados em 850 regiões (prefixo de CEP de 3 dígitos). O frete segue o piso mínimo da ANTT (Res. 6.084/2026) e o custo fixo vem de um levantamento de aluguel de galpões (fev/2026).

```
bronze ──▶ silver ──▶ gold ──▶ instância ──▶ solvers ──▶ avaliação única
HTML/CSV   tabelas    frete,     imutável     guloso       custo, viabilidade,
brutos +   tipadas    aluguel                 busca local  pedido não atendido
hash/data             rastreável              LNS · MILP
                      ao bronze               PL (limite)
```

A coleta de páginas respeita o `robots.txt`, usa user-agent identificado, espera 5 s entre requisições e grava hash, data e status ([`lake/bronze.py`](src/alocacao_capacitada/lake/bronze.py)). Ela usa o [Cavuca](https://github.com/EstevezCodando/Cavuca), que é **dependência opcional**: sem ele, tudo o mais funciona e basta fornecer outra função de busca ao coletor.

### 3. Matheurística (LNS)

Para instâncias maiores, o MILP sozinho perde fôlego. O LNS reotimiza, a cada iteração, só uma vizinhança de regiões com um subproblema exato, partindo de uma boa solução. O tamanho da vizinhança virou parâmetro; os testes levaram o padrão a 50 regiões.

### 4. Validação externa

Antes de confiar nos números, o modelo foi confrontado com resultados publicados:

- **OR-Library:** o modelo com demanda divisível reproduz os 7 ótimos publicados (cap41–44, 51, 61, 71). Isso valida a formulação, não o LNS.
- **Holmberg et al. (1999):** 71 instâncias de fonte única com ótimo publicado, o benchmark certo para este problema.

A validação revelou problemas reais: em cap41–44 e cap51 há clientes com demanda maior que a capacidade de qualquer centro, o que as torna inviáveis com fonte única; alguns arquivos de Holmberg trazem cabeçalhos de e-mail de 1995 e bytes nulos, que o carregador ignora; e coeficientes de frete de um blog diferiam dos oficiais em até 8%, o que tornou obrigatória a fonte primária.

### 5. Sensibilidade e demanda incerta

Análise de sensibilidade nos parâmetros e um modelo de dois estágios com demanda incerta (SAA, VSS e EVPI), para saber se vale planejar sob incerteza ou se o determinístico basta.

### 6. Dados territoriais, aprendizado de máquina e rede nacional

Uma camada de dados abertos (IBGE/Censo 2022, Ipeadata, malhas do IBGE, OpenStreetMap e rotas OSRM) alimenta:

- um **modelo supervisionado de demanda** por município (Poisson e gradient boosting);
- uma **previsão de população** com validação contra o Censo 2022;
- uma **clusterização** dos municípios;
- um **score de candidatos** para pré-selecionar locais antes do MILP;
- uma **rede nacional** (400 nós, 60 candidatos) com cenários de demanda futura.

## Resultados

### Olist: distância até o limite inferior (quanto menor, melhor)

Frete ANTT de 2 eixos. 30 s em 50×15 e 100×30; 60 s em 200×50. LNS com 3 a 5 sementes.

| Instância (regiões × centros) | Guloso | Busca local | MILP | MILP + aquecimento | LNS (média) |
|---|---:|---:|---:|---:|---:|
| 50 × 15 | 14,05% | 5,55% | 1,70% | 1,15% | **0,85%** |
| 100 × 30 | 4,63% | 3,93% | 1,92% | **1,51%** | 1,57% |
| 200 × 50 | 5,01% | 3,42% | 1,80% | 1,99% | **0,63%** |

O limite do PL garante que o ótimo inteiro está no máximo nessa distância; não é o ótimo em si.

### Holmberg et al. (1999): capacidade apertada, ótimo publicado

71 instâncias, até 30 centros e 200 clientes; 10 s por método; gap em relação ao ótimo publicado de fonte única.

| Método | Gap médio | Mediana | Igual à referência¹ | Provado pelo solver² | Tempo médio |
|---|---:|---:|---:|---:|---:|
| Guloso | 46,56% | 49,13% | 0 | – | 0,0 s |
| Busca local | 3,14% | 2,81% | 3 | – | 0,6 s |
| LNS, vizinhança de 12 | 2,05% | 1,24% | 18 | – | 10,0 s |
| LNS, vizinhança de 60 | 0,59% | 0,06% | 30 (+1 a 0,0095%) | – | 10,1 s |
| **MILP** | **0,20%** | **0,00%** | **61** | **60** | 2,6 s |

¹ Custo idêntico ao publicado (tolerância relativa 10⁻⁶). ² Status `OTIMO` do solver; coincidir com a referência e provar a otimalidade são evidências diferentes.

**Achado central.** Nos recortes do Olist, de capacidade folgada, o LNS fica no nível do MILP ou melhor; com capacidade apertada (Holmberg) o MILP domina, e o LNS só se aproxima com vizinhanças maiores. A folga de capacidade é uma hipótese para a diferença, não um fato isolado: as famílias de instâncias também diferem em geografia, custos e tamanho.

### Custos reais

**Aluguel.** 8.700+ anúncios de 18 cidades (preços pedidos, faixa de 1.001 a 3.000 m²), cobrindo 9 estados; os demais usam a média e são marcados como imputados. O custo fixo de cada centro é `aluguel do estado × capacidade ÷ densidade`, em que a densidade (pedidos por m² por mês) é premissa. Em 100 × 30:

| Densidade | Centros abertos | Custo fixo no custo total |
|---|---:|---:|
| Premissa antiga | 28 | 4,3% |
| 2 pedidos/m²/mês | 25 | 29,8% |
| 5 | 27 | 15,8% |
| 10 | 27 | 9,1% |
| 25 | 29 | 4,2% |

Com aluguel real a rede muda pouco, e o frete continua dominando a decisão.

**Um erro que quase passou.** A primeira tentativa usou 5.000 m² fixos por centro. O modelo abriu 2 centros e deixou 28.806 pedidos sem atendimento, porque em escala Olist um galpão desse tamanho custa mais que a entrega dos pedidos que comporta. A correção foi ligar a área à capacidade; o resultado antigo está em [`results/aluguel_area_fixa.csv`](results/aluguel_area_fixa.csv).

### Sensibilidade e demanda incerta

- **Folga de capacidade** é o único parâmetro que muda a rede de forma relevante (50×15: de 15 centros com folga 1,1 para 9 com folga 3,0).
- **Pedidos por veículo** escala o custo quase na razão inversa; é a premissa de maior impacto e a menos fundamentada.
- **Demanda incerta** (24 execuções): em 21 a política estocástica é igual à determinística (VSS = 0). Nas 3 restantes o VSS saiu negativo, o que é impossível com solução ótima: é gap do solver, não ganho.

### Rede nacional e ML

- **Previsão de população:** o backtest contra a série oficial dá 1,5% de erro e engana; contra o Censo 2022 real, a extrapolação simples erra 9%. O modelo usa taxa encolhida ao grupo, amortecimento e trava de plausibilidade.
- **Clusterização:** k = 3 foi o mais estável (ARI 0,98).
- **Rede nacional:** o projeto ótimo para 2025 custa +47% se a demanda de 2030 base se confirmar e +209% no cenário alto; o projeto robusto custa +2,3% em 2025. Uma previsão melhor de população não mudou a decisão (<1%). Distância viária contra linha reta: +1,3% de custo e apenas 70% dos módulos em comum.
- **Score de candidatos:** a regressão logística com os 60 melhores candidatos fica a 0,2% do melhor custo conhecido; o ranking por demanda local, a 4,0%.

### Mapa interativo

[`docs/index.html`](docs/index.html) traz o mapa da rede (MapLibre GL) com fronteiras do IBGE embutidas, ruas do [OpenFreeMap](https://openfreemap.org/) (dados © OpenStreetMap) e rotas viárias reais pelo OSRM. As rotas são só para visualização: o custo do modelo Olist continua em linha reta, e a estrada alonga o caminho em 1,29× (mediana) e 1,49× (p90).

## Limitações

- O Olist é um marketplace pequeno; escala e regras diferem de operações maiores.
- Capacidades, penalidade de não atendimento, pedidos por veículo e densidade de pedidos por m² são **premissas declaradas**. Nada aqui é ganho medido em operação real.
- A rede nacional usa distâncias do servidor de demonstração do OSRM (com imputação por linha reta × 1,29 quando falta rota); o exemplo Olist usa linha reta.
- O aluguel vem de preços pedidos em portais, não de contratos, e é média por estado.
- Os ótimos de Holmberg foram transcritos manualmente de um visualizador; vale conferir na fonte.
- O LNS não foi comparado com Kong (2021) nos mesmos dados.

## Trabalhos relacionados

| Trabalho | Relação |
|---|---|
| Holmberg, Rönnqvist & Yuan (1999), *An exact algorithm for the CFLP with single sourcing*, EJOR 113(3) | Origem das 71 instâncias |
| Guastaroba & Speranza (2014), *A heuristic for BILP problems: the SSCFLP*, EJOR 238(2) | Valores ótimos publicados e conjuntos de instâncias ([OR-Brescia](https://or-brescia.unibs.it/instances/instances_sscflp)) |
| Kong (2021), [*A matheuristic for the SSCFLP and its variants*](https://arxiv.org/abs/2112.12974) | LNS com subproblemas exatos, da mesma família do usado aqui |
| Ajide (2026), [*Two-Stage Stochastic Optimization for Capacitated Facility Location*](https://optimization-online.org/2026/09/two-stage-stochastic-optimization-for-capacitated-facility-location-under-demand-uncertainty/) | Mesmo arcabouço de VSS e EVPI; reporta VSS alto, aqui ≈ 0 |

## Como executar

```bash
uv sync
uv run pytest
uv run python -m alocacao_capacitada.benchmark --sizes 20x5 50x15
uv run python -m alocacao_capacitada.stage2 --sizes 100x30 --time-limit 60 --seeds 5
uv run python -m alocacao_capacitada.validation.run_holmberg --milp-time 10 --lns-time 10
uv run python -m alocacao_capacitada.analysis.rent_experiment --time-limit 20
uv run python scripts/build_site.py        # regenera docs/index.html
```

Baixe o dataset do Olist no Kaggle e extraia os CSVs em `data/raw/` (fora do git). As instâncias da OR-Library e de Holmberg, o aluguel e as resoluções da ANTT ficam em `data/bronze/` com proveniência. O `Dockerfile` existe, mas não foi testado.

## Estrutura

```
src/alocacao_capacitada/
├── domain/       Instance, Solution, evaluate(): puro
├── solvers/      guloso, busca local, LNS, MILP, PL, model.py
├── data/         Olist e construção da instância
├── lake/         bronze (coleta), silver, gold, frete, aluguel, OSRM
├── territory/    dados territoriais (IBGE, Ipeadata, OSM)
├── ml/           demanda, previsão, clusterização, score
├── network/      rede nacional, cenários e mapa
├── analysis/     sensibilidade, demanda incerta, aluguel real
├── validation/   OR-Library e Holmberg
└── benchmark.py · stage2.py
scripts/          geração do site e do estudo
docs/             página de resultados e estudo integrado
results/          saídas versionadas dos experimentos
```
