# Diagnósticos de validação — resultados

## D1 — subproblemas repetidos reexecutados com limite maior

| Contexto | Execuções | Iterações (mediana) | Taxa de repetição (mediana) | Subproblemas reexecutados |
|---|---:|---:|---:|---:|
| expansao | 16 | 387 | 70.3% | 128 |
| pl | 16 | 266 | 46.9% | 128 |

| Contexto | Limite | esgotado (ótimo provado) | falta de tempo (melhorou) | indeterminado | Tempo mediano até terminar | Variáveis (mediana) |
|---|---:|---:|---:|---:|---:|---:|
| expansao | 5 s | 126 | 1 | 1 | 0.00 s | 202 |
| expansao | 30 s | 126 | 2 | 0 | 0.00 s | 202 |
| pl | 5 s | 126 | 1 | 1 | 0.01 s | 160 |
| pl | 30 s | 126 | 2 | 0 | 0.00 s | 160 |

Quando houve melhoria, o ganho mediano foi de 0.068% do custo.

Por tipo de subproblema, no limite maior:

| Tipo | esgotado (ótimo provado) | falta de tempo (melhorou) |
|---|---:|---:|
| fronteira | 72 | 2 |
| interior | 96 | 0 |
| liberacao | 31 | 2 |
| regret | 53 | 0 |

## D2 e D3 — agrupamento e número de centros candidatos

| Método | Integral primal | Desvio final mediano | Subproblemas por execução |
|---|---:|---:|---:|
| `clns:rotacao` | 0.0332 | 2.46% | 587 |
| `clns:rotacao+alea` | 0.0376 | 3.06% | 1036 |
| `clns:rotacao+k30` | 0.0235 | 1.22% | 259 |
| `hibridopl:rotacao` | 0.0162 | 0.40% | 474 |
| `hibridopl:rotacao+alea` | 0.0163 | 0.41% | 763 |
| `hibridopl:rotacao+k20` | 0.0163 | 0.34% | 215 |
| `hibridopl:rotacao+k30` | 0.0169 | 0.36% | 154 |

Diferença média **variante − referência** (negativo favorece a variante); desvio em pontos percentuais.

| Diag. | Referência | Variante | Métrica | Diferença (IC 95%) | Variante melhor / referência melhor | p | p (Holm) |
|---|---|---|---|---|---|---:|---:|
| D2 | `clns:rotacao` | `clns:rotacao+alea` | integral | +0.0045 (-0.0015 a +0.0101) | 7 / 9 | 0.193 | 0.771 |
| D2 | `hibridopl:rotacao` | `hibridopl:rotacao+alea` | integral | +0.0001 (-0.0003 a +0.0006) | 8 / 8 | 0.706 | 1.000 |
| D3 | `clns:rotacao` | `clns:rotacao+k30` | integral | -0.0097 (-0.0160 a -0.0037) | 12 / 4 | 0.021 | 0.107 |
| D3 | `hibridopl:rotacao` | `hibridopl:rotacao+k20` | integral | +0.0001 (-0.0008 a +0.0011) | 10 / 6 | 0.782 | 1.000 |
| D3 | `hibridopl:rotacao` | `hibridopl:rotacao+k30` | integral | +0.0007 (-0.0002 a +0.0020) | 7 / 9 | 0.404 | 1.000 |
| D2 | `clns:rotacao` | `clns:rotacao+alea` | desvio | +0.5251 (-0.0651 a +1.0801) | 4 / 12 | 0.074 | 0.296 |
| D2 | `hibridopl:rotacao` | `hibridopl:rotacao+alea` | desvio | +0.0229 (-0.0344 a +0.0907) | 6 / 8 | 0.583 | 1.000 |
| D3 | `clns:rotacao` | `clns:rotacao+k30` | desvio | -1.4447 (-2.0292 a -0.8873) | 14 / 2 | 0.000 | 0.002 |
| D3 | `hibridopl:rotacao` | `hibridopl:rotacao+k20` | desvio | -0.0907 (-0.2448 a +0.0518) | 9 / 5 | 0.326 | 0.977 |
| D3 | `hibridopl:rotacao` | `hibridopl:rotacao+k30` | desvio | +0.0111 (-0.1401 a +0.1697) | 8 / 6 | 0.952 | 1.000 |