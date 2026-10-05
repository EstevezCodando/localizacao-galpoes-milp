# Teste fechado — as duas rodadas lado a lado

## Integral primal

| Hipótese | A − B local (IC 95%) | p Holm local | A − B nuvem (IC 95%) | p Holm nuvem | Veredito |
|---|---|---:|---|---:|---|
| H1 `adaptativa:pl` − `completo` | -0.0586 (-0.0836 a -0.0310) | 0.0000 | -0.0285 (-0.0451 a -0.0080) | 0.0000 | confirmada |
| H2 `adaptativa:gnn` − `adaptativa:pl` | +0.0392 (+0.0135 a +0.0706) | 0.0043 | +0.0385 (+0.0129 a +0.0693) | 0.0048 | confirmada |
| H3 `hibridopl:rotacao` − `adaptativa:pl` | -0.0077 (-0.0179 a +0.0007) | 0.6589 | -0.0046 (-0.0135 a +0.0024) | 0.7230 | sem diferença nas duas |
| H4 `adaptativa:pl` − `kernel` | +0.0076 (-0.0012 a +0.0176) | 0.6381 | +0.0058 (-0.0010 a +0.0139) | 0.4077 | sem diferença nas duas |

## Desvio final (p.p.)

| Hipótese | A − B local (IC 95%) | p Holm local | A − B nuvem (IC 95%) | p Holm nuvem | Veredito |
|---|---|---:|---|---:|---|
| H1 `adaptativa:pl` − `completo` | -3.1028 (-4.0677 a -2.1826) | 0.0000 | -0.9024 (-1.3091 a -0.4918) | 0.0001 | confirmada |
| H2 `adaptativa:gnn` − `adaptativa:pl` | +0.5721 (+0.1379 a +1.0465) | 0.0429 | +0.5171 (+0.1343 a +0.9404) | 0.0557 | não replicada |
| H3 `hibridopl:rotacao` − `adaptativa:pl` | -0.5369 (-1.0618 a -0.1164) | 0.0495 | -0.2801 (-0.5935 a +0.0084) | 0.0557 | não replicada |
| H4 `adaptativa:pl` − `kernel` | -0.3189 (-0.6912 a +0.0345) | 0.0495 | -0.2033 (-0.4510 a +0.0408) | 0.4909 | não replicada |

## Tabela do conjunto agregado (64 instâncias sintéticas)

| Método | Integral local | Integral nuvem | Desvio mediano local | Desvio mediano nuvem |
|---|---:|---:|---:|---:|
| `hibridopl:rotacao` | 0.0420 | 0.0280 | 0.35% | 0.30% |
| `kernel` | 0.0422 | 0.0269 | 0.74% | 0.53% |
| `adaptativa:pl` | 0.0497 | 0.0326 | 0.50% | 0.35% |
| `hibrido:rotacao` | 0.0677 | 0.0478 | 0.62% | 0.30% |
| `adaptativa:gnn` | 0.0889 | 0.0712 | 0.64% | 0.54% |
| `completo` | 0.1083 | 0.0611 | 3.72% | 1.27% |

## Medição

| Rodada | Execuções | Mediana CPU/parede | Marcadas (< 0,9) |
|---|---:|---:|---:|
| local | 1390 | 0.957 | 229 |
| nuvem | 1390 | 1.000 | 0 |