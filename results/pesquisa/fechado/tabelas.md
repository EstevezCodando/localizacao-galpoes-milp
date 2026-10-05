# Teste fechado — tabelas

## teste (32 instâncias, 60 s)

| Método | n | Integral primal (IC 95%) | Desvio mediano | Desvio médio | Até 1% | Até 0,1% | V/E/D contra SCIP |
|---|---:|---|---:|---:|---:|---:|---|
| `kernel` | 32 | 0.0364 (0.0230–0.0584) | 0.81% | 1.46% | 56.2% | 37.5% | 22/4/6 |
| `hibridopl:rotacao` | 32 | 0.0385 (0.0247–0.0568) | 0.35% | 0.67% | 65.6% | 28.1% | 27/2/3 |
| `adaptativa:pl` | 32 | 0.0412 (0.0203–0.0754) | 0.29% | 0.82% | 65.6% | 43.8% | 25/4/3 |
| `hibrido:rotacao` | 32 | 0.0506 (0.0322–0.0736) | 0.55% | 0.84% | 68.8% | 25.0% | 27/2/3 |
| `adaptativa:gnn` | 32 | 0.0659 (0.0341–0.1066) | 0.90% | 1.43% | 53.1% | 21.9% | 21/4/7 |
| `completo` | 32 | 0.0794 (0.0662–0.0945) | 3.21% | 3.02% | 25.0% | 15.6% | – |

Certificado de ótimo antes do último estágio: `adaptativa:gnn` 1, `adaptativa:pl` 1

## gen_corredor (16 instâncias, 60 s)

| Método | n | Integral primal (IC 95%) | Desvio mediano | Desvio médio | Até 1% | Até 0,1% | V/E/D contra SCIP |
|---|---:|---|---:|---:|---:|---:|---|
| `hibridopl:rotacao` | 16 | 0.0289 (0.0199–0.0401) | 0.23% | 0.45% | 87.5% | 37.5% | 14/1/1 |
| `adaptativa:pl` | 16 | 0.0311 (0.0218–0.0431) | 0.73% | 1.10% | 62.5% | 31.2% | 11/3/2 |
| `kernel` | 16 | 0.0341 (0.0241–0.0467) | 0.85% | 1.02% | 56.2% | 18.8% | 10/1/5 |
| `hibrido:rotacao` | 16 | 0.0423 (0.0226–0.0749) | 0.18% | 0.36% | 81.2% | 43.8% | 14/1/1 |
| `adaptativa:gnn` | 16 | 0.0540 (0.0222–0.1110) | 0.15% | 0.53% | 81.2% | 37.5% | 12/1/3 |
| `completo` | 16 | 0.1057 (0.0799–0.1337) | 3.59% | 4.77% | 18.8% | 18.8% | – |

Certificado de ótimo antes do último estágio: `adaptativa:gnn` 0, `adaptativa:pl` 0

## gen_escala (16 instâncias, 120 s)

| Método | n | Integral primal (IC 95%) | Desvio mediano | Desvio médio | Até 1% | Até 0,1% | V/E/D contra SCIP |
|---|---:|---|---:|---:|---:|---:|---|
| `kernel` | 16 | 0.0616 (0.0385–0.0885) | 0.52% | 2.10% | 62.5% | 37.5% | 14/1/1 |
| `hibridopl:rotacao` | 16 | 0.0621 (0.0411–0.0862) | 0.58% | 0.83% | 62.5% | 31.2% | 13/0/3 |
| `adaptativa:pl` | 16 | 0.0854 (0.0475–0.1310) | 0.44% | 2.02% | 68.8% | 25.0% | 14/0/2 |
| `hibrido:rotacao` | 16 | 0.1275 (0.0904–0.1737) | 2.19% | 2.18% | 18.8% | 0.0% | 12/0/4 |
| `completo` | 16 | 0.1688 (0.1368–0.1991) | 7.00% | 6.35% | 25.0% | 12.5% | – |
| `adaptativa:gnn` | 16 | 0.1699 (0.0924–0.2650) | 2.03% | 3.65% | 37.5% | 18.8% | 9/4/3 |

Certificado de ótimo antes do último estágio: `adaptativa:gnn` 0, `adaptativa:pl` 0

## holmberg (71 instâncias, 30 s)

| Método | n | Integral primal (IC 95%) | Desvio mediano | Desvio médio | Até 1% | Até 0,1% | V/E/D contra SCIP |
|---|---:|---|---:|---:|---:|---:|---|
| `hibridopl:rotacao` | 71 | 0.0143 (0.0109–0.0183) | 0.00% | 0.13% | 97.2% | 84.5% | 7/57/7 |
| `adaptativa:pl` | 71 | 0.0146 (0.0111–0.0184) | 0.00% | 0.14% | 94.4% | 87.3% | 6/60/5 |
| `kernel` | 71 | 0.0191 (0.0122–0.0276) | 0.00% | 0.18% | 93.0% | 85.9% | 3/63/5 |
| `completo` | 71 | 0.0286 (0.0203–0.0386) | 0.00% | 0.20% | 94.4% | 87.3% | – |
| `hibrido:rotacao` | 71 | 0.0419 (0.0347–0.0497) | 0.00% | 0.39% | 87.3% | 76.1% | 3/53/15 |
| `adaptativa:gnn` | 71 | 0.0481 (0.0390–0.0582) | 0.00% | 0.57% | 90.1% | 84.5% | 1/60/10 |

Certificado de ótimo antes do último estágio: `adaptativa:gnn` 59, `adaptativa:pl` 59

Instâncias em que o ótimo publicado foi atingido (em todas as sementes / em alguma): `adaptativa:gnn` 59/59, `adaptativa:pl` 61/61, `completo` 61/61, `hibrido:rotacao` 53/57, `hibridopl:rotacao` 59/62, `kernel` 60/60

## olist (4 instâncias, 120 s)

| Método | n | Integral primal (IC 95%) | Desvio mediano | Desvio médio | Até 1% | Até 0,1% | V/E/D contra SCIP |
|---|---:|---|---:|---:|---:|---:|---|
| `hibrido:rotacao` | 4 | 0.2882 (0.0833–0.4931) | 0.17% | 0.27% | 100.0% | 25.0% | 3/0/1 |
| `completo` | 4 | 0.2923 (0.0329–0.5516) | 5.87% | 5.26% | 25.0% | 25.0% | – |
| `hibridopl:rotacao` | 4 | 0.3208 (0.1494–0.4923) | 0.18% | 0.35% | 75.0% | 25.0% | 3/0/1 |
| `kernel` | 4 | 0.3932 (0.0209–0.7654) | 4.73% | 4.69% | 50.0% | 25.0% | 1/3/0 |
| `adaptativa:pl` | 4 | 0.4076 (0.1372–0.7959) | 5.87% | 27.95% | 25.0% | 25.0% | 0/3/1 |
| `adaptativa:gnn` | 4 | 0.5308 (0.1325–0.9291) | 5.87% | 27.95% | 25.0% | 25.0% | 0/3/1 |

Certificado de ótimo antes do último estágio: `adaptativa:gnn` 0, `adaptativa:pl` 0

## Agregado (teste + gen_corredor + gen_escala, 64 instâncias)

| Método | n | Integral primal (IC 95%) | Desvio mediano | Desvio médio | Até 1% | Até 0,1% | V/E/D contra SCIP |
|---|---:|---|---:|---:|---:|---:|---|
| `hibridopl:rotacao` | 64 | 0.0420 (0.0320–0.0534) | 0.35% | 0.65% | 70.3% | 31.2% | 54/3/7 |
| `kernel` | 64 | 0.0422 (0.0316–0.0552) | 0.74% | 1.51% | 57.8% | 32.8% | 46/6/12 |
| `adaptativa:pl` | 64 | 0.0497 (0.0330–0.0700) | 0.50% | 1.19% | 65.6% | 35.9% | 50/7/7 |
| `hibrido:rotacao` | 64 | 0.0677 (0.0505–0.0873) | 0.62% | 1.06% | 59.4% | 23.4% | 53/3/8 |
| `adaptativa:gnn` | 64 | 0.0889 (0.0583–0.1239) | 0.64% | 1.76% | 56.2% | 25.0% | 42/9/13 |
| `completo` | 64 | 0.1083 (0.0932–0.1239) | 3.72% | 4.29% | 23.4% | 15.6% | – |

### Hipóteses — integral primal (A − B < 0 favorece A)

| Hipótese | A | B | n | A − B (IC 95%) | A melhor / B melhor | p | p (Holm) |
|---|---|---|---:|---|---|---:|---:|
| H1 | `adaptativa:pl` | `completo` | 64 | -0.0586 (-0.0836 a -0.0310) | 55 / 9 | 0.0000 | 0.0000 |
| H2 | `adaptativa:gnn` | `adaptativa:pl` | 64 | +0.0392 (+0.0135 a +0.0706) | 22 / 42 | 0.0014 | 0.0043 |
| H3 | `hibridopl:rotacao` | `adaptativa:pl` | 64 | -0.0077 (-0.0179 a +0.0007) | 31 / 33 | 0.6589 | 0.6589 |
| H4 | `adaptativa:pl` | `kernel` | 64 | +0.0076 (-0.0012 a +0.0176) | 36 / 28 | 0.3190 | 0.6381 |

### Hipóteses — desvio final, em pontos percentuais

| Hipótese | A | B | n | A − B (IC 95%) | A melhor / B melhor | p | p (Holm) |
|---|---|---|---:|---|---|---:|---:|
| H1 | `adaptativa:pl` | `completo` | 64 | -3.1028 (-4.0677 a -2.1826) | 50 / 7 | 0.0000 | 0.0000 |
| H2 | `adaptativa:gnn` | `adaptativa:pl` | 64 | +0.5721 (+0.1379 a +1.0465) | 15 / 32 | 0.0143 | 0.0429 |
| H3 | `hibridopl:rotacao` | `adaptativa:pl` | 64 | -0.5369 (-1.0618 a -0.1164) | 38 / 19 | 0.0247 | 0.0495 |
| H4 | `adaptativa:pl` | `kernel` | 64 | -0.3189 (-0.6912 a +0.0345) | 33 / 20 | 0.0492 | 0.0495 |

### Sensibilidade: integral primal sem as execuções marcadas por contenção

| Hipótese | A | B | n | A − B (IC 95%) | A melhor / B melhor | p | p (Holm) |
|---|---|---|---:|---|---|---:|---:|
| H1 | `adaptativa:pl` | `completo` | 39 | -0.0440 (-0.0730 a -0.0098) | 35 / 4 | 0.0001 | 0.0005 |
| H2 | `adaptativa:gnn` | `adaptativa:pl` | 38 | +0.0416 (+0.0110 a +0.0826) | 12 / 26 | 0.0020 | 0.0061 |
| H3 | `hibridopl:rotacao` | `adaptativa:pl` | 42 | -0.0054 (-0.0199 a +0.0059) | 15 / 27 | 0.2688 | 0.5377 |
| H4 | `adaptativa:pl` | `kernel` | 41 | +0.0103 (-0.0021 a +0.0260) | 23 / 18 | 0.4642 | 0.5377 |
