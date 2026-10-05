# Teste fechado — tabelas

## teste (32 instâncias, 60 s)

| Método | n | Integral primal (IC 95%) | Desvio mediano | Desvio médio | Até 1% | Até 0,1% | V/E/D contra SCIP |
|---|---:|---|---:|---:|---:|---:|---|
| `hibridopl:rotacao` | 32 | 0.0265 (0.0154–0.0432) | 0.26% | 0.51% | 81.2% | 34.4% | 25/3/4 |
| `kernel` | 32 | 0.0279 (0.0157–0.0489) | 0.69% | 1.15% | 65.6% | 37.5% | 24/2/6 |
| `adaptativa:pl` | 32 | 0.0331 (0.0132–0.0658) | 0.29% | 0.68% | 78.1% | 40.6% | 24/3/5 |
| `hibrido:rotacao` | 32 | 0.0389 (0.0227–0.0593) | 0.34% | 0.69% | 71.9% | 18.8% | 23/2/7 |
| `adaptativa:gnn` | 32 | 0.0514 (0.0206–0.0908) | 0.53% | 0.91% | 68.8% | 21.9% | 21/3/8 |
| `completo` | 32 | 0.0547 (0.0464–0.0637) | 1.83% | 1.92% | 37.5% | 15.6% | – |

Certificado de ótimo antes do último estágio: `adaptativa:gnn` 2, `adaptativa:pl` 2

## gen_corredor (16 instâncias, 60 s)

| Método | n | Integral primal (IC 95%) | Desvio mediano | Desvio médio | Até 1% | Até 0,1% | V/E/D contra SCIP |
|---|---:|---|---:|---:|---:|---:|---|
| `adaptativa:pl` | 16 | 0.0153 (0.0100–0.0213) | 0.09% | 0.69% | 75.0% | 50.0% | 7/2/7 |
| `kernel` | 16 | 0.0160 (0.0106–0.0221) | 0.07% | 0.70% | 75.0% | 50.0% | 9/2/5 |
| `hibridopl:rotacao` | 16 | 0.0169 (0.0116–0.0224) | 0.29% | 0.62% | 81.2% | 37.5% | 11/1/4 |
| `hibrido:rotacao` | 16 | 0.0263 (0.0091–0.0559) | 0.02% | 0.40% | 81.2% | 62.5% | 12/2/2 |
| `adaptativa:gnn` | 16 | 0.0398 (0.0089–0.0973) | 0.04% | 0.38% | 87.5% | 75.0% | 11/2/3 |
| `completo` | 16 | 0.0507 (0.0387–0.0630) | 0.39% | 1.16% | 62.5% | 43.8% | – |

Certificado de ótimo antes do último estágio: `adaptativa:gnn` 1, `adaptativa:pl` 1

## gen_escala (16 instâncias, 120 s)

| Método | n | Integral primal (IC 95%) | Desvio mediano | Desvio médio | Até 1% | Até 0,1% | V/E/D contra SCIP |
|---|---:|---|---:|---:|---:|---:|---|
| `kernel` | 16 | 0.0356 (0.0236–0.0489) | 0.72% | 1.20% | 56.2% | 31.2% | 11/1/4 |
| `hibridopl:rotacao` | 16 | 0.0422 (0.0247–0.0631) | 0.37% | 0.62% | 75.0% | 25.0% | 15/0/1 |
| `adaptativa:pl` | 16 | 0.0491 (0.0256–0.0793) | 0.94% | 1.34% | 50.0% | 25.0% | 13/0/3 |
| `completo` | 16 | 0.0843 (0.0674–0.0997) | 1.43% | 2.00% | 43.8% | 6.2% | – |
| `hibrido:rotacao` | 16 | 0.0871 (0.0483–0.1345) | 0.82% | 1.38% | 62.5% | 18.8% | 12/0/4 |
| `adaptativa:gnn` | 16 | 0.1421 (0.0667–0.2358) | 2.39% | 3.25% | 25.0% | 12.5% | 5/0/11 |

Certificado de ótimo antes do último estágio: `adaptativa:gnn` 0, `adaptativa:pl` 0

## holmberg (71 instâncias, 30 s)

| Método | n | Integral primal (IC 95%) | Desvio mediano | Desvio médio | Até 1% | Até 0,1% | V/E/D contra SCIP |
|---|---:|---|---:|---:|---:|---:|---|
| `adaptativa:pl` | 71 | 0.0054 (0.0038–0.0071) | 0.00% | 0.03% | 98.6% | 95.8% | 3/66/2 |
| `kernel` | 71 | 0.0056 (0.0038–0.0077) | 0.00% | 0.02% | 100.0% | 93.0% | 3/65/3 |
| `hibridopl:rotacao` | 71 | 0.0056 (0.0041–0.0075) | 0.00% | 0.09% | 97.2% | 90.1% | 3/63/5 |
| `completo` | 71 | 0.0106 (0.0073–0.0144) | 0.00% | 0.09% | 95.8% | 94.4% | – |
| `hibrido:rotacao` | 71 | 0.0178 (0.0143–0.0217) | 0.00% | 0.10% | 97.2% | 88.7% | 3/60/8 |
| `adaptativa:gnn` | 71 | 0.0200 (0.0157–0.0248) | 0.00% | 0.11% | 97.2% | 90.1% | 1/64/6 |

Certificado de ótimo antes do último estágio: `adaptativa:gnn` 63, `adaptativa:pl` 66

Instâncias em que o ótimo publicado foi atingido (em todas as sementes / em alguma): `adaptativa:gnn` 63/63, `adaptativa:pl` 67/67, `completo` 66/66, `hibrido:rotacao` 59/63, `hibridopl:rotacao` 63/64, `kernel` 65/65

## olist (4 instâncias, 120 s)

| Método | n | Integral primal (IC 95%) | Desvio mediano | Desvio médio | Até 1% | Até 0,1% | V/E/D contra SCIP |
|---|---:|---|---:|---:|---:|---:|---|
| `kernel` | 4 | 0.0904 (0.0095–0.1799) | 1.49% | 2.68% | 50.0% | 25.0% | 3/1/0 |
| `completo` | 4 | 0.1174 (0.0178–0.2399) | 2.10% | 3.37% | 25.0% | 25.0% | – |
| `adaptativa:pl` | 4 | 0.1377 (0.0865–0.1950) | 4.70% | 4.68% | 25.0% | 25.0% | 0/1/3 |
| `hibridopl:rotacao` | 4 | 0.1827 (0.0640–0.3887) | 0.26% | 0.37% | 100.0% | 50.0% | 3/0/1 |
| `hibrido:rotacao` | 4 | 0.1860 (0.0625–0.3905) | 0.69% | 0.71% | 75.0% | 25.0% | 3/0/1 |
| `adaptativa:gnn` | 4 | 0.2951 (0.1060–0.6222) | 4.70% | 4.68% | 25.0% | 25.0% | 0/1/3 |

Certificado de ótimo antes do último estágio: `adaptativa:gnn` 0, `adaptativa:pl` 0

## Agregado (teste + gen_corredor + gen_escala, 64 instâncias)

| Método | n | Integral primal (IC 95%) | Desvio mediano | Desvio médio | Até 1% | Até 0,1% | V/E/D contra SCIP |
|---|---:|---|---:|---:|---:|---:|---|
| `kernel` | 64 | 0.0269 (0.0190–0.0381) | 0.53% | 1.05% | 65.6% | 39.1% | 44/5/15 |
| `hibridopl:rotacao` | 64 | 0.0280 (0.0199–0.0379) | 0.30% | 0.57% | 79.7% | 32.8% | 51/4/9 |
| `adaptativa:pl` | 64 | 0.0326 (0.0193–0.0502) | 0.35% | 0.85% | 70.3% | 39.1% | 44/5/15 |
| `hibrido:rotacao` | 64 | 0.0478 (0.0325–0.0653) | 0.30% | 0.79% | 71.9% | 29.7% | 47/4/13 |
| `completo` | 64 | 0.0611 (0.0535–0.0685) | 1.27% | 1.75% | 45.3% | 20.3% | – |
| `adaptativa:gnn` | 64 | 0.0712 (0.0414–0.1048) | 0.54% | 1.36% | 62.5% | 32.8% | 37/5/22 |

### Hipóteses — integral primal (A − B < 0 favorece A)

| Hipótese | A | B | n | A − B (IC 95%) | A melhor / B melhor | p | p (Holm) |
|---|---|---|---:|---|---|---:|---:|
| H1 | `adaptativa:pl` | `completo` | 64 | -0.0285 (-0.0451 a -0.0080) | 55 / 9 | 0.0000 | 0.0000 |
| H2 | `adaptativa:gnn` | `adaptativa:pl` | 64 | +0.0385 (+0.0129 a +0.0693) | 19 / 45 | 0.0016 | 0.0048 |
| H3 | `hibridopl:rotacao` | `adaptativa:pl` | 64 | -0.0046 (-0.0135 a +0.0024) | 31 / 33 | 0.7230 | 0.7230 |
| H4 | `adaptativa:pl` | `kernel` | 64 | +0.0058 (-0.0010 a +0.0139) | 38 / 26 | 0.2039 | 0.4077 |

### Hipóteses — desvio final, em pontos percentuais

| Hipótese | A | B | n | A − B (IC 95%) | A melhor / B melhor | p | p (Holm) |
|---|---|---|---:|---|---|---:|---:|
| H1 | `adaptativa:pl` | `completo` | 64 | -0.9024 (-1.3091 a -0.4918) | 44 / 15 | 0.0000 | 0.0001 |
| H2 | `adaptativa:gnn` | `adaptativa:pl` | 64 | +0.5171 (+0.1343 a +0.9404) | 16 / 29 | 0.0186 | 0.0557 |
| H3 | `hibridopl:rotacao` | `adaptativa:pl` | 64 | -0.2801 (-0.5935 a +0.0084) | 36 / 17 | 0.0229 | 0.0557 |
| H4 | `adaptativa:pl` | `kernel` | 64 | -0.2033 (-0.4510 a +0.0408) | 24 / 23 | 0.4909 | 0.4909 |

### Sensibilidade: integral primal sem as execuções marcadas por contenção

| Hipótese | A | B | n | A − B (IC 95%) | A melhor / B melhor | p | p (Holm) |
|---|---|---|---:|---|---|---:|---:|
| H1 | `adaptativa:pl` | `completo` | 64 | -0.0285 (-0.0451 a -0.0080) | 55 / 9 | 0.0000 | 0.0000 |
| H2 | `adaptativa:gnn` | `adaptativa:pl` | 64 | +0.0385 (+0.0129 a +0.0693) | 19 / 45 | 0.0016 | 0.0048 |
| H3 | `hibridopl:rotacao` | `adaptativa:pl` | 64 | -0.0046 (-0.0135 a +0.0024) | 31 / 33 | 0.7230 | 0.7230 |
| H4 | `adaptativa:pl` | `kernel` | 64 | +0.0058 (-0.0010 a +0.0139) | 38 / 26 | 0.2039 | 0.4077 |
