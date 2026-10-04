# Adendo ao pré-registro — método híbrido e protocolo de tempo v3

Escrito em 03/10/2026, depois do piloto de validação v1 (16 instâncias, 1 semente) e **antes**
de qualquer execução do método híbrido. Mudança de direção decidida com o autor.

## Por que mudar

No piloto v1, a expansão adaptativa por GNN da Etapa 1 teve a menor integral primal (0,020) e o
menor desvio final (0,3%). O CLNS achou boas soluções cedo (integral de 0,024 a 0,041, contra
0,062 a 0,074 de SCIP e LNS), mas estagnou e terminou de 1,2% a 2,7% acima do melhor. Os dois
mecanismos são complementares: um fecha a qualidade em domínio reduzido, o outro melhora
localmente mantendo a viabilidade global.

## Método híbrido (congelado antes de rodar)

1. **Expansão adaptativa por GNN** (Etapa 1, sem alteração) em α = **0,5** do orçamento,
   fixado a priori, sem ajuste.
2. **CLNS** no orçamento restante, partindo da solução da etapa 1, reaproveitando o PL já
   resolvido (custos reduzidos para o regret) e sem recontar o tempo.
3. Seletores avaliados: rotação (clássico), aprendido (LightGBM já treinado) e **guiado pela
   GNN**, que ataca onde a solução mais discorda do ranking (centros abertos improváveis,
   centros fechados prováveis por perto). Ele inclui o tipo de subproblema "abertura", que só
   existe quando há GNN. Não há treino novo.

Ablações no mesmo piloto: CLNS com seletor GNN partindo do PL (sem a etapa adaptativa), CLNS
com rotação, ALNS e aprendido, e a expansão adaptativa sozinha.

## Protocolo de tempo v3

- Cada worker fixado num **núcleo físico exclusivo** (psutil), com o núcleo 0 livre para o
  sistema; 3 workers em 4 núcleos físicos.
- **1 thread** em todas as bibliotecas (OMP, MKL, OpenBLAS, threadpoolctl, torch).
- **Parede e CPU** por execução (`perf_counter_ns`, `process_time_ns`); razão CPU/parede abaixo
  de 0,9 marca a execução como suspeita de contenção.
- Carga do sistema, pico de memória e núcleo registrados em cada linha.
- Aquecimento de imports e modelos no inicializador do worker, fora do cronômetro.

## Dados congelados

`data/frozen/LOCK.json` (versionado) guarda o SHA-256 de cada instância e modelo. O executor
verifica a trava antes de rodar e grava o hash agregado no manifesto da execução.

## Hipóteses do piloto v2

- **H-a:** o híbrido com seletor GNN tem integral primal menor que a expansão adaptativa sozinha.
- **H-b:** o seletor guiado pela GNN supera rotação e aprendido dentro do híbrido.

O piloto continua na validação. O teste fechado só roda depois, com o método escolhido aqui.
