# Pré-registro — CLNS: LNS orientado por clusters com seleção aprendida de subproblemas

Escrito em 03/10/2026, **antes** de qualquer execução no split de teste com o CLNS. As escolhas
de configuração vieram só da validação. Desvios posteriores serão listados no relatório.

## Título de trabalho

*Learning to Delegate in Capacitated Facility Location: cluster-oriented large neighborhood
search with learned subproblem selection.*

## Motivação (evidência prévia, toda fora do teste)

- O SCIP 10 não fecha SSCFLP 30×150 do tipo Cornuéjols em 60 s (gap certificado de 1,8% a 3,6%).
- Na Etapa 1, a poda aprendida exigiu manter cerca de 80% dos centros para preservar a melhor
  solução (ρ escolhido na validação).
- Decomposição ingênua (grupos resolvidos isoladamente): união inviável em 3 de 3 instâncias de
  validação; após reparo, 19% a 29% acima do rótulo.
- LNS por clusters com coordenação por capacidade residual: −0,95% contra o rótulo (SCIP 60 s +
  LNS 30 s) em 8 instâncias de validação, com 60 s.
- Na literatura, Learning to Delegate (Li, Yan & Wu, 2021) mostra 10 a 100× pela decomposição e
  1,5 a 2× adicionais pela seleção aprendida em VRP. Não encontramos o equivalente para
  localização capacitada com fonte única, em que a capacidade acopla os subproblemas.

## Configuração congelada (escolhida na validação)

Grupos por k-means sobre perfis de custo, tamanho-alvo de 15 clientes; 0,5 s por subproblema
(aumenta 50% a cada fracasso seguido, até 3×); 10 centros candidatos por cliente livre;
partida pelo arredondamento do PL forte; PL limitado a 25% do orçamento.

## Hipóteses

| ID | Hipótese | Teste | O que refuta |
|---|---|---|---|
| C1 | CLNS (qualquer seletor) supera SCIP completo, warm start clássico e o LNS do projeto em integral primal no mesmo orçamento | Wilcoxon pareado por instância, Holm | Mediana da diferença ≥ 0 |
| C2 | A coordenação global é necessária | CLNS contra decomposição ingênua com reparo | Ingênua igual ou melhor |
| C3 | O seletor aprendido supera rotação, aleatório, **ALNS** e dual no mesmo orçamento | Wilcoxon da integral primal contra o **melhor** seletor não aprendido, Holm | Não supera o melhor baseline |
| C4 | O ganho se mantém fora da distribuição de treino | Corredor (família não vista), 50×200 (escala) e Holmberg (ótimo publicado) | Ganho desaparece |

Métrica primária: integral primal normalizada no orçamento (mínimo acumulado, Berthold 2013).
Secundárias: desvio final contra BKS (ou contra o ótimo publicado no Holmberg), fração até 1% e
0,1%, tempo até 1% (censurado), iterações e atributos de tempo por fase.

## Desenho

- **Splits:** teste 30×150 (32 instâncias, uniforme/clusters × razão 1,5/3,0); gen_corredor
  (16); gen_escala 50×200 (16); Holmberg (71, ótimo publicado).
- **Orçamentos:** 60 s (30×150 e corredor), 120 s (50×200), 30 s (Holmberg).
- **Sementes:** 3 sementes para métodos estocásticos (CLNS, LNS, ingênua); 1 para métodos com
  SCIP de semente fixa (completo, warm_classico, adaptativa:gnn).
- **Seletor aprendido:** LightGBM de ganho por segundo, treinado com rótulos fora da política de
  60 instâncias de TREINO (15 por célula, 60 s cada, 8 candidatos por estado), separando treino
  e validação **por instância**. Sem ajuste no teste.
- **Execução:** 3 processos em 4 núcleos físicos, ordem dos métodos aleatória, registro
  versionado (manifesto com hashes de fontes, dados e modelo; SQLite transacional).

## Ameaças à validade registradas de antemão

Concorrência de CPU entre processos; instâncias sintéticas (Holmberg é a âncora externa);
referência heurística (BKS) fora do Holmberg; orçamento cooperativo (estouros registrados).
