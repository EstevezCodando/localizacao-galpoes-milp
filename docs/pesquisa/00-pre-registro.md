# Pré-registro dos experimentos (03/10/2026)

Escrito antes de abrir qualquer resultado de teste. Desvios posteriores ficam listados no fim de
cada relatório.

## Ambiente

Windows 10, Intel 4 núcleos físicos / 8 lógicos, 1 thread por solver, até 3 processos em
paralelo (contenção registrada como ameaça à validade). Python 3.12, SCIP 10.0 (PySCIPOpt 6.2.1),
HiGHS 1.15, OR-Tools 9.15, PyTorch 2.14 (CPU), LightGBM 4.7. GPU GTX 1650 não usada (inferência
em CPU é o cenário relevante para tempo online; ver P03).

## Portão G0 (cumprido antes dos experimentos)

`tests/test_pesquisa_g0.py`: SCIP coincide com a enumeração exaustiva em 13 microinstâncias
(três famílias, capacidade apertada, custo assimétrico); bound do PL nunca acima do ótimo;
validador detecta violações; poda devolve bound com escopo "reduzido".

## Etapa 1 — hipóteses e critérios

| Hipótese | Teste | Refuta |
|---|---|---|
| H1 poda aprendida + solver melhora o compromisso tempo–qualidade | integral primal e desvio vs BKS a 60 s, pareado contra o completo | não supera o completo nem o ranking clássico |
| H2 relações (GNN) ganham dos agregados tabulares | GNN vs MLP/LightGBM com as mesmas informações | tabular igual ou melhor |
| H3 expansão adaptativa recupera falhas do top-k | adaptativa vs top-k | recuperação precisa de ~100% dos centros |
| H4 warm start neural ajuda sem remover domínio | warm vs completo | completo já acha incumbentes equivalentes |
| H5 generalização | corredor (família nunca vista), 50x200 (escala), Holmberg (distribuição externa, ótimo publicado) | ganho some fora da distribuição |

Métrica primária: integral primal normalizada em [0, T] (Berthold, 2013), com o BKS da instância.
Secundárias: desvio final vs BKS (mediana, IC95 por bootstrap), fração até 0,1% e 1%, tempo até
1% (censurado), taxa de certificação, fração de domínio mantida. Teste: Wilcoxon pareado da
integral primal contra `topk:classico` e contra `completo`, correção de Holm.

Fração de poda: escolhida por ranqueador na VALIDAÇÃO como o menor rho em {0,2..0,8} que mantém
todos os centros da melhor solução em >= 80% das instâncias (critério sem custo de resolução).

Orçamento: T = 60 s (30 s em Holmberg), mesmo para todos, incluindo atributos, inferência, PL,
reparo, todas as resoluções e fallback.

## Frentes de reconstrução (R3)

P02/P03 learn2branch, P04/P05 trocas guiadas, P06 UniFL, P07 NEO-LRP: reconstruções em
ambiente aberto. Os nomes dos métodos originais só são usados com a ressalva "reconstrução";
nenhum número dos artigos é tratado como reproduzido.
