# Pré-registro dos diagnósticos de validação (Etapa B do parecer)

*Registrado em 05/10/2026, antes de rodar. Exploratório: usa as 16 instâncias de **validação**
dos pilotos, não o teste fechado. Nada aqui altera as conclusões confirmatórias; serve para
escolher entre explicações do comportamento do CLNS.*

Ambiente: a mesma instância EC2 `c7i.2xlarge` da segunda rodada do teste fechado, 3 execuções
simultâneas em núcleos dedicados. O código muda em relação ao congelado (opções novas, todas
desligadas por padrão): `grupos(..., modo="aleatorio")`, `k_cand` configurável, `info` em
`reotimizar`, e o relógio da inferência no `clns:gnn` (ponto 3.8 do parecer).

## D1 — Um subproblema que falhou por tempo estava esgotado? (parecer 3.2)

`scripts/diag_repeticao.py`: CLNS com rotação por 30 s, sem memória, em dois contextos (partida
pelo arredondamento do PL; partida pela expansão adaptativa ordenada pelo PL). De cada execução,
até 8 subproblemas que eram repetição de uma falha anterior são reexecutados do mesmo estado com
limites de 5 s e 30 s.

Leitura fixada:

- **esgotado**: o solver termina com ótimo provado e sem melhoria. Falta de alcance da vizinhança.
- **falta de tempo**: há melhoria com o limite maior.
- **indeterminado**: limite atingido sem melhoria.

Se a maioria for "esgotado", a hipótese do alcance ganha apoio e a memória é segura na prática;
se uma fração relevante for "falta de tempo", o gargalo inclui os limites curtos e a memória pode
descartar subproblemas úteis.

## D2 — O agrupamento por perfil de custo importa? (parecer 3.6)

Mesma busca, trocando só a composição dos grupos: perfil de custo contra grupos **aleatórios de
mesmos tamanhos** (`+alea`). Métodos: `clns:rotacao`, `clns:rotacao+alea`, `hibridopl:rotacao`,
`hibridopl:rotacao+alea`. 16 instâncias, 3 sementes, 60 s.

Leitura: diferença pareada (perfil − aleatório) na integral primal e no desvio final, com IC por
bootstrap e Wilcoxon. Sem diferença, o agrupamento por perfil não é o que explica o desempenho do
CLNS, e a contribuição declarada fica restrita à coordenação.

## D3 — O corte de 10 centros por cliente limita a busca?

`hibridopl:rotacao` contra `+k20` e `+k30`; `clns:rotacao` contra `+k30`. Mesmas instâncias,
sementes e orçamento. Subproblemas maiores custam mais por iteração; o que se mede é o saldo.

## Limites

16 instâncias de validação, já usadas três vezes nos pilotos; qualquer efeito encontrado é
indicação para um teste posterior, não confirmação. Correção de Holm sobre as comparações de D2 e
D3 em conjunto (5 comparações por métrica).
