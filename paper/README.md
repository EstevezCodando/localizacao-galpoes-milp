# Artigo — o aprendizado é necessário para reduzir o espaço de busca no SSCFLP?

Manuscrito em inglês, em duas versões:

| Versão | Pasta | Classe | Alvo | Tamanho |
|---|---|---|---|---|
| Completa (texto-mestre) | `journal/` | `elsarticle` (Elsevier) | *Computers & Operations Research* ou *European Journal of Operational Research* | ~30 páginas em formato de revisão |
| Curta | `short/` | `llncs` (Springer) | CPAIOR ou LION | ~15 páginas + referências |

Referências compartilhadas em `refs.bib`. Entradas com `% conferir` têm páginas ou DOI omitidos
de propósito, para confirmar na fonte antes da submissão.

## Como compilar

`refs.bib` fica na raiz de `paper/` e é citado como `\bibliography{refs}`; o BibTeX o encontra
pela variável `BIBINPUTS`, porque caminhos com `../` são recusados em instalações restritas.

- **Overleaf:** enviar o conteúdo de `paper/` para a raiz do projeto (incluindo `latexmkrc`, que
  define `BIBINPUTS`) e escolher o documento principal em Settings → Compiler → Main document
  (`journal/main.tex`, `abnt/main.tex` ou `short/main.tex`).
- **Local (MiKTeX, PowerShell):** dentro da pasta da versão, `$env:BIBINPUTS = "..;"` e depois
  `pdflatex main`, `bibtex main`, `pdflatex main` duas vezes.

## Estado das seções (versão completa)

| Seção | Estado |
|---|---|
| Abstract | redigido; falta a frase com o resultado do teste fechado |
| 1 Introduction | redigida |
| 2 Related work | redigida; ampliar com busca sistemática (Lagrangiana e Benders para SSCFLP) |
| 3 Problem | redigida |
| 4 Method | redigido, com 2 proposições e o algoritmo |
| 5 Experimental design | redigido conforme os pré-registros (`docs/pesquisa/07` e `08`) |
| 6 Results | 6.1 e 6.2 com dados finais da validação; pilotos descritos como pilotos; teste fechado, Holmberg, Olist, ablações e qualidade de medição como `\todo` |
| 7 Discussion | redigida; revisar após o teste fechado |
| 8 Conclusion | redigida, com uma frase pendente |
| Appendix A | reconstruções com números finais |

## Regra do texto

Nenhum número entra no manuscrito sem vir de uma execução registrada em `results/pesquisa/`
(com manifesto e hash dos dados). Resultados de piloto aparecem rotulados como piloto. O que
ainda não foi medido fica marcado com `\todo{}` em vermelho.

## Pendências para submeter

1. Teste fechado: teste, corredor, escala, Holmberg e Olist, com o método congelado (expansão
   ordenada pelo PL, com e sem CLNS, contra Kernel Search e SCIP).
2. Ablações: α ∈ {0,3; 0,5; 0,7}, sem subproblema de abertura, sem partida pelo PL, tamanho de grupo.
3. Vizinhanças de maior alcance no CLNS (a repetição medida indica que o gargalo é o alcance).
4. Versão curta (`short/`): ainda é só o esqueleto.
5. Pacote de reprodutibilidade: repositório público e DOI de arquivo (por exemplo, Zenodo).
6. Afiliação e revisão de inglês.

Feito desde a versão anterior: pilotos 2 e 3 no texto, baseline Kernel Search, controles sem
aprendizado, memória de subproblemas, formulação dual e demonstrações, figuras geradas por script
(`paper/figuras/gerar_figuras.py`).
