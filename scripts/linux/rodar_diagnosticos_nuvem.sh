#!/usr/bin/env bash
# Diagnósticos de validação (docs/pesquisa/12-preregistro-diagnosticos.md) na instância de nuvem.
set -uo pipefail
PY=.venv/bin/python
export PYTHONIOENCODING=utf-8
mkdir -p results/pesquisa/diagnosticos
$PY scripts/diag_repeticao.py --workers 3 > results/pesquisa/diagnosticos/repeticao.log 2>&1
echo "D1 $(date -Is)" >> results/pesquisa/diagnosticos/progresso.txt
$PY -m alocacao_capacitada.pesquisa.exp_clns avaliar --split validacao --tempo 60 --workers 3 \
  --limite 16 --seeds 0 1 2 --metodos clns:rotacao clns:rotacao+alea clns:rotacao+k30 \
  hibridopl:rotacao hibridopl:rotacao+alea hibridopl:rotacao+k20 hibridopl:rotacao+k30 \
  > results/pesquisa/diagnosticos/d2_d3.log 2>&1
grep -o 'results.pesquisa.clns.[0-9a-f]\{20\}' results/pesquisa/diagnosticos/d2_d3.log | tail -1 \
  > results/pesquisa/diagnosticos/run_d2_d3.txt
tar -czf ~/diagnosticos_nuvem.tgz results/pesquisa/diagnosticos "$(cat results/pesquisa/diagnosticos/run_d2_d3.txt)"
echo "fim $(date -Is)" >> results/pesquisa/diagnosticos/progresso.txt
