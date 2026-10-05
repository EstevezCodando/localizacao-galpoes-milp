#!/usr/bin/env bash
# Executa conjuntos do teste fechado (pré-registro 10) com retomada e descarte de execuções
# invalidadas por suspensão. Uso: bash scripts/rodar_fechado.sh conjunto:orcamento [...]
set -uo pipefail
PY=.venv/Scripts/python.exe
METODOS="completo kernel adaptativa:pl adaptativa:gnn hibridopl:rotacao hibrido:rotacao"
export PYTHONIOENCODING=utf-8
mkdir -p results/pesquisa/fechado/logs
for item in "$@"; do
  conj="${item%%:*}"; tempo="${item##*:}"
  log="results/pesquisa/fechado/logs/${conj}.log"
  for tentativa in 1 2 3 4 5 6; do
    $PY scripts/rodar_acordado.py alocacao_capacitada.pesquisa.exp_clns avaliar --split "$conj" \
      --tempo "$tempo" --workers 3 --seeds 0 1 2 --metodos $METODOS >> "$log" 2>&1
    run=$(grep -o 'results.pesquisa.clns.[0-9a-f]\{20\}' "$log" | tail -1 | tr '\\' '/')
    saida=$($PY scripts/descartar_suspensas.py "$conj" "$run")
    echo "$saida (tentativa $tentativa, $run)" | tee -a "$log"
    case "$saida" in *": 0 descartadas") break ;; esac
  done
done
