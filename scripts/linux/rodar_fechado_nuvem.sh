#!/usr/bin/env bash
# Teste fechado completo na instância de nuvem (emenda 9 do pré-registro). Roda os cinco
# conjuntos em sequência, com retomada, empacota os resultados e desliga a máquina ao fim.
# Uso (dentro de tmux, na raiz do repositório):  bash scripts/linux/rodar_fechado_nuvem.sh
set -uo pipefail
sudo shutdown -h +1200 "limite de segurança de 20 h" || true   # teto de custo, mesmo se algo travar
PY=.venv/bin/python
METODOS="completo kernel adaptativa:pl adaptativa:gnn hibridopl:rotacao hibrido:rotacao"
export PYTHONIOENCODING=utf-8
mkdir -p results/pesquisa/fechado_nuvem/logs
{ lscpu; free -h; uname -a; $PY -c "import pyscipopt,torch,highspy,ortools,lightgbm;print(pyscipopt.__version__,torch.__version__,lightgbm.__version__)"; } \
  > results/pesquisa/fechado_nuvem/ambiente.txt 2>&1
for item in teste:60 gen_corredor:60 gen_escala:120 holmberg:30 olist:120; do
  conj="${item%%:*}"; tempo="${item##*:}"
  log="results/pesquisa/fechado_nuvem/logs/${conj}.log"
  for tentativa in 1 2 3; do
    $PY -m alocacao_capacitada.pesquisa.exp_clns avaliar --split "$conj" --tempo "$tempo" \
      --workers 3 --seeds 0 1 2 --metodos $METODOS >> "$log" 2>&1 && break
  done
  grep -o 'results.pesquisa.clns.[0-9a-f]\{20\}' "$log" | tail -1 >> results/pesquisa/fechado_nuvem/runs.txt
  echo "$conj $(date -Is) $(grep -c 'cpu/parede' "$log") execucoes" >> results/pesquisa/fechado_nuvem/progresso.txt
done
tar -czf ~/resultados_nuvem.tgz results/pesquisa/clns results/pesquisa/fechado_nuvem
echo "fim $(date -Is)" >> results/pesquisa/fechado_nuvem/progresso.txt
sudo shutdown -c || true
sudo shutdown -h +2 "teste fechado concluido"
