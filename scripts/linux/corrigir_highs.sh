#!/usr/bin/env bash
# No Linux, ortools e highspy trazem cada um a sua libhighs.so.1 (versões diferentes, mesmo
# SONAME). O carregador reaproveita a primeira que entra no processo e o segundo pacote falha
# com "undefined symbol". No Windows as DLLs são isoladas e o problema não aparece.
# Correção de AMBIENTE (não altera src/, que está congelado): renomeia o SONAME da cópia do
# highspy e aponta a extensão dele para o novo nome, como o auditwheel faz.
# Uso (na raiz do repositório, depois de `uv sync --frozen --group pesquisa`):
#   bash scripts/linux/corrigir_highs.sh
set -euo pipefail
SP=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
command -v patchelf >/dev/null || { sudo apt-get update -qq && sudo apt-get install -y -qq patchelf; }
cd "$SP/highspy"
NOVO=libhighs_highspy.so.1
if [ ! -f "$NOVO" ]; then
  cp -L libhighs.so.1 "$NOVO"
  patchelf --set-soname "$NOVO" "$NOVO"
  for ext in _core*.so; do patchelf --replace-needed libhighs.so.1 "$NOVO" "$ext"; done
fi
cd - >/dev/null
.venv/bin/python - <<'PY'
import highspy
from ortools.linear_solver import pywraplp
import pyscipopt, torch, lightgbm
print("highspy + ortools no mesmo processo: ok", highspy.Highs().version(), pywraplp.__name__)
PY
