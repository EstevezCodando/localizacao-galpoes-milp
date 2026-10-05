"""Roda um módulo Python impedindo a suspensão por ociosidade enquanto ele executa (Windows).

Uma suspensão no meio de uma rodada invalida as execuções em curso: o relógio de parede avança
horas e o solver, limitado por tempo de parede, é interrompido ao acordar. O pedido vale só
enquanto este processo vive e não altera nenhuma configuração do sistema.

uv run python scripts/rodar_acordado.py alocacao_capacitada.pesquisa.exp_clns avaliar ...
"""

from __future__ import annotations

import ctypes
import runpy
import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def main() -> None:
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    modulo = sys.argv[1]
    sys.argv = [modulo, *sys.argv[2:]]
    try:
        runpy.run_module(modulo, run_name="__main__", alter_sys=True)
    finally:
        if sys.platform == "win32":
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


if __name__ == "__main__":
    main()
