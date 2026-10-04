"""Piloto diagnóstico; não altera splits nem resultados dos experimentos registrados."""
import json
import time
from pathlib import Path

from threadpoolctl import threadpool_limits

from alocacao_capacitada.pesquisa.clns import Rotacao, clns
from alocacao_capacitada.pesquisa.decomposicao import construtivo, grupos
from alocacao_capacitada.pesquisa.exato import resolver
from alocacao_capacitada.pesquisa.geradores import Config, gerar
from alocacao_capacitada.pesquisa.heuristicas import lns
from alocacao_capacitada.pesquisa.problema import validar


def main():
    out = Path("results/pesquisa/auditoria_clns_20261003.json")
    if out.exists():
        raise FileExistsError(out)
    rows = []
    with threadpool_limits(limits=1):
        for m, n in [(30, 150), (50, 200), (50, 850)]:
            p, _ = gerar(Config("clusters", m, n, 1.5), 20261003)
            for k in [-1, 0, 1, 3, 5]:
                t0 = time.perf_counter()
                a = construtivo(p)
                inicial = validar(p, a).objetivo
                restante = max(0, 5 - (time.perf_counter() - t0))
                if k == -1:
                    s = lns(p, restante, seed=0)
                    atrib = s.atribuicao
                elif k == 0:
                    s = resolver(p, restante, dica=a)
                    atrib = s.atribuicao
                else:
                    s = clns(p, restante, Rotacao(), tamanho=round(n / k),
                             usar_pl=False, atrib_inicial=a, seed=0)
                    atrib = s.atribuicao
                elapsed = time.perf_counter() - t0
                v = validar(p, atrib) if atrib is not None else None
                nome = "lns" if k == -1 else "scip_warm" if k == 0 else "clns_rotacao"
                row = dict(m=m, n=n, metodo=nome,
                           grupos_alvo=k,
                           grupos_reais=len(grupos(p, round(n/k))) if k > 0 else None,
                           tempo=elapsed, valida=bool(v and v.valida), inicial=inicial,
                           objetivo=v.objetivo if v else None,
                           melhoria_pct=(100 * (1 - v.objetivo / inicial)
                                         if v else None))
                rows.append(row)
                print(json.dumps(row), flush=True)
    out.write_text(json.dumps(dict(orcamento=5, seed_dados=20261003,
                                  seed_solver=0, resultados=rows), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
