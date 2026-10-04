"""Congelamento dos dados de pesquisa: uma trava versionada no git com o hash de cada arquivo.

Os conjuntos (pickles de instâncias e rótulos) ficam em data/processed/pesquisa/, fora do git
por tamanho e por serem reproduzíveis por semente. Para que um resultado publicado aponte para
dados EXATOS, cada conjunto usado em experimento é congelado:

  * data/frozen/LOCK.json (versionado) guarda, por conjunto: arquivos, SHA-256, total de bytes,
    data do congelamento e o commit do código que gerou os dados;
  * os arquivos ficam somente leitura no disco;
  * `verificar(conjunto)` recalcula os hashes e FALHA se houver arquivo alterado, faltando ou a
    mais — os executores chamam isso antes de rodar.

uv run python -m alocacao_capacitada.pesquisa.congelar treino validacao teste ...
uv run python -m alocacao_capacitada.pesquisa.congelar --verificar
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

RAIZ_DADOS = Path("data/processed/pesquisa")
TRAVA = Path("data/frozen/LOCK.json")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _arquivos(conjunto: str) -> list[Path]:
    base = RAIZ_DADOS / conjunto
    if base.is_file():
        return [base]
    return sorted(p for p in base.rglob("*") if p.is_file())


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "desconhecido"


def ler_trava() -> dict:
    return json.loads(TRAVA.read_text(encoding="utf-8")) if TRAVA.exists() else {}


def congelar(conjuntos: list[str], substituir: bool = False) -> dict:
    trava = ler_trava()
    for c in conjuntos:
        arqs = _arquivos(c)
        if not arqs:
            raise FileNotFoundError(f"conjunto vazio ou inexistente: {c}")
        if c in trava and not substituir:
            verificar(c, trava)  # já congelado: só confirma que não mudou
            continue
        trava[c] = {
            "arquivos": {p.relative_to(RAIZ_DADOS).as_posix(): _sha(p) for p in arqs},
            "n": len(arqs), "bytes": sum(p.stat().st_size for p in arqs),
            "congelado_em": dt.datetime.now().isoformat(timespec="seconds"),
            "commit_codigo": _commit(),
        }
        for p in arqs:
            os.chmod(p, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    TRAVA.parent.mkdir(parents=True, exist_ok=True)
    TRAVA.write_text(json.dumps(trava, indent=2, sort_keys=True), encoding="utf-8")
    return trava


def verificar(conjunto: str, trava: dict | None = None) -> str:
    """Devolve o hash agregado do conjunto; levanta ValueError se divergir da trava."""
    trava = trava if trava is not None else ler_trava()
    if conjunto not in trava:
        raise ValueError(f"conjunto {conjunto!r} não está congelado em {TRAVA}")
    esperado = trava[conjunto]["arquivos"]
    atual = {p.relative_to(RAIZ_DADOS).as_posix(): _sha(p) for p in _arquivos(conjunto)}
    faltam = sorted(set(esperado) - set(atual))
    sobram = sorted(set(atual) - set(esperado))
    mudaram = sorted(k for k in set(esperado) & set(atual) if esperado[k] != atual[k])
    if faltam or sobram or mudaram:
        raise ValueError(f"{conjunto}: dados divergem da trava — faltam {faltam[:3]}, "
                         f"a mais {sobram[:3]}, alterados {mudaram[:3]}")
    return hashlib.sha256("".join(esperado[k] for k in sorted(esperado)).encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("conjuntos", nargs="*")
    ap.add_argument("--verificar", action="store_true")
    ap.add_argument("--substituir", action="store_true")
    a = ap.parse_args()
    if a.verificar:
        for c in ler_trava():
            print(c, verificar(c)[:16], "ok")
    else:
        t = congelar(a.conjuntos, a.substituir)
        for c in a.conjuntos:
            print(c, t[c]["n"], "arquivos", t[c]["bytes"] // 1024, "KiB")


if __name__ == "__main__":
    main()
