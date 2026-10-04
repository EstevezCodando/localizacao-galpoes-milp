"""Tempos de fases por execução; sem estado compartilhado entre processos/contextos."""

import time
from contextvars import ContextVar
from functools import wraps

_fases: ContextVar[dict[str, float] | None] = ContextVar("fases_pesquisa", default=None)


def iniciar() -> None:
    _fases.set({})


def registrar(nome: str, segundos: float) -> None:
    fases = _fases.get()
    if fases is not None:
        fases[nome] = fases.get(nome, 0.) + segundos


def finalizar() -> dict[str, float]:
    resultado = dict(_fases.get() or {})
    _fases.set(None)
    return resultado


def cronometrar(nome: str):
    def decorar(func):
        @wraps(func)
        def medida(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                registrar(nome, time.perf_counter() - t0)
        return medida
    return decorar
