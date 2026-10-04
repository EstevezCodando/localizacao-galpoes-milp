"""Medição de tempo e isolamento de processos para os experimentos (v3 do protocolo de tempo).

Problema que resolve: com vários processos em paralelo, o tempo de parede de um método inclui
espera por CPU (contenção), e bibliotecas numéricas podem abrir threads extras sem aviso. Isso
torna comparações de tempo frágeis. Medidas adotadas:

  1. ISOLAMENTO. Cada worker é fixado (afinidade, psutil) num núcleo FÍSICO exclusivo, e todas
     as bibliotecas ficam em 1 thread: variáveis OMP/MKL/OPENBLAS, threadpoolctl (BLAS já
     carregado) e torch.set_num_threads(1). Um núcleo físico fica livre para o sistema.
  2. DUAS MEDIDAS DE TEMPO. Parede (perf_counter_ns), que é o orçamento dos métodos, e CPU do
     processo (process_time_ns), que não conta espera por CPU.
  3. INDICADOR DE CONTENÇÃO. razao_cpu = CPU / parede. Num método de 1 thread limitado por CPU,
     deve ficar perto de 1; abaixo de ~0,9 indica que o processo esperou (contenção ou I/O), e a
     execução é marcada (`contencao_suspeita`).
  4. ESFORÇO DETERMINÍSTICO, independente da máquina, registrado pelos métodos quando existir
     (nós do SCIP, subproblemas resolvidos, iterações).
  5. CONTEXTO DA MÁQUINA. Carga média de CPU do sistema durante a execução e pico de memória do
     processo, gravados em cada linha.

Uso:
    from alocacao_capacitada.pesquisa import medicao
    ProcessPoolExecutor(3, initializer=medicao.inicializar_worker, initargs=(fila,))
    with medicao.Cronometro() as c:
        ...
    linha.update(c.registro())
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

_LIMITE_CONTENCAO = 0.9
_VARS_THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def limitar_threads() -> None:
    """Todas as bibliotecas numéricas em 1 thread (antes e depois de importadas)."""
    for v in _VARS_THREADS:
        os.environ[v] = "1"
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except ImportError:
        pass
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass


def nucleos_fisicos() -> list[int]:
    """Um CPU lógico por núcleo físico (no Windows/Intel com HT, os pares 0-1, 2-3, ...)."""
    import psutil

    logicos = psutil.cpu_count(logical=True) or 1
    fisicos = psutil.cpu_count(logical=False) or logicos
    passo = max(1, logicos // fisicos)
    return list(range(0, logicos, passo))[:fisicos]


def fixar_nucleo(cpu: int) -> None:
    import psutil

    p = psutil.Process()
    p.cpu_affinity([cpu])
    try:  # prioridade um pouco acima do normal: menos preempção por processos de fundo
        p.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS if os.name == "nt" else -5)
    except (AttributeError, psutil.AccessDenied, OSError):
        pass


_CPU_DO_WORKER: int | None = None


def inicializar_worker(fila: object) -> None:
    """Initializer do ProcessPoolExecutor: pega um núcleo físico livre da fila e se fixa nele."""
    global _CPU_DO_WORKER
    limitar_threads()
    cpu = fila.get()  # type: ignore[attr-defined]
    fixar_nucleo(cpu)
    _CPU_DO_WORKER = cpu


def fila_de_nucleos(workers: int) -> object:
    """Fila com `workers` núcleos físicos, deixando ao menos um livre para o sistema."""
    import multiprocessing as mp

    nucleos = nucleos_fisicos()
    if workers >= len(nucleos):
        raise ValueError(f"{workers} workers para {len(nucleos)} núcleos físicos: deixe 1 livre")
    fila = mp.Manager().Queue()
    for c in nucleos[1 : workers + 1]:  # o núcleo 0 fica para o sistema e para a sessão
        fila.put(c)
    return fila


@dataclass
class Cronometro:
    """Mede parede, CPU do processo, pico de memória e carga do sistema de um bloco."""

    parede_ns: int = 0
    cpu_ns: int = 0
    _t0: int = 0
    _c0: int = 0
    carga_sistema: float = float("nan")
    pico_mem_mb: float = float("nan")
    extras: dict[str, float] = field(default_factory=dict)

    def __enter__(self) -> Cronometro:
        try:
            import psutil

            psutil.cpu_percent(None)  # zera o contador de carga do sistema
        except ImportError:
            pass
        self._t0 = time.perf_counter_ns()
        self._c0 = time.process_time_ns()
        return self

    def __exit__(self, *exc: object) -> None:
        self.parede_ns = time.perf_counter_ns() - self._t0
        self.cpu_ns = time.process_time_ns() - self._c0
        try:
            import psutil

            self.carga_sistema = float(psutil.cpu_percent(None))
            mi = psutil.Process().memory_info()
            pico = getattr(mi, "peak_wset", None) or getattr(mi, "rss", 0)
            self.pico_mem_mb = pico / 2**20
        except ImportError:
            pass

    @property
    def parede_s(self) -> float:
        return self.parede_ns / 1e9

    @property
    def cpu_s(self) -> float:
        return self.cpu_ns / 1e9

    @property
    def razao_cpu(self) -> float:
        return self.cpu_ns / self.parede_ns if self.parede_ns else float("nan")

    def registro(self) -> dict[str, float | int | bool | None]:
        return {
            "t_parede_s": self.parede_s,
            "t_cpu_s": self.cpu_s,
            "razao_cpu": self.razao_cpu,
            "contencao_suspeita": bool(self.razao_cpu < _LIMITE_CONTENCAO),
            "carga_sistema_pct": self.carga_sistema,
            "pico_mem_mb": self.pico_mem_mb,
            "nucleo": _CPU_DO_WORKER,
            **self.extras,
        }
