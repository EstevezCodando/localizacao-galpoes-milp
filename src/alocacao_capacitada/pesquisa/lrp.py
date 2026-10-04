"""Frente P07 — Kaleem, Lee, Kwon & Subramanyam: Neural Embedded Mixed-Integer Optimization for
Location-Routing (NEO-LRP). Reconstrução aberta (R3): o código oficial usa Gurobi/Gurobi-ML
(licença indisponível); aqui o mesmo princípio roda em SCIP com ReLU por big-M.

CLRP (estilo Prodhon): depósitos candidatos com custo de abertura O_i e capacidade W_i,
clientes com demanda d_j, veículos de capacidade Q e custo fixo F, custo de rota = 100 * dist.
  custo = sum O_i y_i + sum_i [F * #rotas_i + 100 * comprimento das rotas_i]

Princípio do NEO-LRP: custo de roteamento de um depósito ~ rho( sum_{j atribuído} phi(i, j) )
(Deep Sets). phi(i, j) depende só do par, então z_i = sum_j phi_ij x_ij é LINEAR em x; rho é uma
MLP ReLU pequena codificada no MIP. A avaliação final SEMPRE substitui a estimativa pelas rotas
reais (OR-Tools, orçamento fixo): previsão e custo final ficam separados (doc 09).

Competidores:
  flp_vrp    SSCFLP com custo de atribuição radial 2 * 100 * dist_ij (+ F d_j / Q), depois VRP;
  aprox_cont aproximação contínua clássica (Daganzo, 1984; Beardwood–Halton–Hammersley) na forma
             linear: linha-haul 2 r_ij d_j / Q + termo local k * dist ao vizinho típico;
  neo_ds     Deep Sets embutido (este módulo).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from scipy.spatial.distance import cdist
from torch import nn

torch.set_num_threads(1)
GRADE = 50.0


@dataclass(frozen=True)
class Clrp:
    xy_dep: np.ndarray  # (m, 2)
    xy_cli: np.ndarray  # (n, 2)
    demand: np.ndarray  # (n,)
    cap_dep: np.ndarray  # (m,)
    open_cost: np.ndarray  # (m,)
    q: float
    f_veic: float
    nome: str

    @property
    def m(self) -> int:
        return int(self.xy_dep.shape[0])

    @property
    def n(self) -> int:
        return int(self.xy_cli.shape[0])


def gerar_clrp(n: int, m: int, seed: int, familia: str = "uniforme", q: float = 70.0) -> Clrp:
    rng = np.random.default_rng(seed)
    if familia == "uniforme":
        cli = rng.uniform(0, GRADE, (n, 2))
    else:
        c = rng.uniform(5, 45, (int(rng.integers(2, 5)), 2))
        cli = np.clip(c[rng.integers(len(c), size=n)] + rng.normal(0, 5, (n, 2)), 0, GRADE)
    dep = rng.uniform(0, GRADE, (m, 2))
    d = rng.integers(11, 21, n).astype(float)
    cap = np.full(m, max(d.sum() / m * 2.5, d.max() * 3))
    oc = rng.uniform(4000, 8000, m) * (n / 50) ** 0.5
    return Clrp(dep, cli, d, cap, oc, q, 1000.0, f"clrp_{familia}_n{n}_m{m}_s{seed}")


# ------------------------------------------------------------------ CVRP (rótulo e avaliação)
def validar_rotas(cli: np.ndarray, dep: np.ndarray, dem: np.ndarray, q: float,
                   f_veic: float, rotas: list[list[int]]) -> float:
    """Recalcula cargas, cobertura e custo na métrica inteira 100*dist arredondada."""
    visitas = [j for rota in rotas for j in rota]
    if sorted(visitas) != list(range(len(dem))):
        raise ValueError("Rotas devem visitar cada cliente exatamente uma vez")
    custo = 0.0
    for rota in rotas:
        if not rota:
            continue
        if dem[rota].sum() > q + 1e-9:
            raise ValueError("Capacidade real do veículo excedida")
        pts = np.vstack([dep, cli[rota], dep])
        custo += f_veic + np.rint(100 * np.linalg.norm(np.diff(pts, axis=0), axis=1)).sum()
    return float(custo)


def cvrp(dep: np.ndarray, cli: np.ndarray, dem: np.ndarray, q: float, f_veic: float,
         tempo: float = 1.0, rotas_out: list[list[int]] | None = None) -> tuple[float, int]:
    """Custo (F * rotas + 100 * distância) pelo OR-Tools (GLS, orçamento fixo). Heurístico: o
    rótulo é o custo ENCONTRADO, não o ótimo (doc 03)."""
    t0 = time.perf_counter()
    dem = np.asarray(dem, dtype=float)
    valores = np.concatenate([dem, [q, f_veic]])
    if not np.isfinite(valores).all() or (valores != np.floor(valores)).any():
        raise ValueError("CVRP exige demanda, capacidade e custo fixo inteiros; escale as unidades")
    if q <= 0 or f_veic < 0 or (dem <= 0).any() or (dem > q).any():
        raise ValueError("Demanda/capacidade/custo inválidos para CVRP")
    k = cli.shape[0]
    if dem.shape != (k,) or not np.isfinite(cli).all() or not np.isfinite(dep).all():
        raise ValueError("Coordenadas ou dimensão da demanda inválidas")
    if k == 0:
        return 0.0, 0
    pts = np.vstack([dep[None, :], cli])
    dist = np.rint(100 * cdist(pts, pts)).astype(np.int64)
    nv = min(k, 2 * math.ceil(dem.sum() / q) + 1)
    man = pywrapcp.RoutingIndexManager(k + 1, nv, 0)
    rt = pywrapcp.RoutingModel(man)

    def dcb(a: int, b: int) -> int:
        return int(dist[man.IndexToNode(a), man.IndexToNode(b)])

    cb = rt.RegisterTransitCallback(dcb)
    rt.SetArcCostEvaluatorOfAllVehicles(cb)
    rt.SetFixedCostOfAllVehicles(int(f_veic))
    dem_i = np.concatenate([[0], dem]).astype(np.int64)

    def qcb(a: int) -> int:
        return int(dem_i[man.IndexToNode(a)])

    qc = rt.RegisterUnaryTransitCallback(qcb)
    rt.AddDimensionWithVehicleCapacity(qc, 0, [int(q)] * nv, True, "carga")
    sp = pywrapcp.DefaultRoutingSearchParameters()
    sp.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    sp.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    restante = tempo - (time.perf_counter() - t0)
    if restante < 0.001:
        raise TimeoutError("Orçamento CVRP esgotado na montagem")
    sp.time_limit.FromMilliseconds(int(restante * 1000))
    sol = rt.SolveWithParameters(sp)
    if sol is None:
        raise RuntimeError("CVRP sem solução")
    rotas = []
    for v in range(nv):
        idx = sol.Value(rt.NextVar(rt.Start(v)))
        rota = []
        while not rt.IsEnd(idx):
            rota.append(man.IndexToNode(idx) - 1)
            idx = sol.Value(rt.NextVar(idx))
        if rota:
            rotas.append(rota)
    custo = validar_rotas(cli, dep, dem, q, f_veic, rotas)
    if abs(custo - sol.ObjectiveValue()) > 1e-6:
        raise ValueError("Objetivo CVRP diverge do custo recalculado")
    if rotas_out is not None:
        rotas_out.extend(rotas)
    return custo, len(rotas)


def avaliar(inst: Clrp, atrib: np.ndarray, tempo_por_dep: float = 2.0,
            *, tempo_total: float | None = None) -> dict[str, float]:
    """Custo final REAL: abertura + rotas construídas por depósito; confere capacidades."""
    t0 = time.perf_counter()
    atrib = np.asarray(atrib)
    if (atrib.shape != (inst.n,) or not np.issubdtype(atrib.dtype, np.integer)
            or ((atrib < 0) | (atrib >= inst.m)).any()):
        raise ValueError("Atribuição CLRP incompleta ou inválida")
    abertos = np.unique(atrib)
    carga = np.bincount(atrib, weights=inst.demand, minlength=inst.m)
    if (carga > inst.cap_dep + 1e-6).any():
        raise ValueError("capacidade de depósito violada")
    rot, nr = 0.0, 0
    for k, i in enumerate(abertos):
        orc = (tempo_por_dep if tempo_total is None else
               (tempo_total - (time.perf_counter() - t0)) / (len(abertos) - k))
        c, r = cvrp(inst.xy_dep[i], inst.xy_cli[atrib == i], inst.demand[atrib == i], inst.q,
                    inst.f_veic, orc)
        rot += c
        nr += r
    return {"custo": float(inst.open_cost[abertos].sum() + rot), "abertura":
            float(inst.open_cost[abertos].sum()), "roteamento": rot, "rotas": nr,
            "depositos": int(abertos.size), "t_rotas": time.perf_counter() - t0}


# ------------------------------------------------------------------ Deep Sets
def phi_entrada(dep: np.ndarray, cli: np.ndarray, dem: np.ndarray, q: float) -> np.ndarray:
    rel = (cli - dep[None, :]) / GRADE
    r = np.linalg.norm(rel, axis=1, keepdims=True)
    return np.column_stack([rel, r, dem[:, None] / q, r * dem[:, None] / q]).astype(np.float32)


class DeepSets(nn.Module):
    def __init__(self, fin: int = 5, lat: int = 8, hid: int = 16) -> None:
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(fin, 32), nn.ReLU(), nn.Linear(32, lat))
        self.rho = nn.Sequential(nn.Linear(lat, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, 1))

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        z = torch.stack([self.phi(x).sum(0) for x in xs])
        return self.rho(z).squeeze(1)


ESCALA = 10_000.0  # custo previsto em unidades de 10^4


def amostrar_rotulos(n_inst: int, seed: int, tempo: float = 1.0) -> list[tuple[np.ndarray, float]]:
    """Subconjuntos de treino de instâncias de treino (sementes disjuntas do teste): aleatórios,
    por vizinhança do depósito (decisões 'boas') e espalhados (decisões 'ruins')."""
    rng = np.random.default_rng(seed)
    out = []
    for s in range(n_inst):
        fam = ("uniforme", "clusters")[s % 2]
        inst = gerar_clrp(int(rng.integers(20, 101)), 5, seed * 1000 + s, fam)
        for _ in range(6):
            i = int(rng.integers(inst.m))
            capk = int(inst.cap_dep[i] // inst.demand.mean())
            k = int(rng.integers(1, max(2, min(capk, inst.n)) + 1))
            modo = rng.integers(3)
            dd = np.linalg.norm(inst.xy_cli - inst.xy_dep[i], axis=1)
            if modo == 0:
                S = rng.choice(inst.n, k, replace=False)
            elif modo == 1:
                S = np.argsort(dd)[:k]
            else:
                S = np.argsort(-dd)[:k] if rng.random() < 0.5 else rng.choice(
                    inst.n, k, replace=False, p=dd / dd.sum())
            S = S[np.cumsum(inst.demand[S]) <= inst.cap_dep[i]]
            if S.size == 0:
                continue
            c, _ = cvrp(inst.xy_dep[i], inst.xy_cli[S], inst.demand[S], inst.q, inst.f_veic, tempo)
            out.append((phi_entrada(inst.xy_dep[i], inst.xy_cli[S], inst.demand[S], inst.q),
                        c / ESCALA))
    return out


def treinar_deepsets(tr: list[tuple[np.ndarray, float]], va: list[tuple[np.ndarray, float]],
                     seed: int = 0, epocas: int = 300) -> tuple[DeepSets, list[float]]:
    torch.manual_seed(seed)
    model = DeepSets()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    Xtr = [torch.from_numpy(x) for x, _ in tr]
    ytr = torch.tensor([y for _, y in tr], dtype=torch.float32)
    Xva = [torch.from_numpy(x) for x, _ in va]
    yva = torch.tensor([y for _, y in va], dtype=torch.float32)
    best, state, wait, hist = np.inf, None, 0, []
    rng = np.random.default_rng(seed)
    for _ in range(epocas):
        model.train()
        perm = rng.permutation(len(Xtr))
        for b in range(0, len(perm), 64):
            idx = perm[b : b + 64]
            opt.zero_grad()
            pred = model([Xtr[k] for k in idx])
            loss = ((pred - ytr[idx]) ** 2).mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float((torch.abs(model(Xva) - yva) / yva.clamp(min=1e-6)).mean())
        hist.append(vl)
        if vl < best - 1e-5:
            best, wait, state = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 30:
                break
    if state is not None:
        model.load_state_dict(state)
    return model, hist


# ------------------------------------------------------------------ MIPs de localização
def _base_mip(inst: Clrp, tempo: float):  # type: ignore[no-untyped-def]
    from pyscipopt import Model

    if tempo <= 0:
        raise TimeoutError("Orçamento MIP esgotado")
    md = Model()
    md.hideOutput()
    md.setParam("parallel/maxnthreads", 1)
    md.setParam("timing/clocktype", 2)
    md.setParam("limits/time", tempo)
    y = md.addMatrixVar(inst.m, vtype="B")
    x = md.addMatrixVar((inst.m, inst.n), vtype="B")
    md.addMatrixCons(x.sum(axis=0) == 1)
    md.addMatrixCons((x * inst.demand[None, :]).sum(axis=1) <= inst.cap_dep * y)
    md.addMatrixCons(x - y.reshape(inst.m, 1) <= 0)
    return md, y, x


def _otimizar(md, t0: float, tempo: float) -> None:  # type: ignore[no-untyped-def]
    restante = tempo - (time.perf_counter() - t0)
    if restante <= 0:
        raise TimeoutError("Orçamento MIP esgotado na montagem")
    md.setParam("limits/time", restante * 0.95)
    md.optimize()
    if md.getNSols() == 0:
        raise RuntimeError(f"MIP sem solução: {md.getStatus()}")


def _extrair(md, x, inst: Clrp) -> np.ndarray:  # type: ignore[no-untyped-def]
    s = md.getBestSol()
    xv = np.array([[md.getSolVal(s, x[i, j]) for j in range(inst.n)] for i in range(inst.m)])
    return np.argmax(xv, axis=0)


def mip_linear(inst: Clrp, custo_par: np.ndarray, tempo: float = 60.0) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    md, y, x = _base_mip(inst, tempo)
    md.setObjective((inst.open_cost * y).sum() + (custo_par * x).sum())
    _otimizar(md, t0, tempo)
    return _extrair(md, x, inst), time.perf_counter() - t0


def custo_flp(inst: Clrp) -> np.ndarray:
    r = cdist(inst.xy_dep, inst.xy_cli)
    return 2 * 100 * r + inst.f_veic * inst.demand[None, :] / inst.q


def custo_aprox_continua(inst: Clrp, kappa: float = 0.75) -> np.ndarray:
    """Linha-haul 2 r d/q (viagens radiais rateadas pela carga) + percurso local
    kappa * distância típica entre vizinhos (BHH: ~ sqrt(A/n)) + veículo rateado F d/q."""
    r = cdist(inst.xy_dep, inst.xy_cli)
    nn_d = np.sort(cdist(inst.xy_cli, inst.xy_cli), axis=1)[:, 1:4].mean(axis=1)
    return 100 * (2 * r * inst.demand[None, :] / inst.q + kappa * nn_d[None, :]) \
        + inst.f_veic * inst.demand[None, :] / inst.q


def mip_neo(inst: Clrp, model: DeepSets, tempo: float = 60.0,
            ini: np.ndarray | None = None) -> tuple[np.ndarray, float, int]:
    """Deep Sets embutido: z_i = sum_j phi_ij x_ij (linear); rho por big-M com limites de
    ativação por propagação de intervalos (válidos para todo x binário). `ini` = atribuição
    inicial completa (as ativações são propagadas para formar uma solução do MIP); sem ela o
    SCIP pode não achar solução viável no orçamento (observado no piloto)."""
    t0 = time.perf_counter()
    md, y, x = _base_mip(inst, tempo)
    with torch.no_grad():
        phi = np.stack([model.phi(torch.from_numpy(phi_entrada(inst.xy_dep[i], inst.xy_cli,
                                                               inst.demand, inst.q))).numpy()
                        for i in range(inst.m)]).astype(float)  # (m, n, lat)
    layers = [lyr for lyr in model.rho if isinstance(lyr, nn.Linear)]
    W = [lyr.weight.detach().numpy().astype(float) for lyr in layers]
    B = [lyr.bias.detach().numpy().astype(float) for lyr in layers]
    obj = (inst.open_cost * y).sum()
    nbin = 0
    registro = []  # por depósito: (z vars, [(a, s, w, b) por neurônio oculto], rc)
    for i in range(inst.m):
        lat = phi.shape[2]
        z = [md.addVar(lb=None, ub=None) for _ in range(lat)]
        for k in range(lat):
            md.addCons(z[k] == sum(float(phi[i, j, k]) * x[i, j] for j in range(inst.n)))
        lo = np.minimum(phi[i], 0).sum(axis=0)
        hi = np.maximum(phi[i], 0).sum(axis=0)
        h_prev: list = list(z)
        camadas = []
        for li, (w, b) in enumerate(zip(W, B)):
            pre_lo = np.maximum(w, 0) @ lo + np.minimum(w, 0) @ hi + b
            pre_hi = np.maximum(w, 0) @ hi + np.minimum(w, 0) @ lo + b
            exprs = [sum(float(w[o, k]) * h_prev[k] for k in range(len(h_prev))
                         if not isinstance(h_prev[k], float)) + float(b[o])
                     for o in range(w.shape[0])]
            if li == len(W) - 1:
                pred = exprs[0]
                break
            h_new: list = []
            neur = []
            for o, e in enumerate(exprs):
                if pre_hi[o] <= 0:
                    h_new.append(0.0)
                    neur.append((None, None))
                    continue
                a = md.addVar(lb=0)
                sv = None
                if pre_lo[o] >= 0:
                    md.addCons(a == e)
                else:
                    sv = md.addVar(vtype="B")
                    nbin += 1
                    md.addCons(a >= e)
                    md.addCons(a <= e - float(pre_lo[o]) * (1 - sv))
                    md.addCons(a <= float(pre_hi[o]) * sv)
                h_new.append(a)
                neur.append((a, sv))
            camadas.append(neur)
            lo = np.maximum(pre_lo, 0)
            hi = np.maximum(pre_hi, 0)
            h_prev = h_new
        big = float(max(abs(pre_lo[0]), abs(pre_hi[0]))) + 1.0
        rc = md.addVar(lb=0)
        md.addCons(rc >= pred - big * (1 - y[i]))
        obj = obj + ESCALA * rc
        registro.append((z, camadas, rc))
    md.setObjective(obj)
    if ini is not None:
        sol = md.createSol()
        abertos = set(np.unique(ini).tolist())
        for i in range(inst.m):
            md.setSolVal(sol, y[i], 1.0 if i in abertos else 0.0)
            for j in range(inst.n):
                md.setSolVal(sol, x[i, j], 1.0 if ini[j] == i else 0.0)
            z, camadas, rc = registro[i]
            zv = phi[i][ini == i].sum(axis=0)
            for k, var in enumerate(z):
                md.setSolVal(sol, var, float(zv[k]))
            h = zv
            for (w, b), neur in zip(zip(W[:-1], B[:-1]), camadas):
                pre = w @ h + b
                h = np.maximum(pre, 0)
                for o, (a, sv) in enumerate(neur):
                    if a is not None:
                        md.setSolVal(sol, a, float(h[o]))
                    if sv is not None:
                        md.setSolVal(sol, sv, 1.0 if pre[o] > 0 else 0.0)
            pred_v = float((W[-1] @ h + B[-1])[0])
            md.setSolVal(sol, rc, max(0.0, pred_v) if i in abertos else 0.0)
        md.addSol(sol, free=True)
    _otimizar(md, t0, tempo)
    return _extrair(md, x, inst), time.perf_counter() - t0, nbin


def prever(model: DeepSets, inst: Clrp, atrib: np.ndarray) -> float:
    """Custo previsto pelo surrogate para uma configuração (para medir o erro 'explorado')."""
    tot = 0.0
    with torch.no_grad():
        for i in np.unique(atrib):
            S = atrib == i
            x = torch.from_numpy(
                phi_entrada(inst.xy_dep[i], inst.xy_cli[S], inst.demand[S], inst.q))
            tot += float(model([x])[0]) * ESCALA
    return tot + float(inst.open_cost[np.unique(atrib)].sum())


def busca_surrogate(inst: Clrp, model: DeepSets, ini: np.ndarray, tempo: float = 60.0
                    ) -> tuple[np.ndarray, float, int]:
    """Caminho B do doc 09: busca local de realocação (cliente j do depósito i para k) com o
    custo avaliado pelo surrogate. z_i = sum phi_ij é mantido incrementalmente, então cada
    iteração avalia todas as n*m realocações com um único lote de rho. Abre depósito quando a
    realocação compensa o custo de abertura; fecha quando o último cliente sai."""
    t0 = time.perf_counter()
    with torch.no_grad():
        phi = torch.stack([model.phi(torch.from_numpy(phi_entrada(inst.xy_dep[i], inst.xy_cli,
                                                                  inst.demand, inst.q)))
                           for i in range(inst.m)])  # (m, n, lat)

        def rho(z: torch.Tensor) -> torch.Tensor:
            return torch.clamp(model.rho(z).squeeze(-1), min=0.0) * ESCALA

        a = np.array(ini)
        onehot = torch.zeros(inst.m, inst.n)
        onehot[torch.from_numpy(a), torch.arange(inst.n)] = 1.0
        z = torch.einsum("in,inl->il", onehot, phi)
        cnt = np.bincount(a, minlength=inst.m)
        carga = np.bincount(a, weights=inst.demand, minlength=inst.m)
        it = 0
        while time.perf_counter() - t0 < tempo:
            it += 1
            r = rho(z)  # (m,)
            r = torch.where(torch.from_numpy(cnt > 0), r, torch.zeros_like(r))
            src = torch.from_numpy(a)
            z_out = z[src] - phi[src, torch.arange(inst.n)]  # (n, lat): origem sem j
            r_out = rho(z_out)
            r_out = torch.where(torch.from_numpy(cnt[a] > 1), r_out, torch.zeros_like(r_out))
            z_in = z[:, None, :] + phi  # (m, n, lat): destino k com j
            r_in = rho(z_in)  # (m, n)
            delta = (r_in - r[:, None]) + (r_out - r[src])[None, :]
            delta = delta.numpy() + (cnt == 0)[:, None] * inst.open_cost[:, None]
            delta -= ((cnt[a] == 1) * inst.open_cost[a])[None, :]
            delta[a, np.arange(inst.n)] = np.inf
            delta[carga[:, None] + inst.demand[None, :] > inst.cap_dep[:, None] + 1e-9] = np.inf
            k, j = np.unravel_index(np.argmin(delta), delta.shape)
            if delta[k, j] >= -1e-6:
                break
            origem = a[j]
            z[origem] -= phi[origem, j]
            z[k] += phi[k, j]
            carga[origem] -= inst.demand[j]
            carga[k] += inst.demand[j]
            cnt[origem] -= 1
            cnt[k] += 1
            a[j] = k
    return a, time.perf_counter() - t0, it
