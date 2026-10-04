"""Ranqueadores aprendidos de centros: GNN bipartida, MLP (mesmos agregados) e LightGBM.

GNN (doc 08): grafo centro–cliente, 3 rodadas de passagem de mensagem com arestas
condicionadas (mensagem = MLP([h_origem, atributo_aresta])), agregação média+soma normalizada,
dimensão 64. Não é o grafo variável–restrição do Gasse et al. (P02): aqui o grafo é logístico.

Rótulo: frequência de abertura do centro no pool de soluções a até `tol` do melhor custo
conhecido (doc 07: um único ótimo não define todas as instalações úteis).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from alocacao_capacitada.pesquisa.atributos import Grafo, tabular
from alocacao_capacitada.pesquisa.tempos import cronometrar

torch.set_num_threads(1)


def rotulo_pool(pool: list[tuple[float, np.ndarray]], m: int, tol: float = 0.01) -> np.ndarray:
    """Frequência de abertura nas soluções do pool com custo <= (1+tol) * melhor."""
    if not pool:
        raise ValueError("pool vazio")
    best = min(o for o, _ in pool)
    good = [y for o, y in pool if o <= best * (1 + tol) + 1e-9]
    return np.mean(np.array(good, dtype=float), axis=0).reshape(m)


class _Mlp(nn.Sequential):
    def __init__(self, i: int, h: int, o: int) -> None:
        super().__init__(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


class GnnBipartida(nn.Module):
    def __init__(self, f_fac: int, f_cli: int, f_ed: int, dim: int = 64, camadas: int = 3) -> None:
        super().__init__()
        self.enc_f = _Mlp(f_fac, dim, dim)
        self.enc_c = _Mlp(f_cli, dim, dim)
        self.enc_e = _Mlp(f_ed, dim, dim)
        self.msg_fc = nn.ModuleList([_Mlp(2 * dim, dim, dim) for _ in range(camadas)])
        self.msg_cf = nn.ModuleList([_Mlp(2 * dim, dim, dim) for _ in range(camadas)])
        self.upd_c = nn.ModuleList([_Mlp(3 * dim, dim, dim) for _ in range(camadas)])
        self.upd_f = nn.ModuleList([_Mlp(3 * dim, dim, dim) for _ in range(camadas)])
        self.norm_f = nn.ModuleList([nn.LayerNorm(dim) for _ in range(camadas)])
        self.norm_c = nn.ModuleList([nn.LayerNorm(dim) for _ in range(camadas)])
        self.head = _Mlp(dim, dim, 1)

    @staticmethod
    def _agg(msg: torch.Tensor, idx: torch.Tensor, size: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = torch.zeros(size, msg.shape[1]).index_add_(0, idx, msg)
        cnt = torch.zeros(size).index_add_(0, idx, torch.ones(idx.shape[0])).clamp(min=1)
        mean = s / cnt[:, None]
        return mean, s / cnt.max()

    def forward(self, g: dict[str, torch.Tensor]) -> torch.Tensor:
        hf, hc, he = self.enc_f(g["xf"]), self.enc_c(g["xc"]), self.enc_e(g["xe"])
        s, d = g["src"], g["dst"]
        m, n = hf.shape[0], hc.shape[0]
        for k in range(len(self.msg_fc)):
            mc = self.msg_fc[k](torch.cat([hf[s], he], 1))
            mean, tot = self._agg(mc, d, n)
            hc = self.norm_c[k](hc + self.upd_c[k](torch.cat([hc, mean, tot], 1)))
            mf = self.msg_cf[k](torch.cat([hc[d], he], 1))
            mean, tot = self._agg(mf, s, m)
            hf = self.norm_f[k](hf + self.upd_f[k](torch.cat([hf, mean, tot], 1)))
        return self.head(hf).squeeze(1)


def _tensor(g: Grafo, mu: dict[str, np.ndarray], sd: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        "xf": torch.from_numpy((g.x_fac - mu["f"]) / sd["f"]).float(),
        "xc": torch.from_numpy((g.x_cli - mu["c"]) / sd["c"]).float(),
        "xe": torch.from_numpy((g.x_ed - mu["e"]) / sd["e"]).float(),
        "src": torch.from_numpy(g.src_fac.astype(np.int64)),
        "dst": torch.from_numpy(g.dst_cli.astype(np.int64)),
    }


def _norm_stats(blocks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    allv = np.vstack(blocks)
    mu = allv.mean(axis=0)
    sd = allv.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def _loss(logit: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    pos = y.mean().clamp(0.02, 0.98)
    w = torch.where(y > 0.5, 0.5 / pos, 0.5 / (1 - pos))
    return nn.functional.binary_cross_entropy_with_logits(logit, y, weight=w)


@dataclass
class RanqueadorGnn:
    modelo: GnnBipartida
    mu: dict[str, np.ndarray]
    sd: dict[str, np.ndarray]
    nome: str = "gnn"

    @cronometrar("t_inferencia")
    def score(self, g: Grafo) -> np.ndarray:
        self.modelo.eval()
        with torch.no_grad():
            return self.modelo(_tensor(g, self.mu, self.sd)).numpy()


def treinar_gnn(
    grafos: list[Grafo], rotulos: list[np.ndarray], val: tuple[list[Grafo], list[np.ndarray]],
    epocas: int = 80, seed: int = 0, dim: int = 64, camadas: int = 3, lr: float = 2e-3,
) -> tuple[RanqueadorGnn, list[tuple[int, float, float]]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mu, sd = {}, {}
    mu["f"], sd["f"] = _norm_stats([g.x_fac for g in grafos])
    mu["c"], sd["c"] = _norm_stats([g.x_cli for g in grafos])
    mu["e"], sd["e"] = _norm_stats([g.x_ed for g in grafos])
    g0 = grafos[0]
    model = GnnBipartida(g0.x_fac.shape[1], g0.x_cli.shape[1], g0.x_ed.shape[1], dim, camadas)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    tr = [(_tensor(g, mu, sd), torch.from_numpy(y).float()) for g, y in zip(grafos, rotulos)]
    va = [(_tensor(g, mu, sd), torch.from_numpy(y).float()) for g, y in zip(*val)]
    hist: list[tuple[int, float, float]] = []
    best, best_state, wait = np.inf, None, 0
    for ep in range(epocas):
        model.train()
        tot = 0.0
        for k in rng.permutation(len(tr)):
            g, y = tr[k]
            opt.zero_grad()
            loss = _loss(model(g), y)
            loss.backward()
            opt.step()
            tot += loss.item()
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([float(_loss(model(g), y)) for g, y in va]))
        hist.append((ep, tot / len(tr), vl))
        if vl < best - 1e-4:
            best, wait = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 15:  # parada antecipada na VALIDAÇÃO (teste fechado)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return RanqueadorGnn(model, mu, sd), hist


@dataclass
class RanqueadorTabular:
    modelo: object
    nome: str
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None

    @cronometrar("t_inferencia")
    def score(self, g: Grafo) -> np.ndarray:
        X = tabular(g)
        if self.mu is not None and self.sd is not None:
            X = (X - self.mu) / self.sd
            with torch.no_grad():
                return self.modelo(torch.from_numpy(X).float()).squeeze(1).numpy()  # type: ignore[operator]
        return np.asarray(self.modelo.predict(X))  # type: ignore[attr-defined]


def treinar_lgbm(grafos: list[Grafo], rotulos: list[np.ndarray], seed: int = 0) -> RanqueadorTabular:
    import lightgbm as lgb

    X = np.vstack([tabular(g) for g in grafos])
    y = np.concatenate(rotulos)
    model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31,
                              min_child_samples=20, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=1)
    model.fit(X, y)
    return RanqueadorTabular(model, "lgbm")


def treinar_mlp(
    grafos: list[Grafo], rotulos: list[np.ndarray], val: tuple[list[Grafo], list[np.ndarray]],
    seed: int = 0, epocas: int = 300,
) -> RanqueadorTabular:
    torch.manual_seed(seed)
    X = np.vstack([tabular(g) for g in grafos])
    mu, sd = _norm_stats([X])
    Xt = torch.from_numpy((X - mu) / sd).float()
    yt = torch.from_numpy(np.concatenate(rotulos)).float()
    Xv = torch.from_numpy((np.vstack([tabular(g) for g in val[0]]) - mu) / sd).float()
    yv = torch.from_numpy(np.concatenate(val[1])).float()
    model = nn.Sequential(nn.Linear(X.shape[1], 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
                          nn.Linear(64, 1))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, state, wait = np.inf, None, 0
    for _ in range(epocas):
        model.train()
        for b in torch.randperm(Xt.shape[0]).split(256):
            opt.zero_grad()
            _loss(model(Xt[b]).squeeze(1), yt[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(_loss(model(Xv).squeeze(1), yv))
        if vl < best - 1e-4:
            best, wait, state = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 30:
                break
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    return RanqueadorTabular(model, "mlp", mu, sd)
