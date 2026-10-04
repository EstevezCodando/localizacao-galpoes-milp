"""Manifesto por conteúdo e checkpoint transacional para experimentos retomáveis."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sqlite3
from pathlib import Path

import pandas as pd


def hash_arquivo(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifesto(config: dict, arquivos: list[Path]) -> dict:
    raiz = Path(__file__).resolve().parents[3]
    fontes = sorted((raiz / "src").rglob("*.py"))
    fontes += [raiz / "pyproject.toml", raiz / "uv.lock"]
    return {
        "schema": 2, "config": config,
        "python": platform.python_version(), "plataforma": platform.platform(),
        "versoes": {nome: importlib.metadata.version(nome)
                    for nome in ("numpy", "pandas", "scipy", "pyscipopt", "highspy", "torch",
                                 "lightgbm", "ortools")},
        "fontes": {p.relative_to(raiz).as_posix(): hash_arquivo(p) for p in fontes},
        "entradas": {str(p): hash_arquivo(p) for p in arquivos},
        "relogio": "perf_counter; limite cooperativo, estouros registrados",
        "aquecimento": "imports/modelos carregados uma vez por processo, fora do custo online",
    }


class Registro:
    def __init__(self, raiz: Path, manifest: dict) -> None:
        texto = json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2)
        self.run_id = hashlib.sha256(texto.encode()).hexdigest()[:20]
        self.path = raiz / self.run_id
        self.path.mkdir(parents=True, exist_ok=True)
        arquivo = self.path / "manifesto.json"
        if arquivo.exists() and arquivo.read_text(encoding="utf-8") != texto:
            raise ValueError("Manifesto divergente: não misturar execuções")
        arquivo.write_text(texto, encoding="utf-8")
        self.db = sqlite3.connect(self.path / "checkpoint.sqlite")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS resultados (chave TEXT PRIMARY KEY, linha TEXT)")
        self.db.commit()

    @staticmethod
    def chave(instancia: str, metodo: str, seed_treino: int, seed_solver: int) -> str:
        return json.dumps([instancia, metodo, seed_treino, seed_solver])

    def feitos(self) -> set[str]:
        return {r[0] for r in self.db.execute("SELECT chave FROM resultados")}

    def gravar(self, row: dict) -> None:
        key = self.chave(row["instancia"], row["metodo"], row["seed_treino"], row["seed_solver"])
        with self.db:
            self.db.execute("INSERT INTO resultados VALUES (?, ?)",
                            (key, json.dumps({**row, "run_id": self.run_id})))

    def exportar(self) -> pd.DataFrame:
        df = pd.DataFrame([json.loads(r[0]) for r in
                           self.db.execute("SELECT linha FROM resultados ORDER BY chave")])
        temp = self.path / "resultados.tmp.csv"
        df.to_csv(temp, index=False)
        temp.replace(self.path / "resultados.csv")
        return df

    def close(self) -> None:
        self.db.close()
