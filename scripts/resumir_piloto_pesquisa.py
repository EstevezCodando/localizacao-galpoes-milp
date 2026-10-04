"""Exporta métricas no horizonte do manifesto, sem retreinar ou escolher hiperparâmetros."""

import argparse
import json
from pathlib import Path

import pandas as pd

from alocacao_capacitada.pesquisa.analise import preparar, resumo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifesto.json").read_text(encoding="utf-8"))
    horizonte = manifest["config"]["tempo"]
    raw = pd.read_csv(args.run / "resultados.csv")
    dados = preparar(raw, horizonte)
    dados.drop(columns="traj").to_csv(args.run / "metricas.csv", index=False)
    tabela = resumo(dados, horizonte).sort_values("integral_primal")
    tabela.to_csv(args.run / "resumo.csv", index=False)
    print(tabela[["metodo", "n", "n_instancias", "validas", "desvio_mediano",
                  "integral_primal", "atinge_1%", "t_total_med"]].to_string(index=False))
    print("Sucesso no prazo:", int(dados.sucesso_no_prazo.sum()), "/", len(dados))
    print("Estouro máximo (s):", raw.estouro_s.max())
    print("Execuções com estouro > 0,05 s:", int((raw.estouro_s > .05).sum()))
    print("Motivos de fallback:", raw.motivo.dropna().value_counts().to_dict())


if __name__ == "__main__":
    main()
