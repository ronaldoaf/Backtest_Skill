"""Protocolo da skill backtest-lighter: contas do short, custos, controle e critério."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills" / "backtest-lighter" / "scripts"))

import protocolo as P  # noqa: E402


def _df(precos, freq="h"):
    idx = pd.date_range("2024-01-01", periods=len(precos), freq=freq, tz="UTC")
    p = np.asarray(precos, float)
    return pd.DataFrame({"open": p, "high": p, "low": p, "close": p}, index=idx)


def test_short_composto_e_custos():
    df = _df([100, 100, 50, 50])
    tr = pd.DataFrame({"i": [1], "j": [2], "d": [-1]})
    r = P.retornos_trades(df, tr, "btc", mult_fund=0)
    assert abs(r.bruto.iloc[0] - 0.5) < 1e-12           # 1 - P1/P0, não log(P0/P1)=0,69
    assert abs(r.ret.iloc[0] - (0.5 - 2 * P.CUSTO["btc"])) < 1e-12


def test_funding_long_paga_short_recebe():
    df = _df([100] * 25)
    tr = pd.DataFrame({"i": [0, 0], "j": [24, 24], "d": [1, -1]})
    r = P.retornos_trades(df, tr, "wti", mult_custo=0)
    dia = P.FUND["wti"] / 100 / 365
    assert abs(r.ret.iloc[0] + dia) < 1e-12 and abs(r.ret.iloc[1] - dia) < 1e-12


def test_controle_sem_sinal_t_pequeno():
    rng = np.random.default_rng(3)
    df = _df(100 * np.exp(np.cumsum(rng.normal(0, 0.005, 5000))))
    ii = rng.integers(0, 4900, 300)
    tr = P.retornos_trades(df, pd.DataFrame({"i": ii, "j": ii + 24, "d": 1}), "btc")
    assert abs(P.controle_aleatorio(df, tr, "btc", nsim=10)["t"]) < 3


def test_divisoes_e_filtro_treino():
    df = _df(np.arange(1000) + 1.0)
    tr, va, oos, partes = P.treino_validacao_oos(df)
    assert len(tr) == 600 and len(va) == 300 and len(oos) == 100 and len(partes) == 6
    assert P.filtro_treino([0.1, 0.1, 0.1, 0.1, -0.05, -0.05], 150)
    assert not P.filtro_treino([2.0, -0.1, -0.1, -0.1, 0.01, 0.01], 150)   # só a melhor parte
    assert not P.filtro_treino([0.1] * 6, 50)


def test_correcoes_multiplos_testes():
    p = [0.001, 0.01, 0.03, 0.5]
    assert P.holm(p) == [True, True, False, False]
    assert P.benjamini_hochberg(p) == [True, True, True, False]


def test_filtro_correlacao_e_carteira():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=400, freq="D")
    a = pd.Series(rng.normal(0.002, 0.01, 400), idx)
    d = {"a": a, "a2": a + rng.normal(0, 0.001, 400), "b": pd.Series(rng.normal(0.001, 0.01, 400), idx)}
    ac = P.filtro_correlacao(d)
    assert "b" in ac and not {"a", "a2"} <= set(ac)
    c = P.carteira({"a": a, "b": d["b"]}, teto=0.6)
    assert abs(c.iloc[0] - (a.iloc[0] + d["b"].iloc[0]) / 2) < 1e-12


def test_aprovar_reprova_passeio_aleatorio():
    rng = np.random.default_rng(1)
    df = _df(100 * np.exp(np.cumsum(rng.normal(0, 0.006, 24 * 400))))

    def estr(d):
        ii = np.arange(1, len(d) - 30, 48)
        return pd.DataFrame({"i": ii, "j": ii + 24, "d": 1})

    r = P.aprovar(estr, df, "eth", nsim=5)
    assert set(r["itens"]) == {"tercos_positivos", "sharpe_vs_bh", "t_controle", "custo4x_positivo",
                               "sem_funding_positivo"}
    assert r["aprovada"] is False or r["t"] > 2


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
    print("ok")
