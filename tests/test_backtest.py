"""Testes de skills/backtest/scripts/backtest.py e do exemplo de ponta a ponta.

Rode com `python -m pytest tests` ou, sem pytest, `python tests/test_backtest.py`.
"""
from __future__ import annotations

import math
import runpy
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "backtest" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import backtest as B  # noqa: E402
from estatistica import sharpe  # noqa: E402


def _precos(seed=0, T=400):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, T))
    abertura = np.r_[100.0, close[:-1]] * (1 + rng.normal(0, 0.004, T))
    return rng, close, abertura


def _sinal_com_viradas(rng, T):
    """Sinal com entradas, saídas, viradas de lado e mudanças de tamanho."""
    blocos = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=T // 7 + 1)
    return np.repeat(blocos, 7)[:T]


# ---------------------------------------------------------------------------
# Convenção de timing
# ---------------------------------------------------------------------------

def test_abertura_bate_com_simulacao_evento_a_evento():
    rng, close, abertura = _precos(1)
    sinal = _sinal_com_viradas(rng, len(close))
    res = B.retornos_estrategia(sinal, close, abertura)
    assert res["execucao"] == "abertura_t1"
    capital, posicao, curva = 1.0, 0.0, []
    for t in range(len(close)):
        if t > 0:
            capital *= 1 + posicao * (abertura[t] / close[t - 1] - 1)
            posicao = sinal[t - 1]             # sinal de t-1 executa na abertura de t
        capital *= 1 + posicao * (close[t] / abertura[t] - 1)
        curva.append(capital)
    assert np.allclose(np.cumprod(1 + res["retorno"]), curva, rtol=0, atol=1e-12)


def test_fechamento_t1_e_fechamento_t():
    rng, close, _ = _precos(2)
    sinal = _sinal_com_viradas(rng, len(close))
    r = np.r_[0.0, close[1:] / close[:-1] - 1]
    t1 = B.retornos_estrategia(sinal, close)
    assert t1["execucao"] == "fechamento_t1"
    assert np.allclose(t1["retorno"], np.r_[0, 0, sinal[:-2]] * r)
    t0 = B.retornos_estrategia(sinal, close, execucao="fechamento_t")
    assert np.allclose(t0["retorno"], np.r_[0, sinal[:-1]] * r)


def test_custos_por_giro_e_caixa_rende_rf():
    rng, close, abertura = _precos(3)
    sinal = _sinal_com_viradas(rng, len(close))
    sem = B.retornos_estrategia(sinal, close, abertura)
    com = B.retornos_estrategia(sinal, close, abertura, custo_por_lado=0.001)
    giro = np.abs(np.diff(com["posicao"], prepend=0.0))
    assert np.allclose(sem["retorno"] - com["retorno"], 0.001 * giro)
    rf = 0.0004
    livre = B.retornos_estrategia(sinal, close, abertura, rf_por_periodo=rf)
    total = B.retornos_estrategia(sinal, close, abertura, rf_por_periodo=rf, caixa="total")
    assert np.allclose(livre["retorno"] - sem["retorno"], rf * (1 - np.abs(sem["posicao"])))
    assert np.allclose(total["retorno"] - sem["retorno"], rf)


def test_nan_no_sinal_vira_zero_e_taxa_anual_e_recusada():
    _, close, _ = _precos(4, 50)
    sinal = np.r_[np.full(10, np.nan), np.ones(40)]
    res = B.retornos_estrategia(sinal, close)
    assert np.all(np.asarray(res["posicao"])[:12] == 0)
    try:
        B.retornos_estrategia(sinal, close, rf_por_periodo=0.11)
    except ValueError as e:
        assert "taxa anual" in str(e)
    else:
        raise AssertionError("taxa anual deveria ser recusada")


def test_series_do_pandas_mantem_as_datas():
    try:
        import pandas as pd
    except ImportError:
        return
    _, close, abertura = _precos(5, 60)
    idx = pd.bdate_range("2024-01-01", periods=60)
    res = B.retornos_estrategia(pd.Series(np.ones(60), idx), pd.Series(close, idx), pd.Series(abertura, idx))
    assert res["retorno"].index.equals(idx)


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------

def test_trades_reproduzem_a_curva_sem_custos():
    for execucao in ("abertura_t1", "fechamento_t1", "fechamento_t"):
        rng, close, abertura = _precos(6)
        sinal = _sinal_com_viradas(rng, len(close))
        res = B.retornos_estrategia(sinal, close, abertura, execucao=execucao)
        assert math.isclose(np.prod(1 + res["trades"]["retorno"]), np.prod(1 + res["retorno"]),
                            rel_tol=1e-12)


def test_trades_simples():
    close = np.array([10, 10, 11, 12, 12, 11, 10, 10, 10.0])
    sinal = np.array([0, 1, 1, 0, 0, -1, -1, 0, 0.0])
    res = B.retornos_estrategia(sinal, close, execucao="fechamento_t")
    tr = res["trades"]
    assert list(tr["lado"]) == [1, -1]
    assert list(tr["inicio"]) == [2, 6] and list(tr["duracao"]) == [2, 2]
    assert np.allclose(tr["retorno"], [12 / 10 - 1, 1 - 10 / 11])
    assert not tr["aberto"].any()
    custo = B.retornos_estrategia(sinal, close, execucao="fechamento_t", custo_por_lado=0.01)["trades"]
    assert np.allclose(custo["retorno"][0], 0.99 * 1.2 * 0.99 - 1)


# ---------------------------------------------------------------------------
# Custos e carregamento
# ---------------------------------------------------------------------------

def test_breakeven_zera_o_excesso_medio():
    rng, close, abertura = _precos(7, 1000)
    sinal = _sinal_com_viradas(rng, len(close))
    rf = 0.0003
    s = B.sensibilidade_custos(sinal, close, abertura, custo_por_lado=0.0005, rf_por_periodo=rf)
    be = s["custo_breakeven"]
    res = B.retornos_estrategia(sinal, close, abertura, custo_por_lado=abs(be), rf_por_periodo=rf)
    if be > 0:
        assert abs(np.mean(res["retorno"] - rf)) < 1e-12
    assert [c["multiplicador"] for c in s["cenarios"]] == [0, 1, 2, 3]
    assert "breakeven" in B.formatar_sensibilidade(s)


def test_custo_por_lado_soma_componentes():
    c = B.custo_por_lado(comissao=0.0005, spread_bps=10, slippage_bps=5, comissao_fixa=5,
                         valor_medio_ordem=10_000, taxas=0.00002)
    assert math.isclose(c["total"], 0.0005 + 0.0005 + 0.0005 + 0.0005 + 0.00002)


def test_carregamento_funding_e_aluguel():
    rng, close, abertura = _precos(8, 300)
    sinal = _sinal_com_viradas(rng, 300)
    base = B.retornos_estrategia(sinal, close, abertura)
    fin = B.retornos_estrategia(sinal, close, abertura, financiamento_por_periodo=0.0003)
    pos = np.asarray(base["posicao"])
    assert np.allclose(base["retorno"] - fin["retorno"], 0.0003 * pos)   # comprado paga, vendido recebe
    alug = B.retornos_estrategia(sinal, close, abertura, aluguel_por_periodo=0.0002)
    assert np.allclose(base["retorno"] - alug["retorno"], 0.0002 * np.maximum(-pos, 0))
    try:
        B.retornos_estrategia(sinal, close, abertura, aluguel_por_periodo=0.05)
    except ValueError as e:
        assert "taxa anual" in str(e)
    else:
        raise AssertionError("taxa anual deveria ser recusada")
    com = B.sensibilidade_custos(sinal, close, abertura, aluguel_por_periodo=0.0002)
    sem = B.sensibilidade_custos(sinal, close, abertura)
    assert com["custo_breakeven"] < sem["custo_breakeven"]


# ---------------------------------------------------------------------------
# Walk-forward e robustez
# ---------------------------------------------------------------------------

def test_walk_forward_escolhe_so_com_o_passado():
    rng = np.random.default_rng(9)
    T = 1000
    base = rng.normal(0, 0.01, (T, 3))
    base[:500, 0] += 0.003      # config 0 boa na primeira metade
    base[500:, 1] += 0.003      # config 1 boa na segunda
    grade = [{"k": 0}, {"k": 1}, {"k": 2}]
    wf = B.walk_forward_parametros(lambda k: base[:, k], grade, janela_treino=200, janela_teste=100)
    for w in wf["janelas"]:
        treino = slice(w["inicio_teste"] - 200, w["inicio_teste"])
        x = base[treino]
        assert w["indice_configuracao"] == int(np.argmax(x.mean(0) / x.std(0, ddof=1)))
    assert len(wf["retorno_fora"]) == T - 200
    assert wf["janelas"][0]["parametros"] == {"k": 0}
    assert wf["parametros_atuais"] == {"k": 1}
    assert wf["n_tentativas"] == 3 and wf["matriz_retornos"].shape == (T, 3)
    cfg = wf["configuracao_por_periodo"]
    assert np.all(cfg[:200] == -1)
    assert np.allclose(np.asarray(wf["retorno_fora"]), base[np.arange(200, T), cfg[200:]])


def test_walk_forward_recebe_dict_e_guarda_posicoes():
    rng, close, abertura = _precos(10, 600)

    def f(n):
        media = np.convolve(close, np.ones(n) / n)[:len(close)]
        media[:n - 1] = np.nan
        return B.retornos_estrategia((close > media).astype(float), close, abertura)

    wf = B.walk_forward_parametros(f, {"n": [5, 10, 20]}, janela_treino=200, janela_teste=50,
                                   ancorada=True)
    assert wf["matriz_posicoes"].shape == (600, 3)
    assert np.isfinite(wf["sharpe_fora"])


def test_robustez_detecta_pico_isolado_e_aceita_plato():
    rng = np.random.default_rng(11)
    ruido = rng.normal(0, 0.01, 2000)

    def pico(a, b):
        return ruido + (0.004 if (a, b) == (3, 3) else -0.001)

    grade = {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]}
    r = B.robustez_parametros(pico, grade, {"a": 3, "b": 3})
    assert any("Pico isolado" in a for a in r["alertas"])

    def plato(a, b):
        return ruido + 0.002 - 0.0001 * (abs(a - 3) + abs(b - 3))

    r = B.robustez_parametros(plato, grade, {"a": 3, "b": 3})
    assert r["alertas"] == []
    assert "**" in B.mapa_calor(r, "a", "b")


# ---------------------------------------------------------------------------
# Regimes, look-ahead e dimensionamento
# ---------------------------------------------------------------------------

def test_quebrar_por_e_regimes():
    rng, close, _ = _precos(12, 600)
    r = rng.normal(0, 0.01, 600)
    q = B.quebrar_por(r, np.repeat(["a", "b"], 300))
    assert q["a"]["n_periodos"] == 300 and math.isclose(q["b"]["retorno_total"], np.prod(1 + r[300:]) - 1)
    tend = B.regime_tendencia(close, janela=50)
    assert set(tend[:50]) == {"aquecimento"} and set(tend[50:]) <= {"alta", "baixa"}
    vol = B.regime_volatilidade(r, janela=20)
    assert set(vol[20:]) == {"vol baixa", "vol média", "vol alta"}


def test_futuro_truncado_pega_media_centrada():
    try:
        import pandas as pd
    except ImportError:
        return
    _, close, _ = _precos(13, 300)
    df = pd.DataFrame({"fechamento": close})
    causal = lambda d: (d["fechamento"] > d["fechamento"].rolling(20).mean()).astype(float)  # noqa: E731
    centrada = lambda d: (d["fechamento"] > d["fechamento"].rolling(20, center=True).mean()).astype(float)  # noqa: E731
    normalizada = lambda d: (d["fechamento"] - d["fechamento"].mean()) / d["fechamento"].std()  # noqa: E731
    assert B.teste_futuro_truncado(causal, df)["ok"]
    assert not B.teste_futuro_truncado(centrada, df)["ok"]
    assert not B.teste_futuro_truncado(normalizada, df)["ok"]


def test_atraso_alerta_com_look_ahead():
    rng, close, abertura = _precos(14, 1000)
    r = np.r_[0.0, close[1:] / close[:-1] - 1]
    espiao = np.sign(np.r_[r[2:], 0, 0])       # usa o retorno de t+2: look-ahead
    assert B.teste_atraso(espiao, close)["alerta"]
    honesto = _sinal_com_viradas(rng, 1000)
    assert not B.teste_atraso(honesto, close)["alerta"]


def test_vol_alvo_usa_so_o_passado_e_kelly():
    rng = np.random.default_rng(15)
    r = rng.normal(0, 0.01, 500)
    s = np.ones(500)
    a = B.posicao_vol_alvo(s, r, vol_alvo=0.10, janela=60, alavancagem_max=2)
    r2 = r.copy()
    r2[300:] *= 5
    b = B.posicao_vol_alvo(s, r2, vol_alvo=0.10, janela=60, alavancagem_max=2)
    assert np.allclose(a[:300], b[:300]) and np.all(a[:60] == 0)
    assert 0.5 < a[100] < 0.8                  # 10% / (1% x raiz de 252) ~ 0,63
    k = B.kelly_estrategia(rng.normal(0.001, 0.01, 2000), fracao=0.25)
    assert 5 < k["kelly_completo"] < 20 and math.isclose(k["alavancagem_sugerida"], 0.25 * k["kelly_completo"])


# ---------------------------------------------------------------------------
# Demonstrações
# ---------------------------------------------------------------------------

def test_demonstracao_roda():
    runpy.run_path(str(SCRIPTS / "backtest.py"), run_name="__main__")


def test_exemplo_completo_roda():
    try:
        import pandas  # noqa: F401
    except ImportError:
        return
    sys.path.insert(0, str(SCRIPTS))
    import exemplo_completo
    res = exemplo_completo.main(rapido=True)
    cripto = exemplo_completo.main(rapido=True, periodos_por_ano=365)
    assert cripto["wf"]["janelas"][0]["inicio_teste"] == 730
    assert res["trunc"]["ok"]
    assert res["veredito"] in ("Rejeitada", "Inconclusiva", "Promissora (incubar)",
                               "Aprovada para operar com tamanho reduzido")
    assert res["wf"]["n_tentativas"] == 25
    assert np.isfinite(sharpe(res["wf"]["retorno_fora"]))


if __name__ == "__main__":
    testes = [(nome, f) for nome, f in sorted(globals().items()) if nome.startswith("test_")]
    for nome, f in testes:
        f()
        print("ok ", nome)
    print(f"{len(testes)} testes passaram")
