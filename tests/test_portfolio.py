"""Testes de skills/backtest/scripts/portfolio.py.

Rode com `python -m pytest tests` ou, sem pytest, `python tests/test_portfolio.py`.
"""
from __future__ import annotations

import math
import runpy
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "backtest" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import portfolio as P  # noqa: E402

CDI = 1.11 ** (1 / 252) - 1


def _levanta(funcao, *args, **kwargs) -> str:
    try:
        funcao(*args, **kwargs)
    except ValueError as erro:
        return str(erro)
    raise AssertionError(f"{funcao.__name__} deveria levantar ValueError")


def _amostra(rng, T, mu, cov):
    return rng.multivariate_normal(mu, cov, size=T)


def _cov_exemplo():
    sd = np.array([0.010, 0.015, 0.020, 0.008])
    C = np.array([[1.0, 0.3, 0.1, 0.0],
                  [0.3, 1.0, 0.5, 0.2],
                  [0.1, 0.5, 1.0, 0.1],
                  [0.0, 0.2, 0.1, 1.0]])
    return C * np.outer(sd, sd)


# ---------------------------------------------------------------------------
# Projeção e limites
# ---------------------------------------------------------------------------

def test_projecao_respeita_limites_e_soma():
    rng = np.random.default_rng(0)
    for _ in range(200):
        N = rng.integers(2, 12)
        lo, hi = np.zeros(N), np.full(N, max(1 / N + 0.01, rng.uniform(0.1, 1)))
        v = rng.normal(0, 1, N)
        w = P._projetar(v, lo, hi, 1.0)
        assert math.isclose(w.sum(), 1.0, abs_tol=1e-9)
        assert np.all(w >= lo - 1e-12) and np.all(w <= hi + 1e-12)
        # é a projeção: nenhum ponto viável aleatório fica mais perto de v
        for _ in range(5):
            u = P._projetar(rng.normal(0, 1, N), lo, hi, 1.0)
            assert np.sum((w - v) ** 2) <= np.sum((u - v) ** 2) + 1e-9


def test_limites_impossiveis_levantam_erro():
    R = np.random.default_rng(1).normal(0, 0.01, (300, 4))
    assert "Limites impossíveis" in _levanta(P.pesos_portfolio, R, "iguais", peso_max=0.2)
    assert "metodo" in _levanta(P.pesos_portfolio, R, "markowitz")


def test_pesos_respeitam_limites_em_todos_os_metodos():
    rng = np.random.default_rng(2)
    R = _amostra(rng, 500, np.full(4, 0.0005), _cov_exemplo())
    for m in P.METODOS:
        res = P.pesos_portfolio(R, m, peso_max=0.35, rf_por_periodo=CDI)
        w = np.array(list(res["pesos"].values()))
        assert np.all(w <= 0.35 + 1e-9) and np.all(w >= -1e-12), m
        if m != "kelly":
            assert math.isclose(w.sum(), 1.0, abs_tol=1e-8), m


# ---------------------------------------------------------------------------
# Métodos contra soluções conhecidas
# ---------------------------------------------------------------------------

def test_risk_parity_iguala_contribuicoes():
    S = _cov_exemplo()
    w = P._risk_parity(S)
    rc = w * (S @ w)
    assert np.allclose(rc / rc.sum(), 0.25, atol=1e-8)
    orc = np.array([0.4, 0.3, 0.2, 0.1])
    w = P._risk_parity(S, orc)
    rc = w * (S @ w)
    assert np.allclose(rc / rc.sum(), orc, atol=1e-8)


def test_diagonal_risk_parity_e_hrp_sao_vol_inversa():
    sd = np.array([0.01, 0.02, 0.04, 0.005])
    S = np.diag(sd ** 2)
    assert np.allclose(P._risk_parity(S), (1 / sd) / (1 / sd).sum(), atol=1e-9)
    ivp = (1 / sd ** 2) / (1 / sd ** 2).sum()
    assert np.allclose(P._hrp(S), ivp, atol=1e-12)        # HRP com correlação zero = variância inversa


def test_hrp_separa_grupos():
    # dois grupos de estratégias muito correlacionadas: cada grupo recebe ~metade do capital
    C = np.full((5, 5), 0.0)
    C[:3, :3] = 0.9
    C[3:, 3:] = 0.9
    np.fill_diagonal(C, 1.0)
    S = C * 0.01 ** 2
    arvore = P._arvore_hrp(C)
    assert sorted(map(sorted, (P._folhas(arvore[0]), P._folhas(arvore[1])))) == [[0, 1, 2], [3, 4]]
    w = P._hrp(S)
    assert math.isclose(w[:3].sum(), 0.5, abs_tol=0.05)


def test_min_variancia_sem_limites_ativos_bate_formula():
    sd = np.array([0.010, 0.012, 0.015, 0.011])
    C = np.array([[1.0, 0.3, 0.1, 0.0], [0.3, 1.0, 0.2, 0.2], [0.1, 0.2, 1.0, 0.1], [0.0, 0.2, 0.1, 1.0]])
    S = C * np.outer(sd, sd)
    um = np.ones(4)
    analitica = np.linalg.solve(S, um) / (um @ np.linalg.solve(S, um))
    assert np.all(analitica > 0)
    proj = lambda v: P._projetar(v, np.zeros(4), np.ones(4), 1.0)  # noqa: E731
    w = P._quadratico(S, np.zeros(4), proj, np.full(4, 0.25))
    assert np.allclose(w, analitica, atol=1e-7)


def test_max_sharpe_sem_limites_ativos_bate_formula():
    S = _cov_exemplo()
    mu = np.array([0.0004, 0.0006, 0.0008, 0.0003])
    analitica = np.linalg.solve(S, mu)
    analitica /= analitica.sum()
    assert np.all(analitica > 0)
    w = P._max_sharpe(mu, S, np.zeros(4), np.ones(4))
    sr = lambda x: (mu @ x) / math.sqrt(x @ S @ x)  # noqa: E731
    assert math.isclose(sr(w), sr(analitica), rel_tol=1e-7)
    assert np.allclose(w, analitica, atol=1e-4)


def test_max_sharpe_com_limites_ganha_de_qualquer_ponto_viavel():
    rng = np.random.default_rng(13)
    S = _cov_exemplo()
    lo, hi = np.zeros(4), np.full(4, 0.35)
    for _ in range(10):
        mu = rng.normal(0.0003, 0.0003, 4)
        w = P._max_sharpe(mu, S, lo, hi)
        assert np.all(w <= hi + 1e-9) and math.isclose(w.sum(), 1.0)
        if mu @ w <= 0:
            continue
        sr = lambda x: (mu @ x) / math.sqrt(x @ S @ x)  # noqa: E731
        for _ in range(300):
            u = P._projetar(rng.dirichlet(np.ones(4)), lo, hi, 1.0)
            assert sr(u) <= sr(w) + 1e-9


def test_max_sharpe_sem_media_positiva_vira_min_variancia():
    S = _cov_exemplo()
    w = P._max_sharpe(np.full(4, -0.001), S, np.zeros(4), np.ones(4))
    proj = lambda v: P._projetar(v, np.zeros(4), np.ones(4), 1.0)  # noqa: E731
    assert np.allclose(w, P._quadratico(S, np.zeros(4), proj, np.full(4, 0.25)), atol=1e-7)


def test_kelly_fracionado_bate_formula_e_deixa_sobra_no_cdi():
    rng = np.random.default_rng(3)
    S = _cov_exemplo()
    R = CDI + _amostra(rng, 3000, np.array([0.0004, 0.0003, 0.0005, 0.0002]), S)
    rf = np.full(len(R), CDI)
    mu = (R - CDI).mean(axis=0)
    Sh = P._cov(R, True)
    completo = np.linalg.solve(Sh, mu)
    w = P._pesos(R, rf, "kelly", 0.0, 100.0, True, 0.0, 0.5, 100.0, None)
    if np.all(completo > 0):
        assert np.allclose(w, 0.5 * completo, rtol=1e-5)
    limitado = P.pesos_portfolio(R, "kelly", rf_por_periodo=CDI, encolher_media=0.0)
    assert sum(limitado["pesos"].values()) <= 0.5 + 1e-9            # meio Kelly, alavancagem 1
    assert math.isclose(limitado["caixa"], 1 - sum(limitado["pesos"].values()))


def test_encolher_media_total_iguala_as_medias():
    mu = np.array([0.001, 0.0, -0.0005])
    assert np.allclose(P._media_encolhida(mu, 1.0), mu.mean())
    assert np.allclose(P._media_encolhida(mu, 0.0), mu)
    assert "encolher_media" in _levanta(P._media_encolhida, mu, 1.5)


# ---------------------------------------------------------------------------
# Covariância de Ledoit-Wolf
# ---------------------------------------------------------------------------

def test_ledoit_wolf_preserva_vols_e_reduz_erro_com_poucos_dados():
    rng = np.random.default_rng(4)
    N = 10
    C = np.full((N, N), 0.2)
    np.fill_diagonal(C, 1)
    sd = np.linspace(0.005, 0.02, N)
    S = C * np.outer(sd, sd)
    erro_lw = erro_am = 0.0
    for _ in range(50):
        X = _amostra(rng, 40, np.zeros(N), S)
        lw, delta = P.covariancia_ledoit_wolf(X)
        assert 0 <= delta <= 1
        assert np.allclose(np.sqrt(np.diag(lw)), X.std(axis=0, ddof=1))
        erro_lw += np.sum((lw - S) ** 2)
        erro_am += np.sum((np.cov(X, rowvar=False) - S) ** 2)
    assert erro_lw < erro_am
    # com muitos dados, quase não encolhe
    _, delta = P.covariancia_ledoit_wolf(_amostra(rng, 20000, np.zeros(N), S))
    assert delta < 0.05


# ---------------------------------------------------------------------------
# Simulação e walk-forward
# ---------------------------------------------------------------------------

def test_simulacao_deriva_custo_e_caixa_no_cdi():
    R = np.array([[0.10, 0.00], [0.00, 0.00], [-0.05, 0.02]])
    # sem rebalancear: comprar e segurar 50/50
    res = P.simular_portfolio(R, [0.5, 0.5], rebalancear_a_cada=None, custo_giro=0.0)
    final = 0.5 * 1.10 * 1.00 * 0.95 + 0.5 * 1.00 * 1.00 * 1.02
    assert math.isclose(np.prod(1 + res["retornos"]), final, rel_tol=1e-12)
    # rebalanceando todo período, com custo: giro inicial 1 e depois o desvio da deriva
    res = P.simular_portfolio(R, [0.5, 0.5], rebalancear_a_cada=1, custo_giro=0.01)
    w_deriva = np.array([0.55, 0.5]) / 1.05
    giro2 = np.abs(np.array([0.5, 0.5]) - w_deriva).sum()
    esperado = [1.05 * 0.99 - 1, 0.0 * 0 + (1 - 0.01 * giro2) - 1,
                (1 + 0.5 * -0.05 + 0.5 * 0.02) - 1]
    assert np.allclose(res["retornos"][:2], esperado[:2], atol=1e-12)
    # caixa rende rf
    res = P.simular_portfolio(R, [0.3, 0.2], rebalancear_a_cada=1, custo_giro=0.0, rf_por_periodo=0.001)
    assert math.isclose(res["retornos"][1], 0.5 * 0.001, rel_tol=1e-9)


def test_faixa_evita_rebalanceamentos_pequenos():
    rng = np.random.default_rng(5)
    R = rng.normal(0.0003, 0.005, (1000, 3))
    sem = P.simular_portfolio(R, [1 / 3] * 3, rebalancear_a_cada=5, custo_giro=0.001)
    com = P.simular_portfolio(R, [1 / 3] * 3, rebalancear_a_cada=5, custo_giro=0.001, faixa=0.05)
    assert com["n_rebalanceamentos"] < sem["n_rebalanceamentos"]
    assert com["custo_anual"] < sem["custo_anual"]


def test_walk_forward_nao_olha_o_futuro():
    rng = np.random.default_rng(6)
    R = _amostra(rng, 900, np.full(4, 0.0004), _cov_exemplo())
    base = P.walk_forward_portfolio(R, "max_sharpe", janela=300, rebalancear_a_cada=50)
    alterado = R.copy()
    corte = 650
    alterado[corte:] = rng.normal(0.01, 0.05, alterado[corte:].shape)      # muda só o futuro
    mudou = P.walk_forward_portfolio(alterado, "max_sharpe", janela=300, rebalancear_a_cada=50)
    for a, b in zip(base["pesos"], mudou["pesos"]):
        if a["periodo"] <= corte:
            assert a == b
    assert np.allclose(base["retornos"][:corte - 300], mudou["retornos"][:corte - 300])
    assert len(base["retornos"]) == 600


def test_walk_forward_ancorado_e_vol_alvo():
    rng = np.random.default_rng(7)
    R = CDI + _amostra(rng, 1500, np.full(4, 0.0004), _cov_exemplo())
    res = P.walk_forward_portfolio(R, "risk_parity", janela=500, rebalancear_a_cada=21,
                                   ancorada=True, vol_alvo=0.05, rf_por_periodo=CDI)
    vol = res["retornos"].std(ddof=1) * math.sqrt(252)
    assert 0.04 < vol < 0.06
    assert all(p["caixa"] > 0 for p in res["pesos"])          # sem alavancagem: sobra no CDI
    assert "janela" in _levanta(P.walk_forward_portfolio, R, janela=5)
    assert "desconhecidos" in _levanta(P.walk_forward_portfolio, R, janela=500, peso_maximo=0.3)


def test_walk_forward_aceita_pandas_e_confere_datas():
    try:
        import pandas as pd
    except ImportError:
        return
    rng = np.random.default_rng(8)
    idx = pd.bdate_range("2018-01-01", periods=800)
    df = pd.DataFrame(rng.normal(0.0004, 0.01, (800, 3)), index=idx, columns=["a", "b", "c"])
    res = P.walk_forward_portfolio(df, "hrp", janela=252, rf_por_periodo=pd.Series(CDI, index=idx))
    assert res["indice"][0] == idx[252] and res["pesos"][0]["data"] == idx[252]
    assert set(res["pesos"][0]) >= {"a", "b", "c", "caixa"}
    msg = _levanta(P.walk_forward_portfolio, df, janela=252,
                   rf_por_periodo=pd.Series(CDI, index=idx + pd.Timedelta(days=1)))
    assert "mesmas datas" in msg


# ---------------------------------------------------------------------------
# Diversificação
# ---------------------------------------------------------------------------

def test_diversificacao_de_estrategias_independentes():
    rng = np.random.default_rng(9)
    R = rng.normal(0, 0.01, (20000, 4))
    d = P.analise_diversificacao(R)
    assert math.isclose(d["razao_diversificacao"], 2.0, rel_tol=0.02)     # sqrt(4)
    assert math.isclose(d["n_efetivo_apostas"], 4.0, rel_tol=0.02)
    assert d["n_efetivo_autovalores"] > 3.9
    assert math.isclose(d["perdas_conjuntas_media"], 0.10, abs_tol=0.01)
    assert not d["pares_redundantes"]


def test_diversificacao_detecta_redundancia_e_perdas_conjuntas():
    rng = np.random.default_rng(10)
    T = 5000
    base = rng.normal(0, 0.01, T)
    crise = rng.random(T) < 0.05
    R = np.column_stack([base + rng.normal(0, 0.003, T),        # quase a mesma aposta que a 2
                         base + rng.normal(0, 0.003, T),
                         rng.normal(0, 0.01, T),
                         rng.normal(0, 0.01, T)])
    R[crise, 2:] -= 0.03                                        # 3 e 4 só se parecem nas crises
    d = P.analise_diversificacao(R)
    assert ("E1", "E2") == d["pares_redundantes"][0][:2]
    assert d["perdas_conjuntas"]["E3"]["E4"] > 0.4
    assert [p[:2] for p in d["pares_perdem_juntos"]] == [("E3", "E4")]
    pesos = {"E1": 0.1, "E2": 0.1, "E3": 0.4, "E4": 0.4}
    rc = P.analise_diversificacao(R, pesos=pesos)["contribuicao_risco"]
    assert math.isclose(sum(rc.values()), 1.0)


# ---------------------------------------------------------------------------
# Comparação de métodos
# ---------------------------------------------------------------------------

def test_comparar_metodos_inclui_iguais_e_alerta_overfitting():
    rng = np.random.default_rng(11)
    T, N = 1000, 6
    R = CDI + rng.normal(0.0002, 0.01, (T, N))            # todas iguais: nada bate 1/N
    res = P.comparar_metodos(R, metodos=("max_sharpe", "risk_parity"), janela=250,
                             rebalancear_a_cada=21, rf_por_periodo=CDI, n_sim=300)
    assert list(res["resultados"]) == ["iguais", "max_sharpe", "risk_parity"]
    assert len(res["resultados"]["iguais"]["retornos"]) == T - 250
    texto = " ".join(res["alertas"])
    assert "Nenhum método bateu 1/N" in texto
    assert "'max_sharpe' usa médias" in texto
    assert all(0 <= c["p_holm"] <= 1 for c in res["contra_iguais"])
    md = P.formatar_comparacao(res)
    assert "| Método x 1/N |" in md and "Alertas:" in md


def test_comparar_metodos_amostra_curta_usa_bootstrap():
    rng = np.random.default_rng(12)
    R = rng.normal(0.008, 0.04, (156, 3))                  # 13 anos mensais
    res = P.comparar_metodos(R, metodos=("hrp",), janela=48, rebalancear_a_cada=3,
                             periodos_por_ano=12, n_sim=300)
    assert res["contra_iguais"][0]["teste"] == "bootstrap estudentizado"
    assert any("Janela de 48" in a for a in res["alertas"])


def test_demonstracao_roda():
    runpy.run_path(str(SCRIPTS / "portfolio.py"), run_name="__main__")


if __name__ == "__main__":
    testes = [(nome, f) for nome, f in sorted(globals().items()) if nome.startswith("test_")]
    for nome, f in testes:
        f()
        print("ok ", nome)
    print(f"{len(testes)} testes passaram")
