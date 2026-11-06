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
        for lo in (0.0, 0.1):
            res = P.pesos_portfolio(R, m, peso_min=lo, peso_max=0.35, rf_por_periodo=CDI)
            w = np.array(list(res["pesos"].values()))
            assert np.all(w <= 0.35 + 1e-9) and np.all(w >= lo - 1e-9), (m, lo)
            if m != "kelly":
                assert math.isclose(w.sum(), 1.0, abs_tol=1e-8), m


def test_limites_por_nome_e_proporcionais():
    rng = np.random.default_rng(21)
    sd = np.array([0.003, 0.010, 0.012, 0.015, 0.02])
    C = np.full((5, 5), 0.3)
    np.fill_diagonal(C, 1)
    R = _amostra(rng, 2000, np.zeros(5), C * np.outer(sd, sd))
    livre = np.array(list(P.pesos_portfolio(R, "risk_parity")["pesos"].values()))
    assert livre[0] > 0.3
    limitado = np.array(list(P.pesos_portfolio(R, "risk_parity", peso_max=0.3)["pesos"].values()))
    assert math.isclose(limitado[0], 0.3) and math.isclose(limitado.sum(), 1.0)
    # as que não bateram no limite mantêm as proporções do risk parity
    assert np.allclose(limitado[1:] / limitado[1:].sum(), livre[1:] / livre[1:].sum())
    por_nome = P.pesos_portfolio(R, "risk_parity", peso_max={f"E{i + 1}": 0.3 for i in range(5)})
    assert np.allclose(list(por_nome["pesos"].values()), limitado)
    assert "faltam" in _levanta(P.pesos_portfolio, R, "hrp", peso_max={"E1": 0.5})
    # limites diferentes por estratégia, inclusive mínimos
    lo, hi = np.array([0.0, 0.1, 0.0, 0.2, 0.0]), np.array([0.25, 0.5, 0.5, 0.5, 0.15])
    w = P._limitar(livre, lo, hi)
    assert math.isclose(w.sum(), 1.0) and np.all(w >= lo - 1e-12) and np.all(w <= hi + 1e-12)


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
    for _ in range(20):
        mu = rng.normal(0.0, 0.0006, 4)                     # muitas vezes com média 1/N negativa
        w = P._max_sharpe(mu, S, lo, hi)
        assert np.all(w <= hi + 1e-9) and math.isclose(w.sum(), 1.0)
        if P._maior_media(mu, lo, hi) @ mu <= 0:           # nenhuma carteira de média positiva
            continue
        sr = lambda x: (mu @ x) / math.sqrt(x @ S @ x)  # noqa: E731
        assert mu @ w > 0
        for _ in range(300):
            u = P._projetar(rng.dirichlet(np.ones(4)), lo, hi, 1.0)
            assert sr(u) <= sr(w) + 1e-9
        assert sr(P._maior_media(mu, lo, hi)) <= sr(w) + 1e-12


def test_max_sharpe_com_media_1_n_abaixo_do_cdi():
    # 1/N tem média negativa, mas a estratégia 4 sozinha tem Sharpe positivo
    S = _cov_exemplo()
    sd = np.sqrt(np.diag(S))
    mu = np.array([-0.88, -0.28, -2.07, 0.88]) / math.sqrt(252) * sd
    assert mu.mean() < 0
    w = P._max_sharpe(mu, S, np.zeros(4), np.ones(4))
    sr = (mu @ w) / math.sqrt(w @ S @ w) * math.sqrt(252)
    assert sr >= 0.88 - 1e-9 and w[3] > 0.5


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
    w = P._pesos(R, rf, "kelly", -100.0, 100.0, True, 0.0, 0.5, 100.0, None)
    assert np.allclose(w, 0.5 * completo, rtol=1e-6)               # sem limites: f × Kelly
    # meio Kelly passa de 100%: com alavancagem 1 investe 100% (não 50%), no ótimo restrito
    assert 0.5 * completo.sum() > 1
    limitado = P.pesos_portfolio(R, "kelly", rf_por_periodo=CDI, encolher_media=0.0)
    wl = np.array(list(limitado["pesos"].values()))
    assert math.isclose(wl.sum(), 1.0, abs_tol=1e-9)
    assert math.isclose(limitado["caixa"], 1 - wl.sum())
    obj = lambda x: mu @ x - x @ Sh @ x / (2 * 0.5)  # noqa: E731
    for _ in range(300):
        u = rng.dirichlet(np.ones(4)) * rng.uniform(0, 1)
        assert obj(u) <= obj(wl) + 1e-12
    # Kelly fraco: sobra no CDI
    fraco = P.pesos_portfolio(CDI + _amostra(rng, 3000, np.full(4, 0.00002), S), "kelly",
                              rf_por_periodo=CDI, encolher_media=0.0)
    assert 0 < fraco["caixa"] < 1


def test_quadratico_para_no_otimo():
    # com momento, dois iterados iguais num vértice não garantem o ótimo: confere o resíduo
    rng = np.random.default_rng(2024)
    for _ in range(300):
        N = int(rng.integers(3, 8))
        B = rng.normal(0, 1, (N, 2))
        sd = rng.uniform(0.005, 0.02, N)
        C = B @ B.T + np.diag(rng.uniform(0.05, 1.0, N))
        d = np.sqrt(np.diag(C))
        S = C / np.outer(d, d) * np.outer(sd, sd)
        mu = rng.uniform(-0.02, 0.1, N) * sd
        for L in (1.0, 2.0):
            proj = lambda v: P._projetar_ate(v, np.zeros(N), np.ones(N), L)  # noqa: E731
            w = P._quadratico(S, mu, proj, np.zeros(N))
            passo = 1 / np.linalg.eigvalsh(S)[-1]
            assert np.max(np.abs(proj(w - passo * (S @ w - mu)) - w)) < 1e-9


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
    assert "kelly" in _levanta(P.walk_forward_portfolio, R, "kelly", janela=500, vol_alvo=0.05)
    # pesos atuais: mesmos parâmetros, com as últimas linhas (ancorada = todas)
    atual = res["pesos_atuais"]
    w = np.array([atual[n] for n in res["nomes"]])
    S = P.covariancia_ledoit_wolf(R)[0]
    assert math.isclose(math.sqrt(w @ S @ w * 252), 0.05, rel_tol=1e-9)
    assert math.isclose(atual["caixa"], 1 - w.sum())
    movel = P.walk_forward_portfolio(R, "hrp", janela=500, peso_max=0.4)
    direto = P.pesos_portfolio(R[-500:], "hrp", peso_max=0.4)["pesos"]
    assert all(math.isclose(movel["pesos_atuais"][n], direto[n]) for n in direto)
    assert movel["pesos_atuais_ate"] == len(R) - 1


def test_faixa_marca_rebalanceamentos_nao_executados():
    rng = np.random.default_rng(14)
    R = _amostra(rng, 1200, np.full(3, 0.0003), _cov_exemplo()[:3, :3])
    res = P.walk_forward_portfolio(R, "iguais", janela=200, rebalancear_a_cada=5, faixa=0.05)
    executados = [p["executado"] for p in res["pesos"]]
    assert executados[0] and not all(executados)


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
    assert res["pesos_atuais_ate"] == idx[-1]
    msg = _levanta(P.walk_forward_portfolio, df, janela=252,
                   rf_por_periodo=pd.Series(CDI, index=idx + pd.Timedelta(days=1)))
    assert "mesmas datas" in msg
    comp = P.comparar_metodos(df, metodos=("hrp",), janela=252, rf_por_periodo=pd.Series(CDI, index=idx),
                              n_sim=100)
    hist = comp["resultados"]["hrp"]["pesos"]
    assert hist[0]["data"] == idx[252] and comp["resultados"]["hrp"]["indice"][0] == idx[252]
    texto = P.formatar_comparacao(comp)
    assert "Pesos atuais para operar" in texto and str(idx[-1].date()) in texto


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


def test_perdas_conjuntas_sem_alarme_falso():
    rng = np.random.default_rng(15)
    # estratégias que ficam paradas na maior parte dos dias, independentes entre si
    T = 2520
    esparsa = lambda: np.where(rng.random(T) < 0.08, rng.normal(0.001, 0.02, T), 0.0)  # noqa: E731
    d = P.analise_diversificacao(np.column_stack([esparsa(), esparsa(), rng.normal(0, 0.01, T)]))
    assert not d["pares_perdem_juntos"]
    assert d["dias_ruins"]["E1"] < 0.1 * T                      # dias parados não são dias ruins
    # amostras curtas de normais (sem dependência extra nas caudas): poucos alarmes
    C = np.full((5, 5), 0.5)
    np.fill_diagonal(C, 1)
    alarmes = sum(bool(P.analise_diversificacao(_amostra(rng, 156, np.zeros(5), C * 1e-4),
                                                n_sim=100)["pares_perdem_juntos"])
                  for _ in range(20))
    assert alarmes <= 3


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
