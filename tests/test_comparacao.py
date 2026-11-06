"""Testes de skills/backtest/scripts/comparacao.py.

Rode com `python -m pytest tests` ou, sem pytest, `python tests/test_comparacao.py`.
"""
from __future__ import annotations

import math
import runpy
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "backtest" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import comparacao as C  # noqa: E402


def _levanta(funcao, *args, **kwargs) -> str:
    try:
        funcao(*args, **kwargs)
    except ValueError as erro:
        return str(erro)
    raise AssertionError(f"{funcao.__name__} deveria levantar ValueError")


def _par_correlacionado(rng, T, mu_a, mu_b, rho=0.6, sd=0.01, gl=None):
    """Duas séries com correlação rho; caudas gordas (t de Student) se gl for dado."""
    gerar = (lambda n: rng.standard_t(gl, n) / math.sqrt(gl / (gl - 2))) if gl else rng.standard_normal
    z0, z1, z2 = gerar(T), gerar(T), gerar(T)
    a = mu_a + sd * (math.sqrt(rho) * z0 + math.sqrt(1 - rho) * z1)
    b = mu_b + sd * (math.sqrt(rho) * z0 + math.sqrt(1 - rho) * z2)
    return a, b


# ---------------------------------------------------------------------------
# Entradas
# ---------------------------------------------------------------------------

def test_series_desalinhadas_levantam_erro():
    msg = _levanta(C.tabela_comparativa, {"a": np.zeros(10), "b": np.zeros(9)})
    assert "Alinhe" in msg
    assert "tamanhos diferentes" in _levanta(C.teste_diferenca_sharpe, np.ones(10), np.ones(9))


def test_nan_levanta_erro():
    a = np.r_[np.nan, np.random.default_rng(0).normal(size=50)]
    assert "NaN" in _levanta(C.teste_diferenca_sharpe, a, np.ones(51))
    assert "NaN" in _levanta(C.model_confidence_set, {"a": a, "b": np.ones(51)})


def test_matriz_com_nomes_e_dict_dao_o_mesmo_resultado():
    rng = np.random.default_rng(1)
    R = rng.normal(0.0005, 0.01, (300, 3))
    t1 = C.tabela_comparativa(R, nomes=["x", "y", "z"])
    t2 = C.tabela_comparativa({"x": R[:, 0], "y": R[:, 1], "z": R[:, 2]})
    assert t1 == t2
    assert math.isclose(t1["correlacao"]["x"]["x"], 1.0)
    texto = C.formatar_tabela(t1)
    assert "| sharpe |" in texto and "Correlação" in texto


# ---------------------------------------------------------------------------
# Diferença de Sharpe
# ---------------------------------------------------------------------------

def test_series_identicas_nao_tem_diferenca():
    r = np.random.default_rng(2).normal(0.0005, 0.01, 500)
    assert C.teste_diferenca_sharpe(r, r)["p_valor"] == 1.0
    assert C.bootstrap_diferenca_sharpe(r, r, n_sim=200)["p_valor"] == 1.0


def test_diferenca_sharpe_e_antissimetrica_e_anualizada():
    rng = np.random.default_rng(3)
    a, b = _par_correlacionado(rng, 1000, 0.0008, 0.0002)
    ab, ba = C.teste_diferenca_sharpe(a, b), C.teste_diferenca_sharpe(b, a)
    assert math.isclose(ab["diferenca"], -ba["diferenca"])
    assert math.isclose(ab["p_valor"], ba["p_valor"])
    sr = lambda x: x.mean() / x.std() * math.sqrt(252)  # noqa: E731 (ddof=0, como no teste)
    assert math.isclose(ab["diferenca"], sr(a) - sr(b), rel_tol=1e-9)


def test_teste_hac_calibrado_sob_h0_com_caudas_gordas():
    """Mesmo Sharpe verdadeiro, retornos t(4) correlacionados: ~5% de rejeições a 5%."""
    rng = np.random.default_rng(4)
    rejeicoes = sum(
        C.teste_diferenca_sharpe(*_par_correlacionado(rng, 750, 0.0004, 0.0004, gl=4))["p_valor"] < 0.05
        for _ in range(400))
    assert 0.02 <= rejeicoes / 400 <= 0.09, rejeicoes


def test_teste_hac_detecta_diferenca_real():
    rng = np.random.default_rng(5)
    a, b = _par_correlacionado(rng, 2520, 0.0010, 0.0, rho=0.8)
    res = C.teste_diferenca_sharpe(a, b)
    assert res["p_valor"] < 0.01 and res["ic95"][0] > 0


def test_correlacao_alta_aumenta_o_poder():
    """A mesma diferença de médias é mais fácil de detectar entre séries muito correlacionadas."""
    rng = np.random.default_rng(6)
    ep = {rho: np.mean([C.teste_diferenca_sharpe(*_par_correlacionado(rng, 1000, 0.0005, 0.0, rho=rho))
                        ["erro_padrao"] for _ in range(20)]) for rho in (0.0, 0.9)}
    assert ep[0.9] < 0.5 * ep[0.0]


def test_bootstrap_concorda_com_hac():
    rng = np.random.default_rng(7)
    a, b = _par_correlacionado(rng, 1500, 0.0007, 0.0001)
    hac = C.teste_diferenca_sharpe(a, b)
    boot = C.bootstrap_diferenca_sharpe(a, b, n_sim=2000)
    assert math.isclose(hac["diferenca"], boot["diferenca"], rel_tol=1e-3)  # ddof 0 x 1
    assert abs(hac["p_valor"] - boot["p_valor"]) < 0.05
    assert boot["ic"][0] < boot["diferenca"] < boot["ic"][1]


# ---------------------------------------------------------------------------
# Holm e pares
# ---------------------------------------------------------------------------

def test_holm_valores_conhecidos():
    assert np.allclose(C.holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
    assert np.allclose(C.holm([0.5, 0.9]), [1.0, 1.0])


def test_comparar_pares_cobre_todos_os_pares():
    R = np.random.default_rng(8).normal(0.0003, 0.01, (400, 4))
    linhas = C.comparar_pares(R)
    assert len(linhas) == 6
    assert all(l["p_holm"] >= l["p_valor"] for l in linhas)
    assert "metodo" in _levanta(C.comparar_pares, R, metodo="xyz")


# ---------------------------------------------------------------------------
# SPA / Reality Check
# ---------------------------------------------------------------------------

def test_spa_calibrado_sob_h0():
    """10 candidatas sem vantagem sobre o benchmark: a melhor não deve passar mais de ~5%."""
    rng = np.random.default_rng(9)
    rejeicoes = 0
    for _ in range(150):
        bench = rng.normal(0.0003, 0.01, 500)
        cand = bench[:, None] + rng.normal(0, 0.005, (500, 10))
        rejeicoes += C.teste_spa(cand, bench, n_sim=300, seed=int(rng.integers(1e9)))["p_spa"] < 0.05
    assert rejeicoes / 150 <= 0.10, rejeicoes


def test_spa_detecta_candidata_boa_e_e_menos_conservador_que_rc():
    rng = np.random.default_rng(10)
    T = 1000
    bench = rng.normal(0.0003, 0.01, T)
    cand = bench[:, None] + rng.normal(0, 0.005, (T, 30))
    cand[:, :25] -= 0.002            # 25 candidatas muito ruins
    cand[:, 29] += 0.0006            # uma com vantagem real
    res = C.teste_spa(cand, bench, n_sim=1000)
    assert res["melhor"] == "E30"
    assert res["p_spa"] < 0.05
    assert res["p_spa_inferior"] <= res["p_spa"] <= res["p_reality_check"] + 1e-12


# ---------------------------------------------------------------------------
# Model Confidence Set
# ---------------------------------------------------------------------------

def test_mcs_exclui_estrategias_claramente_piores():
    rng = np.random.default_rng(11)
    T = 1500
    base = rng.normal(0, 0.01, T)
    r = {"melhor": 0.0010 + base + rng.normal(0, 0.004, T),
         "media": 0.0000 + base + rng.normal(0, 0.004, T),
         "pior": -0.0010 + base + rng.normal(0, 0.004, T)}
    res = C.model_confidence_set(r, n_sim=1000)
    assert res["mcs"] == ["melhor"]
    assert res["ordem_eliminacao"][0] == "pior"
    assert res["p_valores"]["melhor"] == 1.0


def test_mcs_mantem_estrategias_equivalentes():
    rng = np.random.default_rng(12)
    R = rng.normal(0.0004, 0.01, (750, 5))
    res = C.model_confidence_set(R, n_sim=1000)
    assert len(res["mcs"]) >= 4
    p = [res["p_valores"][n] for n in res["ordem_eliminacao"]]
    assert p == sorted(p)        # p-valores MCS não decrescem ao longo das eliminações


def test_mcs_normalizar_vol_compara_risco_ajustado():
    """B rende mais que A só por ter o dobro de alavancagem: com normalizar_vol, empatam."""
    rng = np.random.default_rng(13)
    a = rng.normal(0.0004, 0.01, 2000)
    r = {"A": a, "B_2x": 2 * a}
    assert C.model_confidence_set(r, n_sim=500)["mcs"] == ["B_2x"]
    assert sorted(C.model_confidence_set(r, n_sim=500, normalizar_vol=True)["mcs"]) == ["A", "B_2x"]


def test_demonstracao_roda():
    runpy.run_path(str(SCRIPTS / "comparacao.py"), run_name="__main__")


if __name__ == "__main__":
    testes = [(nome, f) for nome, f in sorted(globals().items()) if nome.startswith("test_")]
    for nome, f in testes:
        f()
        print("ok ", nome)
    print(f"{len(testes)} testes passaram")
