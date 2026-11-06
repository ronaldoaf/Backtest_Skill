"""Testes de regressão de skills/backtest/scripts/estatistica.py.

Rode com `python -m pytest tests` ou, sem pytest, `python tests/test_estatistica.py`.
"""
from __future__ import annotations

import math
import runpy
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "backtest" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import estatistica as E  # noqa: E402


def _levanta(funcao, *args, **kwargs) -> str:
    try:
        funcao(*args, **kwargs)
    except ValueError as erro:
        return str(erro)
    raise AssertionError(f"{funcao.__name__} deveria levantar ValueError")


def _posicoes_persistentes(rng, T, duracao=20):
    blocos = rng.choice([-1.0, 1.0], size=T // duracao + 1)
    return np.r_[0.0, 0.0, np.repeat(blocos, duracao)[:T - 2]]


# ---------------------------------------------------------------------------
# Drawdown a partir do capital inicial
# ---------------------------------------------------------------------------

def test_drawdown_conta_o_capital_inicial():
    assert math.isclose(E.max_drawdown([-0.10, -0.10, 0.05]), 0.9 * 0.9 - 1)
    assert math.isclose(E.max_drawdown([-0.5]), -0.5)
    assert E.duracao_max_drawdown([-0.10, -0.10, 0.05]) == 3


def test_drawdown_sem_perda_inicial_nao_muda():
    assert math.isclose(E.max_drawdown([0.10, -0.05, 0.02]), -0.05)
    assert E.max_drawdown([0.01, 0.02]) == 0.0
    assert E.duracao_max_drawdown([0.10, -0.05, 0.02]) == 2


# ---------------------------------------------------------------------------
# Validação de entradas
# ---------------------------------------------------------------------------

def test_nan_levanta_erro_com_dica():
    r = np.r_[np.nan, np.full(50, 0.001)]
    for funcao in (E.sharpe, E.resumo_metricas, E.bootstrap_metricas, E.max_drawdown, E.cagr):
        assert "fillna" in _levanta(funcao, r)
    assert "NaN" in _levanta(E.teste_permutacao, r, np.full(51, 0.001))
    assert "NaN" in _levanta(E.pbo_cscv, np.c_[r, r], n_blocos=4)


def test_tamanhos_diferentes_levanta_erro():
    assert "tamanhos" in _levanta(E.teste_permutacao, np.ones(10), np.ones(11))
    assert "formato" in _levanta(E.teste_permutacao_selecao, np.ones((10, 3)), np.ones(11))


# ---------------------------------------------------------------------------
# Permutação
# ---------------------------------------------------------------------------

def _permutacao_original(p, r, n_perm, seed):
    """Implementação anterior, para conferir que custo zero não muda o resultado."""
    real = E.sharpe(p * r)
    rng = np.random.default_rng(seed)
    sims = np.array([E.sharpe(np.roll(p, rng.integers(1, len(p))) * r) for _ in range(n_perm)])
    return real, float((np.sum(sims >= real) + 1) / (n_perm + 1))


def test_permutacao_sem_custo_igual_a_versao_anterior():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0003, 0.012, 600)
    p = _posicoes_persistentes(rng, 600)
    novo = E.teste_permutacao(p, r, n_perm=200, seed=7)
    real, p_valor = _permutacao_original(p, r, 200, 7)
    assert math.isclose(novo["sharpe_real"], real)
    assert novo["p_valor"] == p_valor


def test_permutacao_com_custo_usa_retorno_liquido():
    rng = np.random.default_rng(2)
    r = rng.normal(0.0003, 0.012, 600)
    p = _posicoes_persistentes(rng, 600)
    custo = 0.001
    liquido = p * r - custo * np.abs(np.diff(p, prepend=0.0))
    out = E.teste_permutacao(p, r, n_perm=50, custo_por_giro=custo)
    assert math.isclose(out["sharpe_real"], E.sharpe(liquido))


def test_permutacao_com_selecao_calibrada_sob_h0():
    """Na melhor de 30 configurações de ruído, o teste com seleção mantém ~5% de falsos positivos,
    enquanto o teste simples aplicado à melhor aprova ruído com frequência."""
    rng = np.random.default_rng(3)
    T, N, reps = 500, 30, 60
    p_sel, p_simples = [], []
    for i in range(reps):
        a = rng.normal(0.0003, 0.012, T)
        P = np.column_stack([_posicoes_persistentes(rng, T) for _ in range(N)])
        out = E.teste_permutacao_selecao(P, a, n_perm=200, seed=i)
        p_sel.append(out["p_valor"])
        p_simples.append(E.teste_permutacao(P[:, out["melhor_configuracao"]], a, n_perm=200,
                                            seed=i)["p_valor"])
    assert np.mean(np.array(p_sel) < 0.05) <= 0.15
    assert np.mean(np.array(p_simples) < 0.05) >= 0.5


def test_permutacao_com_selecao_detecta_vantagem_real():
    rng = np.random.default_rng(4)
    T, N = 1000, 20
    a = rng.normal(0.0003, 0.012, T)
    P = np.column_stack([_posicoes_persistentes(rng, T) for _ in range(N)])
    a = a + 0.002 * P[:, 5]                       # a configuração 5 prevê parte do retorno
    out = E.teste_permutacao_selecao(P, a, n_perm=300)
    assert out["melhor_configuracao"] == 5
    assert out["p_valor"] < 0.01


def test_permutacao_com_selecao_aceita_um_ativo_por_configuracao():
    rng = np.random.default_rng(5)
    T, N = 400, 4
    R = rng.normal(0, 0.01, (T, N))
    P = np.column_stack([_posicoes_persistentes(rng, T) for _ in range(N)])
    out = E.teste_permutacao_selecao(P, R, n_perm=10_000)
    assert out["n_permutacoes"] == T - 2 * (T // 20) + 1   # todos os deslocamentos: teste exato
    assert 0 < out["p_valor"] <= 1


# ---------------------------------------------------------------------------
# Convenção de timing documentada em references/dados-e-lookahead.md
# ---------------------------------------------------------------------------

def test_formula_de_execucao_na_abertura_bate_com_simulacao_evento_a_evento():
    try:
        import pandas as pd
    except ImportError:
        return  # pandas é opcional para o script; sem ele, o teste não se aplica
    rng = np.random.default_rng(6)
    n = 300
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, n)))
    open_ = close.shift(1) * (1 + rng.normal(0, 0.005, n))
    open_.iloc[0] = 100.0
    sinal = pd.Series(np.sign(rng.normal(size=n)))

    # Fórmula da documentação
    p_nova = sinal.shift(1).fillna(0)
    p_antiga = sinal.shift(2).fillna(0)
    gap = (open_ / close.shift(1) - 1).fillna(0)
    intradia = close / open_ - 1
    retorno = (1 + p_antiga * gap) * (1 + p_nova * intradia) - 1

    # Simulação evento a evento: o sinal de t-1 executa na abertura de t
    capital, posicao, curva = 1.0, 0.0, []
    for t in range(n):
        if t > 0:
            capital *= 1 + posicao * (open_[t] / close[t - 1] - 1)
            posicao = sinal[t - 1]
        capital *= 1 + posicao * (close[t] / open_[t] - 1)
        curva.append(capital)
    assert np.allclose((1 + retorno).cumprod().to_numpy(), curva, rtol=0, atol=1e-12)


def test_resumo_e_bootstrap_descontam_o_rf():
    rng = np.random.default_rng(21)
    r = rng.normal(0.0006, 0.01, 1000)
    rf_t = np.full(1000, 0.0004)
    assert math.isclose(E.resumo_metricas(r, rf_por_periodo=rf_t)["sharpe"], E.sharpe(r - rf_t))
    assert math.isclose(E.resumo_metricas(r, rf_por_periodo=0.0004)["sharpe"], E.sharpe(r - rf_t))
    com = E.bootstrap_metricas(r, n_sim=200, rf_por_periodo=rf_t)
    sem = E.bootstrap_metricas(r, n_sim=200)
    assert com["sharpe_ic"][1] < sem["sharpe_ic"][1] and com["cagr_ic"] == sem["cagr_ic"]


def test_demonstracao_roda():
    runpy.run_path(str(SCRIPTS / "estatistica.py"), run_name="__main__")


if __name__ == "__main__":
    testes = [(nome, f) for nome, f in sorted(globals().items()) if nome.startswith("test_")]
    for nome, f in testes:
        f()
        print("ok ", nome)
    print(f"{len(testes)} testes passaram")
