"""Funções estatísticas para validar backtests.

Dependência: apenas numpy (e a biblioteca padrão).

Convenções:
- `retornos`: array 1D de retornos simples por período (ex.: diários), líquidos de custos.
- `periodos_por_ano`: 252 para pregões diários, 365 para cripto, 52 semanal, 12 mensal.

Uso rápido:
    import numpy as np
    from estatistica import resumo_metricas, bootstrap_metricas, deflated_sharpe_ratio

    r = np.asarray(serie_de_retornos)
    print(resumo_metricas(r))
    print(bootstrap_metricas(r))

Execute `python estatistica.py` para uma demonstração com dados sintéticos.
"""
from __future__ import annotations

import itertools
import math
from statistics import NormalDist

import numpy as np

_N = NormalDist()
EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# Métricas básicas
# ---------------------------------------------------------------------------

def sharpe(retornos, periodos_por_ano: int = 252, rf_por_periodo: float = 0.0) -> float:
    r = np.asarray(retornos, dtype=float) - rf_por_periodo
    sd = r.std(ddof=1)
    return float(r.mean() / sd * math.sqrt(periodos_por_ano)) if sd > 0 else float("nan")


def sortino(retornos, periodos_por_ano: int = 252, alvo: float = 0.0) -> float:
    r = np.asarray(retornos, dtype=float) - alvo
    downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    return float(r.mean() / downside * math.sqrt(periodos_por_ano)) if downside > 0 else float("nan")


def curva_capital(retornos) -> np.ndarray:
    return np.cumprod(1.0 + np.asarray(retornos, dtype=float))


def drawdowns(retornos) -> np.ndarray:
    eq = curva_capital(retornos)
    pico = np.maximum.accumulate(eq)
    return eq / pico - 1.0


def max_drawdown(retornos) -> float:
    return float(drawdowns(retornos).min())


def duracao_max_drawdown(retornos) -> int:
    """Maior número de períodos consecutivos abaixo do pico anterior."""
    dd = drawdowns(retornos)
    maior = atual = 0
    for v in dd:
        atual = atual + 1 if v < 0 else 0
        maior = max(maior, atual)
    return maior


def cagr(retornos, periodos_por_ano: int = 252) -> float:
    r = np.asarray(retornos, dtype=float)
    anos = len(r) / periodos_por_ano
    final = curva_capital(r)[-1]
    return float(final ** (1.0 / anos) - 1.0) if anos > 0 and final > 0 else float("nan")


def maior_sequencia_perdas(resultados) -> int:
    maior = atual = 0
    for v in np.asarray(resultados, dtype=float):
        atual = atual + 1 if v < 0 else 0
        maior = max(maior, atual)
    return maior


def resumo_metricas(retornos, periodos_por_ano: int = 252) -> dict:
    r = np.asarray(retornos, dtype=float)
    mdd = max_drawdown(r)
    c = cagr(r, periodos_por_ano)
    return {
        "n_periodos": int(len(r)),
        "cagr": c,
        "volatilidade": float(r.std(ddof=1) * math.sqrt(periodos_por_ano)),
        "sharpe": sharpe(r, periodos_por_ano),
        "sortino": sortino(r, periodos_por_ano),
        "max_drawdown": mdd,
        "calmar": float(c / abs(mdd)) if mdd < 0 else float("nan"),
        "duracao_max_dd_periodos": duracao_max_drawdown(r),
        "assimetria": _assimetria(r),
        "curtose": _curtose(r),
    }


def metricas_trades(resultados_trades) -> dict:
    """Métricas a partir do resultado (P&L ou retorno) de cada trade."""
    t = np.asarray(resultados_trades, dtype=float)
    ganhos, perdas = t[t > 0], t[t < 0]
    acerto = len(ganhos) / len(t) if len(t) else float("nan")
    gm = ganhos.mean() if len(ganhos) else 0.0
    pm = -perdas.mean() if len(perdas) else 0.0
    return {
        "n_trades": int(len(t)),
        "taxa_acerto": acerto,
        "ganho_medio": float(gm),
        "perda_media": float(pm),
        "payoff": float(gm / pm) if pm > 0 else float("nan"),
        "expectativa": float(t.mean()) if len(t) else float("nan"),
        "profit_factor": float(ganhos.sum() / -perdas.sum()) if len(perdas) else float("nan"),
        "maior_sequencia_perdas": maior_sequencia_perdas(t),
    }


def _assimetria(r: np.ndarray) -> float:
    d = r - r.mean()
    s = d.std()
    return float(np.mean(d ** 3) / s ** 3) if s > 0 else float("nan")


def _curtose(r: np.ndarray) -> float:
    """Curtose não-excessiva (normal = 3)."""
    d = r - r.mean()
    s = d.std()
    return float(np.mean(d ** 4) / s ** 4) if s > 0 else float("nan")


# ---------------------------------------------------------------------------
# Bootstrap / Monte Carlo
# ---------------------------------------------------------------------------

def _indices_bloco(n: int, tamanho_bloco: int, rng: np.random.Generator) -> np.ndarray:
    """Índices para bootstrap em blocos circulares."""
    n_blocos = math.ceil(n / tamanho_bloco)
    inicios = rng.integers(0, n, size=n_blocos)
    idx = (inicios[:, None] + np.arange(tamanho_bloco)[None, :]) % n
    return idx.ravel()[:n]


def bootstrap_metricas(retornos, periodos_por_ano: int = 252, n_sim: int = 5000,
                       tamanho_bloco: int = 10, nivel: float = 0.95, seed: int = 42) -> dict:
    """IC do Sharpe, CAGR e drawdown máximo via bootstrap em blocos."""
    r = np.asarray(retornos, dtype=float)
    rng = np.random.default_rng(seed)
    sh, cg, dd = np.empty(n_sim), np.empty(n_sim), np.empty(n_sim)
    for i in range(n_sim):
        amostra = r[_indices_bloco(len(r), tamanho_bloco, rng)]
        sh[i] = sharpe(amostra, periodos_por_ano)
        cg[i] = cagr(amostra, periodos_por_ano)
        dd[i] = max_drawdown(amostra)
    a = (1 - nivel) / 2
    q = lambda x: (float(np.nanquantile(x, a)), float(np.nanquantile(x, 1 - a)))  # noqa: E731
    return {
        "sharpe_ic": q(sh),
        "cagr_ic": q(cg),
        "max_drawdown_ic": q(dd),
        "prob_sharpe_positivo": float(np.mean(sh > 0)),
    }


def monte_carlo_drawdown(resultados_trades, n_sim: int = 10000, seed: int = 42,
                         percentis=(50, 90, 95, 99)) -> dict:
    """Embaralha a ordem dos trades (retornos por trade) e mede drawdown e sequência de perdas.

    Use os percentis altos como limites de desligamento no monitoramento (passo 12).
    """
    t = np.asarray(resultados_trades, dtype=float)
    rng = np.random.default_rng(seed)
    dds, seqs = np.empty(n_sim), np.empty(n_sim)
    for i in range(n_sim):
        amostra = rng.choice(t, size=len(t), replace=True)
        dds[i] = max_drawdown(amostra)
        seqs[i] = maior_sequencia_perdas(amostra)
    return {
        "max_drawdown": {p: float(np.percentile(dds, 100 - p)) for p in percentis},
        "maior_sequencia_perdas": {p: float(np.percentile(seqs, p)) for p in percentis},
    }


# ---------------------------------------------------------------------------
# Permutação e estratégias aleatórias
# ---------------------------------------------------------------------------

def teste_permutacao(posicoes, retornos_ativo, periodos_por_ano: int = 252, n_perm: int = 5000,
                     circular: bool = True, seed: int = 42) -> dict:
    """p-valor do Sharpe da estratégia contra sinais embaralhados.

    `posicoes[t]` deve ser a posição mantida durante o retorno `retornos_ativo[t]`
    (ou seja, já defasada: posicao = sinal.shift(1)).
    `circular=True` desloca a série de posições (preserva a estrutura dos sinais);
    `False` embaralha cada período independentemente.
    """
    p = np.asarray(posicoes, dtype=float)
    r = np.asarray(retornos_ativo, dtype=float)
    real = sharpe(p * r, periodos_por_ano)
    rng = np.random.default_rng(seed)
    n = len(p)
    sims = np.empty(n_perm)
    for i in range(n_perm):
        pp = np.roll(p, rng.integers(1, n)) if circular else rng.permutation(p)
        sims[i] = sharpe(pp * r, periodos_por_ano)
    return {
        "sharpe_real": real,
        "p_valor": float((np.sum(sims >= real) + 1) / (n_perm + 1)),
        "percentil": float(np.mean(sims < real) * 100),
    }


# ---------------------------------------------------------------------------
# Correção para múltiplos testes
# ---------------------------------------------------------------------------

def sharpe_maximo_esperado(n_tentativas: int, var_sharpes: float) -> float:
    """Sharpe máximo esperado por sorte entre N tentativas (mesma frequência dos dados)."""
    if n_tentativas < 2:
        return 0.0
    z1 = _N.inv_cdf(1 - 1.0 / n_tentativas)
    z2 = _N.inv_cdf(1 - 1.0 / (n_tentativas * math.e))
    return math.sqrt(var_sharpes) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def probabilistic_sharpe_ratio(retornos, sr_referencia: float = 0.0) -> float:
    """PSR: P(Sharpe verdadeiro > referência). Sharpe NÃO anualizado (por período)."""
    r = np.asarray(retornos, dtype=float)
    sr = r.mean() / r.std(ddof=1)
    g3, g4 = _assimetria(r), _curtose(r)
    den = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return float(_N.cdf((sr - sr_referencia) * math.sqrt(len(r) - 1) / den))


def deflated_sharpe_ratio(retornos, n_tentativas: int, sharpes_tentativas=None,
                          var_sharpes: float | None = None) -> dict:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    retornos: retornos por período da estratégia escolhida.
    n_tentativas: número de configurações (efetivamente independentes) testadas.
    sharpes_tentativas: Sharpes NÃO anualizados de todas as tentativas (para estimar a variância),
                        ou informe var_sharpes diretamente.
    """
    if var_sharpes is None:
        if sharpes_tentativas is None or len(sharpes_tentativas) < 2:
            raise ValueError("Informe sharpes_tentativas (>= 2 valores) ou var_sharpes.")
        var_sharpes = float(np.var(np.asarray(sharpes_tentativas, dtype=float), ddof=1))
    sr0 = sharpe_maximo_esperado(n_tentativas, var_sharpes)
    r = np.asarray(retornos, dtype=float)
    return {
        "sharpe_por_periodo": float(r.mean() / r.std(ddof=1)),
        "sharpe_max_esperado_sorte": sr0,
        "dsr": probabilistic_sharpe_ratio(r, sr0),
    }


def pbo_cscv(matriz_retornos, n_blocos: int = 16, metrica=None) -> dict:
    """Probabilidade de Overfitting do Backtest via CSCV.

    matriz_retornos: array T x N (T períodos, N configurações candidatas).
    n_blocos: número par de blocos S. Combinações = C(S, S/2) (16 -> 12870).
    metrica: função(retornos_1d) -> float; padrão = média/desvio (Sharpe por período).
    """
    M = np.asarray(matriz_retornos, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2:
        raise ValueError("matriz_retornos deve ser T x N com N >= 2.")
    if n_blocos % 2:
        raise ValueError("n_blocos deve ser par.")
    if metrica is None:
        def metrica(x):
            sd = x.std(ddof=1)
            return x.mean() / sd if sd > 0 else 0.0
    blocos = np.array_split(np.arange(M.shape[0]), n_blocos)
    n = M.shape[1]
    logits = []
    for comb in itertools.combinations(range(n_blocos), n_blocos // 2):
        treino = np.concatenate([blocos[i] for i in comb])
        teste = np.concatenate([blocos[i] for i in range(n_blocos) if i not in comb])
        perf_tr = np.array([metrica(M[treino, j]) for j in range(n)])
        perf_te = np.array([metrica(M[teste, j]) for j in range(n)])
        melhor = int(np.argmax(perf_tr))
        rank = (np.sum(perf_te < perf_te[melhor]) + 0.5 * (np.sum(perf_te == perf_te[melhor]) - 1) + 1) / (n + 1)
        logits.append(math.log(rank / (1 - rank)))
    logits = np.asarray(logits)
    return {"pbo": float(np.mean(logits <= 0)), "n_combinacoes": int(len(logits)),
            "logit_mediano": float(np.median(logits))}


# ---------------------------------------------------------------------------
# Tamanho de amostra
# ---------------------------------------------------------------------------

def erro_padrao_sharpe_anual(sharpe_anual: float, anos: float) -> float:
    return math.sqrt((1 + sharpe_anual ** 2 / 2) / anos)


def anos_necessarios(sharpe_anual: float, t_stat: float = 2.0) -> float:
    return (t_stat / sharpe_anual) ** 2 if sharpe_anual > 0 else float("inf")


# ---------------------------------------------------------------------------
# Demonstração
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T = 252 * 8
    ativo = rng.normal(0.0003, 0.012, T)
    sinal = np.sign(rng.normal(size=T))
    pos = np.roll(sinal, 1)
    pos[0] = 0
    estrat = pos * ativo - 0.0002 * (np.abs(np.diff(pos, prepend=0)) > 0)

    print("Métricas:", {k: round(v, 4) if isinstance(v, float) else v
                        for k, v in resumo_metricas(estrat).items()})
    print("Bootstrap:", bootstrap_metricas(estrat, n_sim=500))
    print("Permutação:", teste_permutacao(pos, ativo, n_perm=500))

    # 50 configurações aleatórias: o melhor Sharpe parece bom, mas é sorte.
    cfgs = np.column_stack([np.roll(np.sign(rng.normal(size=T)), 1) * ativo for _ in range(50)])
    srs = cfgs.mean(0) / cfgs.std(0, ddof=1)
    melhor = cfgs[:, np.argmax(srs)]
    print("Melhor Sharpe anual entre 50 aleatórias:", round(sharpe(melhor), 2))
    print("DSR:", deflated_sharpe_ratio(melhor, n_tentativas=50, sharpes_tentativas=srs))
    print("PBO:", pbo_cscv(cfgs, n_blocos=10))
    print("Monte Carlo (trades sintéticos):",
          monte_carlo_drawdown(rng.normal(0.002, 0.02, 200), n_sim=1000))
