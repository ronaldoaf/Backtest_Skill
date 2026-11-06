"""Funções estatísticas para validar backtests.

Dependência: apenas numpy (e a biblioteca padrão).

Convenções:
- `retornos`: array 1D de retornos simples por período (ex.: diários), líquidos de custos.
- `posicoes[t]`: posição em vigor durante `retornos_ativo[t]`, montada pela convenção de timing
  de `references/dados-e-lookahead.md`. Com sinal no fechamento de t e execução no fechamento
  de t+1: `posicoes = sinal.shift(2).fillna(0)` e `retornos_ativo = close.pct_change().fillna(0)`.
- `periodos_por_ano`: 252 para pregões diários, 365 para cripto, 52 semanal, 12 mensal.
- Entradas com NaN ou infinito levantam ValueError. Remova as linhas iniciais criadas por
  `shift`/`pct_change` (ou preencha posições com 0) antes de chamar.

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
# Validação de entradas
# ---------------------------------------------------------------------------

def _vetor(x, nome: str = "retornos", minimo: int = 2) -> np.ndarray:
    """Converte para array 1D de float e falha alto com NaN, infinito ou série curta."""
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{nome} deve ser 1D; recebeu formato {a.shape}.")
    _checar_finitos(a, nome)
    if len(a) < minimo:
        raise ValueError(f"{nome} precisa de pelo menos {minimo} valores; recebeu {len(a)}.")
    return a


def _matriz(x, nome: str, minimo_linhas: int = 2) -> np.ndarray:
    """Converte para array 2D (T x N) de float e falha alto com NaN ou infinito."""
    a = np.asarray(x, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"{nome} deve ser 2D (T x N); recebeu formato {a.shape}.")
    _checar_finitos(a, nome)
    if a.shape[0] < minimo_linhas:
        raise ValueError(f"{nome} precisa de pelo menos {minimo_linhas} linhas; recebeu {a.shape[0]}.")
    return a


def _checar_finitos(a: np.ndarray, nome: str) -> None:
    ruins = ~np.isfinite(a)
    if ruins.any():
        primeiro = int(np.argmax(ruins.ravel()))
        raise ValueError(
            f"{nome} tem {int(ruins.sum())} valor(es) NaN ou infinito (primeiro na posição "
            f"{primeiro}). Linhas iniciais de shift/pct_change: use .dropna() nos retornos ou "
            f".fillna(0) nas posições antes de chamar.")


# ---------------------------------------------------------------------------
# Métricas básicas
# ---------------------------------------------------------------------------

def sharpe(retornos, periodos_por_ano: int = 252, rf_por_periodo: float = 0.0) -> float:
    r = _vetor(retornos) - rf_por_periodo
    sd = r.std(ddof=1)
    return float(r.mean() / sd * math.sqrt(periodos_por_ano)) if sd > 0 else float("nan")


def sortino(retornos, periodos_por_ano: int = 252, alvo: float = 0.0) -> float:
    r = _vetor(retornos) - alvo
    downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    return float(r.mean() / downside * math.sqrt(periodos_por_ano)) if downside > 0 else float("nan")


def curva_capital(retornos) -> np.ndarray:
    """Capital ao fim de cada período, partindo de 1,0 (o capital inicial não entra no array)."""
    return np.cumprod(1.0 + _vetor(retornos, minimo=1))


def drawdowns(retornos) -> np.ndarray:
    """Queda em relação ao pico anterior, contando o capital inicial (1,0) como primeiro pico."""
    eq = curva_capital(retornos)
    pico = np.maximum.accumulate(np.maximum(eq, 1.0))
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
    r = _vetor(retornos, minimo=1)
    anos = len(r) / periodos_por_ano
    final = curva_capital(r)[-1]
    return float(final ** (1.0 / anos) - 1.0) if anos > 0 and final > 0 else float("nan")


def maior_sequencia_perdas(resultados) -> int:
    maior = atual = 0
    for v in _vetor(resultados, "resultados", minimo=1):
        atual = atual + 1 if v < 0 else 0
        maior = max(maior, atual)
    return maior


def resumo_metricas(retornos, periodos_por_ano: int = 252) -> dict:
    r = _vetor(retornos)
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
    t = _vetor(resultados_trades, "resultados_trades", minimo=1)
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
    """IC do Sharpe, CAGR e drawdown máximo via bootstrap em blocos.

    Vale para a série fora da amostra ou para uma estratégia definida antes de ver os dados.
    Na melhor de N configurações, o IC fica otimista: use DSR, PBO ou teste_permutacao_selecao.
    """
    r = _vetor(retornos)
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
    """Reamostra os trades com reposição e mede drawdown máximo e maior sequência de perdas.

    `resultados_trades` é o retorno de cada trade sobre o capital total (fração, não P&L em R$),
    com trades sem sobreposição no tempo. Com trades simultâneos, use bootstrap em blocos dos
    retornos diários. Use os percentis altos como limites de desligamento (passo 12).
    """
    t = _vetor(resultados_trades, "resultados_trades")
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

def _giro(posicoes: np.ndarray) -> np.ndarray:
    """|variação da posição| em cada período, contando a entrada inicial a partir de zero."""
    return np.abs(np.diff(posicoes, axis=0, prepend=np.zeros_like(posicoes[:1])))


def _sharpe_colunas(R: np.ndarray) -> np.ndarray:
    """Sharpe por período de cada coluna de uma matriz T x N (0 quando o desvio é zero)."""
    sd = R.std(axis=0, ddof=1)
    return np.divide(R.mean(axis=0), sd, out=np.zeros(R.shape[1]), where=sd > 0)


def teste_permutacao(posicoes, retornos_ativo, periodos_por_ano: int = 252, n_perm: int = 5000,
                     circular: bool = True, seed: int = 42, custo_por_giro: float = 0.0) -> dict:
    """p-valor do Sharpe de UMA estratégia contra posições deslocadas ou embaralhadas.

    Vale para a série fora da amostra ou para uma estratégia definida antes de ver os dados.
    Para a melhor de N configurações, use teste_permutacao_selecao.

    `posicoes[t]` é a posição em vigor durante `retornos_ativo[t]`, montada pela convenção de
    timing (ex.: execução no fechamento de t+1 -> `sinal.shift(2).fillna(0)`). Não use
    `sinal.shift(1)` com retornos de fechamento a fechamento: isso executa ao preço do sinal.
    `circular=True` desloca a série de posições (preserva exposição, número e duração dos trades;
    equivale a comparar com estratégias aleatórias de mesma estrutura); `False` embaralha cada
    período independentemente. `custo_por_giro` (fração por unidade de giro) acompanha as posições,
    de modo que o Sharpe real e o simulado são líquidos de custos.
    """
    p = _vetor(posicoes, "posicoes")
    r = _vetor(retornos_ativo, "retornos_ativo")
    if len(p) != len(r):
        raise ValueError(f"posicoes ({len(p)}) e retornos_ativo ({len(r)}) têm tamanhos diferentes.")
    c = custo_por_giro * _giro(p)
    real = sharpe(p * r - c, periodos_por_ano)
    rng = np.random.default_rng(seed)
    n = len(p)
    sims = np.empty(n_perm)
    for i in range(n_perm):
        if circular:
            k = rng.integers(1, n)
            pp, cc = np.roll(p, k), np.roll(c, k)
        else:
            idx = rng.permutation(n)
            pp, cc = p[idx], c[idx]
        sims[i] = sharpe(pp * r - cc, periodos_por_ano)
    return {
        "sharpe_real": real,
        "p_valor": float((np.sum(sims >= real) + 1) / (n_perm + 1)),
        "percentil": float(np.mean(sims < real) * 100),
    }


def teste_permutacao_selecao(matriz_posicoes, retornos_ativo, periodos_por_ano: int = 252,
                             n_perm: int = 2000, deslocamento_minimo: int | None = None,
                             seed: int = 42, custo_por_giro: float = 0.0) -> dict:
    """p-valor da MELHOR de N configurações, corrigido pela seleção (permutação do máximo).

    Use quando a estratégia foi escolhida entre várias configurações no mesmo período.
    O mesmo deslocamento circular é aplicado aos retornos de todas as configurações, e o melhor
    Sharpe real é comparado com o melhor Sharpe de cada deslocamento (ideia do White's Reality
    Check, com permutação no lugar do bootstrap). Passe TODAS as configurações testadas,
    inclusive as descartadas: omiti-las deixa o teste otimista.

    matriz_posicoes: T x N, coluna j = posições da configuração j, pela convenção de timing.
    retornos_ativo: T (mesmo ativo para todas) ou T x N (um ativo por configuração).
    deslocamento_minimo: deslocamentos menores geram quase cópias da estratégia e tiram poder;
                         padrão max(1, T // 20). Use pelo menos a duração média dos trades.
    Quando n_perm >= número de deslocamentos possíveis, todos são usados (teste exato).
    """
    P = _matriz(matriz_posicoes, "matriz_posicoes", minimo_linhas=3)
    T, n_cfg = P.shape
    R = np.asarray(retornos_ativo, dtype=float)
    R = _vetor(R, "retornos_ativo")[:, None] if R.ndim == 1 else _matriz(R, "retornos_ativo")
    if R.shape[0] != T or R.shape[1] not in (1, n_cfg):
        raise ValueError(f"retornos_ativo deve ter formato ({T},) ou ({T}, {n_cfg}); recebeu {R.shape}.")
    d_min = max(1, T // 20) if deslocamento_minimo is None else int(deslocamento_minimo)
    if not 1 <= d_min <= T // 2:
        raise ValueError(f"deslocamento_minimo deve estar entre 1 e {T // 2}.")
    C = custo_por_giro * _giro(P)
    sr_real = _sharpe_colunas(P * R - C)
    melhor = int(np.argmax(sr_real))
    candidatos = np.arange(d_min, T - d_min + 1)
    rng = np.random.default_rng(seed)
    ks = candidatos if n_perm >= len(candidatos) else rng.choice(candidatos, n_perm, replace=False)
    sims = np.array([_sharpe_colunas(P * np.roll(R, k, axis=0) - C).max() for k in ks])
    raiz = math.sqrt(periodos_por_ano)
    return {
        "melhor_configuracao": melhor,
        "sharpe_melhor": float(sr_real[melhor] * raiz),
        "p_valor": float((np.sum(sims >= sr_real[melhor]) + 1) / (len(ks) + 1)),
        "n_configuracoes": int(n_cfg),
        "n_permutacoes": int(len(ks)),
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
    r = _vetor(retornos, minimo=3)
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
    r = _vetor(retornos, minimo=3)
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
    M = _matriz(matriz_retornos, "matriz_retornos")
    if M.shape[1] < 2:
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
    fmt = lambda d: {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()}  # noqa: E731

    # 1) Estratégia definida antes de ver os dados, com vantagem real pequena.
    #    sinal[t] usa dados até o fechamento de t, a ordem executa no fechamento de t+1 e a posição
    #    vale durante o retorno de t+2: posicoes = sinal.shift(2).fillna(0).
    sinal = np.sign(rng.normal(size=T))
    pos = np.r_[0.0, 0.0, sinal[:-2]]
    ativo = rng.normal(0.0003, 0.012, T) + 0.0012 * pos   # o sinal prevê parte do retorno de t+2
    custo = 0.0001                                        # 1 bp por unidade de giro
    estrat = pos * ativo - custo * np.abs(np.diff(pos, prepend=0.0))
    print("1) Estratégia definida antes dos dados (vantagem real)")
    print("   Métricas:", fmt(resumo_metricas(estrat)))
    print("   Bootstrap:", fmt(bootstrap_metricas(estrat, n_sim=500)))
    print("   Permutação:", fmt(teste_permutacao(pos, ativo, n_perm=500, custo_por_giro=custo)))

    # 2) Seleção: a melhor de 50 configurações aleatórias num ativo sem vantagem nenhuma.
    ruido = rng.normal(0.0003, 0.012, T)
    P = np.column_stack([np.r_[0.0, 0.0, np.sign(rng.normal(size=T - 2))] for _ in range(50)])
    cfgs = P * ruido[:, None]
    srs = cfgs.mean(0) / cfgs.std(0, ddof=1)
    j = int(np.argmax(srs))
    print("2) Melhor de 50 configurações aleatórias (pura sorte)")
    print("   Sharpe anual da melhor:", round(sharpe(cfgs[:, j]), 2))
    print("   Permutação simples (inválida após seleção):",
          fmt(teste_permutacao(P[:, j], ruido, n_perm=500)))
    print("   Permutação com seleção:", fmt(teste_permutacao_selecao(P, ruido, n_perm=500)))
    print("   DSR:", fmt(deflated_sharpe_ratio(cfgs[:, j], n_tentativas=50, sharpes_tentativas=srs)))
    print("   PBO:", fmt(pbo_cscv(cfgs, n_blocos=10)))

    print("3) Monte Carlo (trades sintéticos):",
          monte_carlo_drawdown(rng.normal(0.002, 0.02, 200), n_sim=1000))
