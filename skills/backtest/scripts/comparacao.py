"""Comparação estatística entre estratégias.

Dependência: apenas numpy (e `estatistica.py`, na mesma pasta). Aceita objetos do pandas sem
importá-lo.

Responde a perguntas do tipo "a estratégia A é melhor que a B?" e "qual destas N estratégias
é de fato a melhor?" sem se enganar com a sorte. Métricas lado a lado não bastam: duas séries
de 5 anos com Sharpe 1,2 e 0,9 costumam ser estatisticamente indistinguíveis.

Convenções (as mesmas de `estatistica.py`):
- `retornos`: DataFrame (uma coluna por estratégia), dict {nome: série 1D} ou matriz T x N (com
  `nomes`), retornos simples por período, líquidos de custos, **alinhados nas mesmas datas**.
  Séries de tamanhos diferentes, ou Series/DataFrames do pandas com datas diferentes, levantam
  ValueError: alinhe antes (ex.: `pd.DataFrame(retornos).dropna()`).
- `rf_por_periodo`: taxa livre de risco POR PERÍODO (número ou série alinhada às mesmas datas),
  descontada antes de calcular Sharpe. No Brasil, use o CDI: `(1 + cdi_anual) ** (1 / 252) - 1`
  ou a série diária histórica. Com rf = 0, uma estratégia de baixa volatilidade que rende
  CDI + 1% ao ano parece ter Sharpe 12, e as comparações de Sharpe saem erradas.
- NaN ou infinito levantam ValueError.
- `periodos_por_ano`: 252 diário, 365 cripto, 52 semanal, 12 mensal.
- `tamanho_bloco`: blocos do bootstrap circular. Padrão: round(T ** (1/3)), que dá 3 a 5 em
  séries mensais e 10 a 14 em diárias. Aumente se os retornos tiverem autocorrelação forte.

Funções:
- `tabela_comparativa` / `formatar_tabela`: métricas lado a lado e matriz de correlação.
- `teste_diferenca_sharpe`: Sharpe de A − Sharpe de B pelo método delta com covariância HAC
  (Ledoit e Wolf, 2008), robusto a caudas gordas e autocorrelação.
- `bootstrap_diferenca_sharpe`: mesma pergunta por bootstrap pareado em blocos, estudentizado
  (Ledoit e Wolf, 2008). Use este em amostras curtas (menos de ~120 períodos).
- `comparar_pares`: todos os pares, com p-valores corrigidos por Holm.
- `teste_spa`: a melhor de N estratégias supera um benchmark? (Hansen SPA e White Reality Check).
- `model_confidence_set`: conjunto das estratégias que não dá para descartar como piores
  (Hansen, Lunde e Nason, 2011).

Uso rápido:
    from comparacao import formatar_tabela, tabela_comparativa, teste_diferenca_sharpe, model_confidence_set

    df = pd.DataFrame({"momentum": r_mom, "reversao": r_rev, "buy_hold": r_bh}).dropna()
    cdi = cdi_diario.reindex(df.index)             # mesma data, taxa por período
    print(formatar_tabela(tabela_comparativa(df, rf_por_periodo=cdi)))
    print(teste_diferenca_sharpe(df["momentum"], df["buy_hold"], rf_por_periodo=cdi))
    print(model_confidence_set(df, rf_por_periodo=cdi))

Execute `python comparacao.py` para uma demonstração com dados sintéticos.
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estatistica import (  # noqa: E402
    _indices_bloco, _matriz, _vetor, resumo_metricas, sharpe, sortino,
)

_N = NormalDist()
CRITERIOS = ("sharpe", "retorno")


# ---------------------------------------------------------------------------
# Entradas
# ---------------------------------------------------------------------------

def _indice(x):
    """Índice de datas de uma Series/DataFrame do pandas; None para arrays e listas."""
    idx = getattr(x, "index", None)
    return None if idx is None or callable(idx) else idx


def _mesmo_indice(i1, i2) -> bool:
    if hasattr(i1, "equals"):
        return bool(i1.equals(i2))
    a, b = np.asarray(i1), np.asarray(i2)
    return a.shape == b.shape and bool(np.all(a == b))


def _checar_datas(indices: dict) -> None:
    """Falha alto quando séries do pandas não estão nas mesmas datas (arrays sem índice passam)."""
    com_indice = [(n, i) for n, i in indices.items() if i is not None]
    for n, i in com_indice[1:]:
        n0, i0 = com_indice[0]
        if not _mesmo_indice(i0, i):
            try:
                comuns = f" ({len(np.intersect1d(np.asarray(i0), np.asarray(i)))} datas em comum)"
            except (TypeError, ValueError):
                comuns = ""
            raise ValueError(
                f"'{n0}' e '{n}' não estão nas mesmas datas{comuns}. Alinhe antes, por exemplo "
                f"df = pd.DataFrame(retornos).dropna() e rf = cdi.reindex(df.index).")


def _painel(retornos, nomes=None, minimo_colunas: int = 2):
    """Converte DataFrame, dict {nome: série} ou matriz T x N em (matriz, nomes, índice de datas)."""
    if isinstance(retornos, dict):
        nomes = [str(k) for k in retornos]
        series = [np.asarray(v, dtype=float).ravel() for v in retornos.values()]
        tamanhos = {n: len(s) for n, s in zip(nomes, series)}
        if len(set(tamanhos.values())) > 1:
            raise ValueError(
                f"As séries têm tamanhos diferentes {tamanhos}. Alinhe pelas mesmas datas antes "
                f"(ex.: pd.DataFrame(retornos).dropna()).")
        indices = {n: _indice(v) for n, v in zip(nomes, retornos.values())}
        _checar_datas(indices)
        indice = next((i for i in indices.values() if i is not None), None)
        R = _matriz(np.column_stack(series) if series else np.empty((0, 0)), "retornos")
    else:
        colunas = getattr(retornos, "columns", None)
        if nomes is None and colunas is not None:
            nomes = list(colunas)
        indice = _indice(retornos)
        R = _matriz(retornos, "retornos")
        nomes = [str(n) for n in nomes] if nomes is not None else [f"E{i + 1}" for i in range(R.shape[1])]
        if len(nomes) != R.shape[1]:
            raise ValueError(f"{len(nomes)} nomes para {R.shape[1]} colunas.")
    if R.shape[1] < minimo_colunas:
        raise ValueError(f"Compare pelo menos {minimo_colunas} estratégias; recebeu {R.shape[1]}.")
    if len(set(nomes)) != len(nomes):
        raise ValueError("Nomes de estratégias repetidos.")
    return R, nomes, indice


def _rf(rf_por_periodo, T: int, indice, periodos_por_ano: int) -> np.ndarray:
    """Taxa livre de risco por período como vetor de tamanho T, conferindo datas e escala."""
    if np.ndim(rf_por_periodo) == 0:
        rf = np.full(T, float(rf_por_periodo))
        if not np.all(np.isfinite(rf)):
            raise ValueError("rf_por_periodo deve ser um número finito.")
    else:
        rf = _vetor(rf_por_periodo, "rf_por_periodo", minimo=1)
        if len(rf) != T:
            raise ValueError(f"rf_por_periodo tem {len(rf)} valores e os retornos têm {T}. "
                             f"Alinhe pelas mesmas datas (ex.: cdi.reindex(df.index)).")
        _checar_datas({"retornos": indice, "rf_por_periodo": _indice(rf_por_periodo)})
    if (1 + rf.mean()) ** periodos_por_ano - 1 > 1.0:
        raise ValueError(
            f"rf_por_periodo dá mais de 100% ao ano com periodos_por_ano={periodos_por_ano}: "
            f"parece uma taxa anual. Converta: (1 + taxa_anual) ** (1 / {periodos_por_ano}) - 1.")
    return rf


def _par(a, b, rf_por_periodo=0.0, periodos_por_ano: int = 252) -> np.ndarray:
    """Matriz T x 2 dos retornos de A e B em excesso de rf."""
    va, vb = _vetor(a, "retornos_a", minimo=3), _vetor(b, "retornos_b", minimo=3)
    if len(va) != len(vb):
        raise ValueError(
            f"retornos_a ({len(va)}) e retornos_b ({len(vb)}) têm tamanhos diferentes. "
            f"Alinhe pelas mesmas datas antes.")
    ia, ib = _indice(a), _indice(b)
    _checar_datas({"retornos_a": ia, "retornos_b": ib})
    rf = _rf(rf_por_periodo, len(va), ia if ia is not None else ib, periodos_por_ano)
    return np.column_stack([va, vb]) - rf[:, None]


def _bloco(tamanho_bloco, T: int) -> int:
    if tamanho_bloco is None:
        return max(1, int(round(T ** (1 / 3))))
    if int(tamanho_bloco) != tamanho_bloco or not 1 <= tamanho_bloco <= max(1, T // 2):
        raise ValueError(f"tamanho_bloco deve ser inteiro entre 1 e {max(1, T // 2)}; recebeu {tamanho_bloco}.")
    return int(tamanho_bloco)


def _checar_prob(valor: float, nome: str) -> None:
    if not 0 < valor < 1:
        raise ValueError(f"{nome} deve estar entre 0 e 1 (ex.: 0.10); recebeu {valor}.")


def _lag_newey_west(T: int) -> int:
    """Regra usual de Newey e West (1994) para a defasagem do kernel de Bartlett."""
    return int(math.floor(4 * (T / 100.0) ** (2.0 / 9.0)))


# ---------------------------------------------------------------------------
# Tabela lado a lado
# ---------------------------------------------------------------------------

def tabela_comparativa(retornos, nomes=None, periodos_por_ano: int = 252,
                       rf_por_periodo=0.0) -> dict:
    """Métricas de cada estratégia (resumo_metricas) e matriz de correlação dos retornos.

    Sharpe e Sortino descontam `rf_por_periodo`; CAGR, volatilidade e drawdown usam os retornos
    como estão. Correlação alta (> 0,7) indica que as estratégias apostam no mesmo fator:
    compará-las é mais fácil (o ruído comum se cancela), mas combiná-las diversifica pouco.
    """
    R, nomes, indice = _painel(retornos, nomes, minimo_colunas=1)
    rf = _rf(rf_por_periodo, len(R), indice, periodos_por_ano)
    metricas = {}
    for i, n in enumerate(nomes):
        m = resumo_metricas(R[:, i], periodos_por_ano)
        m["sharpe"] = sharpe(R[:, i] - rf, periodos_por_ano)
        m["sortino"] = sortino(R[:, i] - rf, periodos_por_ano)
        metricas[n] = m
    corr = np.corrcoef(R, rowvar=False) if R.shape[1] > 1 else np.ones((1, 1))
    return {
        "metricas": metricas,
        "correlacao": {n: {m: float(corr[i, j]) for j, m in enumerate(nomes)}
                       for i, n in enumerate(nomes)},
        "rf_anual": float((1 + rf.mean()) ** periodos_por_ano - 1),
    }


def formatar_tabela(tabela: dict, casas: int = 3) -> str:
    """Tabela Markdown com uma linha por métrica e uma coluna por estratégia, mais a correlação."""
    nomes = list(tabela["metricas"])
    metricas = list(next(iter(tabela["metricas"].values())))
    fmt = lambda v: f"{v:.{casas}f}" if isinstance(v, float) else str(v)  # noqa: E731
    linhas = ["| Métrica | " + " | ".join(nomes) + " |", "|---" * (len(nomes) + 1) + "|"]
    for m in metricas:
        linhas.append(f"| {m} | " + " | ".join(fmt(tabela["metricas"][n][m]) for n in nomes) + " |")
    if "rf_anual" in tabela:
        rf = tabela["rf_anual"]
        linhas += ["", f"Sharpe e Sortino em excesso da taxa livre de risco (média de {rf:.1%} ao ano)."
                   if rf else "Sharpe e Sortino sem desconto de taxa livre de risco (rf = 0)."]
    if len(nomes) > 1:
        linhas += ["", "Correlação dos retornos:", "",
                   "| | " + " | ".join(nomes) + " |", "|---" * (len(nomes) + 1) + "|"]
        for n in nomes:
            linhas.append(f"| {n} | " + " | ".join(fmt(tabela["correlacao"][n][m]) for m in nomes) + " |")
    return "\n".join(linhas)


# ---------------------------------------------------------------------------
# Diferença de Sharpe entre duas estratégias
# ---------------------------------------------------------------------------

def _momentos(X: np.ndarray):
    """Sharpe por período (desvio ddof=0), gradiente de SR_a − SR_b e momentos centrados (T x 4)."""
    mu = X.mean(axis=0)
    gama = (X ** 2).mean(axis=0)
    var = gama - mu ** 2
    if np.any(var <= 0):
        return None
    grad = np.array([gama[0] / var[0] ** 1.5, -gama[1] / var[1] ** 1.5,
                     -mu[0] / (2 * var[0] ** 1.5), mu[1] / (2 * var[1] ** 1.5)])
    return mu / np.sqrt(var), grad, np.column_stack([X - mu, X ** 2 - gama])


def _momentos_ou_erro(X: np.ndarray):
    m = _momentos(X)
    if m is None:
        raise ValueError(
            "Uma das séries tem variância zero (depois de descontar rf); o Sharpe dela não está "
            "definido. Para testar se uma estratégia bate o caixa, use teste_spa com o CDI como "
            "benchmark e rf_por_periodo.")
    return m


def _ep_hac(Y: np.ndarray, grad: np.ndarray, L: int) -> float:
    """Erro padrão da diferença de Sharpe (por período) com kernel de Bartlett de defasagem L."""
    T = len(Y)
    psi = Y.T @ Y / T
    for k in range(1, min(L, T - 1) + 1):
        g = Y[k:].T @ Y[:-k] / T
        psi += (1 - k / (L + 1)) * (g + g.T)
    return float(math.sqrt(max(grad @ psi @ grad, 0.0) / T))


def teste_diferenca_sharpe(retornos_a, retornos_b, periodos_por_ano: int = 252,
                           lag: int | None = None, rf_por_periodo=0.0) -> dict:
    """H0: Sharpe(A) = Sharpe(B). Método delta com covariância HAC (Ledoit e Wolf, 2008).

    Ao contrário de Jobson-Korkie/Memmel, não supõe retornos normais e independentes, e já
    aproveita a correlação entre A e B (séries muito correlacionadas permitem detectar
    diferenças menores). Defasagem do kernel de Bartlett: `lag` (padrão Newey-West).

    Tamanho em simulação (rejeições a 5% com H0 verdadeira): 4% a 6% em séries diárias de 1 a
    10 anos (caudas t(4), GARCH, vols diferentes); 6% a 8,5% com autocorrelação 0,3; 6% a 11%
    com 36 a 120 meses. Abaixo de ~120 períodos, decida por `bootstrap_diferenca_sharpe`.

    Vale para estratégias definidas antes de ver os dados ou para séries fora da amostra.
    Comparar a melhor de N configurações com outra herda o viés de seleção: use `teste_spa`
    ou `model_confidence_set` com todas as candidatas.
    Sharpes reportados com desvio ddof=1, como em `tabela_comparativa`.
    """
    X = _par(retornos_a, retornos_b, rf_por_periodo, periodos_por_ano)
    T = len(X)
    sr, grad, Y = _momentos_ou_erro(X)
    dif = float(sr[0] - sr[1])
    L = _lag_newey_west(T) if lag is None else int(lag)
    if L < 0:
        raise ValueError("lag deve ser >= 0.")
    ep = _ep_hac(Y, grad, L)
    z = dif / ep if ep > 0 else (0.0 if dif == 0 else math.copysign(math.inf, dif))
    p = 2 * (1 - _N.cdf(abs(z))) if math.isfinite(z) else 0.0
    c = math.sqrt((T - 1) / T * periodos_por_ano)   # ddof=1 e anualização
    return {
        "sharpe_a": float(sr[0] * c),
        "sharpe_b": float(sr[1] * c),
        "diferenca": dif * c,
        "erro_padrao": ep * c,
        "ic95": (float((dif - 1.96 * ep) * c), float((dif + 1.96 * ep) * c)),
        "z": float(z),
        "p_valor": float(p),
        "correlacao": float(np.corrcoef(X, rowvar=False)[0, 1]),
        "lag": L,
    }


def bootstrap_diferenca_sharpe(retornos_a, retornos_b, periodos_por_ano: int = 252,
                               n_sim: int = 5000, tamanho_bloco: int | None = None,
                               nivel: float = 0.95, seed: int = 42, rf_por_periodo=0.0) -> dict:
    """IC e p-valor de Sharpe(A) − Sharpe(B) por bootstrap estudentizado (Ledoit e Wolf, 2008).

    "Pareado": os mesmos índices reamostram A e B em blocos circulares, o que preserva a
    correlação entre elas e a autocorrelação. "Estudentizado": cada reamostra calcula a
    diferença E o seu erro padrão (pelos blocos), e o p-valor compara |dif* − dif| / ep* com
    |dif| / ep. Em simulação com H0 verdadeira e 36 a 250 períodos, rejeitou 3% a 5% a 5%,
    onde o teste HAC rejeitou 5% a 9%; com autocorrelação 0,3, 6% a 7% (aumente tamanho_bloco).

    Use este quando houver menos de ~120 períodos. IC simétrico em torno da diferença.
    `prob_a_melhor` = fração das reamostras em que A teve Sharpe maior; serve para comunicar
    ("A foi melhor em 93% das reamostras"), mas não é a probabilidade de A ser melhor nem p-valor.
    """
    X = _par(retornos_a, retornos_b, rf_por_periodo, periodos_por_ano)
    T = len(X)
    b = _bloco(tamanho_bloco, T)
    _checar_prob(nivel, "nivel")
    sr, grad, Y = _momentos_ou_erro(X)
    dif = float(sr[0] - sr[1])
    ep = _ep_hac(Y, grad, _lag_newey_west(T))
    t_obs = abs(dif) / ep if ep > 0 else (0.0 if dif == 0 else math.inf)
    k = T // b
    rng = np.random.default_rng(seed)
    difs, ts = np.full(n_sim, np.nan), np.full(n_sim, np.nan)
    for i in range(n_sim):
        m = _momentos(X[_indices_bloco(T, b, rng)])
        if m is None:
            continue
        s, g, Ys = m
        S = Ys[:k * b].reshape(k, b, 4).sum(axis=1)       # somas por bloco (Götze e Künsch)
        v = g @ (S.T @ S / (k * b)) @ g / T
        difs[i] = s[0] - s[1]
        if v > 0:
            ts[i] = abs(difs[i] - dif) / math.sqrt(v)
    ts = ts[np.isfinite(ts)]
    p = (np.sum(ts >= t_obs) + 1) / (len(ts) + 1)
    q = float(np.quantile(ts, nivel)) if len(ts) else 0.0
    c = math.sqrt((T - 1) / T * periodos_por_ano)
    return {
        "diferenca": dif * c,
        "erro_padrao": ep * c,
        "ic": (float((dif - q * ep) * c), float((dif + q * ep) * c)),
        "p_valor": float(min(1.0, p)),
        "prob_a_melhor": float(np.mean(difs[np.isfinite(difs)] > 0)),
        "tamanho_bloco": b,
    }


def holm(p_valores) -> np.ndarray:
    """p-valores ajustados por Holm-Bonferroni (controla a chance de qualquer falso positivo)."""
    p = _vetor(p_valores, "p_valores", minimo=1)
    m = len(p)
    ordem = np.argsort(p)
    ajust = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[ordem]))
    saida = np.empty(m)
    saida[ordem] = ajust
    return saida


def comparar_pares(retornos, nomes=None, periodos_por_ano: int = 252,
                   metodo: str = "hac", rf_por_periodo=0.0, **kwargs) -> list[dict]:
    """Testa a diferença de Sharpe em todos os pares, com p-valores corrigidos por Holm.

    `metodo`: "hac" (teste_diferenca_sharpe) ou "bootstrap" (bootstrap_diferenca_sharpe; use
    abaixo de ~120 períodos). Com N estratégias há N(N−1)/2 pares; sem a correção, algum par
    "significativo" aparece por acaso. Reporte `p_holm`, não `p_valor`. `ic` não é corrigido
    pela multiplicidade.
    """
    R, nomes, indice = _painel(retornos, nomes)
    rf = _rf(rf_por_periodo, len(R), indice, periodos_por_ano)
    teste = {"hac": teste_diferenca_sharpe, "bootstrap": bootstrap_diferenca_sharpe}.get(metodo)
    if teste is None:
        raise ValueError('metodo deve ser "hac" ou "bootstrap".')
    linhas = []
    for i, j in itertools.combinations(range(R.shape[1]), 2):
        res = teste(R[:, i], R[:, j], periodos_por_ano=periodos_por_ano, rf_por_periodo=rf, **kwargs)
        linhas.append({"a": nomes[i], "b": nomes[j], "diferenca_sharpe": res["diferenca"],
                       "ic": res["ic95"] if metodo == "hac" else res["ic"],
                       "p_valor": res["p_valor"]})
    for linha, p in zip(linhas, holm([l["p_valor"] for l in linhas])):
        linha["p_holm"] = float(p)
    return linhas


# ---------------------------------------------------------------------------
# A melhor de N contra um benchmark: Reality Check e SPA
# ---------------------------------------------------------------------------

def _medias_bootstrap(D: np.ndarray, n_sim: int, tamanho_bloco: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Médias de cada coluna em n_sim reamostragens em blocos circulares (n_sim x K)."""
    T = len(D)
    return np.stack([D[_indices_bloco(T, tamanho_bloco, rng)].mean(axis=0) for _ in range(n_sim)])


def _checar_criterio(criterio: str) -> None:
    if criterio not in CRITERIOS:
        raise ValueError(f'criterio deve ser "sharpe" ou "retorno"; recebeu {criterio!r}.')


def _diferencas_spa(C: np.ndarray, b: np.ndarray, rf: np.ndarray, criterio: str) -> np.ndarray:
    """Desempenho de cada candidata menos o do benchmark, por período (maior é melhor)."""
    if criterio == "retorno":
        return C - b[:, None]
    E, e = C - rf[:, None], b - rf
    sd = E.std(axis=0, ddof=1)
    if np.any(sd <= 0):
        raise ValueError("Uma candidata tem volatilidade zero depois de descontar rf; o Sharpe dela "
                         "não está definido. Use criterio='retorno'.")
    sd_b = float(e.std(ddof=1))
    if sd_b == 0:      # benchmark constante (caixa): bater o caixa = excesso médio sobre ele positivo
        return (E - e[:, None]) / sd * float(np.median(sd))
    if sd_b < 0.05 * float(np.median(sd)):
        raise ValueError(
            "O benchmark quase não oscila perto das candidatas (parece caixa ou renda fixa curta). "
            "Para testar se alguma candidata bate o caixa, passe rf_por_periodo igual ao benchmark "
            "(ex.: a série do CDI nos dois) ou use criterio='retorno'.")
    return E / sd * sd_b - e[:, None]          # candidatas na volatilidade do benchmark


def teste_spa(retornos_candidatas, retornos_benchmark, nomes=None, n_sim: int = 5000,
              tamanho_bloco: int | None = None, seed: int = 42, criterio: str = "sharpe",
              rf_por_periodo=0.0, periodos_por_ano: int = 252) -> dict:
    """H0: nenhuma candidata é melhor que o benchmark.

    Recebe TODAS as candidatas testadas (inclusive as descartadas) e o benchmark, todos
    alinhados: buy and hold, a estratégia atual ou, para "ganha do caixa", a série do CDI (nesse
    caso passe também `rf_por_periodo` = CDI).
    `criterio`:
    - "sharpe" (padrão): desconta rf e escala cada candidata à volatilidade do benchmark antes de
      comparar médias, o que equivale a comparar Sharpe. Uma candidata alavancada não ganha só
      por ter mais risco.
    - "retorno": compara retornos médios brutos (use só se o risco já for o mesmo).
    Retorna três p-valores:
    - `p_spa`: Hansen (2005), versão consistente. Use este. Em simulação, rejeita 6% a 7% a 5%
      quando todas as candidatas empatam com o benchmark: trate p entre 0,03 e 0,05 como limítrofe.
    - `p_spa_inferior`: mais liberal (limite inferior).
    - `p_reality_check`: White (2000). Conservador: candidatas muito ruins inflam o p-valor;
      o SPA corrige isso.

    Diferença para `teste_permutacao_selecao` (estatistica.py): aquele precisa das posições e
    testa contra estratégias aleatórias; este só precisa dos retornos e testa contra um benchmark.
    """
    _checar_criterio(criterio)
    C, nomes, indice = _painel(retornos_candidatas, nomes, minimo_colunas=1)
    b = _vetor(retornos_benchmark, "retornos_benchmark")
    if len(b) != len(C):
        raise ValueError(f"benchmark ({len(b)}) e candidatas ({len(C)}) têm tamanhos diferentes.")
    i_b = _indice(retornos_benchmark)
    _checar_datas({"candidatas": indice, "retornos_benchmark": i_b})
    rf = _rf(rf_por_periodo, len(C), indice if indice is not None else i_b, periodos_por_ano)
    D = _diferencas_spa(C, b, rf, criterio)
    T = len(D)
    bloco = _bloco(tamanho_bloco, T)
    dbar = D.mean(axis=0)
    rng = np.random.default_rng(seed)
    boot = _medias_bootstrap(D, n_sim, bloco, rng)
    omega = np.sqrt(T) * boot.std(axis=0, ddof=1)
    if np.any(omega <= 0):
        raise ValueError("Uma candidata é idêntica ao benchmark (diferença com variância zero).")
    t_obs = np.sqrt(T) * dbar / omega
    estat = max(float(t_obs.max()), 0.0)
    limite = -np.sqrt(omega ** 2 / T * 2 * math.log(math.log(max(T, 16))))
    centros = {
        "p_spa": np.where(dbar >= limite, dbar, 0.0),
        "p_spa_inferior": np.maximum(dbar, 0.0),
    }
    saida = {}
    for chave, mu in centros.items():
        t_boot = np.maximum((np.sqrt(T) * (boot - mu) / omega).max(axis=1), 0.0)
        saida[chave] = float(np.mean(t_boot >= estat))
    # White (2000): estatística não studentizada, centrada na média amostral
    rc_obs = max(float(np.sqrt(T) * dbar.max()), 0.0)
    rc_boot = np.maximum((np.sqrt(T) * (boot - dbar)).max(axis=1), 0.0)
    saida["p_reality_check"] = float(np.mean(rc_boot >= rc_obs))
    k = int(np.argmax(t_obs))
    return {"melhor": nomes[k], "t_melhor": float(t_obs[k]),
            "excesso_medio_melhor": float(dbar[k]), **saida, "n_candidatas": len(nomes),
            "criterio": criterio, "tamanho_bloco": bloco}


# ---------------------------------------------------------------------------
# Model Confidence Set
# ---------------------------------------------------------------------------

def model_confidence_set(retornos, nomes=None, alfa: float = 0.10, n_sim: int = 5000,
                         tamanho_bloco: int | None = None, criterio: str = "sharpe",
                         seed: int = 42, rf_por_periodo=0.0, periodos_por_ano: int = 252) -> dict:
    """Conjunto de estratégias que contém a melhor com confiança 1 − alfa (Hansen et al., 2011).

    Elimina, uma a uma, a estratégia de pior desempenho relativo (estatística T_max, bootstrap
    em blocos) enquanto a hipótese "todas no conjunto são equivalentes" for rejeitada.
    Um conjunto grande não é falha do método: significa que os dados não distinguem as
    estratégias.
    `criterio`:
    - "sharpe" (padrão): desconta rf e escala cada série à mesma volatilidade antes de comparar
      médias, o que equivale a comparar Sharpe. Passe o CDI em `rf_por_periodo`.
    - "retorno": compara retornos médios brutos e favorece a estratégia mais alavancada; use só
      se o risco já for o mesmo.
    Em simulação, com todas equivalentes, exclui alguma por engano em 9% a 13% das vezes a
    alfa = 10%.

    Retorna `mcs` (nomes que ficam), `p_valores` (p-valor MCS de cada estratégia; fica no
    conjunto quem tem p ≥ alfa) e `ordem_eliminacao` (da pior para a melhor).
    """
    _checar_criterio(criterio)
    _checar_prob(alfa, "alfa")
    R, nomes, indice = _painel(retornos, nomes)
    rf = _rf(rf_por_periodo, len(R), indice, periodos_por_ano)
    if criterio == "sharpe":
        E = R - rf[:, None]
        sd = E.std(axis=0, ddof=1)
        if np.any(sd <= 0):
            zeradas = [n for n, s in zip(nomes, sd) if s <= 0]
            raise ValueError(f"{zeradas} têm volatilidade zero depois de descontar rf; o Sharpe não "
                             f"está definido. Use criterio='retorno'.")
        R = E / sd * sd.mean()
    bloco = _bloco(tamanho_bloco, len(R))
    perdas = -R
    rng = np.random.default_rng(seed)
    boot = _medias_bootstrap(perdas, n_sim, bloco, rng)   # n_sim x m
    lbar = perdas.mean(axis=0)

    # Elimina até sobrar uma; o p-valor MCS de cada estratégia é o maior p-valor visto até a
    # eliminação dela. Fica no conjunto quem tem p-valor MCS >= alfa.
    vivos = list(range(R.shape[1]))
    p_mcs, ordem, p_acum = {}, [], 0.0
    while len(vivos) > 1:
        v = np.array(vivos)
        d_obs = lbar[v] - lbar[v].mean()
        d_boot = boot[:, v] - boot[:, v].mean(axis=1, keepdims=True)
        sd = d_boot.std(axis=0, ddof=1)
        sd = np.where(sd > 0, sd, np.inf)
        t_obs = d_obs / sd
        t_boot = ((d_boot - d_obs) / sd).max(axis=1)
        p_acum = max(p_acum, float(np.mean(t_boot >= t_obs.max())))
        pior = vivos[int(np.argmax(t_obs))]
        p_mcs[nomes[pior]] = p_acum
        ordem.append(nomes[pior])
        vivos.remove(pior)
    p_mcs[nomes[vivos[0]]] = 1.0
    return {"mcs": [n for n in nomes if p_mcs[n] >= alfa],
            "p_valores": {n: p_mcs[n] for n in nomes},
            "ordem_eliminacao": ordem, "alfa": alfa, "criterio": criterio, "tamanho_bloco": bloco}


# ---------------------------------------------------------------------------
# Demonstração
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    def _fmt(d):
        arred = lambda v: round(v, 3) if isinstance(v, float) else v  # noqa: E731
        return {k: tuple(map(arred, v)) if isinstance(v, tuple) else arred(v) for k, v in d.items()}

    rng = np.random.default_rng(1)
    T = 252 * 6
    cdi = (1.11) ** (1 / 252) - 1                         # CDI de 11% ao ano, por pregão
    fator = rng.normal(0, 0.010, T)                        # risco comum às estratégias
    r = {
        "boa": cdi + 0.0006 + 0.8 * fator + rng.normal(0, 0.006, T),
        "parecida": cdi + 0.0004 + 0.8 * fator + rng.normal(0, 0.006, T),
        "ruim": cdi - 0.0002 + 0.5 * fator + rng.normal(0, 0.010, T),
        "buy_hold": cdi + 0.0002 + fator,
    }
    print(formatar_tabela(tabela_comparativa(r, rf_por_periodo=cdi)))
    print("\nDiferença de Sharpe boa x parecida (HAC):",
          _fmt(teste_diferenca_sharpe(r["boa"], r["parecida"], rf_por_periodo=cdi)))
    print("Mesma pergunta por bootstrap estudentizado:",
          _fmt(bootstrap_diferenca_sharpe(r["boa"], r["parecida"], n_sim=1000, rf_por_periodo=cdi)))
    print("\nTodos os pares (Holm):")
    for linha in comparar_pares(r, rf_por_periodo=cdi):
        print("  ", _fmt(linha))
    candidatas = {k: v for k, v in r.items() if k != "buy_hold"}
    print("\nSPA contra buy and hold (Sharpe):",
          _fmt(teste_spa(candidatas, r["buy_hold"], n_sim=1000, rf_por_periodo=cdi)))
    print("Model Confidence Set (10%, Sharpe):", _fmt(model_confidence_set(r, n_sim=1000, rf_por_periodo=cdi)))
