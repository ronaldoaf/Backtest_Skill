"""Comparação estatística entre estratégias.

Dependência: apenas numpy (e `estatistica.py`, na mesma pasta).

Responde a perguntas do tipo "a estratégia A é melhor que a B?" e "qual destas N estratégias
é de fato a melhor?" sem se enganar com a sorte. Métricas lado a lado não bastam: duas séries
de 5 anos com Sharpe 1,2 e 0,9 costumam ser estatisticamente indistinguíveis.

Convenções (as mesmas de `estatistica.py`):
- `retornos`: dict {nome: série 1D} ou matriz T x N (com `nomes`), retornos simples por período,
  líquidos de custos, **alinhados nas mesmas datas**. Alinhe antes (ex.: `df.dropna()` num
  DataFrame com uma coluna por estratégia). Séries de tamanhos diferentes levantam ValueError.
- NaN ou infinito levantam ValueError.
- `periodos_por_ano`: 252 diário, 365 cripto, 52 semanal, 12 mensal.

Funções:
- `tabela_comparativa`: métricas lado a lado e matriz de correlação.
- `teste_diferenca_sharpe`: Sharpe de A − Sharpe de B pelo método delta com covariância HAC
  (Ledoit e Wolf, 2008), robusto a caudas gordas e autocorrelação.
- `bootstrap_diferenca_sharpe`: mesma pergunta por bootstrap pareado em blocos (amostras curtas).
- `comparar_pares`: todos os pares, com p-valores corrigidos por Holm.
- `teste_spa`: a melhor de N estratégias supera um benchmark? (Hansen SPA e White Reality Check).
- `model_confidence_set`: conjunto das estratégias que não dá para descartar como piores
  (Hansen, Lunde e Nason, 2011).

Uso rápido:
    from comparacao import tabela_comparativa, teste_diferenca_sharpe, model_confidence_set

    r = {"momentum": r_mom, "reversao": r_rev, "buy_hold": r_bh}
    print(formatar_tabela(tabela_comparativa(r)))
    print(teste_diferenca_sharpe(r["momentum"], r["buy_hold"]))
    print(model_confidence_set(r))

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
    _indices_bloco, _matriz, _vetor, resumo_metricas,
)

_N = NormalDist()


# ---------------------------------------------------------------------------
# Entradas
# ---------------------------------------------------------------------------

def _painel(retornos, nomes=None, minimo_colunas: int = 2) -> tuple[np.ndarray, list[str]]:
    """Converte dict {nome: série} ou matriz T x N em (matriz, nomes), checando alinhamento."""
    if isinstance(retornos, dict):
        nomes = [str(k) for k in retornos]
        series = [np.asarray(v, dtype=float).ravel() for v in retornos.values()]
        tamanhos = {n: len(s) for n, s in zip(nomes, series)}
        if len(set(tamanhos.values())) > 1:
            raise ValueError(
                f"As séries têm tamanhos diferentes {tamanhos}. Alinhe pelas mesmas datas antes "
                f"(ex.: pd.DataFrame(retornos).dropna()).")
        R = _matriz(np.column_stack(series) if series else np.empty((0, 0)), "retornos")
    else:
        R = _matriz(retornos, "retornos")
        nomes = [str(n) for n in nomes] if nomes is not None else [f"E{i + 1}" for i in range(R.shape[1])]
        if len(nomes) != R.shape[1]:
            raise ValueError(f"{len(nomes)} nomes para {R.shape[1]} colunas.")
    if R.shape[1] < minimo_colunas:
        raise ValueError(f"Compare pelo menos {minimo_colunas} estratégias; recebeu {R.shape[1]}.")
    if len(set(nomes)) != len(nomes):
        raise ValueError("Nomes de estratégias repetidos.")
    return R, nomes


def _par(a, b) -> np.ndarray:
    a, b = _vetor(a, "retornos_a", minimo=3), _vetor(b, "retornos_b", minimo=3)
    if len(a) != len(b):
        raise ValueError(
            f"retornos_a ({len(a)}) e retornos_b ({len(b)}) têm tamanhos diferentes. "
            f"Alinhe pelas mesmas datas antes.")
    return np.column_stack([a, b])


def _lag_newey_west(T: int) -> int:
    """Regra usual de Newey e West (1994) para a defasagem do kernel de Bartlett."""
    return int(math.floor(4 * (T / 100.0) ** (2.0 / 9.0)))


# ---------------------------------------------------------------------------
# Tabela lado a lado
# ---------------------------------------------------------------------------

def tabela_comparativa(retornos, nomes=None, periodos_por_ano: int = 252) -> dict:
    """Métricas de cada estratégia (resumo_metricas) e matriz de correlação dos retornos.

    Correlação alta (> 0,7) indica que as estratégias apostam no mesmo fator: compará-las é
    mais fácil (o ruído comum se cancela), mas combiná-las diversifica pouco.
    """
    R, nomes = _painel(retornos, nomes, minimo_colunas=1)
    corr = np.corrcoef(R, rowvar=False) if R.shape[1] > 1 else np.ones((1, 1))
    return {
        "metricas": {n: resumo_metricas(R[:, i], periodos_por_ano) for i, n in enumerate(nomes)},
        "correlacao": {n: {m: float(corr[i, j]) for j, m in enumerate(nomes)}
                       for i, n in enumerate(nomes)},
    }


def formatar_tabela(tabela: dict, casas: int = 3) -> str:
    """Tabela Markdown com uma linha por métrica e uma coluna por estratégia, mais a correlação."""
    nomes = list(tabela["metricas"])
    metricas = list(next(iter(tabela["metricas"].values())))
    fmt = lambda v: f"{v:.{casas}f}" if isinstance(v, float) else str(v)  # noqa: E731
    linhas = ["| Métrica | " + " | ".join(nomes) + " |", "|---" * (len(nomes) + 1) + "|"]
    for m in metricas:
        linhas.append(f"| {m} | " + " | ".join(fmt(tabela["metricas"][n][m]) for n in nomes) + " |")
    if len(nomes) > 1:
        linhas += ["", "Correlação dos retornos:", "",
                   "| | " + " | ".join(nomes) + " |", "|---" * (len(nomes) + 1) + "|"]
        for n in nomes:
            linhas.append(f"| {n} | " + " | ".join(fmt(tabela["correlacao"][n][m]) for m in nomes) + " |")
    return "\n".join(linhas)


# ---------------------------------------------------------------------------
# Diferença de Sharpe entre duas estratégias
# ---------------------------------------------------------------------------

def _sharpe_por_periodo(R: np.ndarray) -> np.ndarray:
    sd = R.std(axis=0, ddof=1)
    return np.divide(R.mean(axis=0), sd, out=np.full(R.shape[1], np.nan), where=sd > 0)


def teste_diferenca_sharpe(retornos_a, retornos_b, periodos_por_ano: int = 252,
                           lag: int | None = None) -> dict:
    """H0: Sharpe(A) = Sharpe(B). Método delta com covariância HAC (Ledoit e Wolf, 2008).

    Ao contrário de Jobson-Korkie/Memmel, não supõe retornos normais e independentes, e já
    aproveita a correlação entre A e B (séries muito correlacionadas permitem detectar
    diferenças menores). Defasagem do kernel de Bartlett: `lag` (padrão Newey-West).
    Em amostras curtas (menos de ~250 períodos) confirme com `bootstrap_diferenca_sharpe`.

    Vale para estratégias definidas antes de ver os dados ou para séries fora da amostra.
    Comparar a melhor de N configurações com outra herda o viés de seleção: use `teste_spa`
    ou `model_confidence_set` com todas as candidatas.
    """
    X = _par(retornos_a, retornos_b)
    T = len(X)
    mu = X.mean(axis=0)
    gama = (X ** 2).mean(axis=0)
    var = gama - mu ** 2
    if np.any(var <= 0):
        raise ValueError("Uma das séries tem variância zero; o Sharpe não está definido.")
    sr = mu / np.sqrt(var)
    dif = float(sr[0] - sr[1])
    # Gradiente de f(mu_a, mu_b, gama_a, gama_b) = SR_a − SR_b
    grad = np.array([gama[0] / var[0] ** 1.5, -gama[1] / var[1] ** 1.5,
                     -mu[0] / (2 * var[0] ** 1.5), mu[1] / (2 * var[1] ** 1.5)])
    Y = np.column_stack([X - mu, X ** 2 - gama])
    L = _lag_newey_west(T) if lag is None else int(lag)
    psi = Y.T @ Y / T
    for k in range(1, min(L, T - 1) + 1):
        g = Y[k:].T @ Y[:-k] / T
        psi += (1 - k / (L + 1)) * (g + g.T)
    ep = float(math.sqrt(max(grad @ psi @ grad, 0.0) / T))
    z = dif / ep if ep > 0 else (0.0 if dif == 0 else math.copysign(math.inf, dif))
    p = 2 * (1 - _N.cdf(abs(z))) if math.isfinite(z) else 0.0
    anual = math.sqrt(periodos_por_ano)
    return {
        "sharpe_a": float(sr[0] * anual),
        "sharpe_b": float(sr[1] * anual),
        "diferenca": dif * anual,
        "erro_padrao": ep * anual,
        "ic95": (float((dif - 1.96 * ep) * anual), float((dif + 1.96 * ep) * anual)),
        "z": float(z),
        "p_valor": float(p),
        "correlacao": float(np.corrcoef(X, rowvar=False)[0, 1]),
        "lag": L,
    }


def bootstrap_diferenca_sharpe(retornos_a, retornos_b, periodos_por_ano: int = 252,
                               n_sim: int = 5000, tamanho_bloco: int = 10, nivel: float = 0.95,
                               seed: int = 42) -> dict:
    """IC e p-valor de Sharpe(A) − Sharpe(B) por bootstrap pareado em blocos circulares.

    "Pareado": os mesmos índices reamostram A e B, o que preserva a correlação entre elas.
    p-valor bilateral pela inversão do intervalo (menor nível em que o IC exclui zero).
    """
    X = _par(retornos_a, retornos_b)
    if np.any(X.std(axis=0, ddof=1) == 0):
        raise ValueError("Uma das séries tem variância zero; o Sharpe não está definido.")
    rng = np.random.default_rng(seed)
    anual = math.sqrt(periodos_por_ano)
    sr = _sharpe_por_periodo(X)
    difs = np.empty(n_sim)
    for i in range(n_sim):
        s = _sharpe_por_periodo(X[_indices_bloco(len(X), tamanho_bloco, rng)])
        difs[i] = s[0] - s[1]
    difs = difs[np.isfinite(difs)] * anual
    a = (1 - nivel) / 2
    p = min(1.0, 2 * min(np.mean(difs <= 0), np.mean(difs >= 0)))
    return {
        "diferenca": float((sr[0] - sr[1]) * anual),
        "ic": (float(np.quantile(difs, a)), float(np.quantile(difs, 1 - a))),
        "p_valor": float(p),
        "prob_a_melhor": float(np.mean(difs > 0)),
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
                   metodo: str = "hac", **kwargs) -> list[dict]:
    """Testa a diferença de Sharpe em todos os pares, com p-valores corrigidos por Holm.

    `metodo`: "hac" (teste_diferenca_sharpe) ou "bootstrap" (bootstrap_diferenca_sharpe).
    Com N estratégias há N(N−1)/2 pares; sem a correção, algum par "significativo" aparece
    por acaso. Reporte `p_holm`, não `p_valor`.
    """
    R, nomes = _painel(retornos, nomes)
    teste = {"hac": teste_diferenca_sharpe, "bootstrap": bootstrap_diferenca_sharpe}.get(metodo)
    if teste is None:
        raise ValueError('metodo deve ser "hac" ou "bootstrap".')
    linhas = []
    for i, j in itertools.combinations(range(R.shape[1]), 2):
        res = teste(R[:, i], R[:, j], periodos_por_ano=periodos_por_ano, **kwargs)
        linhas.append({"a": nomes[i], "b": nomes[j], "diferenca_sharpe": res["diferenca"],
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


def teste_spa(retornos_candidatas, retornos_benchmark, nomes=None, n_sim: int = 5000,
              tamanho_bloco: int = 10, seed: int = 42) -> dict:
    """H0: nenhuma candidata tem retorno médio maior que o benchmark.

    Recebe TODAS as candidatas testadas (inclusive as descartadas) e o benchmark (buy and hold,
    estratégia atual, zero para "ganha do caixa"), todos alinhados. Compara retornos médios por
    período; para comparar retorno ajustado ao risco, normalize antes as séries à mesma
    volatilidade. Retorna três p-valores:
    - `p_spa`: Hansen (2005), versão consistente. Use este.
    - `p_spa_inferior`: mais liberal (limite inferior).
    - `p_reality_check`: White (2000). Conservador: candidatas muito ruins inflam o p-valor;
      o SPA corrige isso.

    Diferença para `teste_permutacao_selecao` (estatistica.py): aquele precisa das posições e
    testa contra estratégias aleatórias; este só precisa dos retornos e testa contra um benchmark.
    """
    C, nomes = _painel(retornos_candidatas, nomes, minimo_colunas=1)
    b = _vetor(retornos_benchmark, "retornos_benchmark")
    if len(b) != len(C):
        raise ValueError(f"benchmark ({len(b)}) e candidatas ({len(C)}) têm tamanhos diferentes.")
    D = C - b[:, None]
    T = len(D)
    dbar = D.mean(axis=0)
    rng = np.random.default_rng(seed)
    boot = _medias_bootstrap(D, n_sim, tamanho_bloco, rng)
    omega = np.sqrt(T) * boot.std(axis=0, ddof=1)
    if np.any(omega <= 0):
        raise ValueError("Uma candidata é idêntica ao benchmark (diferença com variância zero).")
    t_obs = np.sqrt(T) * dbar / omega
    estat = max(float(t_obs.max()), 0.0)
    limite = -np.sqrt(omega ** 2 / T * 2 * math.log(math.log(T)))
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
            "excesso_medio_melhor": float(dbar[k]), **saida, "n_candidatas": len(nomes)}


# ---------------------------------------------------------------------------
# Model Confidence Set
# ---------------------------------------------------------------------------

def model_confidence_set(retornos, nomes=None, alfa: float = 0.10, n_sim: int = 5000,
                         tamanho_bloco: int = 10, normalizar_vol: bool = False,
                         seed: int = 42) -> dict:
    """Conjunto de estratégias que contém a melhor com confiança 1 − alfa (Hansen et al., 2011).

    Elimina, uma a uma, a estratégia de pior retorno médio enquanto a hipótese "todas no
    conjunto são equivalentes" for rejeitada (estatística T_max, bootstrap em blocos).
    Um conjunto grande não é falha do método: significa que os dados não distinguem as
    estratégias. `normalizar_vol=True` escala cada série à mesma volatilidade antes, de modo
    que a comparação passa a ser de retorno ajustado ao risco (equivalente ao Sharpe).

    Retorna `mcs` (nomes que ficam), `p_valores` (p-valor MCS de cada estratégia; fica no
    conjunto quem tem p ≥ alfa) e `ordem_eliminacao` (da pior para a melhor).
    """
    R, nomes = _painel(retornos, nomes)
    if normalizar_vol:
        sd = R.std(axis=0, ddof=1)
        if np.any(sd <= 0):
            raise ValueError("Uma das séries tem variância zero.")
        R = R / sd * sd.mean()
    perdas = -R
    rng = np.random.default_rng(seed)
    boot = _medias_bootstrap(perdas, n_sim, tamanho_bloco, rng)   # n_sim x m
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
            "ordem_eliminacao": ordem, "alfa": alfa}


# ---------------------------------------------------------------------------
# Demonstração
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    T = 252 * 6
    fator = rng.normal(0, 0.010, T)                        # risco comum às estratégias
    r = {
        "boa": 0.0006 + 0.8 * fator + rng.normal(0, 0.006, T),
        "parecida": 0.0004 + 0.8 * fator + rng.normal(0, 0.006, T),
        "ruim": -0.0002 + 0.5 * fator + rng.normal(0, 0.010, T),
        "buy_hold": 0.0002 + fator,
    }
    print(formatar_tabela(tabela_comparativa(r)))
    print("\nDiferença de Sharpe boa x parecida (HAC):",
          {k: round(v, 3) if isinstance(v, float) else v
           for k, v in teste_diferenca_sharpe(r["boa"], r["parecida"]).items()})
    print("Mesma pergunta por bootstrap:", bootstrap_diferenca_sharpe(r["boa"], r["parecida"], n_sim=1000))
    print("\nTodos os pares (Holm):")
    for linha in comparar_pares(r):
        print("  ", {k: round(v, 3) if isinstance(v, float) else v for k, v in linha.items()})
    candidatas = {k: v for k, v in r.items() if k != "buy_hold"}
    spa = teste_spa(candidatas, r["buy_hold"], n_sim=1000)
    print("\nSPA contra buy and hold:", {k: round(v, 4) if isinstance(v, float) else v for k, v in spa.items()})
    print("Model Confidence Set (10%):", model_confidence_set(r, n_sim=1000, normalizar_vol=True))
