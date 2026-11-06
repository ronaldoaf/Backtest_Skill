"""Protocolo de aprovação usado no projeto Backtests Lighter (só numpy e pandas).

Entrada: um DataFrame de barras (índice DatetimeIndex UTC, colunas open/high/low/close) e uma
lista de trades com i (barra de entrada), j (barra de saída, exclusiva), d (+1 long, -1 short).
Convenção: sinal no fechamento da barra i-1, execução na ABERTURA da barra i; saída na
abertura da barra j. Retorno do trade composto corretamente para o vendido (1 - P1/P0),
nunca log(P0/P1).

Custos (CUSTO, FUND) são os médios medidos na Lighter; atualize com
`lighter_dados.py custos` antes de um estudo novo.

    python protocolo.py        # demonstração com dados sintéticos
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

ATIVOS = ("xau", "us100", "wti", "btc", "eth")
# custo de UMA perna (fração): spread medido no livro da Lighter, ~US$1k
CUSTO = {"xau": 0.75e-4, "us100": 1.2e-4, "wti": 1.0e-4, "btc": 0.5e-4, "eth": 0.35e-4}
# funding anual médio que o long paga (%); o short recebe. Histórico 2025-26 da Lighter.
FUND = {"xau": 13.8, "us100": 4.5, "wti": 67.7, "btc": 10.9, "eth": 11.2}
ANO = 365 * 86400


# ---------------------------------------------------------------- trades e retornos
def retornos_trades(df: pd.DataFrame, trades: pd.DataFrame, ativo: str, mult_custo: float = 1.0,
                    mult_fund: float = 1.0) -> pd.DataFrame:
    """Retorno líquido por trade: bruto - 2 pernas de custo - funding pró-rata (long paga)."""
    o = df["open"].to_numpy(float)
    t = df.index
    tr = trades.copy()
    p0, p1 = o[tr.i.to_numpy()], o[np.minimum(tr.j.to_numpy(), len(o) - 1)]
    tr["bruto"] = tr.d * (p1 / p0 - 1)
    dur = (t[np.minimum(tr.j.to_numpy(), len(o) - 1)] - t[tr.i.to_numpy()]).total_seconds().to_numpy()
    tr["funding"] = FUND[ativo] * mult_fund / 100 * dur / ANO * tr.d
    tr["ret"] = tr.bruto - 2 * CUSTO[ativo] * mult_custo - tr.funding
    tr["t_in"], tr["t_out"] = t[tr.i.to_numpy()], t[np.minimum(tr.j.to_numpy(), len(o) - 1)]
    return tr


def diario(df: pd.DataFrame, trades: pd.DataFrame) -> pd.Series:
    """Retorno diário da estratégia contando os dias parados (custo/funding no dia da saída)."""
    idx = pd.date_range(df.index[0].normalize(), df.index[-1].normalize(), freq="D", tz=df.index.tz)
    r = pd.Series(0.0, index=idx)
    lr = np.log1p(trades.ret.clip(lower=-0.999))
    for d, x in zip(trades.t_out.dt.normalize(), lr):
        r[d] += x
    return np.expm1(r)


# ---------------------------------------------------------------- métricas
def metricas(r: pd.Series, periodos: int = 365) -> dict:
    r = r.dropna()
    eq = np.exp(np.log1p(r).cumsum())
    anos = max((r.index[-1] - r.index[0]).days / 365.25, 1e-9)
    sd = r.std()
    return {"cagr": float(eq.iloc[-1] ** (1 / anos) - 1),
            "sharpe": float(r.mean() / sd * math.sqrt(periodos)) if sd > 0 else float("nan"),
            "dd": float((eq / eq.cummax() - 1).min()),
            "exposicao_dias": float((r != 0).mean())}


def buy_hold_diario(df: pd.DataFrame, ativo: str) -> pd.Series:
    """B&H com funding do long, para comparar Sharpe do mesmo jeito."""
    c = df["close"].resample("1D").last().dropna()
    return c.pct_change().fillna(0.0) - FUND[ativo] / 100 / 365


# ---------------------------------------------------------------- controle aleatório
def controle_aleatorio(df: pd.DataFrame, trades: pd.DataFrame, ativo: str, nsim: int = 20,
                       semente: int = 0) -> dict:
    """Cada trade real vs nsim entradas sorteadas com o MESMO lado e a MESMA duração.

    t = (média real - média controle) / erro padrão da diferença pareada por trade.
    """
    rng = np.random.default_rng(semente)
    n = len(df)
    dur = (trades.j - trades.i).to_numpy()
    sims = np.empty((len(trades), nsim))
    for k in range(nsim):
        ini = rng.integers(1, np.maximum(n - dur, 2))
        fake = pd.DataFrame({"i": ini, "j": ini + dur, "d": trades.d.to_numpy()})
        sims[:, k] = retornos_trades(df, fake, ativo).ret.to_numpy()
    real = trades.ret.to_numpy()
    dif = real - sims.mean(1)
    t = dif.mean() / (dif.std(ddof=1) / math.sqrt(len(dif))) if len(dif) > 2 and dif.std() > 0 else float("nan")
    return {"media_real": float(real.mean()), "media_controle": float(sims.mean()), "t": float(t)}


# ---------------------------------------------------------------- divisões
def tercos(df: pd.DataFrame) -> list[pd.DataFrame]:
    a, b = len(df) // 3, 2 * len(df) // 3
    return [df.iloc[:a], df.iloc[a:b], df.iloc[b:]]


def treino_validacao_oos(df: pd.DataFrame, partes: int = 10, treino: int = 6, validacao: int = 3):
    """Período em `partes` iguais: as primeiras `treino` partes, depois `validacao`, o resto é OOS
    (não ler até a decisão final). Devolve (treino, validação, oos, lista das partes do treino)."""
    cortes = np.linspace(0, len(df), partes + 1).astype(int)
    p = [df.iloc[cortes[k]:cortes[k + 1]] for k in range(partes)]
    return (df.iloc[:cortes[treino]], df.iloc[cortes[treino]:cortes[treino + validacao]],
            df.iloc[cortes[treino + validacao]:], p[:treino])


def filtro_treino(rets_partes: list[float], n_trades: int, min_pos: int = 4, min_trades: int = 100) -> bool:
    """Exclusão só no treino: geo > 0, ≥ min_pos partes positivas, positiva sem a melhor parte,
    e ≥ min_trades trades."""
    lr = np.log1p(np.asarray(rets_partes))
    return bool(lr.sum() > 0 and (lr > 0).sum() >= min_pos and lr.sum() - lr.max() > 0
                and n_trades >= min_trades)


# ---------------------------------------------------------------- critério de aprovação
def aprovar(estrategia, df: pd.DataFrame, ativo: str, nsim: int = 20) -> dict:
    """Critério do projeto (desde 2026-10-23). `estrategia(df) -> trades` (i, j, d).

    1. líquido positivo em cada terço (cada terço rodado de novo);
    2. Sharpe diário >= Sharpe do B&H (com funding);
    3. t > 2 contra o controle aleatório de mesmo lado/duração;
    4. positivo com spread 4x e com funding zerado (dois cenários);
    5. (fora daqui) correlação < 0,5 com as já aprovadas: `filtro_correlacao`.
    """
    tr = retornos_trades(df, estrategia(df), ativo)
    m = metricas(diario(df, tr))
    mb = metricas(buy_hold_diario(df, ativo))
    ter = []
    for p in tercos(df):
        tp = retornos_trades(p, estrategia(p), ativo)
        ter.append(metricas(diario(p, tp))["cagr"] if len(tp) else float("nan"))
    ctl = controle_aleatorio(df, tr, ativo, nsim)
    c4 = metricas(diario(df, retornos_trades(df, tr[["i", "j", "d"]], ativo, mult_custo=4)))["cagr"]
    f0 = metricas(diario(df, retornos_trades(df, tr[["i", "j", "d"]], ativo, mult_fund=0)))["cagr"]
    itens = {"tercos_positivos": all(x > 0 for x in ter), "sharpe_vs_bh": m["sharpe"] >= mb["sharpe"],
             "t_controle": ctl["t"] > 2, "custo4x_positivo": c4 > 0, "sem_funding_positivo": f0 > 0}
    return {"ativo": ativo, "trades": len(tr), **m, "cagr_bh": mb["cagr"], "sharpe_bh": mb["sharpe"],
            "tercos": ter, "t": ctl["t"], "cagr_custo4x": c4, "cagr_sem_funding": f0,
            "itens": itens, "aprovada": all(itens.values())}


def filtro_correlacao(diarios: dict[str, pd.Series], limite: float = 0.5) -> list[str]:
    """Entra por ordem de Sharpe; recusa quem tem correlação >= limite com alguma já aceita."""
    df = pd.DataFrame(diarios).fillna(0.0)
    ordem = sorted(df, key=lambda c: -(df[c].mean() / df[c].std() if df[c].std() > 0 else -1e9))
    aceitas: list[str] = []
    for c in ordem:
        if all(abs(df[c].corr(df[a])) < limite for a in aceitas):
            aceitas.append(c)
    return aceitas


# ---------------------------------------------------------------- correção por múltiplos testes
def holm(pvalores: list[float], alfa: float = 0.05) -> list[bool]:
    p = np.asarray(pvalores)
    ordem, m, rej = np.argsort(p), len(p), np.zeros(len(p), bool)
    for k, idx in enumerate(ordem):
        if p[idx] > alfa / (m - k):
            break
        rej[idx] = True
    return rej.tolist()


def benjamini_hochberg(pvalores: list[float], q: float = 0.05) -> list[bool]:
    p = np.asarray(pvalores)
    m, ordem = len(p), np.argsort(p)
    ok = p[ordem] <= q * np.arange(1, m + 1) / m
    k = np.max(np.nonzero(ok)[0]) + 1 if ok.any() else 0
    rej = np.zeros(m, bool)
    rej[ordem[:k]] = True
    return rej.tolist()


def p_de_t(t: float) -> float:
    """p unilateral aproximado (normal) para o t do controle."""
    return 0.5 * math.erfc(t / math.sqrt(2))


# ---------------------------------------------------------------- carteira
def carteira(diarios: dict[str, pd.Series], pesos: str = "igual", teto: float | None = None) -> pd.Series:
    """Retorno diário da carteira, rebalanceada diariamente. pesos: 'igual' ou 'inv_vol'
    (vol medida até o dia anterior, janela 90d). teto: peso máximo por estratégia."""
    df = pd.DataFrame(diarios).fillna(0.0)
    if pesos == "inv_vol":
        w = 1 / df.rolling(90, min_periods=20).std().shift(1)
        w = w.replace([np.inf], np.nan).fillna(1.0)
    else:
        w = pd.DataFrame(1.0, index=df.index, columns=df.columns)
    w = w.div(w.sum(axis=1), axis=0)
    if teto:
        for _ in range(10):
            w = w.clip(upper=teto)
            w = w.div(w.sum(axis=1), axis=0)
    return (w * df).sum(axis=1)


def _demo():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2020-01-01", periods=24 * 365 * 4, freq="h", tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.006, len(idx))))
    df = pd.DataFrame({"open": np.r_[c[0], c[:-1]], "close": c}, index=idx)
    df["high"], df["low"] = df[["open", "close"]].max(axis=1), df[["open", "close"]].min(axis=1)

    def cruzamento(d):
        f, s = d.close.rolling(24).mean(), d.close.rolling(96).mean()
        sinal = np.sign(f - s).fillna(0).to_numpy()
        trocas = np.nonzero(np.diff(sinal))[0] + 1
        ii = trocas + 1
        jj = np.r_[ii[1:], len(d) - 1]
        t = pd.DataFrame({"i": ii, "j": jj, "d": sinal[trocas]})
        return t[(t.i < t.j) & (t.d != 0)].reset_index(drop=True)

    for a in ATIVOS:
        r = aprovar(cruzamento, df, a, nsim=5)
        print(f"{a:6} CAGR {r['cagr']:+6.1%} (B&H {r['cagr_bh']:+6.1%})  Sharpe {r['sharpe']:+.2f}  "
              f"t {r['t']:+.2f}  aprovada={r['aprovada']}")


if __name__ == "__main__":
    _demo()
