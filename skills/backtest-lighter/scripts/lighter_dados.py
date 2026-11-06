"""Custos e funding reais da Lighter, e candles longos da API pública.

Duas fontes:
- VPS do usuário (gravador do livro, histórico curto desde ~out/2026): spread, custo de
  execução por US$1k/5k/20k, profundidade, funding, OI. URL em LIGHTER_VPS_URL
  (padrão http://217.217.251.30:4003), token em DATA_API_TOKEN, passado como ?token=
  (o header Authorization é bloqueado pelo proxy). Se HTTPS_PROXY existir, a chamada usa
  `curl -p -x $HTTPS_PROXY`.
- API pública da Lighter (mainnet.zklighter.elliot.ai): candles e funding horários longos.

    python lighter_dados.py custos            # tabela de custo/funding dos 5 ativos (VPS)
    python lighter_dados.py candles BTC 1h    # baixa candles da API pública -> BTC_1h.csv
    python lighter_dados.py funding XAU       # baixa funding horário -> XAU_funding.csv
"""
from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
import time

ATIVOS = ("XAU", "US100", "WTI", "BTC", "ETH")
MARKET_ID = {"ETH": 0, "BTC": 1, "PAXG": 48, "XAU": 92, "WTI": 145, "US100": 181}
API = "https://mainnet.zklighter.elliot.ai/api/v1"
VPS = os.environ.get("LIGHTER_VPS_URL", "http://217.217.251.30:4003")
RES_S = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}


def _curl(url: str, proxy: bool) -> str:
    cmd = ["curl", "-s", "--max-time", "120"]
    if proxy and os.environ.get("HTTPS_PROXY"):
        cmd += ["-p", "-x", os.environ["HTTPS_PROXY"]]
    for k in range(5):
        r = subprocess.run(cmd + [url], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout and "upstream request failed" not in r.stdout:
            return r.stdout
        time.sleep(2 * (k + 1))
    raise RuntimeError(f"falhou: {url.split('token=')[0]}")


def vps(endpoint: str, **params) -> str:
    """Chama a VPS e devolve o texto (use format='csv' para séries)."""
    token = os.environ.get("DATA_API_TOKEN")
    if not token:
        raise RuntimeError("DATA_API_TOKEN não definido")
    q = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
    return _curl(f"{VPS}/{endpoint}?{q}&token={token}", proxy=True)


def vps_csv(endpoint: str, **params) -> list[dict]:
    params.setdefault("format", "csv")
    txt = vps(endpoint, **params)
    if txt.lstrip().startswith("{"):
        raise RuntimeError(txt[:200])
    return list(csv.DictReader(io.StringIO(txt)))


def _mediana(xs: list[float]) -> float:
    xs = sorted(x for x in xs if x == x)
    if not xs:
        return float("nan")
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def _serie_dias(endpoint: str, sym: str, dias: int, every: str) -> list[dict]:
    """Busca dia a dia (consultas longas na VPS estouram o tempo do proxy)."""
    rows = []
    for k in range(dias, 0, -1):
        try:
            rows += vps_csv(endpoint, sym=sym, **{"from": f"-{k}d", "to": f"-{k - 1}d" if k > 1 else "now"},
                            every=every, limit=200)
        except RuntimeError as e:
            print(f"  aviso: {endpoint} {sym} dia -{k}: {e}", file=sys.stderr)
    return rows


def custos_reais(sym: str, dias: int = 3, every: str = "1h") -> dict:
    """Mediana do custo de UMA perna (fração do preço) e funding anual que o long paga (%).

    funding_rate da Lighter é % por hora (positivo = long paga); anual = média × 8760.
    """
    m = _serie_dias("metrics", sym, dias, every)
    s = _serie_dias("stats", sym, dias, every)
    f = lambda rows, c: [float(r[c]) for r in rows if r.get(c) not in (None, "")]
    fr = f(s, "funding_rate")
    return {
        "sym": sym,
        "amostras": len(m),
        "spread_bps": _mediana(f(m, "spread_bps")),
        "custo_1k": _mediana(f(m, "cost_1000")),
        "custo_5k": _mediana(f(m, "cost_5000")),
        "custo_20k": _mediana(f(m, "cost_20000")),
        "funding_anual_pct": (sum(fr) / len(fr) * 8760) if fr else float("nan"),
    }


def candles_publicos(sym: str, tf: str = "1h", inicio: int = 1_700_000_000) -> list[dict]:
    """Candles da API pública, paginando para trás (500 por chamada). t em ms."""
    mid, fim, d = MARKET_ID[sym], int(time.time()), {}
    while fim > inicio:
        j = json.loads(_curl(f"{API}/candles?market_id={mid}&resolution={tf}&start_timestamp={inicio}"
                             f"&end_timestamp={fim}&count_back=500", proxy=False) or "{}")
        c = j.get("c") or []
        if len(c) < 2:
            break
        for x in c:
            d[x["t"]] = x
        novo = min(x["t"] for x in c) // 1000 - 1
        if novo >= fim:
            break
        fim = novo
        time.sleep(0.6)
    return [d[t] for t in sorted(d)]


def funding_publico(sym: str, inicio: int = 1_700_000_000) -> list[dict]:
    """Funding horário: timestamp (s), rate (% por hora, sinal pela direção: + = long paga)."""
    mid, fim, d = MARKET_ID[sym], int(time.time()), {}
    while fim > inicio:
        j = json.loads(_curl(f"{API}/fundings?market_id={mid}&resolution=1h&start_timestamp={inicio}"
                             f"&end_timestamp={fim}&count_back=500", proxy=False) or "{}")
        c = j.get("fundings") or []
        if len(c) < 2:
            break
        for x in c:
            r = float(x["rate"])
            d[x["timestamp"]] = {"timestamp": x["timestamp"],
                                 "rate": r if x.get("direction", "long") == "long" else -r}
        novo = min(x["timestamp"] for x in c) - 1
        if novo >= fim:
            break
        fim = novo
        time.sleep(2)
    return [d[t] for t in sorted(d)]


def _salvar(rows: list[dict], nome: str) -> None:
    if not rows:
        print("nada baixado")
        return
    with open(nome, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{nome}: {len(rows)} linhas")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "custos"
    if cmd == "custos":
        print(f"{'ativo':6} {'spread bp':>9} {'1k bp':>7} {'5k bp':>7} {'20k bp':>7} {'funding a.a.':>12}")
        for a in ATIVOS:
            c = custos_reais(a)
            print(f"{a:6} {c['spread_bps']:9.2f} {c['custo_1k']*1e4:7.2f} {c['custo_5k']*1e4:7.2f} "
                  f"{c['custo_20k']*1e4:7.2f} {c['funding_anual_pct']:11.1f}%")
    elif cmd == "candles":
        _salvar(candles_publicos(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "1h"),
                f"{sys.argv[2]}_{sys.argv[3] if len(sys.argv) > 3 else '1h'}.csv")
    elif cmd == "funding":
        _salvar(funding_publico(sys.argv[2]), f"{sys.argv[2]}_funding.csv")
    else:
        print(__doc__)
