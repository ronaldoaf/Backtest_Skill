"""Motor vetorizado de backtest e ferramentas dos passos 3, 4, 6, 7, 9 e de dimensionamento.

Dependência: apenas numpy (e `estatistica.py` e `comparacao.py`, na mesma pasta). Aceita
objetos do pandas sem importá-lo; quando os preços vêm como Series, as séries devolvidas
também são Series com as mesmas datas.

Faz o que a skill pede em texto, para que o código de cada backtest não reinvente (e não erre)
as mesmas partes:
- `retornos_estrategia`: retorno líquido pela convenção de timing de
  `references/dados-e-lookahead.md` (abertura de t+1, fechamento de t+1 ou fechamento de t),
  custos por unidade de giro, caixa rendendo CDI e a lista de trades (para `metricas_trades` e
  `monte_carlo_drawdown`).
- `sensibilidade_custos`: tabela de custos x0, x1, x2, x3 e custo de breakeven (passo 4).
- `custo_por_lado_b3`, `simular_ir`, `imposto_renda_mensal`: custos e imposto de renda no
  mercado brasileiro (ver `references/custos.md`). Valores de referência: confira os vigentes.
- `walk_forward_parametros`: otimiza na janela k e testa na k+1, devolvendo a série fora da
  amostra concatenada, a eficiência, os parâmetros escolhidos e a matriz de todas as
  tentativas (para DSR, PBO e permutação com seleção) (passo 6).
- `robustez_parametros` / `mapa_calor`: grade em torno dos parâmetros escolhidos e indicadores
  de platô ou pico isolado (passo 7).
- `quebrar_por`, `por_ano`, `regime_tendencia`, `regime_volatilidade`: resultado por regime e
  por ano (passo 9).
- `teste_atraso`, `teste_futuro_truncado`: testes práticos de look-ahead (passo 3).
- `posicao_vol_alvo`, `kelly_estrategia`: dimensionamento de uma estratégia.

Convenções:
- `sinal[t]` é calculado com dados até o fechamento de t, inclusive, e vale -1, 0, 1 ou a
  fração do capital (ex.: 0.5, ou 1.5 com alavancagem). NaN no sinal (aquecimento de
  indicadores) vira 0, sem posição.
- `close`, `abertura`: preços do mesmo ativo nas mesmas datas, sem NaN.
- `rf_por_periodo`: CDI por período (número ou série alinhada), como em `comparacao.py`.
- `periodos_por_ano`: 252 diário, 365 cripto, 52 semanal, 12 mensal.

Uso rápido:
    from backtest import retornos_estrategia, sensibilidade_custos, walk_forward_parametros

    sinal = (close.rolling(20).mean() > close.rolling(100).mean()).astype(float)
    res = retornos_estrategia(sinal, close, abertura=open_, custo_por_lado=0.0005,
                              rf_por_periodo=cdi)
    res["retorno"]            # retorno líquido por período (Series, se close for Series)
    res["trades"]["retorno"]  # retorno de cada trade sobre o capital

Execute `python backtest.py` para uma demonstração com dados sintéticos e
`python exemplo_completo.py` para o roteiro de ponta a ponta.
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comparacao import _indice, _rf  # noqa: E402
from estatistica import _vetor, cagr, max_drawdown, sharpe  # noqa: E402

EXECUCOES = ("abertura_t1", "fechamento_t1", "fechamento_t")
CAIXA = ("livre", "total", "nenhum")

# Tarifas da B3 (negociação + liquidação) por lado, pessoa física, faixa de menor volume.
# Valores de referência: confira a tabela vigente em b3.com.br antes de usar.
TARIFAS_B3 = {
    "acoes_swing": 0.00030,      # 0,005% negociação + 0,025% liquidação
    "acoes_day_trade": 0.00023,  # 0,005% negociação + 0,018% liquidação
}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _atrasar(x: np.ndarray, k: int) -> np.ndarray:
    """x deslocado k períodos para a frente (o valor de t passa a valer em t + k), com zeros."""
    if k <= 0:
        return x.copy()
    return np.r_[np.zeros(min(k, len(x))), x[:-k]] if len(x) > k else np.zeros(len(x))


def _serie(valores: np.ndarray, indice):
    """Devolve Series do pandas com `indice` quando houver, senão o próprio array."""
    if indice is None:
        return valores
    import pandas as pd  # só chega aqui se a entrada já era do pandas
    return pd.Series(valores, index=indice)


def _sinal(sinal) -> np.ndarray:
    s = np.asarray(sinal, dtype=float).ravel()
    if np.isinf(s).any():
        raise ValueError("sinal tem valores infinitos.")
    return np.nan_to_num(s, nan=0.0)


def _excesso(r: np.ndarray, rf_por_periodo, indice, periodos_por_ano: int) -> np.ndarray:
    return r - _rf(rf_por_periodo, len(r), indice, periodos_por_ano)


def gerar_grade(parametros: dict) -> list[dict]:
    """Produto cartesiano: {"rapida": [10, 20], "lenta": [50, 100]} -> lista de 4 dicts."""
    nomes = list(parametros)
    return [dict(zip(nomes, valores)) for valores in itertools.product(*parametros.values())]


# ---------------------------------------------------------------------------
# Passo 3/4: retorno da estratégia pela convenção de timing
# ---------------------------------------------------------------------------

def retornos_estrategia(sinal, close, abertura=None, execucao: str | None = None,
                        custo_por_lado: float = 0.0, rf_por_periodo=0.0, caixa: str = "livre",
                        periodos_por_ano: int = 252, calcular_trades: bool = True) -> dict:
    """Retorno líquido de uma estratégia pela convenção de timing de `dados-e-lookahead.md`.

    execucao:
    - "abertura_t1" (padrão quando `abertura` é passada): a ordem do sinal de t executa na
      abertura de t+1; a posição antiga ainda carrega o gap overnight.
    - "fechamento_t1" (padrão sem `abertura`): executa no fechamento de t+1.
    - "fechamento_t": executa no próprio fechamento do sinal. Só vale se o sinal puder ser
      calculado antes do leilão de fechamento; diga ao usuário que essa premissa foi usada.
    custo_por_lado: fração cobrada por unidade de giro (0.0005 = 5 bps), no período em que a
      posição muda; inclui corretagem, tarifas, metade do spread e slippage (`custo_por_lado_b3`).
    caixa: o que o capital rende além da posição, com `rf_por_periodo` (CDI):
    - "livre" (padrão): a parte não usada, (1 − |posição|), rende CDI; acima de 100% de exposição
      paga CDI. Ações compradas à vista.
    - "total": o capital inteiro rende CDI, porque a posição não consome caixa (futuros, ou
      long-short com margem aplicada). Nesse caso a posição é o nocional sobre o capital.
    - "nenhum": caixa parado não rende nada (raramente realista no Brasil).

    Retorna `retorno` (líquido), `retorno_bruto` (só a posição), `custos`, `rf_caixa`,
    `posicao` (em vigor no período, a que paga o custo), `giro`, `giro_anual`, `exposicao`
    (fração dos períodos posicionado) e `trades` (dict de arrays: `retorno` sobre o capital no
    início do trade, líquido de custos e sem o CDI do caixa; `inicio`, `fim`, `lado`, `duracao`,
    `aberto`). Nas convenções de fechamento, `fim` é o período em que a saída paga custo; na
    abertura, o período cujo gap ainda pertence ao trade.
    """
    if execucao is None:
        execucao = "abertura_t1" if abertura is not None else "fechamento_t1"
    if execucao not in EXECUCOES:
        raise ValueError(f"execucao deve ser uma de {EXECUCOES}; recebeu {execucao!r}.")
    if caixa not in CAIXA:
        raise ValueError(f"caixa deve ser um de {CAIXA}; recebeu {caixa!r}.")
    if not 0 <= custo_por_lado < 0.05:
        raise ValueError(f"custo_por_lado é uma fração por lado (0.0005 = 5 bps); recebeu {custo_por_lado}.")
    indice = _indice(close)
    c = _vetor(close, "close")
    if (c <= 0).any():
        raise ValueError("close tem preços zero ou negativos.")
    s = _sinal(sinal)
    if len(s) != len(c):
        raise ValueError(f"sinal ({len(s)}) e close ({len(c)}) têm tamanhos diferentes.")
    T = len(c)
    r_close = np.r_[0.0, c[1:] / c[:-1] - 1.0]

    if execucao == "abertura_t1":
        if abertura is None:
            raise ValueError("execucao='abertura_t1' exige a série de abertura.")
        o = _vetor(abertura, "abertura")
        if len(o) != T:
            raise ValueError(f"abertura ({len(o)}) e close ({T}) têm tamanhos diferentes.")
        posicao = _atrasar(s, 1)                     # vale da abertura ao fechamento de t
        p_antiga = _atrasar(s, 2)                    # carrega o gap do fechamento de t-1 à abertura de t
        gap = np.r_[0.0, o[1:] / c[:-1] - 1.0]
        intradia = c / o - 1.0
        comp_gap, comp_pos = p_antiga * gap, posicao * intradia
        bruto = (1 + comp_gap) * (1 + comp_pos) - 1
    else:
        posicao = _atrasar(s, 2 if execucao == "fechamento_t1" else 1)
        comp_gap, comp_pos = np.zeros(T), posicao * r_close
        bruto = comp_pos

    giro = np.abs(np.diff(posicao, prepend=0.0))
    custos = custo_por_lado * giro
    rf = _rf(rf_por_periodo, T, indice, periodos_por_ano)
    if caixa == "livre":
        rf_caixa = rf * (1 - np.abs(posicao))
    elif caixa == "total":
        rf_caixa = rf.copy()
    else:
        rf_caixa = np.zeros(T)
    liquido = bruto - custos + rf_caixa

    res = {
        "retorno": _serie(liquido, indice),
        "retorno_bruto": _serie(bruto, indice),
        "custos": _serie(custos, indice),
        "rf_caixa": _serie(rf_caixa, indice),
        "posicao": _serie(posicao, indice),
        "giro": _serie(giro, indice),
        "giro_anual": float(giro.mean() * periodos_por_ano),
        "exposicao": float(np.mean(posicao != 0)),
        "execucao": execucao,
        "caixa": caixa,
        "sinal": _serie(s, indice),
    }
    if calcular_trades:
        res["trades"] = _trades(posicao, comp_gap, comp_pos, custo_por_lado)
        res["n_trades"] = int(len(res["trades"]["retorno"]))
    return res


def _trades(P: np.ndarray, comp_gap: np.ndarray, comp_pos: np.ndarray, custo: float) -> dict:
    """Agrupa os períodos em trades: trechos com posição de mesmo sinal (mudar o tamanho não fecha).

    O gap do período t pertence à posição de t-1 (abertura de t+1); a saída paga o custo da
    posição antiga e a entrada o da nova (uma virada divide o custo entre os dois trades).
    """
    fator, inicio, fim, lado, dur = [], [], [], [], []
    atual = None
    for t in range(len(P)):
        ant = P[t - 1] if t else 0.0
        agora = P[t]
        if atual is not None:
            fator[atual] *= 1 + comp_gap[t]
        if ant != 0 and agora != 0 and np.sign(ant) == np.sign(agora):
            fator[atual] *= (1 - custo * abs(agora - ant)) * (1 + comp_pos[t])
            fim[atual] = t
            dur[atual] += 1
            continue
        if atual is not None:
            fator[atual] *= 1 - custo * abs(ant)
            fim[atual] = t
            atual = None
        if agora != 0:
            fator.append((1 - custo * abs(agora)) * (1 + comp_pos[t]))
            inicio.append(t)
            fim.append(t)
            lado.append(int(np.sign(agora)))
            dur.append(1)
            atual = len(fator) - 1
    aberto = np.zeros(len(fator), dtype=bool)
    if atual is not None:
        aberto[atual] = True
    return {
        "retorno": np.asarray(fator, dtype=float) - 1.0,
        "inicio": np.asarray(inicio, dtype=int),
        "fim": np.asarray(fim, dtype=int),
        "lado": np.asarray(lado, dtype=int),
        "duracao": np.asarray(dur, dtype=int),
        "aberto": aberto,
    }


# ---------------------------------------------------------------------------
# Passo 4: custos
# ---------------------------------------------------------------------------

def sensibilidade_custos(sinal, close, abertura=None, execucao: str | None = None,
                         custo_por_lado: float = 0.0005, multiplicadores=(0, 1, 2, 3),
                         rf_por_periodo=0.0, caixa: str = "livre",
                         periodos_por_ano: int = 252) -> dict:
    """Tabela de custos (x0 só como referência, x1 base, x2, x3) e custo de breakeven.

    `custo_breakeven` é o custo por lado em que o retorno médio em excesso de `rf_por_periodo`
    vira zero (sem CDI, o retorno líquido zero). `margem` = breakeven / custo base; abaixo de 2,
    a estratégia é frágil (`fragil=True`). Sharpe em excesso do CDI.
    """
    base = retornos_estrategia(sinal, close, abertura, execucao, 0.0, rf_por_periodo, caixa,
                               periodos_por_ano, calcular_trades=False)
    indice = _indice(close)
    rf = _rf(rf_por_periodo, len(np.asarray(base["giro"])), indice, periodos_por_ano)
    bruto = np.asarray(base["retorno"])          # sem custo, com o CDI do caixa
    giro = np.asarray(base["giro"])
    cenarios = []
    for m in multiplicadores:
        c = custo_por_lado * m
        liq = bruto - c * giro
        cenarios.append({
            "multiplicador": m,
            "custo_por_lado": c,
            "sharpe": sharpe(liq - rf, periodos_por_ano),
            "cagr": cagr(liq, periodos_por_ano),
            "max_drawdown": max_drawdown(liq),
        })
    giro_medio = giro.mean()
    be = float((bruto - rf).mean() / giro_medio) if giro_medio > 0 else float("inf")
    margem = be / custo_por_lado if custo_por_lado > 0 else float("inf")
    return {
        "cenarios": cenarios,
        "custo_breakeven": be,
        "margem": margem,
        "fragil": bool(margem < 2),
        "giro_anual": float(giro_medio * periodos_por_ano),
    }


def formatar_sensibilidade(res: dict) -> str:
    linhas = ["| Cenário | Custo por lado (bps) | Sharpe | CAGR | Max DD |", "|---|---|---|---|---|"]
    nomes = {0: "Sem custos (só referência)", 1: "Base"}
    for c in res["cenarios"]:
        nome = nomes.get(c["multiplicador"], f"Custos x{c['multiplicador']:g}")
        linhas.append(f"| {nome} | {c['custo_por_lado'] * 1e4:.1f} | {c['sharpe']:.2f} | "
                      f"{c['cagr']:.1%} | {c['max_drawdown']:.1%} |")
    be = res["custo_breakeven"]
    linhas += ["", f"Custo de breakeven: {be * 1e4:.1f} bps por lado ({res['margem']:.1f}x o custo base)"
               f"{'. FRÁGIL: menos de 2x o custo base.' if res['fragil'] else '.'}"
               f" Giro anual: {res['giro_anual']:.1f}x."]
    return "\n".join(linhas)


def custo_por_lado_b3(mercado: str = "acoes_swing", corretagem_por_ordem: float = 0.0,
                      valor_medio_ordem: float | None = None, spread_bps: float = 0.0,
                      slippage_bps: float = 0.0, tarifa: float | None = None) -> dict:
    """Custo por lado (fração do valor negociado) para ações na B3.

    Soma a tarifa da B3 (`TARIFAS_B3`, ou `tarifa` para sobrescrever), a corretagem fixa
    dividida pelo valor médio da ordem, metade do spread e o slippage. Os valores de
    `TARIFAS_B3` são de referência: confira a tabela vigente da B3 e da corretora. Futuros
    (WIN, WDO) cobram por contrato: use `tarifa` = custo por contrato / valor nocional do
    contrato (pontos × R$ por ponto: R$ 0,20 no WIN, R$ 10 no WDO).
    """
    if tarifa is None:
        if mercado not in TARIFAS_B3:
            raise ValueError(f"mercado deve ser um de {tuple(TARIFAS_B3)} ou passe `tarifa`.")
        tarifa = TARIFAS_B3[mercado]
    if corretagem_por_ordem and not valor_medio_ordem:
        raise ValueError("Com corretagem fixa por ordem, informe valor_medio_ordem (R$).")
    corretagem = corretagem_por_ordem / valor_medio_ordem if corretagem_por_ordem else 0.0
    comp = {
        "tarifa_b3": float(tarifa),
        "corretagem": float(corretagem),
        "meio_spread": spread_bps / 2 / 1e4,
        "slippage": slippage_bps / 1e4,
    }
    comp["total"] = float(sum(comp.values()))
    return comp


# ---------------------------------------------------------------------------
# Imposto de renda (Brasil)
# ---------------------------------------------------------------------------

def imposto_renda_mensal(lucro_por_mes, aliquota: float = 0.15, vendas_por_mes=None,
                         isencao_vendas: float | None = None, prejuizo_inicial: float = 0.0,
                         darf_minimo: float = 10.0) -> dict:
    """Imposto de renda mês a mês, com compensação de prejuízo, isenção e DARF mínimo (R$).

    lucro_por_mes: resultado realizado (vendas fechadas) de cada mês, em R$, de UMA categoria
      (swing trade e day trade se compensam só dentro da própria categoria: rode uma vez para
      cada). Regras de referência: ações swing 15% com isenção para vendas de ações à vista até
      R$ 20 mil no mês (`isencao_vendas=20000`, com `vendas_por_mes`); day trade 20% sem
      isenção; ETFs e BDRs 15% sem isenção; FIIs 20% sem isenção. Confira a legislação vigente.
    Em mês isento, o lucro não paga e não consome prejuízo acumulado; prejuízo sempre acumula.
    Imposto abaixo de `darf_minimo` passa para o mês seguinte. O IRRF ("dedo-duro") é
    antecipação do mesmo imposto e não muda o total; ignore-o no backtest.
    """
    lucros = np.asarray(lucro_por_mes, dtype=float).ravel()
    vendas = None if vendas_por_mes is None else np.asarray(vendas_por_mes, dtype=float).ravel()
    if vendas is not None and len(vendas) != len(lucros):
        raise ValueError("vendas_por_mes e lucro_por_mes têm tamanhos diferentes.")
    if isencao_vendas is not None and vendas is None:
        raise ValueError("A isenção depende do total vendido no mês: passe vendas_por_mes.")
    prejuizo, pendente = float(prejuizo_inicial), 0.0
    imposto, acumulado, isento = [], [], []
    for i, lucro in enumerate(lucros):
        mes_isento = isencao_vendas is not None and vendas[i] <= isencao_vendas
        devido = 0.0
        if lucro < 0:
            prejuizo += -lucro
        elif lucro > 0 and not mes_isento:
            usado = min(prejuizo, lucro)
            prejuizo -= usado
            devido = (lucro - usado) * aliquota
        devido += pendente
        if 0 < devido < darf_minimo:
            pendente, devido = devido, 0.0
        else:
            pendente = 0.0
        imposto.append(devido)
        acumulado.append(prejuizo)
        isento.append(bool(mes_isento))
    return {"imposto": np.asarray(imposto), "prejuizo_acumulado": np.asarray(acumulado),
            "isento": np.asarray(isento), "total": float(np.sum(imposto)),
            "pendente": pendente}


def simular_ir(resultado: dict, meses, capital_inicial: float = 100_000.0,
               aliquota: float = 0.15, isencao_vendas: float | None = None,
               darf_minimo: float = 10.0) -> dict:
    """Aplica o imposto sobre os trades fechados de `retornos_estrategia` e devolve os retornos após IR.

    meses: rótulo do mês de cada período (ex.: `close.index.to_period("M")`), mesmo tamanho.
    O lucro de cada trade (retorno × capital no início dele) é realizado no mês do fim do
    trade; o imposto do mês sai do capital no último período do mês (na prática, o DARF vence
    no fim do mês seguinte: a aproximação é levemente conservadora). Trades ainda abertos no fim
    não pagam. O CDI do caixa tributado na fonte não entra aqui. Com `isencao_vendas`, as
    vendas do mês são estimadas pelo valor da posição comprada na saída (e na entrada das
    vendidas): use só para ações à vista em swing trade.
    """
    r = _vetor(resultado["retorno"], "retorno", minimo=1)
    m = np.asarray(meses).ravel()
    if len(m) != len(r):
        raise ValueError(f"meses ({len(m)}) e retornos ({len(r)}) têm tamanhos diferentes.")
    tr = resultado.get("trades")
    if tr is None:
        raise ValueError("Rode retornos_estrategia com calcular_trades=True.")
    P = np.abs(np.asarray(resultado["posicao"], dtype=float))
    fecha = {}
    for k in range(len(tr["retorno"])):
        if not tr["aberto"][k]:
            fecha.setdefault(int(tr["fim"][k]), []).append(k)
    capital = float(capital_inicial)
    cap_inicio = {}
    lucro_mes = vendas_mes = 0.0
    lucros, vendas, fim_mes = [], [], []
    caps = np.empty(len(r))
    for t in range(len(r)):
        for k in np.flatnonzero(tr["inicio"] == t):
            cap_inicio[int(k)] = capital
        capital *= 1 + r[t]
        for k in fecha.get(t, []):
            ganho = tr["retorno"][k] * cap_inicio[k]
            lucro_mes += ganho
            tamanho = P[tr["inicio"][k]] * cap_inicio[k]
            vendas_mes += tamanho + ganho if tr["lado"][k] > 0 else tamanho
        if t == len(r) - 1 or m[t + 1] != m[t]:
            lucros.append(lucro_mes)
            vendas.append(vendas_mes)
            fim_mes.append(t)
            lucro_mes = vendas_mes = 0.0
            ir = imposto_renda_mensal(lucros, aliquota, vendas if isencao_vendas else None,
                                      isencao_vendas, darf_minimo=darf_minimo)
            capital -= ir["imposto"][-1]
        caps[t] = capital
    anterior = np.r_[capital_inicial, caps[:-1]]
    apos = caps / anterior - 1
    total = float(ir["imposto"].sum())
    lucro_bruto = float(np.sum(np.maximum(lucros, 0)))
    return {
        "retorno": _serie(apos, _indice(resultado["retorno"])),
        "imposto_por_mes": ir["imposto"],
        "lucro_por_mes": np.asarray(lucros),
        "meses": m[fim_mes],
        "imposto_total": total,
        "prejuizo_a_compensar": float(ir["prejuizo_acumulado"][-1]),
        "aliquota_efetiva_sobre_lucros": total / lucro_bruto if lucro_bruto > 0 else float("nan"),
        "capital_final": float(caps[-1]),
    }


# ---------------------------------------------------------------------------
# Passo 6: walk-forward de parâmetros
# ---------------------------------------------------------------------------

def _rodar(funcao, cfg):
    saida = funcao(**cfg)
    if isinstance(saida, dict):
        return saida["retorno"], saida.get("posicao")
    return saida, None


def walk_forward_parametros(funcao, grade, janela_treino: int, janela_teste: int | None = None,
                            ancorada: bool = False, criterio="sharpe", rf_por_periodo=0.0,
                            periodos_por_ano: int = 252) -> dict:
    """Otimiza na janela k, testa na k+1, avança e concatena só os trechos de teste.

    funcao(**params) devolve o retorno líquido por período da configuração na amostra inteira
    (array/Series de tamanho T, ou o dict de `retornos_estrategia`). Os indicadores precisam ser
    causais (só dados até t), o que `teste_futuro_truncado` confere; assim calcular na amostra
    inteira e fatiar não vaza informação.
    grade: lista de dicts ou dict de listas (`gerar_grade`). Cada configuração é uma tentativa.
    criterio: "sharpe" (em excesso de rf, padrão), "retorno" (média) ou função(r) -> float.
    janela_teste: padrão janela_treino // 4.

    Retorna `retorno_fora` (concatenado, o resultado oficial) e `indice_fora`, `janelas` (por
    janela: períodos, parâmetros escolhidos, Sharpe no treino e no teste), `sharpe_fora`,
    `sharpe_dentro` (média dos Sharpes das escolhidas no próprio treino: otimista),
    `eficiencia` (fora / dentro; abaixo de ~0,5 indica overfitting), `fracao_janelas_positivas`,
    `trocas_de_parametros`, `parametros_atuais` (escolhidos com a última janela, para operar),
    `configuracao_por_periodo` (índice na grade da configuração em vigor em cada período fora
    da amostra, -1 no treino inicial: monta o sinal composto para trades e permutação),
    `matriz_retornos` (T x N, todas as configurações: para `pbo_cscv`), `matriz_posicoes`
    (quando a função devolve o dict; para `teste_permutacao_selecao`), `sharpes_tentativas`
    (Sharpe por período de cada configuração, para `deflated_sharpe_ratio`) e `n_tentativas`.
    """
    grade = gerar_grade(grade) if isinstance(grade, dict) else list(grade)
    if len(grade) < 1:
        raise ValueError("grade vazia.")
    saidas = [_rodar(funcao, cfg) for cfg in grade]
    indice = _indice(saidas[0][0])
    M = np.column_stack([_vetor(r, "retorno") for r, _ in saidas])
    T, N = M.shape
    pos = [p for _, p in saidas]
    Pm = np.column_stack([np.asarray(p, dtype=float) for p in pos]) if all(p is not None for p in pos) else None
    rf = _rf(rf_por_periodo, T, indice, periodos_por_ano)
    X = M - rf[:, None]
    janela_teste = janela_teste or max(1, janela_treino // 4)
    if not 2 <= janela_treino < T:
        raise ValueError(f"janela_treino deve estar entre 2 e {T - 1}; recebeu {janela_treino}.")

    def pontuar(linhas):
        if callable(criterio):
            return np.array([criterio(M[linhas, j]) for j in range(N)])
        if criterio == "retorno":
            return X[linhas].mean(0)
        if criterio != "sharpe":
            raise ValueError("criterio deve ser 'sharpe', 'retorno' ou uma função.")
        sd = X[linhas].std(0, ddof=1)
        return np.divide(X[linhas].mean(0), sd, out=np.full(N, -np.inf), where=sd > 0)

    raiz = math.sqrt(periodos_por_ano)
    janelas, fora_idx = [], []
    inicio = janela_treino
    while inicio < T:
        fim = min(inicio + janela_teste, T)
        treino = np.arange(0 if ancorada else inicio - janela_treino, inicio)
        teste = np.arange(inicio, fim)
        j = int(np.argmax(pontuar(treino)))
        sd_tr = X[treino, j].std(ddof=1)
        janelas.append({
            "inicio_teste": int(inicio), "fim_teste": int(fim),
            "data_inicio_teste": None if indice is None else indice[inicio],
            "parametros": grade[j], "indice_configuracao": j,
            "sharpe_treino": float(X[treino, j].mean() / sd_tr * raiz) if sd_tr > 0 else float("nan"),
            "sharpe_teste": sharpe(X[teste, j], periodos_por_ano) if len(teste) > 2 else float("nan"),
            "retorno_teste": float(np.prod(1 + M[teste, j]) - 1),
            "excesso_medio_teste": float(X[teste, j].mean()),
        })
        fora_idx.append((teste, j))
        inicio = fim
    retorno_fora = np.concatenate([M[t, j] for t, j in fora_idx])
    rf_fora = np.concatenate([rf[t] for t, _ in fora_idx])
    sh_fora = sharpe(retorno_fora - rf_fora, periodos_por_ano)
    sh_dentro = float(np.nanmean([w["sharpe_treino"] for w in janelas]))
    ultimo = np.arange(0 if ancorada else T - janela_treino, T)
    atual = int(np.argmax(pontuar(ultimo)))
    escolhas = [w["indice_configuracao"] for w in janelas]
    cfg_periodo = np.full(T, -1)
    for t, j in fora_idx:
        cfg_periodo[t] = j
    sd_all = X.std(0, ddof=1)
    inicio_fora = janela_treino
    return {
        "retorno_fora": _serie(retorno_fora, None if indice is None else indice[inicio_fora:]),
        "rf_fora": rf_fora,
        "janelas": janelas,
        "sharpe_fora": sh_fora,
        "sharpe_dentro": sh_dentro,
        "eficiencia": sh_fora / sh_dentro if sh_dentro > 0 else float("nan"),
        "fracao_janelas_positivas": float(np.mean([w["excesso_medio_teste"] > 0 for w in janelas])),
        "trocas_de_parametros": int(np.sum(np.diff(escolhas) != 0)),
        "parametros_atuais": grade[atual],
        "configuracao_por_periodo": cfg_periodo,
        "matriz_retornos": M,
        "matriz_posicoes": Pm,
        "sharpes_tentativas": np.divide(X.mean(0), sd_all, out=np.zeros(N), where=sd_all > 0),
        "n_tentativas": N,
        "grade": grade,
    }


# ---------------------------------------------------------------------------
# Passo 7: robustez de parâmetros
# ---------------------------------------------------------------------------

def robustez_parametros(funcao, grade: dict, escolhido: dict, rf_por_periodo=0.0,
                        periodos_por_ano: int = 252) -> dict:
    """Sharpe de cada ponto da grade e indicadores de platô em torno de `escolhido`.

    grade: dict {parâmetro: lista ordenada de valores} (±20 a 50% em torno do escolhido), que
    precisa conter os valores de `escolhido`. Vizinhos são os pontos a um passo em um único
    parâmetro. Rode na amostra em que os parâmetros foram escolhidos (treino/validação).
    Alertas: pico isolado (pior vizinho < metade do escolhido), mediana da grade < metade do
    escolhido, fração de Sharpe positivo < 70%, escolhido na borda da grade (não dá para ver se
    há platô do outro lado).
    """
    nomes = list(grade)
    valores = {n: list(grade[n]) for n in nomes}
    for n in nomes:
        if escolhido.get(n) not in valores[n]:
            raise ValueError(f"O valor escolhido de {n!r} ({escolhido.get(n)!r}) não está na grade.")
    pontos = {}
    for combo in itertools.product(*(range(len(valores[n])) for n in nomes)):
        cfg = {n: valores[n][i] for n, i in zip(nomes, combo)}
        r, _ = _rodar(funcao, cfg)
        r = _vetor(r, "retorno")
        pontos[combo] = sharpe(_excesso(r, rf_por_periodo, _indice(r), periodos_por_ano),
                               periodos_por_ano)
    centro = tuple(valores[n].index(escolhido[n]) for n in nomes)
    sh0 = pontos[centro]
    viz = []
    for k in range(len(nomes)):
        for d in (-1, 1):
            p = list(centro)
            p[k] += d
            if tuple(p) in pontos:
                viz.append(pontos[tuple(p)])
    todos = np.array([v for v in pontos.values() if np.isfinite(v)])
    borda = [n for n, i in zip(nomes, centro) if len(valores[n]) > 1 and i in (0, len(valores[n]) - 1)]
    mediana = float(np.median(todos))
    pior = float(min(viz)) if viz else float("nan")
    alertas = []
    if sh0 > 0 and viz and pior < 0.5 * sh0:
        alertas.append(f"Pico isolado: o pior vizinho tem Sharpe {pior:.2f} contra {sh0:.2f} do escolhido.")
    if sh0 > 0 and mediana < 0.5 * sh0:
        alertas.append(f"Mediana da grade ({mediana:.2f}) abaixo da metade do escolhido.")
    if np.mean(todos > 0) < 0.7:
        alertas.append(f"Só {np.mean(todos > 0):.0%} da grade tem Sharpe positivo (< 70%).")
    if borda:
        alertas.append(f"Escolhido na borda da grade em {borda}: estenda a grade para ver o platô.")
    return {
        "pontos": [({n: valores[n][i] for n, i in zip(nomes, c)}, v) for c, v in pontos.items()],
        "sharpe_escolhido": sh0,
        "mediana_grade": mediana,
        "fracao_positiva": float(np.mean(todos > 0)),
        "pior_vizinho": pior,
        "media_vizinhos": float(np.mean(viz)) if viz else float("nan"),
        "alertas": alertas,
        "_valores": valores,
        "_escolhido": dict(escolhido),
    }


def mapa_calor(res: dict, linha: str, coluna: str, casas: int = 2) -> str:
    """Tabela Markdown do Sharpe para dois parâmetros, com os demais fixos no escolhido (*)."""
    valores, esc = res["_valores"], res["_escolhido"]
    fixos = {n: v for n, v in esc.items() if n not in (linha, coluna)}
    tab = {}
    for cfg, v in res["pontos"]:
        if all(cfg[n] == fixos[n] for n in fixos):
            tab[(cfg[linha], cfg[coluna])] = v
    out = [f"| {linha} \\ {coluna} | " + " | ".join(str(c) for c in valores[coluna]) + " |",
           "|---" * (len(valores[coluna]) + 1) + "|"]
    for a in valores[linha]:
        cel = []
        for b in valores[coluna]:
            txt = f"{tab[(a, b)]:.{casas}f}"
            cel.append(f"**{txt}**" if a == esc[linha] and b == esc[coluna] else txt)
        out.append(f"| {a} | " + " | ".join(cel) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Passo 9: regimes e ano a ano
# ---------------------------------------------------------------------------

def quebrar_por(retornos, rotulos, rf_por_periodo=0.0, periodos_por_ano: int = 252) -> dict:
    """Métricas por rótulo (regime, ano, lado). Sharpe em excesso de rf."""
    r = _vetor(retornos, minimo=1)
    rot = np.asarray(rotulos).ravel()
    if len(rot) != len(r):
        raise ValueError(f"rotulos ({len(rot)}) e retornos ({len(r)}) têm tamanhos diferentes.")
    rf = _rf(rf_por_periodo, len(r), _indice(retornos), periodos_por_ano)
    saida = {}
    for g in dict.fromkeys(rot.tolist()):
        m = rot == g
        x = r[m]
        saida[g] = {
            "n_periodos": int(m.sum()),
            "fracao_do_tempo": float(m.mean()),
            "retorno_total": float(np.prod(1 + x) - 1),
            "retorno_anual": float(np.prod(1 + x) ** (periodos_por_ano / len(x)) - 1),
            "sharpe": sharpe(x - rf[m], periodos_por_ano) if len(x) > 2 else float("nan"),
            "max_drawdown": max_drawdown(x),
        }
    return saida


def por_ano(retornos, rf_por_periodo=0.0, periodos_por_ano: int = 252) -> dict:
    """quebrar_por com o ano de cada data (exige Series do pandas com DatetimeIndex)."""
    idx = _indice(retornos)
    if idx is None or not hasattr(idx, "year"):
        raise ValueError("por_ano exige uma Series com DatetimeIndex; use quebrar_por com os anos.")
    return quebrar_por(retornos, np.asarray(idx.year), rf_por_periodo, periodos_por_ano)


def formatar_quebra(quebra: dict, titulo: str = "Grupo") -> str:
    out = [f"| {titulo} | Períodos | % do tempo | Retorno anual | Sharpe | Max DD |", "|---|---|---|---|---|---|"]
    for g, m in quebra.items():
        out.append(f"| {g} | {m['n_periodos']} | {m['fracao_do_tempo']:.0%} | {m['retorno_anual']:.1%} | "
                   f"{m['sharpe']:.2f} | {m['max_drawdown']:.1%} |")
    return "\n".join(out)


def regime_tendencia(close, janela: int = 200) -> np.ndarray:
    """'alta' / 'baixa' pelo fechamento anterior acima ou abaixo da média móvel de `janela`."""
    c = _vetor(close, "close")
    acum = np.cumsum(np.r_[0.0, c])
    mm = np.full(len(c), np.nan)
    mm[janela - 1:] = (acum[janela:] - acum[:-janela]) / janela
    acima = _atrasar(np.where(np.isnan(mm), np.nan, (c > mm).astype(float)), 1)
    acima[:janela] = np.nan
    return np.where(np.isnan(acima), "aquecimento", np.where(acima == 1, "alta", "baixa"))


def regime_volatilidade(retornos_ativo, janela: int = 63) -> np.ndarray:
    """Tercis da volatilidade realizada de `janela` períodos até t-1: 'vol baixa/média/alta'.

    Os cortes dos tercis usam a amostra inteira: serve para descrever o resultado, não como sinal.
    """
    r = _vetor(retornos_ativo, "retornos_ativo")
    vol = np.full(len(r), np.nan)
    for t in range(janela, len(r)):
        vol[t] = r[t - janela:t].std(ddof=1)
    ok = ~np.isnan(vol)
    q1, q2 = np.quantile(vol[ok], [1 / 3, 2 / 3])
    rot = np.where(vol <= q1, "vol baixa", np.where(vol <= q2, "vol média", "vol alta"))
    return np.where(ok, rot, "aquecimento")


# ---------------------------------------------------------------------------
# Passo 3: testes práticos de look-ahead
# ---------------------------------------------------------------------------

def teste_atraso(sinal, close, abertura=None, execucao: str | None = None,
                 custo_por_lado: float = 0.0, atraso_extra: int = 1, rf_por_periodo=0.0,
                 caixa: str = "livre", periodos_por_ano: int = 252) -> dict:
    """Sharpe com o sinal atrasado em mais `atraso_extra` períodos.

    Queda drástica (razão < 0,3 com Sharpe original > 1) sugere dependência de informação do
    mesmo instante: investigue look-ahead antes de seguir. Estratégias muito rápidas perdem
    desempenho legitimamente com o atraso; aí a pergunta é se a execução assumida é realista.
    """
    s = _sinal(sinal)
    kw = dict(abertura=abertura, execucao=execucao, custo_por_lado=custo_por_lado,
              rf_por_periodo=rf_por_periodo, caixa=caixa, periodos_por_ano=periodos_por_ano,
              calcular_trades=False)
    rf = _rf(rf_por_periodo, len(s), _indice(close), periodos_por_ano)
    a = np.asarray(retornos_estrategia(s, close, **kw)["retorno"]) - rf
    b = np.asarray(retornos_estrategia(_atrasar(s, atraso_extra), close, **kw)["retorno"]) - rf
    sa, sb = sharpe(a, periodos_por_ano), sharpe(b, periodos_por_ano)
    razao = sb / sa if sa > 0 else float("nan")
    return {"sharpe_original": sa, "sharpe_atrasado": sb, "razao": razao,
            "alerta": bool(sa > 1 and razao < 0.3)}


def _fatiar(dados, n: int):
    if hasattr(dados, "iloc"):
        return dados.iloc[:n]
    if isinstance(dados, dict):
        return {k: _fatiar(v, n) for k, v in dados.items()}
    if isinstance(dados, tuple):
        return tuple(_fatiar(v, n) for v in dados)
    return np.asarray(dados)[:n]


def teste_futuro_truncado(funcao_sinal, dados, cortes=None, tolerancia: float = 1e-9) -> dict:
    """Recalcula o sinal com os dados cortados em cada data de `cortes` e compara com o completo.

    funcao_sinal(dados) -> sinal (array/Series do mesmo tamanho dos dados). `dados`: DataFrame,
    Series, array, dict ou tupla de séries. Sinais até o corte precisam ser idênticos; qualquer
    diferença é look-ahead (média centrada, bfill, normalização global, dado revisado).
    Padrão: cortes em 50%, 75% e 90% da amostra.
    """
    completo = np.asarray(funcao_sinal(dados), dtype=float).ravel()
    T = len(completo)
    cortes = cortes or [T // 2, (3 * T) // 4, (9 * T) // 10]
    por_corte, primeira = {}, None
    for n in cortes:
        parcial = np.asarray(funcao_sinal(_fatiar(dados, n)), dtype=float).ravel()
        if len(parcial) != n:
            raise ValueError(f"funcao_sinal devolveu {len(parcial)} valores para {n} linhas.")
        a, b = completo[:n], parcial
        iguais = (np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= tolerancia)
        dif = np.flatnonzero(~iguais)
        por_corte[n] = int(len(dif))
        if len(dif) and (primeira is None or dif[0] < primeira):
            primeira = int(dif[0])
    total = int(sum(por_corte.values()))
    return {"ok": total == 0, "divergencias_por_corte": por_corte, "primeira_divergencia": primeira}


# ---------------------------------------------------------------------------
# Dimensionamento de uma estratégia
# ---------------------------------------------------------------------------

def posicao_vol_alvo(sinal, retornos_ativo, vol_alvo: float = 0.10, janela: int = 63,
                     alavancagem_max: float = 1.0, periodos_por_ano: int = 252):
    """Sinal escalado para a volatilidade anual `vol_alvo`, com a vol estimada até t.

    retornos_ativo: retornos do ativo de fechamento a fechamento (`close.pct_change()`), sem
    defasar: o valor de t é conhecido no fechamento de t, junto com o sinal. Passe o resultado
    para `retornos_estrategia`, que aplica a convenção de timing. Antes de `janela` períodos o
    sinal fica zerado.
    """
    s = _sinal(sinal)
    r = np.nan_to_num(np.asarray(retornos_ativo, dtype=float).ravel(), nan=0.0)
    if len(r) != len(s):
        raise ValueError("sinal e retornos_ativo têm tamanhos diferentes.")
    escala = np.zeros(len(s))
    for t in range(janela, len(s)):
        v = r[t - janela + 1:t + 1].std(ddof=1) * math.sqrt(periodos_por_ano)
        escala[t] = min(vol_alvo / v, alavancagem_max) if v > 0 else 0.0
    return _serie(s * escala, _indice(sinal))


def kelly_estrategia(retornos, rf_por_periodo=0.0, fracao: float = 0.25,
                     periodos_por_ano: int = 252) -> dict:
    """Kelly de uma estratégia: alavancagem f* = média / variância do retorno em excesso do CDI.

    Devolve o Kelly completo, o IC 95% aproximado (o erro da média domina) e a alavancagem
    sugerida = `fracao` × Kelly (padrão 1/4). Use como teto, nunca o Kelly completo: se a média
    verdadeira for metade da estimada, o Kelly completo vira o dobro do ótimo.
    """
    x = _excesso(_vetor(retornos), rf_por_periodo, _indice(retornos), periodos_por_ano)
    var = x.var(ddof=1)
    if var <= 0:
        raise ValueError("Variância zero.")
    k = x.mean() / var
    ep = x.std(ddof=1) / math.sqrt(len(x)) / var
    return {"kelly_completo": float(k), "kelly_ic95": (float(k - 1.96 * ep), float(k + 1.96 * ep)),
            "alavancagem_sugerida": float(max(0.0, fracao * k)), "fracao": fracao}


# ---------------------------------------------------------------------------
# Demonstração
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    T = 252 * 6
    close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.015, T))
    abertura = close * (1 + rng.normal(0, 0.004, T))
    cdi = (1 + 0.11) ** (1 / 252) - 1

    def media_movel(x, n):
        out = np.full(len(x), np.nan)
        acum = np.cumsum(np.r_[0.0, x])
        out[n - 1:] = (acum[n:] - acum[:-n]) / n
        return out

    sinal = (media_movel(close, 20) > media_movel(close, 100)).astype(float)
    res = retornos_estrategia(sinal, close, abertura, custo_por_lado=0.0005, rf_por_periodo=cdi)
    print("Execução:", res["execucao"], "| trades:", res["n_trades"],
          "| giro anual:", round(res["giro_anual"], 1), "| exposição:", round(res["exposicao"], 2))
    print(formatar_sensibilidade(sensibilidade_custos(sinal, close, abertura, custo_por_lado=0.0005,
                                                      rf_por_periodo=cdi)))
    print("Teste do atraso:", teste_atraso(sinal, close, abertura, rf_por_periodo=cdi))
    print("Custo B3 (swing, spread 10 bps, slippage 5 bps):",
          custo_por_lado_b3(spread_bps=10, slippage_bps=5))
    print("IR mensal (lucros 1000, -3000, 5000):",
          imposto_renda_mensal([1000, -3000, 5000])["imposto"])
