"""Exemplo de ponta a ponta: os 12 passos da skill numa estratégia de médias móveis.

Dependências: numpy e pandas. Use como modelo: troque `calcular_sinal` e a grade pela sua
estratégia e mantenha o resto. Imprime um rascunho do relatório de `references/relatorio.md`.

    python exemplo_completo.py                 # dados sintéticos
    python exemplo_completo.py precos.csv      # seus dados
    python exemplo_completo.py --rapido        # menos simulações (para testes)

CSV: colunas `data`, `abertura`, `fechamento` e, opcional, `cdi` com a taxa DIÁRIA em % (o
formato da série 12 do Banco Central, ex.: 0.040168). Sem `cdi`, usa 11% ao ano constante.
Os preços devem estar ajustados por proventos (passo 2).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import (  # noqa: E402
    _atrasar, custo_por_lado_b3, formatar_quebra, formatar_sensibilidade, gerar_grade,
    kelly_estrategia, mapa_calor, por_ano, quebrar_por, regime_tendencia, regime_volatilidade,
    retornos_estrategia, robustez_parametros, sensibilidade_custos, simular_ir, teste_atraso,
    teste_futuro_truncado, walk_forward_parametros,
)
from estatistica import (  # noqa: E402
    bootstrap_metricas, deflated_sharpe_ratio, metricas_trades, monte_carlo_drawdown, pbo_cscv,
    resumo_metricas, teste_permutacao, teste_permutacao_selecao,
)

# ---------------------------------------------------------------------------
# Passo 1: hipótese e regras, definidas antes de rodar qualquer coisa
# ---------------------------------------------------------------------------
HIPOTESE = ("Tendências de médio prazo persistem porque investidores reagem devagar a "
            "notícias (subreação). Do outro lado: quem opera reversão à média cedo demais.")
GRADE = {"rapida": [10, 15, 20, 30, 40], "lenta": [60, 80, 100, 150, 200]}
JANELA_TREINO, JANELA_TESTE = 504, 126           # 2 anos de treino, 6 meses de teste
FRACAO_HOLDOUT = 0.20


def calcular_sinal(df: pd.DataFrame, rapida: int, lenta: int) -> pd.Series:
    """1 comprado quando a média rápida está acima da lenta, 0 fora. Só dados até t."""
    c = df["fechamento"]
    return (c.rolling(rapida).mean() > c.rolling(lenta).mean()).astype(float)


def dados_sinteticos(anos: int = 12, seed: int = 7) -> pd.DataFrame:
    """Preços com regimes de alta e de baixa persistentes (a vantagem que a hipótese supõe) e CDI."""
    rng = np.random.default_rng(seed)
    T = 252 * anos
    regime = np.cumsum(rng.random(T) < 1 / 250) % 2      # troca de regime a cada ~250 pregões
    r = np.where(regime == 0, 0.0015, -0.0012) + rng.normal(0.0, 0.016, T)
    fechamento = 50 * np.cumprod(1 + r)
    abertura = np.r_[50, fechamento[:-1]] * (1 + rng.normal(0, 0.005, T))
    datas = pd.bdate_range("2012-01-02", periods=T)
    cdi_anual = 0.10 + 0.03 * np.sin(np.arange(T) / 600)
    return pd.DataFrame({"abertura": abertura, "fechamento": fechamento,
                         "cdi": ((1 + cdi_anual) ** (1 / 252) - 1) * 100}, index=datas)


def carregar_csv(caminho: str) -> pd.DataFrame:
    df = pd.read_csv(caminho, parse_dates=["data"]).set_index("data").sort_index()
    faltam = {"abertura", "fechamento"} - set(df.columns)
    if faltam:
        raise ValueError(f"Faltam colunas no CSV: {faltam}.")
    if "cdi" not in df:
        df["cdi"] = ((1 + 0.11) ** (1 / 252) - 1) * 100
    return df.dropna(subset=["abertura", "fechamento"])


def main(caminho_csv: str | None = None, rapido: bool = False) -> dict:
    n_sim = 300 if rapido else 5000
    df = carregar_csv(caminho_csv) if caminho_csv else dados_sinteticos()
    cdi = df["cdi"].fillna(0) / 100                # taxa por período
    print("# Backtest: cruzamento de médias móveis\n")
    print(f"## 1. Hipótese\n{HIPOTESE}\nRegras: comprado se MM(rápida) > MM(lenta); grade {GRADE}.\n")

    # Passo 2: dados (aqui só os checks mínimos; ver dados-e-lookahead.md)
    print("## 2. Dados")
    print(f"{df.index[0]:%Y-%m-%d} a {df.index[-1]:%Y-%m-%d}, {len(df)} pregões. "
          f"Saltos > 15% sem evento: {int((df['fechamento'].pct_change().abs() > 0.15).sum())}.\n")

    # Passo 4 (custos) vem antes do walk-forward: todos os resultados já são líquidos
    custo = custo_por_lado_b3("acoes_swing", spread_bps=10, slippage_bps=5)["total"]

    # Passo 5: holdout = os 20% mais recentes, tocado uma vez no fim
    corte = int(len(df) * (1 - FRACAO_HOLDOUT))
    dev, cdi_dev = df.iloc[:corte], cdi.iloc[:corte]

    def estrategia(dados, taxa, **p):
        return retornos_estrategia(calcular_sinal(dados, **p), dados["fechamento"], dados["abertura"],
                                   custo_por_lado=custo, rf_por_periodo=taxa)

    # Passo 3: look-ahead
    p_exemplo = gerar_grade(GRADE)[0]
    trunc = teste_futuro_truncado(lambda d: calcular_sinal(d, **p_exemplo), dev)
    print("## 3. Auditoria de look-ahead")
    print(f"Execução assumida: abertura de t+1. Futuro truncado: {'ok' if trunc['ok'] else 'FALHOU'} "
          f"{trunc['divergencias_por_corte']}.")

    # Passo 6: walk-forward no período de desenvolvimento
    wf = walk_forward_parametros(lambda **p: estrategia(dev, cdi_dev, **p), GRADE,
                                 JANELA_TREINO, JANELA_TESTE, rf_por_periodo=cdi_dev)
    escolhido = wf["parametros_atuais"]
    sinal_esc = calcular_sinal(dev, **escolhido)
    atraso = teste_atraso(sinal_esc, dev["fechamento"], dev["abertura"], custo_por_lado=custo,
                          rf_por_periodo=cdi_dev)
    print(f"Atraso extra de 1 pregão: Sharpe {atraso['sharpe_original']:.2f} -> "
          f"{atraso['sharpe_atrasado']:.2f}{' (ALERTA)' if atraso['alerta'] else ''}.\n")

    print("## 4. Custos")
    print(f"Custo por lado: {custo * 1e4:.1f} bps (tarifa B3 + meio spread + slippage).\n")
    print(formatar_sensibilidade(sensibilidade_custos(sinal_esc, dev["fechamento"], dev["abertura"],
                                                      custo_por_lado=custo, rf_por_periodo=cdi_dev)))

    print("\n## 5–6. Divisão e walk-forward")
    print(f"Desenvolvimento: {dev.index[0]:%Y-%m-%d} a {dev.index[-1]:%Y-%m-%d}; holdout: "
          f"{df.index[corte]:%Y-%m-%d} em diante. Janela móvel de {JANELA_TREINO} pregões, teste de "
          f"{JANELA_TESTE}.")
    print(f"Sharpe fora {wf['sharpe_fora']:.2f}, dentro {wf['sharpe_dentro']:.2f}, eficiência "
          f"{wf['eficiencia']:.2f}; janelas positivas {wf['fracao_janelas_positivas']:.0%}; "
          f"trocas de parâmetro {wf['trocas_de_parametros']} em {len(wf['janelas'])} janelas; "
          f"parâmetros atuais {escolhido}.\n")

    # Passo 7: robustez em torno dos parâmetros escolhidos
    rob = robustez_parametros(lambda **p: estrategia(dev, cdi_dev, **p), GRADE, escolhido,
                              rf_por_periodo=cdi_dev)
    print("## 7. Robustez de parâmetros (Sharpe no desenvolvimento)\n")
    print(mapa_calor(rob, "rapida", "lenta"))
    print("\nAlertas:", rob["alertas"] or "nenhum", "\n")

    # Passo 8: testes estatísticos
    oos = wf["retorno_fora"]
    rf_oos = cdi_dev.iloc[JANELA_TREINO:]
    boot = bootstrap_metricas(oos, n_sim=n_sim, rf_por_periodo=rf_oos.to_numpy())
    # Permutação: sinal composto (a configuração em vigor em cada janela), retorno de fechamento
    # a fechamento com execução no fechamento de t+1 (aproximação da abertura de t+1).
    sinais = [calcular_sinal(dev, **cfg).to_numpy() for cfg in wf["grade"]]
    cfg_t = wf["configuracao_por_periodo"]
    composto = np.array([sinais[j][t] if j >= 0 else 0.0 for t, j in enumerate(cfg_t)])
    r_ativo = (dev["fechamento"].pct_change().fillna(0) - cdi_dev).to_numpy()
    pos = _atrasar(np.nan_to_num(composto), 2)
    perm = teste_permutacao(pos[JANELA_TREINO:], r_ativo[JANELA_TREINO:], n_perm=n_sim,
                            custo_por_giro=custo)
    P = np.column_stack([_atrasar(np.nan_to_num(s), 2) for s in sinais])
    perm_sel = teste_permutacao_selecao(P, r_ativo, n_perm=min(n_sim, 2000), custo_por_giro=custo)
    M = wf["matriz_retornos"] - cdi_dev.to_numpy()[:, None]
    j_melhor = int(np.argmax(wf["sharpes_tentativas"]))
    dsr = deflated_sharpe_ratio(M[:, j_melhor], wf["n_tentativas"], wf["sharpes_tentativas"])
    pbo = pbo_cscv(M, n_blocos=10 if rapido else 16)
    print("## 8. Testes estatísticos\n")
    print("| Teste | Resultado |\n|---|---|")
    print(f"| IC 95% Sharpe (bootstrap, fora da amostra) | {boot['sharpe_ic'][0]:.2f} a {boot['sharpe_ic'][1]:.2f} |")
    print(f"| p-valor permutação circular (fora da amostra) | {perm['p_valor']:.3f} |")
    print(f"| p-valor permutação com seleção (melhor de {wf['n_tentativas']}) | {perm_sel['p_valor']:.3f} |")
    print(f"| Nº de tentativas registradas | {wf['n_tentativas']} |")
    print(f"| Deflated Sharpe Ratio | {dsr['dsr']:.2f} |")
    print(f"| PBO | {pbo['pbo']:.2f} |\n")

    # Passo 9: regimes e ano a ano, fora da amostra
    print("## 9. Regimes e anos (fora da amostra)\n")
    tend = regime_tendencia(dev["fechamento"])[JANELA_TREINO:]
    vol = regime_volatilidade(dev["fechamento"].pct_change().fillna(0))[JANELA_TREINO:]
    print(formatar_quebra(quebrar_por(oos, tend, rf_por_periodo=rf_oos), "Tendência"), "\n")
    print(formatar_quebra(quebrar_por(oos, vol, rf_por_periodo=rf_oos), "Volatilidade"), "\n")
    print(formatar_quebra(por_ano(oos, rf_por_periodo=rf_oos), "Ano"), "\n")

    # Holdout: uma única vez, com tudo congelado (parâmetros atuais do walk-forward)
    res_full = estrategia(df, cdi, **escolhido)
    hold = res_full["retorno"].iloc[corte:]
    rf_hold = cdi.iloc[corte:]
    m_hold = resumo_metricas(hold, rf_por_periodo=rf_hold.to_numpy())
    boot_hold = bootstrap_metricas(hold, n_sim=n_sim, rf_por_periodo=rf_hold.to_numpy())

    # Passo 10: métricas
    trades = res_full["trades"]
    fora = trades["inicio"] >= JANELA_TREINO
    mt = metricas_trades(trades["retorno"][fora & ~trades["aberto"]])
    m_oos = resumo_metricas(oos, rf_por_periodo=rf_oos.to_numpy())
    bh = df["fechamento"].pct_change().fillna(0).iloc[JANELA_TREINO:corte]
    m_bh = resumo_metricas(bh, rf_por_periodo=rf_oos.to_numpy())
    print("## 10. Métricas (fora da amostra, líquidas; Sharpe em excesso do CDI)\n")
    print("| Métrica | Estratégia | Buy and hold | Holdout |\n|---|---|---|---|")
    for k in ("cagr", "volatilidade", "sharpe", "sortino", "max_drawdown", "calmar"):
        print(f"| {k} | {m_oos[k]:.3f} | {m_bh[k]:.3f} | {m_hold[k]:.3f} |")
    print(f"\nTrades fora da amostra: {mt['n_trades']}, acerto {mt['taxa_acerto']:.0%}, payoff "
          f"{mt['payoff']:.2f}. Exposição {res_full['exposicao']:.0%}, giro anual {res_full['giro_anual']:.1f}x.")
    kelly = kelly_estrategia(oos, rf_por_periodo=rf_oos.to_numpy())
    print(f"Kelly completo {kelly['kelly_completo']:.2f} (IC {kelly['kelly_ic95'][0]:.2f} a "
          f"{kelly['kelly_ic95'][1]:.2f}); 1/4 de Kelly sugere até {kelly['alavancagem_sugerida']:.0%} do capital.")
    ir = simular_ir(res_full, df.index.to_period("M"), capital_inicial=100_000, aliquota=0.15,
                    isencao_vendas=20_000)
    print(f"Imposto de renda (swing, 15%, isenção de R$ 20 mil em vendas): R$ {ir['imposto_total']:,.0f} "
          f"no período; CAGR após IR {resumo_metricas(ir['retorno'])['cagr']:.1%} contra "
          f"{resumo_metricas(res_full['retorno'])['cagr']:.1%} antes.\n")

    # Passos 11–12: limites de desligamento a partir do Monte Carlo dos trades
    mc = monte_carlo_drawdown(trades["retorno"][fora], n_sim=n_sim) if fora.sum() > 1 else None
    print("## 11–12. Incubação e desligamento")
    if mc:
        print(f"Drawdown limite (P95 Monte Carlo): {mc['max_drawdown'][95]:.1%}; sequência de perdas "
              f"limite (P99): {mc['maior_sequencia_perdas'][99]:.0f} trades.")

    # Veredito: regra simples a partir dos testes; o relatório real justifica em texto
    aprovados = sum([boot["sharpe_ic"][0] > 0, perm["p_valor"] < 0.05, perm_sel["p_valor"] < 0.05,
                     dsr["dsr"] >= 0.95, pbo["pbo"] < 0.2, boot_hold["sharpe_ic"][0] > 0,
                     not rob["alertas"]])
    veredito = ("Rejeitada" if aprovados <= 1 else "Inconclusiva" if aprovados <= 4
                else "Promissora (incubar)" if aprovados <= 6 else "Aprovada para operar com tamanho reduzido")
    print(f"\n## Veredito\n{veredito} ({aprovados} de 7 critérios). Holdout: Sharpe "
          f"{m_hold['sharpe']:.2f}, IC {boot_hold['sharpe_ic'][0]:.2f} a {boot_hold['sharpe_ic'][1]:.2f}.")
    return {"wf": wf, "robustez": rob, "bootstrap": boot, "permutacao": perm,
            "permutacao_selecao": perm_sel, "dsr": dsr, "pbo": pbo, "holdout": m_hold,
            "trunc": trunc, "veredito": veredito, "ir": ir}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else None, rapido="--rapido" in sys.argv)
