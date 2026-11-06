---
name: backtest-lighter
description: Protocolo de backtest do projeto Backtests Lighter. Use ao testar, aprovar ou combinar estratégias para operar na exchange Lighter em XAU (ouro), US100 (Nasdaq), WTI (petróleo), BTC e ETH, com custos e funding reais da Lighter, controle aleatório, terços, treino/validação/OOS e filtro de correlação. Complementa a skill backtest, que tem a teoria e as ferramentas estatísticas gerais.
---

# Backtest na Lighter (XAU, US100, WTI, BTC, ETH)

Este é o método que o projeto já usa, condensado para não ser reinventado a cada estudo. Para teoria, walk-forward, Deflated Sharpe, PBO, SPA e otimização de carteira, use a skill `backtest` (mesmo repositório). Aqui ficam as regras específicas da Lighter e o critério de aprovação.

## Regras do projeto (não negociáveis)

1. **Sempre os 5 ativos:** XAU, US100, WTI, BTC e ETH. Reporte uma linha por ativo, inclusive os que reprovam. (Os estudos antigos do TradingView usavam ouro primeiro e só estendiam se ele passasse; o padrão atual é testar os 5.)
2. **Templates e exemplos não se testam.** Scripts que são placeholder (ex.: "3Commas Bot", cruzamento de MA de exemplo, webhooks) são pulados com uma linha dizendo por quê.
3. **Sinal no fechamento, execução na abertura da barra seguinte.** Stops/alvos intrabar resolvidos no M1; se stop e alvo caem no mesmo minuto, o stop executa primeiro. Retornos a 1x sobre o nocional.
4. **Custo sempre ligado:** spread por perna + funding pró-rata (long paga, short recebe). Nunca reporte resultado bruto como principal.
5. **Short composto certo:** retorno do vendido é `1 - P1/P0`, não `log(P0/P1)`. Esse erro inflou o CAGR das aprovadas antigas em ~30 pontos/ano (mediana).
6. **Melhoria de infraestrutura que acelere os testes ou descarte estratégias mais cedo pode ser feita direto**; registre no relatório.

## Dados

| O quê | Onde |
| --- | --- |
| Candles M1 longos (2017/2020 → 2026) | repositório `tradingview_backtests_lighter`, pasta `data/` (`tvlib.load_m1v(nome)`): Dukascopy para XAU/US100/WTI, Binance perp para BTC/ETH |
| Candles e funding horário da própria Lighter (desde ~out/2025) | API pública: `python scripts/lighter_dados.py candles BTC 1h`, `... funding XAU` |
| Custo real de execução, spread, profundidade, funding atual, OI | VPS do usuário, `http://217.217.251.30:4003` (histórico curto, desde ~out/2026) |

**VPS:** token na variável `DATA_API_TOKEN`, sempre como `?token=` na URL (o header Authorization é bloqueado pelo proxy) e com `curl -p -x $HTTPS_PROXY`. Endpoints: `/snapshot` (estado atual, rápido), `/markets`, `/days`, `/candles`, `/metrics` (spread, `cost_1000/5000/20000` = custo de UMA perna em fração do preço, profundidade), `/stats` (mark/index, OI, `funding_rate` em % por hora, positivo = long paga), `/trades`, `/depth`. Consultas longas estouram o tempo do proxy: busque um dia por chamada, com `every=1h` e `limit` pequeno (é o que `custos_reais` faz; 3 dias × 5 ativos leva alguns minutos). Para um número rápido, use `/snapshot?sym=XAU,US100,WTI,BTC,ETH`.

IDs de mercado na API pública: ETH 0, BTC 1, PAXG 48, XAU 92, WTI 145, US100 181.

## Custos de referência

`scripts/protocolo.py` traz `CUSTO` (perna, fração) e `FUND` (% a.a. que o long paga), os mesmos de `study_engulf_psar.py`:

| Ativo | Spread/perna | Funding a.a. (long) |
| --- | --- | --- |
| XAU | 0,75 bp | 13,8% |
| US100 | 1,2 bp | 4,5% |
| WTI | 1,0 bp | 67,7% |
| BTC | 0,5 bp | 10,9% |
| ETH | 0,35 bp | 11,2% |

Medição na VPS em 2026-11-06 (3–7 dias, mediana): custo por perna US$1k ≈ XAU 0,4 bp, US100 0,6, WTI 0,6, BTC 0,2, ETH 0,3; funding ≈ XAU 5,5%, US100 4,8%, WTI −10% (virou), BTC 10,5%, ETH 7,7%. Os valores da tabela são conservadores; antes de um estudo novo rode `python scripts/lighter_dados.py custos` e, se o tamanho operado for maior, use `cost_5000`/`cost_20000`. O funding do WTI muda de sinal e de ordem de grandeza: sempre rode o cenário sem funding (item 4 abaixo) e um com funding 0,5x.

## Benchmarks

1. **Buy & hold** do ativo, com o funding do long, Sharpe diário calculado do mesmo jeito.
2. **Controle aleatório:** para cada trade real, N entradas sorteadas com o mesmo lado e a mesma duração (`controle_aleatorio`). Reporte o t da diferença. É o teste que separa sinal de beta.

## Critério de aprovação (`aprovar`)

Uma célula da grade só é aprovada se:

1. líquido **positivo em cada terço** do período (cada terço rodado de novo, não fatiado da curva);
2. **Sharpe diário ≥ Sharpe do B&H** (dias parados contam como zero);
3. **t > 2** contra o controle aleatório;
4. **positivo com spread 4x** e **positivo com funding zerado**;
5. **correlação < 0,5** com as já aprovadas, entrando por ordem de Sharpe (`filtro_correlacao`).

Conte todas as células testadas. Com centenas de células, ~2% passam o t > 2 por acaso: aplique `holm` ou `benjamini_hochberg` sobre `p_de_t(t)` e diga quantas sobrevivem. Os terços são os mesmos dados usados para escolher: a lista é candidata a paper trading, não aprovação para dinheiro real.

## Treino / validação / OOS (seleção de muitas estratégias)

Quando o estudo escolhe entre muitas estratégias ou monta carteira (`treino_validacao_oos`):

- período em 10 partes iguais: 1–6 treino, 7–9 validação, 10 OOS **não lido** até a decisão final;
- exclusão só no treino (`filtro_treino`): geo > 0, ≥ 4/6 partes positivas, positiva sem a melhor parte, ≥ 100 trades;
- melhor célula por estratégia escolhida pela média geométrica do treino;
- a validação decide K (número de estratégias), teto de peso e esquema de pesos; o OOS é lido uma vez e reportado como está.

## Carteira

`carteira(diarios, pesos="igual"|"inv_vol", teto=...)` dá o retorno diário rebalanceado. O projeto compara K = 6, 8, 10, 12 com tetos 0,2–0,5 e pesos igual vs vol inversa; escolha na validação, nunca no treino. Para HRP, risk parity, Kelly e custos de rebalanceamento, use `portfolio.py` da skill `backtest`.

## Armadilhas que já custaram tempo

- `lookahead_on` em `security()` do Pine: principal causa de curva publicada boa.
- Sempre-no-mercado não é estratégia; o benchmark é o B&H, não zero.
- Acerto alto sem stop vem da saída assimétrica; RR < 1 com acerto ~50% perde por construção (dá para reprovar lendo o código).
- "ML"/"rede neural" que são constantes cravadas.
- Funding do WTI mata long com exposição alta (ou ajuda o short): olhe antes de rodar a grade.
- `DatetimeIndex.asi8` no pandas 3 vem em milissegundos; use `as_unit` antes de aritmética.
- Indicadores do Pine (`ta.rsi`, `ta.atr`, `ta.hma`...) inicializam com Wilder/RMA; use os equivalentes de `tvlib.py`.

## Relatório

Use o modelo de `references/relatorio.md` da skill `backtest`, com estas adições: tabela com os 5 ativos (CAGR vs B&H, Sharpe vs B&H, DD, trades, t do controle, terços, spread 4x, sem funding, veredito), número total de células testadas e quantas sobrevivem à correção de múltiplos testes, e os custos usados (tabela ou medição da VPS com data).

## Arquivos

| Arquivo | Conteúdo |
| --- | --- |
| `scripts/protocolo.py` | `retornos_trades`, `diario`, `metricas`, `buy_hold_diario`, `controle_aleatorio`, `tercos`, `treino_validacao_oos`, `filtro_treino`, `aprovar`, `filtro_correlacao`, `holm`, `benjamini_hochberg`, `p_de_t`, `carteira`; `python protocolo.py` roda uma demonstração nos 5 ativos |
| `scripts/lighter_dados.py` | `vps`, `vps_csv`, `custos_reais` (VPS), `candles_publicos`, `funding_publico` (API pública) |
