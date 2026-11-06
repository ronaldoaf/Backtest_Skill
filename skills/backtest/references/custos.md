# Custos e execução realistas (passo 4)

Muitas "vantagens" morrem aqui. Modele custos **por trade**, não como desconto anual genérico. Em `scripts/backtest.py`, `retornos_estrategia(..., custo_por_lado=...)` cobra o custo por unidade de giro no período em que a posição muda, e `sensibilidade_custos` / `formatar_sensibilidade` montam a tabela abaixo com o custo de breakeven.

## Componentes

| Componente | Como modelar | Observações |
|---|---|---|
| Corretagem e taxas | Valor fixo por ordem + % do volume (emolumentos, liquidação, registro) | Use a tabela real da corretora/bolsa do usuário |
| Spread | Metade do spread bid-ask por lado | Use spread histórico; na falta, estime pela volatilidade intradiária (ex.: estimador de Corwin-Schultz ou Roll) |
| Slippage | Ticks ou bps por lado, maior em ordens a mercado e em stops | Stops em gap executam no preço de abertura, não no stop |
| Impacto de mercado | Proporcional à participação no volume, ex.: `custo ≈ k · σ · sqrt(Q / ADV)` | Define a **capacidade** da estratégia |
| Financiamento | Juros de margem, custo de aluguel de ações (short), funding em perpétuos cripto, carrego de futuros | Cobrar por dia de posição |
| Impostos | Alíquota sobre ganho, regras de day trade vs. swing, compensação de prejuízo | Importa para comparar com buy and hold |
| Câmbio | Spread e IOF em ativos no exterior | |

## Regras de execução

- Ordens a mercado: preço da próxima barra mais slippage.
- Ordens limitadas: só considerar executada se o preço **atravessou** o limite (não apenas tocou), e aceitar que parte não executa.
- Limite de participação: não negociar mais que ~1 a 10% do volume médio da barra/dia. Acima disso, reduzir o tamanho ou modelar impacto.
- Restrições: lote mínimo, tick size, circuit breakers, ações sem aluguel disponível, limites de alavancagem.

## Análise de sensibilidade obrigatória

Rode e reporte uma tabela:

| Cenário | Custo por lado | Sharpe | CAGR | Max DD |
|---|---|---|---|---|
| Sem custos (só referência) | 0 | | | |
| Base | x bps | | | |
| Custos x2 | 2x bps | | | |
| Custos x3 | 3x bps | | | |

Calcule também o **custo de breakeven**: o custo por lado em que o retorno líquido vira zero (com CDI, em que o retorno empata com o CDI). Compare com o custo real. Se a margem for pequena (breakeven < 2x o custo base), sinalize a estratégia como frágil.

## Turnover

Reporte o giro anual. Estratégias de alto giro precisam de vantagem bruta proporcionalmente maior: `retorno líquido ≈ retorno bruto − turnover × custo por giro`.

## Mercado brasileiro

Valores de referência para pessoa física; tarifas e regras mudam, então confira a tabela vigente da B3, da corretora e da Receita antes de usar e diga ao usuário quais valores assumiu.

### Custos por operação

| Item | Referência | Na skill |
|---|---|---|
| Tarifas B3, ações à vista (swing trade) | 0,005% negociação + 0,025% liquidação = 0,030% por lado | `custo_por_lado_b3("acoes_swing")` |
| Tarifas B3, ações (day trade) | cerca de 0,023% por lado, menor com volume alto | `custo_por_lado_b3("acoes_day_trade")` |
| Corretagem | muitas corretoras cobram zero em ações; fixa por ordem pesa em ordens pequenas | `corretagem_por_ordem` e `valor_medio_ordem` |
| Spread e slippage | metade do spread por lado; ações de menor liquidez passam fácil de 10 a 30 bps | `spread_bps`, `slippage_bps` |
| Minicontratos (WIN, WDO) | tarifa por contrato; converta pelo nocional: pontos × R$ 0,20 (WIN) ou × R$ 10 (WDO) | `tarifa=` custo por contrato / nocional |
| Aluguel de ações (BTC) para vender a descoberto | taxa anual do doador mais a da B3, cobrada por dia; proventos pagos ao doador | inclua no retorno da posição vendida |

### Caixa e CDI

Com o CDI na casa de 10% a 15% ao ano, o caixa parado é a referência que importa. Em `retornos_estrategia`, `caixa="livre"` (padrão) faz a parte não alocada render CDI, `caixa="total"` faz o capital inteiro render CDI (futuros e long-short com margem aplicada) e alavancagem acima de 100% paga CDI. Sem isso, uma estratégia que fica fora do mercado metade do tempo parece pior do que é, e o Sharpe sai errado (use sempre o excesso do CDI; ver `comparacao.md`).

### Imposto de renda

| Operação | Alíquota | Isenção | Compensação de prejuízo |
|---|---|---|---|
| Ações, swing trade | 15% | vendas de ações à vista até R$ 20 mil no mês | com ganhos de swing trade em bolsa |
| Ações, day trade | 20% | nenhuma | só com day trade |
| Futuros e opções | 15% (swing) / 20% (day trade) | nenhuma | na mesma categoria |
| ETFs de ações e BDRs | 15% | nenhuma | |
| FIIs (ganho na venda) | 20% | nenhuma | só com FIIs |

O imposto é mensal, sobre o ganho realizado, pago por DARF até o fim do mês seguinte (DARF abaixo de R$ 10 acumula). O IRRF ("dedo-duro") é antecipação e não muda o total. `imposto_renda_mensal` aplica essas regras a uma série de lucros mensais em R$ (rode uma vez por categoria) e `simular_ir` aplica sobre os trades de `retornos_estrategia` e devolve os retornos após IR. Compare com buy and hold também após IR: segurar uma ação não realiza ganho e adia o imposto, o que favorece o buy and hold contra estratégias de giro alto.
