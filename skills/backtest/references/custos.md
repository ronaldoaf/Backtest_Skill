# Custos e execução realistas (passo 4)

Muitas "vantagens" morrem aqui. Modele custos **por trade**, não como desconto anual genérico. Em `scripts/backtest.py`, `retornos_estrategia(..., custo_por_lado=...)` cobra o custo por unidade de giro no período em que a posição muda, e `sensibilidade_custos` / `formatar_sensibilidade` montam a tabela abaixo com o custo de breakeven.

## Componentes

| Componente | Como modelar | Observações |
|---|---|---|
| Comissão e taxas | Valor fixo por ordem ou contrato + % do volume (taxas de bolsa, regulatórias, maker/taker) | Use a tabela real da corretora/exchange do usuário |
| Spread | Metade do spread bid-ask por lado | Use spread histórico; na falta, estime pela volatilidade intradiária (ex.: estimador de Corwin-Schultz ou Roll) |
| Slippage | Ticks ou bps por lado, maior em ordens a mercado e em stops | Stops em gap executam no preço de abertura, não no stop |
| Impacto de mercado | Proporcional à participação no volume, ex.: `custo ≈ k · σ · sqrt(Q / ADV)` | Define a **capacidade** da estratégia |
| Financiamento | Juros de margem, custo de aluguel de ações (short), funding em perpétuos cripto, carrego de futuros | Cobrar por dia de posição |
| Impostos | Dependem da jurisdição do usuário (ganho de curto vs. longo prazo, compensação de perdas) | Importa para comparar com buy and hold; a skill não simula |
| Câmbio | Spread de conversão quando a conta é em outra moeda | |

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

Calcule também o **custo de breakeven**: o custo por lado em que o retorno líquido vira zero (com `rf_por_periodo`, em que o retorno empata com a taxa livre de risco). Compare com o custo real. Se a margem for pequena (breakeven < 2x o custo base), sinalize a estratégia como frágil.

## Turnover

Reporte o giro anual. Estratégias de alto giro precisam de vantagem bruta proporcionalmente maior: `retorno líquido ≈ retorno bruto − turnover × custo por giro`.

## Referências por mercado

Valores típicos para conta de varejo, para usar como ponto de partida conservador quando o usuário não informar os dele. Mudam com a corretora, o volume e o tempo: diga ao usuário quais valores assumiu. Monte o custo por lado com `custo_por_lado` e passe o carregamento em `retornos_estrategia`.

| Mercado | Custo por lado (comissão + meio spread + slippage) | Carregamento | Cuidados |
|---|---|---|---|
| Cripto à vista | taker 0,05% a 0,10% + 1 a 5 bps; altcoins de baixa liquidez muito mais | o saldo parado em geral não rende (`caixa="nenhum"` ou rf baixo) | negocia 24/7: `periodos_por_ano=365`, defina o fechamento diário (ex.: 00:00 UTC); exchanges fecham e moedas são deslistadas (viés de sobrevivência) |
| Perpétuos de cripto | taker 0,03% a 0,06% + slippage | funding a cada 8 h (some as do período em `financiamento_por_periodo`; positivo, comprado paga); use a série histórica, que passa de 0,1% por período em euforia | `caixa="total"` (a posição é nocional sobre a margem); liquidação forçada com alavancagem: confira o pior drawdown intradiário contra a margem |
| Forex (pares principais) | spread de 0,5 a 2 pips (≈ 0,5 a 2 bps) no horário líquido, várias vezes maior na virada de Nova York e em notícias | swap (rollover) diário pela diferença de juros das moedas, triplo na quarta-feira | ~260 dias por ano; fins de semana criam gaps na abertura de domingo; `caixa="total"` com a posição como nocional sobre o capital |
| Futuros de commodities | comissão + taxas de bolsa por contrato (alguns USD) dividido pelo nocional (preço × multiplicador), + 1 tick de slippage | rolagem: custo da diferença entre vencimentos e do spread na troca | série contínua ajustada para sinais e contratos reais para o P&L; limites de oscilação travam a saída; `caixa="total"` |
| Ações americanas | comissão em geral zero; taxa da SEC e da FINRA nas vendas (frações de bp); spread de 1 a 5 bps em large caps, 20+ bps em small caps | aluguel para vender a descoberto (`aluguel_por_periodo`: 0,3% a.a. em papéis fáceis, 10%+ nos "hard to borrow") e dividendos pagos pela ponta vendida; margem acima do T-bill | universo point-in-time com deslistadas (CRSP ou equivalente); pré e pós-mercado com liquidez menor |

### Caixa e taxa livre de risco

Quando a conta rende juros (T-bill, juros da corretora), `caixa="livre"` faz a parte não alocada render `rf_por_periodo` e cobra rf de exposição acima de 100%; `caixa="total"` faz o capital inteiro render rf (futuros, perpétuos, forex, long-short com margem aplicada). Com juros de 4% a 5% ao ano, isso muda bastante o retorno de estratégias que ficam fora do mercado parte do tempo, e o Sharpe deve ser medido em excesso de rf. Se o caixa não rende nada (comum em exchanges de cripto), deixe `rf_por_periodo=0`.

### Impostos

Dependem do país do usuário e não são simulados pela skill. Quando importarem para a decisão (por exemplo, estratégia de giro alto contra buy and hold, que adia o imposto), pergunte a jurisdição e aplique a alíquota sobre o ganho realizado por período, separando curto e longo prazo quando a regra fizer essa distinção.
