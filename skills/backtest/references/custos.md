# Custos e execução realistas (passo 4)

Muitas "vantagens" morrem aqui. Modele custos **por trade**, não como desconto anual genérico.

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

Calcule também o **custo de breakeven**: o custo por lado em que o retorno líquido vira zero. Compare com o custo real. Se a margem for pequena (breakeven < 2x o custo base), sinalize a estratégia como frágil.

## Turnover

Reporte o giro anual. Estratégias de alto giro precisam de vantagem bruta proporcionalmente maior: `retorno líquido ≈ retorno bruto − turnover × custo por giro`.
