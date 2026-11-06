# Métricas (passo 10)

Reporte sempre **líquido de custos**, sobre o período fora da amostra (walk-forward + holdout), e ao lado do benchmark.

## Retorno e risco
| Métrica | Definição | Nota |
|---|---|---|
| CAGR | `(capital_final / capital_inicial)^(1/anos) − 1` | |
| Volatilidade anual | `std(retornos) · sqrt(periodos_por_ano)` | 252 para diário em ações, 365 para cripto |
| Sharpe | `(média(r − rf) / std(r)) · sqrt(periodos_por_ano)` | Informe a taxa livre de risco usada |
| Sortino | Igual ao Sharpe, mas com desvio apenas dos retornos abaixo de zero (ou do alvo) | |
| Calmar | `CAGR / |drawdown máximo|` | |
| Drawdown máximo | Maior queda do pico ao vale da curva de capital | Reporte também a **data** |
| Duração do drawdown | Maior tempo entre um pico e a recuperação | Muitas vezes mais doloroso que a profundidade |
| Ulcer index | Raiz da média dos drawdowns ao quadrado | Penaliza drawdowns longos |
| VaR / CVaR 95% | Perda em percentil 5 e média das perdas além dele | |
| Assimetria e curtose | Momentos dos retornos | Estratégias de venda de volatilidade parecem ótimas até a cauda |

## Trades

`retornos_estrategia` devolve os trades em `res["trades"]` (retorno de cada um sobre o capital, início, fim, lado e duração). Passe `res["trades"]["retorno"]` (sem os trades ainda abertos) para `metricas_trades` e `monte_carlo_drawdown`.

| Métrica | Definição |
|---|---|
| Número de trades | Total e por ano |
| Taxa de acerto | trades vencedores / total |
| Payoff | ganho médio / perda média |
| Expectativa por trade | `acerto · ganho_médio − (1 − acerto) · perda_média` (em R$ e em múltiplos do risco) |
| Profit factor | soma dos ganhos / soma das perdas |
| Duração média | Tempo médio em posição, separado para ganhos e perdas |
| Maior sequência de perdas | Comparar com a esperada pelo Monte Carlo |
| MAE / MFE | Excursão adversa/favorável máxima, útil para avaliar stops e alvos |

## Operacionais
- **Exposição:** % do tempo posicionado e exposição bruta/líquida média.
- **Turnover:** giro anual da carteira.
- **Capacidade:** capital máximo antes que o impacto de mercado consuma metade da vantagem (ver `custos.md`).
- **Alavancagem** máxima e média.
- **Concentração:** contribuição dos 5/10 melhores trades ou ativos para o resultado total.

## Sinais de alerta
- Sharpe muito alto (> 2 a 3 em diário) com poucos trades: suspeitar de look-ahead ou overfitting.
- Taxa de acerto muito alta com payoff muito baixo: risco de cauda (perdas raras e grandes).
- Lucro concentrado em poucos trades ou em um único período.
- Curva de capital suave demais: verificar se preços são ilíquidos/estáticos ou se há reavaliação defasada.
- Amostra pequena: menos de ~100 trades ou menos de ~3 anos de dados diários.
