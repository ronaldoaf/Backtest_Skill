# Paper trading, incubação e monitoramento (passos 11 e 12)

## Passo 11: incubação

Objetivo: confirmar que a estratégia funciona fora do laboratório e que o backtest representa a execução real.

Plano mínimo a propor:
1. **Congelar** código, parâmetros e universo. Versionar (commit/hash) o que entra em incubação.
2. **Duração:** o suficiente para gerar trades com algum poder estatístico (ex.: ≥ 30 trades ou ≥ 3 a 6 meses, o que for maior). Estratégias lentas exigem mais tempo; deixe claro que incubação curta só detecta erros grosseiros.
3. **Modo:** paper trading com dados ao vivo, ou dinheiro real com tamanho mínimo (melhor para medir slippage e problemas operacionais).
4. **Reconciliação diária/semanal:**
   - os sinais ao vivo são idênticos aos que o backtest geraria para os mesmos dados?
   - preço de execução real vs. preço assumido no backtest (slippage observado);
   - retorno ao vivo vs. retorno do backtest no mesmo período (tracking error);
   - falhas operacionais: dados atrasados, ordens rejeitadas, horários.
5. **Critério de aprovação:** definido antes. Ex.: retorno ao vivo dentro do intervalo de 90% simulado pelo Monte Carlo para a mesma duração, e slippage real ≤ 1,5x o assumido.

Se o slippage real for maior, refaça o backtest com o custo observado antes de aumentar o tamanho.

## Passo 12: monitoramento e desligamento

Defina **por escrito, antes de operar**, as regras de desligamento. Decidir isso durante um drawdown é decidir com emoção.

### Limites derivados do Monte Carlo (passo 8)
- **Drawdown:** pausar se o drawdown ao vivo exceder o percentil 95 (ou 99) do drawdown máximo simulado.
- **Sequência de perdas:** pausar se a sequência de perdas consecutivas exceder o percentil 99 simulado.
- **Desempenho móvel:** Sharpe móvel de 6 a 12 meses abaixo de um limite (ex.: abaixo do percentil 5 da distribuição simulada para essa janela).
- **Tempo submerso:** duração do drawdown acima do percentil 95 simulado.

### Monitoramento estrutural
- A hipótese do passo 1 continua válida? (mudança de regra de mercado, tributação, microestrutura, concorrência explorando o mesmo efeito)
- Custos e liquidez mudaram?
- Correlação com outras estratégias da carteira aumentou?

### Rotina
- Painel com curva ao vivo sobreposta ao cone de Monte Carlo.
- Revisão periódica (ex.: mensal) com o checklist acima.
- Registro de toda intervenção manual.
- Plano de retomada: o que precisa acontecer para religar (ex.: nova rodada de validação com os dados recentes, sem reotimizar às pressas).

### Dimensionamento
- Começar pequeno e aumentar em degraus conforme o desempenho ao vivo confirma o backtest.
- Considerar fração de Kelly bem conservadora (ex.: ≤ 1/4 de Kelly) ou volatility targeting, nunca o Kelly cheio estimado do backtest.
