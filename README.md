# Backtest_Skill

Skill para o Claude executar backtests de estratégias de trading e investimento da melhor forma possível: o objetivo é descobrir se uma estratégia tem vantagem real, não produzir a curva de capital mais bonita.

## O que a skill cobre

- **Roteiro de 12 passos** ([`SKILL.md`](skills/backtest/SKILL.md)): hipótese, dados limpos, sem look-ahead, custos, divisão de dados, walk-forward, robustez, testes estatísticos, regimes, métricas, paper trading e monitoramento, com regras de ouro e um veredito obrigatório (*Rejeitada*, *Inconclusiva*, *Promissora* ou *Aprovada com tamanho reduzido*).
- **Referências** ([`references/`](skills/backtest/references)): checklists de cada passo, custos por mercado (cripto, forex, commodities, ações americanas), testes estatísticos, comparação de estratégias, portfólio e o modelo do relatório final.
- **Scripts** ([`scripts/`](skills/backtest/scripts), só `numpy`):
  - `backtest.py`: motor vetorizado com a convenção de timing (abertura ou fechamento de *t+1*), custos por giro, carregamento (funding, swap, aluguel), rendimento do caixa e extração de trades; sensibilidade e breakeven de custos; walk-forward de parâmetros; robustez e mapa de calor; regimes e ano a ano; testes de look-ahead; alvo de volatilidade e Kelly.
  - `estatistica.py`: métricas, bootstrap, permutação simples e com correção de seleção, Deflated Sharpe Ratio, PBO e Monte Carlo de drawdown.
  - `comparacao.py`: diferença de Sharpe (Ledoit-Wolf e bootstrap estudentizado), pares com Holm, Hansen SPA/White Reality Check e Model Confidence Set.
  - `portfolio.py`: 1/N, vol inversa, risk parity, HRP, mínima variância, max Sharpe e Kelly fracionado, com walk-forward dos pesos, custos de rebalanceamento, limites, diversificação e alertas de overfitting.
  - `exemplo_completo.py`: os 12 passos de ponta a ponta numa estratégia de médias móveis, com rascunho do relatório.


## Skill `backtest-lighter`

[`skills/backtest-lighter`](skills/backtest-lighter/SKILL.md) é o protocolo do projeto Backtests Lighter: testar sempre em XAU, US100, WTI, BTC e ETH, com spread e funding reais da Lighter, controle aleatório de mesmo lado e duração, terços, spread 4x, sem funding, filtro de correlação, treino/validação/OOS em 10 partes e carteiras. Usa a skill `backtest` para a parte estatística geral.

- `scripts/protocolo.py`: custos por trade (short composto certo), métricas, controle aleatório, critério de aprovação, Holm/BH, filtro de correlação e carteira. `python protocolo.py` roda uma demonstração.
- `scripts/lighter_dados.py`: custos e funding medidos na VPS (`python lighter_dados.py custos`, precisa de `DATA_API_TOKEN`) e candles/funding longos da API pública da Lighter.

## Instalação

- **Claude Code:** copie `skills/backtest/` para `~/.claude/skills/backtest/` (pessoal) ou `.claude/skills/backtest/` (no projeto).
- **Claude.ai:** gere o pacote com `python ferramentas/validar_skill.py --zip` e envie `dist/backtest.zip` em *Configurações > Capacidades > Skills*. O mesmo pacote fica disponível como artefato em cada execução do CI.

## Como usar

Peça ao Claude algo como "faça o backtest desta estratégia", "compare estas duas estratégias" ou "quanto capital dar a cada uma destas estratégias". A skill é carregada pela descrição e conduz os passos; o resultado sai no formato de `references/relatorio.md`.

Para ver o fluxo inteiro rodando:

```bash
pip install numpy pandas
cd skills/backtest/scripts
python exemplo_completo.py              # dados sintéticos
python exemplo_completo.py precos.csv   # colunas data, abertura, fechamento e, opcional, rf (% ao ano, ex.: DTB3 do FRED)
python exemplo_completo.py btc.csv --periodos=365   # cripto
```

Cada script também roda sozinho (`python backtest.py`, `python estatistica.py`, ...) com uma demonstração.

## Testes e validação

```bash
pip install numpy pandas pytest
python ferramentas/validar_skill.py   # cabeçalho, arquivos citados e nomes de funções da documentação
python -m pytest tests                # ou python tests/test_<módulo>.py, sem pytest
```

Os testes conferem as fórmulas de execução contra uma simulação evento a evento, a calibração dos testes estatísticos sob a hipótese nula, os custos de carregamento, o walk-forward sem olhar o futuro e o exemplo de ponta a ponta. O workflow em `.github/workflows/testes.yml` roda tudo isso a cada push e pull request.

## Limitações

- O motor é vetorizado e diário: ordens limitadas, stops intradiários, filas e execução parcial pedem um motor orientado a eventos (a skill orienta a usar um nesses casos).
- Os custos por mercado são valores típicos de referência: confira os da corretora ou exchange. Impostos dependem da jurisdição e não são simulados.
- Não há fonte de dados embutida: os preços (ajustados por dividendos e splits, sem viés de sobrevivência) e a taxa livre de risco, se usada, vêm do usuário.
