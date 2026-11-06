# Backtest_Skill

Skill para o Claude executar backtests de estratégias de trading e investimento da melhor forma possível.

## Conteúdo

- [`skills/backtest/`](skills/backtest/SKILL.md): roteiro de 12 passos (hipótese, dados limpos, sem look-ahead, custos, divisão de dados, walk-forward, robustez, testes estatísticos, regimes, métricas, paper trading e monitoramento), com referências detalhadas e um script Python de estatística (bootstrap, permutação, Deflated Sharpe Ratio, PBO, Monte Carlo).

## Instalação

- **Claude Code:** copie `skills/backtest/` para `~/.claude/skills/backtest/` (pessoal) ou `.claude/skills/backtest/` (no projeto).
- **Claude.ai:** compacte a pasta `skills/backtest/` em um .zip e envie em *Configurações > Capacidades > Skills*.

O script `scripts/estatistica.py` depende apenas de `numpy`. Rode `python skills/backtest/scripts/estatistica.py` para ver uma demonstração.

## Testes

`python -m pytest tests` (ou `python tests/test_estatistica.py`, sem pytest) roda os testes de regressão do script e confere a fórmula de execução na abertura de *t+1* contra uma simulação evento a evento (essa parte usa `pandas`, se estiver instalado).
