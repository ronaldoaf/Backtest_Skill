# Backtest_Skill

Skill para o Claude executar backtests de estratégias de trading e investimento da melhor forma possível.

## Conteúdo

- [`skills/backtest/`](skills/backtest/SKILL.md): roteiro de 12 passos (hipótese, dados limpos, sem look-ahead, custos, divisão de dados, walk-forward, robustez, testes estatísticos, regimes, métricas, paper trading e monitoramento), com referências detalhadas, um script Python de estatística (bootstrap, permutação, Deflated Sharpe Ratio, PBO, Monte Carlo) e um módulo de comparação de estratégias (diferença de Sharpe por Ledoit-Wolf e bootstrap pareado, pares com Holm, Hansen SPA/White Reality Check, Model Confidence Set).

## Instalação

- **Claude Code:** copie `skills/backtest/` para `~/.claude/skills/backtest/` (pessoal) ou `.claude/skills/backtest/` (no projeto).
- **Claude.ai:** compacte a pasta `skills/backtest/` em um .zip e envie em *Configurações > Capacidades > Skills*.

Os scripts `scripts/estatistica.py` e `scripts/comparacao.py` dependem apenas de `numpy`. Rode `python skills/backtest/scripts/estatistica.py` ou `python skills/backtest/scripts/comparacao.py` para ver uma demonstração.

## Testes

`python -m pytest tests` (ou `python tests/test_estatistica.py` e `python tests/test_comparacao.py`, sem pytest) roda os testes de regressão dos scripts, verifica a calibração dos testes de comparação sob a hipótese nula e confere a fórmula de execução na abertura de *t+1* contra uma simulação evento a evento (essa parte usa `pandas`, se estiver instalado).
