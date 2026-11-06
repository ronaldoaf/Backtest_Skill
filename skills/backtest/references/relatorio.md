# Modelo de relatório final

Use esta estrutura ao entregar o resultado. Seja direto e não esconda testes reprovados.

```markdown
# Backtest: <nome da estratégia>

## Veredito
<Rejeitada | Inconclusiva | Promissora (incubar) | Aprovada para operar com tamanho reduzido>
<2–3 frases justificando com os testes decisivos.>

## 1. Hipótese
- Por que a vantagem deveria existir:
- Quem está do outro lado:
- Regras (entrada, saída, tamanho, universo, frequência):
- Definida antes de ver os dados? sim/não

## 2. Dados
- Fonte, período, frequência, universo:
- Viés de sobrevivência tratado? Ajustes de proventos/splits:
- Problemas encontrados e correções:

## 3. Auditoria de look-ahead
- Execução assumida (linha da convenção de timing: fechamento de t+1, abertura de t+1 ou fechamento de t com a ressalva):
- Testes feitos (atraso extra, futuro truncado, sinal aleatório) e resultado:

## 4. Custos
- Premissas por lado (corretagem, spread, slippage, impacto, financiamento, impostos):
- Tabela de sensibilidade (x1, x2, x3) e custo de breakeven:

## 5–6. Divisão e walk-forward
- Treino / validação / holdout (datas):
- Esquema do walk-forward (ancorado/móvel, tamanhos):
- Eficiência walk-forward, estabilidade dos parâmetros, % janelas positivas:
- Holdout: rodado uma vez em <data>? Resultado:

## 7. Robustez de parâmetros
- Grade testada, mapa de calor, existe platô?
- Outros testes (ruído, retirar melhores trades, outra data de rebalanceamento):

## 8. Testes estatísticos
| Teste | Resultado | Status |
|---|---|---|
| vs. benchmark (alfa, beta) | | |
| IC 95% Sharpe (bootstrap, fora da amostra) | | |
| p-valor permutação circular (fora da amostra) | | |
| p-valor permutação com seleção (melhor de N) | | |
| Nº de tentativas registradas | | |
| Deflated Sharpe Ratio | | |
| PBO | | |

Se houve comparação entre estratégias (ver `comparacao.md`):

| Comparação | Diferença de Sharpe (IC 95%) | p-valor | Método |
|---|---|---|---|
| A x B | | | HAC / bootstrap estudentizado |
| Melhor de N x benchmark | | p_spa = | SPA |
| MCS (90%) | {estratégias que ficam} | | T_max |

- Taxa livre de risco usada no Sharpe (ex.: CDI diário):
- Correlação entre as estratégias:
- Critério de desempate, se os testes não distinguem:

Se houver portfólio de estratégias (ver `portfolio.md`):

| Método | Sharpe fora | Eficiência | Max DD | Giro anual | Diferença x 1/N (IC) | p Holm |
|---|---|---|---|---|---|---|
| 1/N | | | | | — | — |
| <escolhido> | | | | | | |

- Correlação média, pares redundantes, perdas conjuntas (observado x normal), apostas efetivas:
- Walk-forward dos pesos (janela, rebalanceamento, faixa, custo_giro, limites, vol_alvo) e métodos testados:
- MCS (90%), método escolhido, pesos atuais e critério de desempate:

## 9. Regimes e mercados
- Tabela por regime e por ano:
- Outros ativos/mercados sem reotimizar:

## 10. Métricas (fora da amostra, líquidas)
| Métrica | Estratégia | Benchmark |
|---|---|---|
| CAGR | | |
| Volatilidade | | |
| Sharpe / Sortino / Calmar | | |
| Max drawdown (data) / duração | | |
| Nº trades / acerto / payoff / expectativa | | |
| Exposição / turnover / capacidade | | |

## 11. Plano de incubação
- Duração, modo, critérios de aprovação:

## 12. Regras de desligamento
- Drawdown limite (P95 Monte Carlo):
- Sequência de perdas limite:
- Sharpe móvel limite:
- Revisão periódica:

## Limitações e riscos conhecidos
- ...
```
