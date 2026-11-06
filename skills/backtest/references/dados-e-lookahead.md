# Dados limpos e ausência de look-ahead (passos 2 e 3)

## Passo 2: checklist de qualidade dos dados

### Viés de sobrevivência
- O universo deve ser o que existia **em cada data**, incluindo ativos deslistados, falidos, incorporados e que saíram do índice.
- Composição de índices: use a composição histórica (point-in-time), não a atual.
- Se o dado de deslistados não estiver disponível, avise que o resultado tende a ser otimista e quantifique quando possível.

### Ajustes de preço
- Splits, grupamentos e bonificações devem estar ajustados.
- Dividendos/JCP: para retorno total, use preço ajustado ou some os proventos no caixa. Para sinais baseados em níveis de preço (ex.: rompimento de máxima), confira se o ajuste não cria níveis que nunca foram negociados.
- Ordens limitadas e stops devem ser simulados sobre o preço **não ajustado** daquele dia, ou com ajuste consistente.
- Futuros: use série contínua com rolagem documentada (back-adjusted para sinais, contratos reais para P&L), e inclua o custo da rolagem.

### Tempo
- Padronize fuso horário e saiba se o timestamp marca abertura ou fechamento do candle.
- Feriados e pregões curtos: não preencha com valores inventados.
- Dados intradiários: confira horário de leilão de abertura/fechamento.

### Integridade
- Procure lacunas, duplicatas, preços zero ou negativos, saltos acima de N desvios sem evento corporativo, volume zero.
- Registre quantas linhas foram removidas ou corrigidas e por quê.
- Compare uma amostra com uma segunda fonte.

### Documentação mínima
Fonte, período, frequência, universo, tratamento de ajustes, tratamento de faltantes, data do download.

## Passo 3: auditoria de look-ahead

### Convenção de timing (use sempre esta)

O sinal de *t* é calculado com dados até o fechamento de *t*, inclusive. A tabela diz como montar o retorno da estratégia em pandas, usando só valores passados em cada linha. `close` e `open` são as séries de preços; `sinal` vale -1, 0 ou 1 (ou o peso da posição).

| Execução | Retorno da estratégia | Quando usar |
|---|---|---|
| Fechamento de *t+1* | `sinal.shift(2).fillna(0) * close.pct_change()` | Padrão para dados só de fechamento |
| Abertura de *t+1* | Fórmula abaixo | Padrão quando há preço de abertura |
| Fechamento de *t* | `sinal.shift(1).fillna(0) * close.pct_change()` | **Só** se o sinal puder ser calculado antes do leilão de fechamento (ex.: com o preço de minutos antes) e a ordem for executável no leilão. Diga ao usuário que essa premissa foi usada |

`retornos_estrategia` em `scripts/backtest.py` implementa as três linhas (`execucao="abertura_t1"`, `"fechamento_t1"` ou `"fechamento_t"`), com custos e caixa no CDI, e foi conferida contra uma simulação evento a evento. Use-a em vez de reescrever as fórmulas; elas ficam abaixo para auditoria.

Execução na abertura de *t+1*: a posição nova vale do pregão seguinte em diante, e a antiga ainda carrega o gap overnight.

```python
p_nova = sinal.shift(1).fillna(0)     # posição que vale da abertura ao fechamento de t
p_antiga = sinal.shift(2).fillna(0)   # posição que carrega o gap do fechamento de t-1 à abertura de t
gap = (open / close.shift(1) - 1).fillna(0)
intradia = close / open - 1
retorno_estrategia = (1 + p_antiga * gap) * (1 + p_nova * intradia) - 1
```

Custos entram por unidade de giro, no candle em que a posição muda, com `posicao` sendo a posição em vigor naquele candle (`p_nova` no caso da abertura, `sinal.shift(2).fillna(0)` no fechamento de *t+1*):

```python
giro = posicao.diff().abs().fillna(posicao.abs())   # inclui a entrada inicial
retorno_liquido = retorno_estrategia - custo_por_lado * giro
```

`sinal * close.pct_change()`, sem `shift`, é look-ahead puro: a estratégia lucra com o movimento que gerou o próprio sinal. `sinal.shift(1) * close.pct_change()` executa no fechamento de *t*, ao mesmo preço do sinal, e só vale com a ressalva da última linha da tabela.

### Erros clássicos
| Erro | Como aparece | Correção |
|---|---|---|
| Executar no mesmo preço do sinal | Sinal no `close[t]`, entrada no `close[t]`, ou seja, `sinal.shift(1) * close.pct_change()` | Entrar em `open[t+1]` ou `close[t+1]`, pelas fórmulas da convenção de timing |
| Esquecer o `shift` | `sinal * close.pct_change()` | Usar as fórmulas da convenção de timing |
| Indicador centrado | `rolling(..., center=True)`, filtros suavizadores bidirecionais | Somente janelas que terminam em *t* |
| `bfill` / interpolação | Preenche passado com valor futuro | Só `ffill` |
| Fundamentos pela data de referência | Balanço do 4T usado em 31/12 | Usar data de **divulgação** (+1 dia) |
| Normalização global | `(x - x.mean()) / x.std()` sobre a série inteira | Estatísticas expandindo no tempo ou ajustadas só no treino |
| Seleção de universo com dados futuros | "Ações que estão no índice hoje" | Composição point-in-time |
| Máxima/mínima do dia | Stop e alvo atingidos no mesmo candle, assumindo o melhor | Assumir o pior caso, ou usar dados de menor granularidade |
| Rótulos de ML sobrepostos | Rótulo de *t* usa retorno até *t+h* e vaza para o treino | Purging e embargo (ver `validacao.md`) |
| Reotimização usando o período de teste | Parâmetros escolhidos com o histórico completo | Walk-forward |

### Testes práticos para pegar vazamento
1. **Teste do atraso** (`teste_atraso`): atrase todos os sinais em mais 1 barra. Uma queda drástica de desempenho (ex.: Sharpe de 3 para 0,3) sugere que a estratégia depende de informação do mesmo instante.
2. **Teste do futuro truncado** (`teste_futuro_truncado`): calcule os sinais com dados até a data *T* e depois com dados até *T+N*. Os sinais até *T* devem ser idênticos. Pega média centrada, `bfill`, normalização global e dados revisados.
3. **Teste do sinal aleatório:** troque o sinal por ruído com a mesma frequência. O resultado deve ficar próximo de zero menos custos; se ficar positivo, há vazamento no motor.
4. **Leitura linha a linha:** para 3 a 5 trades, imprima os dados disponíveis no momento da decisão e confira manualmente.
