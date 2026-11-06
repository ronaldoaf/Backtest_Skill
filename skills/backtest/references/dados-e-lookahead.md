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

### Erros clássicos
| Erro | Como aparece | Correção |
|---|---|---|
| Executar no mesmo preço do sinal | Sinal no `close[t]`, entrada no `close[t]` | Entrar em `open[t+1]` (ou `close[t+1]`) |
| Esquecer o `shift` | `posicao = sinal` em vez de `sinal.shift(1)` | `retorno_estrategia = posicao.shift(1) * retorno` |
| Indicador centrado | `rolling(..., center=True)`, filtros suavizadores bidirecionais | Somente janelas que terminam em *t* |
| `bfill` / interpolação | Preenche passado com valor futuro | Só `ffill` |
| Fundamentos pela data de referência | Balanço do 4T usado em 31/12 | Usar data de **divulgação** (+1 dia) |
| Normalização global | `(x - x.mean()) / x.std()` sobre a série inteira | Estatísticas expandindo no tempo ou ajustadas só no treino |
| Seleção de universo com dados futuros | "Ações que estão no índice hoje" | Composição point-in-time |
| Máxima/mínima do dia | Stop e alvo atingidos no mesmo candle, assumindo o melhor | Assumir o pior caso, ou usar dados de menor granularidade |
| Rótulos de ML sobrepostos | Rótulo de *t* usa retorno até *t+h* e vaza para o treino | Purging e embargo (ver `validacao.md`) |
| Reotimização usando o período de teste | Parâmetros escolhidos com o histórico completo | Walk-forward |

### Testes práticos para pegar vazamento
1. **Teste do atraso:** atrase todos os sinais em mais 1 barra. Uma queda drástica de desempenho (ex.: Sharpe de 3 para 0,3) sugere que a estratégia depende de informação do mesmo instante.
2. **Teste do futuro truncado:** calcule os sinais com dados até a data *T* e depois com dados até *T+N*. Os sinais até *T* devem ser idênticos.
3. **Teste do sinal aleatório:** troque o sinal por ruído com a mesma frequência. O resultado deve ficar próximo de zero menos custos; se ficar positivo, há vazamento no motor.
4. **Leitura linha a linha:** para 3 a 5 trades, imprima os dados disponíveis no momento da decisão e confira manualmente.
