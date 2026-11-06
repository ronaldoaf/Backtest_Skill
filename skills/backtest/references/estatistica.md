# Testes estatísticos (passo 8)

Pergunta central: **qual a chance de um resultado tão bom quanto este aparecer por sorte, dado tudo o que foi tentado?** As funções citadas estão em `scripts/estatistica.py`.

## 0. Em que amostra cada teste vale

Um teste calibrado para uma estratégia definida antes de olhar os dados aprova ruído quando é aplicado à melhor de várias configurações. Em simulação com a melhor de 50 estratégias aleatórias (200 repetições), a permutação simples deu p < 0,05 em 81% dos casos e p < 0,01 em 32%.

| Teste | Onde aplicar | Função |
|---|---|---|
| Bootstrap (IC do Sharpe e do CAGR) | Série fora da amostra (walk-forward concatenado ou holdout), ou estratégia definida antes de ver os dados | `bootstrap_metricas` |
| Permutação circular (= estratégias aleatórias) | Idem | `teste_permutacao` |
| Permutação com correção de seleção | Resultado otimizado: posições de **todas** as N configurações testadas, no período em que a escolha foi feita | `teste_permutacao_selecao` |
| Deflated Sharpe Ratio | Configuração escolhida, no período da escolha, com N = todas as tentativas | `deflated_sharpe_ratio` |
| PBO | Retornos de todas as configurações candidatas, no período da escolha | `pbo_cscv` |
| Monte Carlo de drawdown | Série fora da amostra, de preferência | `monte_carlo_drawdown` |

Se o walk-forward foi redesenhado depois de ver o resultado fora da amostra, essa série também passou por seleção: conte cada versão como tentativa e aplique o DSR sobre ela.

## 1. Comparação com benchmark
- Buy and hold do ativo/índice no mesmo período, com os mesmos custos de entrada.
- Calcule alfa e beta (regressão dos retornos da estratégia contra o benchmark) e a correlação. Uma estratégia de beta alto "ganhando" num mercado de alta não demonstrou vantagem.

## 2. Estratégias aleatórias
Compare com milhares de estratégias de **mesma exposição média, mesmo número de trades e mesma duração média**, mas entradas aleatórias. A permutação circular de `teste_permutacao` (seção 4) faz exatamente isso: desloca a série de posições inteira, o que preserva exposição, número e duração dos trades e só quebra o alinhamento com os retornos. Não rode um teste separado: o percentil da estratégia real é 100 × (1 − p-valor), e valem os limiares da permutação.

## 3. Bootstrap / Monte Carlo — `bootstrap_metricas`, `monte_carlo_drawdown`
- **Bootstrap em blocos** dos retornos diários fora da amostra (preserva autocorrelação; bloco ≈ 5 a 20 dias) para intervalo de confiança do Sharpe e do CAGR.
- **Reamostragem/embaralhamento dos trades** para a distribuição do drawdown máximo e da maior sequência de perdas. O drawdown histórico é apenas **uma** trajetória; o percentil 95 simulado é uma expectativa mais honesta e alimenta o passo 12.
- Reporte: IC 95% do Sharpe. Se o IC inclui zero, a estratégia é inconclusiva.

## 4. Teste de permutação — `teste_permutacao`, `teste_permutacao_selecao`
Desloque circularmente (padrão) ou embaralhe a série de posições em relação aos retornos e recalcule a métrica milhares de vezes. p-valor = fração das permutações com métrica ≥ real. O deslocamento circular preserva a estrutura temporal dos sinais. `posicoes[t]` é a posição em vigor durante `retornos[t]`, montada pela convenção de timing de `dados-e-lookahead.md`. Passe `custo_por_giro` para que o Sharpe real e o das permutações sejam líquidos de custos.

**Com seleção** (a melhor de N configurações), use `teste_permutacao_selecao`. Ele recebe as posições de todas as N configurações testadas, aplica o mesmo deslocamento a todas e compara o melhor Sharpe real com o melhor Sharpe de cada permutação. É a ideia do White's Reality Check, com permutação no lugar do bootstrap. Na mesma simulação, deu 5% de falsos positivos a 5%, como esperado, contra 81% do teste simples. Inclua as configurações descartadas: omiti-las deixa o teste otimista.

## 5. Correção para múltiplos testes

Testar N variações e escolher a melhor infla o Sharpe. Por isso é obrigatório **registrar todas as tentativas** (inclusive as feitas em conversas anteriores ou "só para ver").

### Deflated Sharpe Ratio (DSR) — `deflated_sharpe_ratio`
(Bailey & López de Prado, 2014). Corrige o Sharpe observado por:
- número de tentativas independentes N;
- variância dos Sharpes entre as tentativas;
- tamanho da amostra T;
- assimetria e curtose dos retornos.

Sharpe máximo esperado só por sorte entre N tentativas (γ = constante de Euler-Mascheroni ≈ 0,5772):

```
SR0 = sqrt(V[SR]) · ((1 − γ) · Z⁻¹(1 − 1/N) + γ · Z⁻¹(1 − 1/(N·e)))
```

```
DSR = Φ( (SR − SR0) · sqrt(T − 1) / sqrt(1 − skew·SR + (kurt − 1)/4 · SR²) )
```

(SR não anualizado, na frequência dos dados.) Interprete DSR como a probabilidade de o Sharpe verdadeiro ser positivo após a correção. Exigir DSR ≥ 0,95.

Se as tentativas forem correlacionadas, estime o número efetivo de tentativas independentes (ex.: agrupando as variações por correlação dos retornos).

### Probabilidade de Overfitting do Backtest (PBO) — `pbo_cscv`
(Bailey, Borwein, López de Prado e Zhu, 2015). Com uma matriz T × N de retornos (N configurações candidatas):
1. Divida as T linhas em S blocos (S par, ex.: 16).
2. Para cada combinação de S/2 blocos como "treino" e o restante como "teste": escolha a melhor configuração no treino e veja o rank relativo dela no teste.
3. PBO = fração das combinações em que a melhor do treino fica **abaixo da mediana** no teste.

PBO > 0,5: o processo de seleção é pior que escolher ao acaso. Desejável: PBO < 0,1 a 0,2.

### Alternativas
- **White's Reality Check** / **Hansen SPA**: testam se a melhor estratégia supera o benchmark considerando o conjunto testado.
- Correções de Bonferroni/Holm ou controle de FDR (Benjamini-Hochberg) para p-valores de muitas variações.
- Barra prática de Harvey, Liu e Zhu (2016): para uma nova "descoberta", t-stat > 3 em vez de 2.

## 6. Tamanho de amostra
- Erro padrão aproximado do Sharpe anualizado: `sqrt((1 + SR²/2) / anos)`. Com Sharpe 1 e 5 anos, o erro padrão é ~0,55: muito ruído.
- Anos necessários para distinguir Sharpe S de zero com t ≈ 2: aproximadamente `(2 / S)²`. Sharpe 0,5 exige ~16 anos.
- Para trades: menos de ~30 é anedótico; ~100+ começa a ser informativo; considere a dependência entre trades.

## Tabela de decisão sugerida

| Teste | Aprovado | Atenção | Reprovado |
|---|---|---|---|
| IC 95% do Sharpe (bootstrap, fora da amostra) | > 0 com folga | toca 0 | inclui valores bem negativos |
| p-valor permutação circular (fora da amostra; equivale ao percentil vs. aleatórias) | < 0,01 | 0,01–0,05 | > 0,05 |
| p-valor permutação com seleção (melhor de N) | < 0,01 | 0,01–0,05 | > 0,05 |
| DSR | ≥ 0,95 | 0,90–0,95 | < 0,90 |
| PBO | < 0,2 | 0,2–0,5 | > 0,5 |
