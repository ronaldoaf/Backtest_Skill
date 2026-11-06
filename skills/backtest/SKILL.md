---
name: backtest
description: Guia para planejar, executar e validar backtests de estratégias de trading/investimento sem se enganar. Use sempre que o usuário pedir para testar, otimizar ou avaliar uma estratégia com dados históricos, comparar duas ou mais estratégias, montar ou otimizar um portfólio de estratégias operando em conjunto (pesos, risk parity, HRP, Kelly, rebalanceamento), revisar um backtest existente, ou decidir se uma estratégia tem vantagem real (edge) antes de operar.
---

# Backtest robusto

Objetivo: descobrir se uma estratégia tem **vantagem real**, não produzir a curva de capital mais bonita possível. Um backtest é um experimento científico em que o adversário é o próprio pesquisador: cada escolha feita olhando o resultado aumenta o risco de overfitting. Trate todo resultado bom como suspeito até ele sobreviver aos 12 passos abaixo.

## Regras de ouro (nunca violar)

1. **Nenhuma decisão usa informação do futuro.** Sinal no fechamento do candle *t* executa, no mínimo, na abertura de *t+1*. Monte os retornos pela convenção de timing de `references/dados-e-lookahead.md`: `sinal.shift(1) * close.pct_change()` executa no fechamento de *t*, ao preço do próprio sinal, e viola esta regra (a única exceção, sinal calculado antes do leilão de fechamento, está na tabela).
2. **Custos sempre ligados.** Nunca reporte um resultado "sem custos" como principal.
3. **O holdout é tocado uma única vez.** Se você olhou e voltou a mexer, ele deixou de ser holdout; diga isso ao usuário.
4. **Conte todas as tentativas.** Mantenha um registro de cada variação testada (parâmetros, filtros, ativos, períodos). Esse número entra na correção de múltiplos testes.
5. **Reporte o que deu errado.** Períodos ruins, regimes em que a estratégia falha e testes reprovados fazem parte do resultado.

## Fluxo de trabalho

Siga os passos em ordem. Para cada um há detalhes e checklists em `references/`. Ao final, entregue o relatório no formato de `references/relatorio.md`.

**Código:** não reescreva o motor a cada backtest. `scripts/backtest.py` já aplica a convenção de timing, os custos, o carregamento (funding, swap, aluguel), o rendimento do caixa e extrai os trades (`retornos_estrategia`), e faz o walk-forward, a robustez, os regimes e os testes de look-ahead. `scripts/exemplo_completo.py` percorre os 12 passos numa estratégia de exemplo e imprime o rascunho do relatório: copie-o e troque `calcular_sinal` e a grade pela estratégia do usuário. Os scripts dependem só de `numpy` (o exemplo usa também `pandas`).

### Fase 1: Antes de tocar nos dados

**1. Hipótese antes dos dados.** Peça ou escreva, em uma ou duas frases, *por que* a vantagem deveria existir (viés comportamental, fricção estrutural, prêmio de risco, fluxo forçado) e *quem perde* do outro lado. Defina regras de entrada/saída, universo, frequência e horizonte **antes** de rodar qualquer coisa. Registre a hipótese no início do relatório. Sem hipótese, avise que o resultado é exploratório e exige uma barra estatística muito mais alta.

**2. Dados limpos e honestos.** Ver `references/dados-e-lookahead.md`. Verifique viés de sobrevivência (inclua deslistados), ajustes de dividendos/splits, rolagem de futuros, fuso e horário de fechamento (cripto negocia 24/7, forex fecha às 17h de Nova York), dados faltantes e outliers. Documente a fonte e o período.

**3. Sem look-ahead.** Ver `references/dados-e-lookahead.md`. Use a convenção de timing de lá e diga ao usuário qual execução foi assumida. Audite o código: indicadores calculados só com dados até *t*, `shift` correto, dados fundamentalistas pela data de **publicação** e não pela data de referência, normalizações/escalas ajustadas só no treino, nenhum `fillna` para trás (`bfill`). Rode `teste_futuro_truncado` (sinais até uma data não mudam quando chegam dados novos) e `teste_atraso`.

**4. Custos realistas.** Ver `references/custos.md`, que tem referências por mercado (cripto, forex, commodities, ações americanas). Comissão, spread, slippage, impacto de mercado proporcional ao volume, carregamento (funding de perpétuos, swap de forex, aluguel para vender a descoberto, rolagem de futuros). Rode também um cenário com custos dobrados: se a vantagem some, ela é frágil (`sensibilidade_custos` dá a tabela e o custo de breakeven). Monte o custo com `custo_por_lado` e o carregamento com `financiamento_por_periodo` e `aluguel_por_periodo` em `retornos_estrategia`.

### Fase 2: Desenvolvimento e validação

**5. Separar dados.** Ver `references/validacao.md`. Divida cronologicamente em treino (desenvolvimento), validação (ajuste) e holdout (teste final, uma vez). Sugestão: 60/20/20, com holdout sendo o período mais recente. Use *purging* e *embargo* quando rótulos ou features se sobrepõem no tempo.

**6. Walk-forward.** Ver `references/validacao.md`. Otimize na janela *k*, teste na janela *k+1*, avance e repita. O resultado oficial é a concatenação apenas dos trechos fora da amostra. Compare o desempenho dentro e fora da amostra (eficiência walk-forward). `walk_forward_parametros` faz isso e guarda a matriz de todas as configurações para o passo 8.

**7. Robustez de parâmetros.** Ver `references/validacao.md`. Faça uma grade em torno dos parâmetros escolhidos (±20 a 50%) e mostre um mapa de calor. Exija um **platô**: vizinhos com desempenho parecido. Pico isolado = overfitting. Prefira poucos parâmetros. `robustez_parametros` e `mapa_calor` dão a grade e os alertas.

### Fase 3: Provar que não é sorte

**8. Testes estatísticos.** Ver `references/estatistica.md` e use `scripts/estatistica.py`. Cada teste vale para uma amostra específica (tabela no início de `estatistica.md`). No mínimo:
- comparar com benchmark (buy and hold, índice);
- na série fora da amostra (walk-forward concatenado ou holdout): bootstrap para o IC do Sharpe e permutação circular dos sinais, que equivale a comparar com estratégias aleatórias de mesma exposição, número e duração de trades;
- sobre o resultado otimizado (a melhor entre várias configurações): **Deflated Sharpe Ratio** com o número total de tentativas registradas (regra de ouro 4), **PBO** e a permutação com correção de seleção (`teste_permutacao_selecao`). Bootstrap e permutação simples aplicados à melhor de N configurações aprovam ruído: na melhor de 50 estratégias aleatórias, cerca de 80% dão p < 0,05;
- Monte Carlo dos trades ou retornos para a distribuição de drawdown, que alimenta o passo 12.

**Comparando estratégias.** Quando a pergunta for "qual é melhor?" (variantes, nova contra a atual, várias candidatas, estratégia contra benchmark), siga `references/comparacao.md` e use `scripts/comparacao.py`. Nunca declare uma vencedora só pela tabela de métricas: alinhe as séries nas mesmas datas (um DataFrame com uma coluna por estratégia), passe a taxa livre de risco por período em `rf_por_periodo` quando o caixa render juros (ex.: T-bill em USD), compare fora da amostra e use o teste que corresponde à pergunta:
- duas estratégias definidas antes de ver os dados: `teste_diferenca_sharpe` (Ledoit-Wolf, HAC); com menos de ~120 períodos, `bootstrap_diferenca_sharpe` (estudentizado), porque o HAC rejeita demais em amostras curtas;
- várias, todas contra todas: `comparar_pares` (p-valores de Holm);
- a melhor de N contra um benchmark: `teste_spa` com todas as candidatas, inclusive as descartadas;
- escolher entre muitas: `model_confidence_set`. Se o conjunto final tiver várias estratégias, diga que os dados não as distinguem e desempate por simplicidade, custo ou capacidade (ou combine-as).
SPA e MCS comparam Sharpe por padrão (`criterio="sharpe"`), para que alavancagem não pareça vantagem.

**Portfólio de estratégias.** Quando o usuário for operar várias estratégias juntas e perguntar quanto capital dar a cada uma, siga `references/portfolio.md` e use `scripts/portfolio.py`. Otimizar pesos é otimizar parâmetros: a referência é 1/N, que não estima nada e é difícil de bater fora da amostra.
- Entrada: retornos **fora da amostra** de cada estratégia (walk-forward concatenado), alinhados, com a taxa livre de risco em `rf_por_periodo` (o caixa rende essa taxa; Sharpe e Kelly em excesso dela).
- Primeiro `analise_diversificacao`: pares redundantes, perdas conjuntas nos piores dias além do que a dependência comum explica, contribuição de risco e número efetivo de apostas. Tirar estratégias redundantes olhando a amostra inteira conta como tentativa. Com menos de ~500 períodos, a ausência de alerta de perdas conjuntas não prova nada.
- Depois `comparar_metodos` (walk-forward com custo de rebalanceamento, limites de peso, teste contra 1/N com Holm, MCS e alertas). Prefira métodos que só usam a covariância (`risk_parity`, `hrp`, `vol_inversa`); `max_sharpe` e `kelly` dependem de médias e só valem se baterem 1/N com folga. Kelly sempre fracionado.
- Fique com 1/N ou o método mais simples dentro do MCS, salvo vitória significativa. Opere os `pesos_atuais` do walk-forward do método escolhido (mesmos parâmetros). Cada método e parâmetro testado conta como tentativa (regra de ouro 4), e o portfólio escolhido passa pelos passos 8 a 12 como uma estratégia.

**9. Diferentes regimes e mercados.** Ver `references/validacao.md`. Quebre o resultado por regime (alta, baixa, lateral; alta e baixa volatilidade; crises) e por ano. Teste em ativos ou mercados parecidos sem reotimizar. Uma vantagem que só existe num ano ou num ativo não é vantagem. Use `quebrar_por`, `por_ano`, `regime_tendencia` e `regime_volatilidade`.

**10. Métricas além do retorno.** Ver `references/metricas.md`. CAGR, volatilidade, Sharpe, Sortino, Calmar, drawdown máximo e sua duração, número de trades, taxa de acerto, payoff, expectativa por trade, exposição, turnover, capacidade. Sinalize amostras pequenas (menos de ~100 trades ou menos de ~3 anos de dados diários são pouco conclusivos). Use `periodos_por_ano` = 252 em ações e futuros, 365 em cripto e ~260 em forex. Passe `rf_por_periodo` (taxa livre de risco, se houver) em `resumo_metricas` e `bootstrap_metricas`, e os trades de `retornos_estrategia` em `metricas_trades`.

### Fase 4: Do papel para o mercado

**11. Paper trading / incubação.** Ver `references/incubacao-e-monitoramento.md`. Proponha um período de incubação (ao vivo sem dinheiro, ou com tamanho mínimo) e como comparar execução real com o backtest (slippage observado, sinais idênticos, desvio do retorno).

**12. Monitoramento.** Ver `references/incubacao-e-monitoramento.md`. Defina **antes** de operar os critérios de desligamento, derivados do Monte Carlo (ex.: drawdown acima do percentil 95 simulado, sequência de perdas improvável, Sharpe móvel abaixo de um limite), e a rotina de revisão. Dimensione pelo risco, não pelo retorno do backtest: alvo de volatilidade (`posicao_vol_alvo`) e, como teto, no máximo 1/4 de Kelly (`kelly_estrategia`, que mostra o IC do Kelly).

## Como se comportar

- **Seja cético por padrão.** Se o Sharpe fora da amostra for maior que ~2 em estratégias diárias simples, procure bug ou look-ahead antes de comemorar.
- **Pergunte o que faltar** (mercado, custos da corretora ou exchange, capital, frequência), mas comece com suposições conservadoras explícitas em vez de travar.
- **Mostre o código e as suposições**, para que o usuário possa auditar. Prefira código vetorizado simples e legível; quando houver ordens complexas, use um motor orientado a eventos.
- **Não otimize no holdout.** Se o usuário pedir para "ajustar só mais um pouco" depois de ver o holdout, explique que o teste perde validade e registre isso.
- **Veredito honesto.** Termine com uma das classificações: *Rejeitada*, *Inconclusiva*, *Promissora (incubar)* ou *Aprovada para operar com tamanho reduzido*, justificando com os testes.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `references/dados-e-lookahead.md` | Passos 2 e 3: qualidade dos dados e auditoria de look-ahead |
| `references/custos.md` | Passo 4: modelagem de custos e execução |
| `references/validacao.md` | Passos 5, 6, 7 e 9: divisão, walk-forward, robustez e regimes |
| `references/estatistica.md` | Passo 8: testes, DSR, PBO, Monte Carlo |
| `references/comparacao.md` | Passo 8: comparar duas ou mais estratégias (diferença de Sharpe, SPA, MCS) |
| `references/portfolio.md` | Passo 8: portfólio de estratégias (alocação, walk-forward dos pesos, diversificação) |
| `references/metricas.md` | Passo 10: definições e limites das métricas |
| `references/incubacao-e-monitoramento.md` | Passos 11 e 12 |
| `references/relatorio.md` | Modelo do relatório final |
| `scripts/backtest.py` | Motor vetorizado: retorno pela convenção de timing com custos, carregamento, rendimento do caixa e trades; sensibilidade e breakeven de custos; montagem do custo por lado; walk-forward de parâmetros; robustez e mapa de calor; regimes e ano a ano; testes de look-ahead; alvo de volatilidade e Kelly |
| `scripts/exemplo_completo.py` | Os 12 passos de ponta a ponta numa estratégia de médias móveis (dados sintéticos ou CSV), com rascunho do relatório: modelo para adaptar |
| `scripts/estatistica.py` | Funções prontas: métricas, bootstrap, permutação simples e com correção de seleção, DSR, PBO |
| `scripts/comparacao.py` | Comparação de estratégias: tabela lado a lado, diferença de Sharpe (HAC e bootstrap), pares com Holm, SPA/Reality Check, Model Confidence Set |
| `scripts/portfolio.py` | Portfólio de estratégias: 1/N, vol inversa, risk parity, HRP, mínima variância, max Sharpe e Kelly fracionado com covariância de Ledoit-Wolf; walk-forward com custos e faixa de rebalanceamento; diversificação; comparação com 1/N e alertas de overfitting |
