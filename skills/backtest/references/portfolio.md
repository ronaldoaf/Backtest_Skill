# Portfólio de estratégias

Use quando o usuário tiver duas ou mais estratégias e quiser operá-las em conjunto: quanto capital dar a cada uma, com que frequência rebalancear e se a combinação vale a pena. Funções em `scripts/portfolio.py` (só numpy; aceita DataFrame e Series do pandas).

A pergunta muda de "esta estratégia tem vantagem?" para "quanto de cada?". A resposta honesta costuma ser modesta: com o histórico típico de uma estratégia, os pesos "ótimos" dentro da amostra são dominados por ruído, e a alocação igual (1/N) é difícil de bater fora dela (DeMiguel, Garlappi e Uppal, 2009). Otimizar pesos é otimizar parâmetros e segue as mesmas regras: walk-forward, contagem de tentativas, comparação com uma referência simples.

## Pré-requisitos

1. **Cada estratégia já passou pelos passos 1 a 10 sozinha.** O portfólio não salva estratégia sem vantagem; só dilui as boas.
2. **Retornos fora da amostra de cada estratégia.** Use o walk-forward concatenado de cada uma, não a curva otimizada. Otimizar pesos sobre curvas dentro da amostra soma o overfitting de cada estratégia ao da alocação: a estratégia mais ajustada parece a melhor e leva o maior peso.
3. **Mesmas datas e mesma base.** DataFrame com uma coluna por estratégia, alinhado (`pd.DataFrame(series).dropna()`). Cada coluna é o retorno da estratégia sobre o capital alocado a ela, líquido dos custos dela. Converta para a mesma frequência antes.
4. **Taxa livre de risco.** Se o caixa rende juros, passe `rf_por_periodo` por período nas mesmas datas (ex.: `tbill.reindex(df.index)` em USD; em cripto, muitas vezes 0). O caixa não alocado rende rf, a alavancagem custa rf, e Sharpe e Kelly usam excesso de rf. Se a estratégia for uma posição comprada financiada (ex.: ações compradas à vista), o retorno dela já embute o custo de oportunidade; se for long-short, em futuros ou perpétuos, o capital parado rende rf e deve entrar no retorno da estratégia.
5. **Estratégias no mesmo ativo.** Se duas estratégias operam o mesmo ativo, na prática as ordens se compensam (netting) e o custo real é menor que a soma dos custos de cada uma; a margem exigida também muda. Os retornos por estratégia continuam válidos para alocar; registre a diferença de custo como ressalva ou simule a carteira consolidada.

## Métodos de alocação

Do mais robusto ao mais frágil. Todos aceitam `peso_min` e `peso_max` e, exceto 1/N, usam a covariância com encolhimento de Ledoit-Wolf (`covariancia_ledoit_wolf`: correlações puxadas para zero na medida do ruído, volatilidades preservadas).

| Método | O que estima | Quando usar | Risco de overfitting |
|---|---|---|---|
| `iguais` (1/N) | nada | sempre, como referência; padrão quando nada o bate | nenhum |
| `vol_inversa` | volatilidades | estratégias com vols muito diferentes e correlações parecidas | baixo |
| `risk_parity` | covariância | contribuições iguais ao risco; bom padrão quando as correlações diferem | baixo |
| `hrp` | correlação (agrupamento) e variâncias | muitas estratégias, grupos de estratégias parecidas, covariância mal condicionada | baixo |
| `min_variancia` | covariância | quando o objetivo é reduzir risco; tende a concentrar nas de menor vol (use `peso_max`) | médio |
| `max_sharpe` | médias e covariância | só se bater 1/N com folga fora da amostra e em vários períodos | alto |
| `kelly` (fracionado) | médias e covariância | dimensionar a exposição total e a parte em caixa; nunca Kelly completo | alto |

Notas:
- **Médias são o problema.** O erro padrão da média anual de uma estratégia com vol de 10% e 5 anos de dados é cerca de 4,5% ao ano, da ordem da própria vantagem. `max_sharpe` e `kelly` encolhem as médias para a média comum (`encolher_media`, padrão 0,5). Com `encolher_media=1`, `max_sharpe` vira a mínima variância. O encolhimento só reduz as diferenças entre as médias: o nível geral, que é o que superdimensiona o Kelly, só é controlado por `fracao_kelly`.
- **Kelly.** `kelly` maximiza μ'w − w'Σw / (2f) em excesso de rf, com f = `fracao_kelly` (padrão 0,5, meio Kelly): sem limites ativos, dá exatamente f × Kelly completo. A soma dos pesos vai até `alavancagem_max` (padrão 1) e `peso_min`/`peso_max` valem para os pesos finais; a sobra fica no caixa. Com estratégias boas, meio Kelly costuma passar de 100%, e o resultado é investir 100% (não 50%). Kelly completo com médias estimadas superdimensiona: se a média verdadeira for metade da estimada, Kelly completo vira o dobro do ótimo, com o crescimento de quem não aposta nada e muito mais risco. Se desconfiar do nível das médias, reduza a fração. Não combine com `vol_alvo` (dá erro): o alvo de vol substituiria o tamanho de Kelly.
- **HRP.** Agrupa as estratégias pela correlação e divide o capital de cima para baixo na árvore. Não inverte a covariância, então não explode com estratégias quase colineares. Esta implementação divide nos ramos da árvore (o artigo original divide a lista ordenada ao meio, o que pode separar estratégias do mesmo grupo).
- **Risk parity com orçamento.** `orcamento_risco` dá a fração do risco de cada estratégia (ex.: menos risco para a mais nova, ainda em incubação).
- **Limites.** `peso_max` é a defesa mais simples contra concentração por erro de estimação: com N estratégias, algo entre 1,5/N e 2/N. Aceitam um número, uma lista na ordem das colunas ou um dict `{nome: limite}`. Nos métodos sem otimização (`iguais`, `vol_inversa`, `risk_parity`, `hrp`), quem passa do limite fica nele e as demais mantêm as proporções do método (o excesso é repartido proporcionalmente, não em partes iguais). Com limite ativo, as contribuições ao risco do risk parity deixam de ser exatamente iguais.
- **Tamanho total.** `vol_alvo` (ex.: 0.08) no walk-forward escala os pesos pela vol estimada na janela, com soma até `alavancagem_max`; a sobra fica no caixa. Serve para definir o tamanho total e para comparar drawdown e CAGR dos métodos no mesmo risco. O Sharpe não depende da escala, mas muda um pouco, porque a exposição passa a variar com a vol estimada.

## Fluxo

1. **Diagnóstico de diversificação** (`analise_diversificacao`) com 1/N. Veja:
   - `correlacao_media` e `pares_redundantes` (correlação > 0,8): estratégias redundantes são a mesma aposta. Para juntar ou tirar uma, decida com dados anteriores ao período de teste (por exemplo, a primeira janela); se decidir olhando a amostra inteira, o conjunto novo é mais uma tentativa e entra na contagem.
   - `perdas_conjuntas` e `pares_perdem_juntos`: chance de uma estratégia ter um dia ruim dado que a outra teve. Dia ruim é um dos 10% piores dias da estratégia, contando só dias com perda (dias parados não contam; `dias_ruins` dá a contagem). O esperado vem de simulações com as mesmas distribuições e a mesma correlação de postos, sem dependência extra nas caudas, e o alerta só aparece quando o excesso passa o que o acaso produz entre todos os pares (`limiar_perdas_conjuntas`). Se aparecer, a diversificação falha nas crises, quando mais importa. Com menos de ~500 períodos (dados mensais, por exemplo) o teste tem pouco poder: a ausência de alerta não prova que as estratégias não caem juntas. Não meça isso pela correlação nos piores dias do portfólio: condicionar na soma cria correlação negativa artificial.
   - `contribuicao_risco`: peso pequeno com fatia grande do risco é concentração escondida.
   - `razao_diversificacao` (1 = nenhuma; √N com N independentes de mesma vol) e `n_efetivo_autovalores` (quantos fatores independentes existem). Cinco estratégias com 1,5 aposta efetiva são, na prática, uma estratégia e meia.
2. **Walk-forward dos métodos** (`comparar_metodos`): para cada rebalanceamento, os pesos usam só a janela anterior (`janela`, móvel ou `ancorada=True`) e valem dali em diante. Entre rebalanceamentos os pesos derivam com os retornos. O resultado oficial é a série concatenada fora da amostra.
   - `janela`: pelo menos 20 × N períodos (ex.: 2 anos de dados diários para até 12 estratégias). Menos que isso gera alerta.
   - `rebalancear_a_cada`: mensal (21 dias úteis, 30 em cripto) é um bom padrão. Rebalancear mais vezes paga mais custo para seguir estimativas ruidosas.
   - `custo_giro`: custo por unidade de capital movimentada entre estratégias (0,001 = 10 bps sobre o valor transferido), adicional ao custo das estratégias. Use o custo real de redimensionar as posições (passo 4). A alocação inicial também paga.
   - `faixa`: banda sem operação (ex.: 0,05); só rebalanceia quando algum peso se afasta mais que isso do alvo. Reduz o giro quase sem mudar o risco.
3. **Ler o resultado** (`formatar_comparacao`):
   - Sharpe fora da amostra de cada método, `sharpe_dentro` (dos pesos na própria janela de estimação, sem custo nem deriva: otimista por construção) e `eficiencia` = fora / dentro. Abaixo de ~0,5, os pesos se ajustam a ruído. Ela é pouco informativa quando o Sharpe dentro é negativo ou perto de zero, e não é calculada para 1/N, que não estima nada.
   - Teste de cada método contra 1/N (`contra_iguais`): HAC de Ledoit-Wolf, ou bootstrap estudentizado com menos de 120 períodos fora da amostra, com p-valores de Holm.
   - Model Confidence Set (90%, Sharpe): métodos que os dados não conseguem descartar. Conjunto grande é o resultado comum; desempate pelo mais simples e de menor giro.
   - `alertas`: lista pronta de problemas (janela curta, eficiência baixa, método baseado em médias, nada bate 1/N, pares redundantes, perdas conjuntas, giro alto).
4. **Decidir.** Fique com 1/N, ou com o método mais simples que estiver no MCS, a não ser que um método bata 1/N com p de Holm < 0,05 e com eficiência razoável. Pesos atuais para operar: `res["resultados"][metodo]["pesos_atuais"]` (ou `walk_forward_portfolio(...)["pesos_atuais"]`), estimados com as últimas linhas e os mesmos parâmetros, inclusive `vol_alvo` e `ancorada`; `formatar_comparacao` já mostra a tabela. No histórico `pesos`, `executado=False` marca rebalanceamentos que a faixa evitou.
5. **Validar o portfólio como estratégia.** A série fora da amostra do método escolhido passa pelos passos 8 a 12: IC do Sharpe, Monte Carlo do drawdown, regimes, incubação e regras de desligamento do portfólio inteiro, além das de cada estratégia.

## Armadilhas

- **Escolher o método olhando o resultado.** Cada método e cada combinação de parâmetros (janela, frequência, limites, `encolher_media`) testado é uma tentativa: entra na contagem da regra de ouro 4 e no Deflated Sharpe Ratio. O MCS e o teste com Holm já tratam os métodos comparados juntos, mas não os que você testou e não reportou.
- **Escolher as estratégias com os mesmos dados.** Se as estratégias do portfólio foram selecionadas entre muitas no mesmo período, o portfólio herda o viés de seleção. Use séries fora da amostra e registre as estratégias descartadas. O mesmo vale para tirar estratégias "redundantes" olhando a correlação da amostra inteira antes do walk-forward.
- **Max Sharpe dentro da amostra.** O Sharpe do portfólio otimizado na amostra inteira é sempre o maior da tabela e não diz nada. Só o walk-forward conta.
- **Correlação estável.** Correlações mudam com o regime e sobem em crises. Olhe `perdas_conjuntas`, quebre o resultado por regime (passo 9) e dimensione pelo pior cenário, não pela média.
- **Esquecer o custo de rebalancear.** Métodos baseados em médias costumam girar várias vezes mais que 1/N. Compare líquido e rode também com custo dobrado.
- **Alavancar para "chegar no alvo".** `vol_alvo` com `alavancagem_max` > 1 toma emprestado a rf; o custo real do financiamento ou da margem costuma ser maior. Inclua o spread no retorno ou não alavanque.
- **Kelly completo.** Nunca. Meio Kelly ou menos, e só como teto da exposição total.

## No relatório

Inclua, depois das métricas de cada estratégia:

| Item | Resultado |
|---|---|
| Correlação média / pares redundantes | |
| Perdas conjuntas nos piores 10% (observado x esperado) | |
| Apostas efetivas (autovalores) / razão de diversificação | |
| Esquema do walk-forward (janela, rebalanceamento, faixa, custo_giro, limites, vol_alvo) | |
| Métodos testados (todos) | |

| Método | Sharpe fora | Sharpe dentro | Eficiência | Max DD | Giro anual | Diferença de Sharpe x 1/N (IC) | p Holm |
|---|---|---|---|---|---|---|---|
| 1/N | | | | | | — | — |
| ... | | | | | | | |

- MCS (90%):
- Método escolhido, pesos atuais e critério de desempate:
- Regras de desligamento do portfólio e de cada estratégia:
