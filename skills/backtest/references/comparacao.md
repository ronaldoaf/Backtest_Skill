# Comparação de estratégias

Use quando a pergunta for "qual é melhor?": duas variantes de uma ideia, a estratégia nova contra a que já opera, várias candidatas para escolher uma, ou uma estratégia contra o benchmark. Funções em `scripts/comparacao.py` (só numpy; aceita DataFrame e Series do pandas).

## Por que métricas lado a lado não bastam

Com dados diários, o erro padrão de um Sharpe anualizado é cerca de `1 / sqrt(anos)`: 0,45 com 5 anos. Sharpe 1,2 e 0,9 estão a menos de um erro padrão de distância. A tabela mostra *o que aconteceu*; o teste diz *se dá para distinguir*. Sempre entregue os dois.

A correlação entre as estratégias importa nos dois sentidos:
- **Alta (> 0,7):** o ruído comum se cancela na diferença, então o teste detecta diferenças menores. Mas combinar as duas diversifica pouco.
- **Baixa:** a diferença é ruidosa e difícil de provar, mas combinar as duas pode ser melhor que escolher uma.

## Pré-requisitos

1. **Mesmas datas.** Monte um DataFrame com uma coluna por estratégia e alinhe (`df = pd.DataFrame(series).dropna()`). Comparar períodos diferentes compara regimes, não estratégias. As funções recusam Series do pandas com datas diferentes; arrays sem índice não têm como ser conferidos.
2. **Taxa livre de risco.** Passe `rf_por_periodo` com a taxa **por período** nas mesmas datas: no Brasil, o CDI diário (`cdi.reindex(df.index)`, ou `(1 + cdi_anual) ** (1 / 252) - 1` como aproximação). Sem isso o Sharpe usa rf = 0, e uma estratégia de baixa volatilidade que rende CDI + 1% ao ano parece ter Sharpe 12. A função recusa uma taxa anual passada por engano.
3. **Mesmo tratamento.** Mesma convenção de timing, mesmos custos e mesmo capital.
4. **Amostra honesta.** Compare séries fora da amostra (walk-forward concatenado ou holdout). Comparar curvas dentro da amostra mede quem foi mais otimizada.
5. **Mesmo risco.** Uma estratégia com o dobro de alavancagem rende o dobro sem ser melhor. Todos os testes comparam Sharpe por padrão; SPA e MCS fazem isso escalando as séries à mesma volatilidade (`criterio="sharpe"`). Use `criterio="retorno"` só quando o risco já for igual ou a pergunta for de retorno bruto.

## Qual teste usar

| Situação | Função | Hipótese nula |
|---|---|---|
| Visão geral, sempre | `tabela_comparativa` + `formatar_tabela` | (descritivo) |
| Duas estratégias definidas antes de ver os dados, 120 períodos ou mais | `teste_diferenca_sharpe` | Sharpe(A) = Sharpe(B) |
| Idem, menos de ~120 períodos (ex.: até 10 anos de dados mensais) | `bootstrap_diferenca_sharpe` | Sharpe(A) = Sharpe(B) |
| Poucas estratégias, todas contra todas | `comparar_pares` (p-valores de Holm) | Cada par igual |
| A melhor de N candidatas contra um benchmark | `teste_spa` | Nenhuma candidata bate o benchmark |
| Muitas candidatas, sem benchmark natural: quais sobrevivem? | `model_confidence_set` | Todas no conjunto são equivalentes |

Os testes com bootstrap usam blocos de `round(T ** (1/3))` períodos (3 a 5 em séries mensais, 10 a 14 em diárias) e devolvem o tamanho usado. Aumente `tamanho_bloco` se os retornos tiverem autocorrelação forte (ex.: posições que duram semanas em dados diários).

### Diferença de Sharpe: `teste_diferenca_sharpe`
Ledoit e Wolf (2008): método delta sobre os momentos das duas séries, com covariância HAC (Newey-West, kernel de Bartlett). Não supõe normalidade nem independência, ao contrário do Jobson-Korkie/Memmel, que rejeita demais com caudas gordas. Em simulação com H0 verdadeira, rejeitou a 5%:

- 4% a 6% em séries diárias de 1 a 10 anos (caudas t(4), GARCH, vols diferentes, correlação até 0,99);
- 6% a 8,5% com autocorrelação 0,3;
- 6% a 11% com 36 a 120 meses. Por isso, abaixo de ~120 períodos, decida pelo bootstrap.

Reporte a diferença anualizada, o IC 95% e o p-valor. Os Sharpes usam desvio com ddof = 1, como na tabela.

### Amostras curtas: `bootstrap_diferenca_sharpe`
Bootstrap estudentizado de Ledoit e Wolf (2008): reamostra A e B com os **mesmos** índices em blocos (preserva a correlação entre elas e a autocorrelação) e, em cada reamostra, divide a diferença pelo próprio erro padrão. Em simulação com 36 a 250 períodos e H0 verdadeira, rejeitou 3% a 5% a 5%, contra 5% a 9% do HAC. O custo é poder: cerca de 4 pontos a menos com 120 meses e 14 com 60, o preço de não declarar vencedoras falsas. Com autocorrelação 0,3, rejeitou 6% a 7%.

Também dá `prob_a_melhor`, a fração das reamostras em que A teve Sharpe maior. Serve para comunicar ("A foi melhor em 93% das reamostras"), mas não é a probabilidade de A ser melhor nem p-valor.

### Muitos pares: `comparar_pares`
Com N estratégias há N(N−1)/2 pares; com 5 estratégias de mesmo Sharpe, algum dos 10 pares deu p < 0,05 em 30% das simulações. Use `p_holm` (4,5% na mesma simulação). Cada linha traz também o IC da diferença, sem correção. Se o objetivo é escolher a melhor, prefira SPA ou MCS, que respondem a essa pergunta diretamente.

### A melhor contra o benchmark: `teste_spa`
Hansen (2005), com o White's Reality Check (2000) junto. Passe **todas** as candidatas testadas, inclusive as descartadas, e o benchmark: buy and hold, a estratégia atual ou, para "ganha do caixa", a série do CDI (nesse caso passe o CDI também em `rf_por_periodo`; um benchmark que quase não oscila sem isso é recusado). Use `p_spa`. Com todas as candidatas empatadas com o benchmark, rejeitou 6% a 7% a 5% em simulação: trate p entre 0,03 e 0,05 como limítrofe. O Reality Check é conservador: incluir candidatas muito ruins aumenta o p-valor dele, e o SPA corrige isso.

Relação com `teste_permutacao_selecao` (estatistica.py): aquele exige as posições e pergunta "a melhor bate estratégias aleatórias com o mesmo perfil de trades?"; o SPA só exige os retornos e pergunta "a melhor bate este benchmark?". Quando os dois se aplicam, rode os dois.

### Model Confidence Set: `model_confidence_set`
Hansen, Lunde e Nason (2011). Elimina a pior estratégia enquanto o teste rejeitar a equivalência do grupo restante e devolve o conjunto que contém a melhor com confiança 1 − alfa (padrão 90%). Cada estratégia recebe um p-valor MCS; fica quem tem p ≥ alfa. Com todas equivalentes, excluiu alguma por engano em 9% a 13% das simulações a alfa = 10%.

Leitura do resultado:
- **Conjunto com uma estratégia:** os dados apontam uma vencedora.
- **Conjunto grande:** os dados não distinguem as estratégias. Isso é uma conclusão, não uma falha: escolha pelo critério mais simples (menos parâmetros, menor custo, maior capacidade) ou combine-as.

## Armadilhas

- **Esquecer o CDI.** Sharpe com rf = 0 favorece estratégias de baixa volatilidade que só carregam o CDI. Num exemplo de 5 anos, CDI + 1% com vol de 1% contra CDI + 15% com vol de 10% deu "A muito melhor, p ≈ 0" com rf = 0 e "indistinguíveis, p = 0,89" em excesso do CDI.
- **Escolher no mesmo período em que se compara.** Se você escolheu as 3 finalistas olhando os mesmos dados, o teste entre elas herda o viés de seleção. Inclua todas as candidatas no SPA/MCS ou compare num período que não foi usado para escolher.
- **Testar só o par mais promissor.** Rodar `teste_diferenca_sharpe` no par com maior diferença visível é seleção. Use `comparar_pares` ou o MCS.
- **Concluir "A é igual a B" a partir de p alto.** p alto significa "não deu para distinguir", não "são iguais". Reporte o IC: se ele vai de −0,8 a +1,1, a amostra não diz nada.
- **Ignorar custos e capacidade.** Sharpe estatisticamente maior com o triplo do giro pode perder depois de custos dobrados (passo 4). Rode a comparação também no cenário de custos dobrados.
- **Comparar períodos ou frequências diferentes.** Converta tudo para a mesma frequência e as mesmas datas.

## No relatório

Diga qual taxa livre de risco foi usada. Inclua a tabela lado a lado, a matriz de correlação e uma tabela de testes, por exemplo:

| Comparação | Diferença de Sharpe (IC 95%) | p-valor | Método |
|---|---|---|---|
| Nova x atual | 0,35 (−0,20; 0,90) | 0,21 | HAC (Ledoit-Wolf) |
| Melhor de 12 x buy and hold | | p_spa = 0,03 | SPA (Sharpe) |
| MCS 90% | {nova, atual, combinada} | | T_max, Sharpe, bloco 10 |

Conclua com a decisão e o critério de desempate quando o teste não distingue as estratégias.
