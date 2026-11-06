# Comparação de estratégias

Use quando a pergunta for "qual é melhor?": duas variantes de uma ideia, a estratégia nova contra a que já opera, várias candidatas para escolher uma, ou uma estratégia contra o benchmark. Funções em `scripts/comparacao.py` (só numpy).

## Por que métricas lado a lado não bastam

O erro padrão do Sharpe anualizado é cerca de `sqrt((1 + SR²/2) / anos)`. Com 5 anos de dados, Sharpe 1,2 e 0,9 estão a menos de um erro padrão de distância. A tabela mostra *o que aconteceu*; o teste diz *se dá para distinguir*. Sempre entregue os dois.

A correlação entre as estratégias importa nos dois sentidos:
- **Alta (> 0,7):** o ruído comum se cancela na diferença, então o teste detecta diferenças menores. Mas combinar as duas diversifica pouco.
- **Baixa:** a diferença é ruidosa e difícil de provar, mas combinar as duas pode ser melhor que escolher uma.

## Pré-requisitos

1. **Mesmas datas.** Alinhe as séries (ex.: `pd.DataFrame(series).dropna()`). Comparar períodos diferentes compara regimes, não estratégias.
2. **Mesmo tratamento.** Mesma convenção de timing, mesmos custos, mesmo capital e mesmo benchmark de caixa.
3. **Amostra honesta.** Compare séries fora da amostra (walk-forward concatenado ou holdout). Comparar curvas dentro da amostra mede quem foi mais otimizada.
4. **Mesmo risco, quando o retorno bruto for o critério.** Uma estratégia com o dobro de alavancagem rende o dobro sem ser melhor. Compare Sharpe, ou escale as séries à mesma volatilidade (`normalizar_vol=True` no MCS).

## Qual teste usar

| Situação | Função | Hipótese nula |
|---|---|---|
| Visão geral, sempre | `tabela_comparativa` + `formatar_tabela` | (descritivo) |
| Duas estratégias definidas antes de ver os dados | `teste_diferenca_sharpe` | Sharpe(A) = Sharpe(B) |
| Idem, amostra curta (< ~250 períodos) ou caudas muito gordas | `bootstrap_diferenca_sharpe` | Sharpe(A) = Sharpe(B) |
| Poucas estratégias, todas contra todas | `comparar_pares` (p-valores de Holm) | Cada par igual |
| A melhor de N candidatas contra um benchmark | `teste_spa` | Nenhuma candidata bate o benchmark |
| Muitas candidatas, sem benchmark natural: quais sobrevivem? | `model_confidence_set` | Todas no conjunto são equivalentes |

### Diferença de Sharpe: `teste_diferenca_sharpe`
Ledoit e Wolf (2008): método delta sobre os momentos das duas séries, com covariância HAC (Newey-West, kernel de Bartlett). Não supõe normalidade nem independência, ao contrário do Jobson-Korkie/Memmel, que rejeita demais com caudas gordas. Em simulações com retornos t(4), correlação 0,6 e autocorrelação 0,3, rejeitou 5% a 7% das vezes a 5% sob H0. Reporte a diferença anualizada, o IC 95% e o p-valor.

`bootstrap_diferenca_sharpe` responde a mesma pergunta reamostrando A e B com os **mesmos** índices em blocos (preserva a correlação entre elas e a autocorrelação). Também dá `prob_a_melhor`, útil para comunicar ("A foi melhor em 93% das reamostragens"), mas que não é p-valor.

### Muitos pares: `comparar_pares`
Com N estratégias há N(N−1)/2 pares; com 6 estratégias, 15 testes, e algum p < 0,05 aparece por acaso. Use `p_holm`. Se o objetivo é escolher a melhor, prefira SPA ou MCS, que respondem a essa pergunta diretamente.

### A melhor contra o benchmark: `teste_spa`
Hansen (2005), com o White's Reality Check (2000) junto. Passe **todas** as candidatas testadas, inclusive as descartadas, e o benchmark (buy and hold, estratégia atual, ou zeros para "ganha do caixa"). Use `p_spa`. O Reality Check é conservador: incluir candidatas muito ruins aumenta o p-valor dele, e o SPA corrige isso. Compara retornos médios; para comparar Sharpe, escale as séries à mesma volatilidade antes.

Relação com `teste_permutacao_selecao` (estatistica.py): aquele exige as posições e pergunta "a melhor bate estratégias aleatórias com o mesmo perfil de trades?"; o SPA só exige os retornos e pergunta "a melhor bate este benchmark?". Quando os dois se aplicam, rode os dois.

### Model Confidence Set: `model_confidence_set`
Hansen, Lunde e Nason (2011). Elimina a pior estratégia enquanto o teste rejeitar a equivalência do grupo restante e devolve o conjunto que contém a melhor com confiança 1 − alfa (padrão 90%). Cada estratégia recebe um p-valor MCS; fica quem tem p ≥ alfa.

Leitura do resultado:
- **Conjunto com uma estratégia:** os dados apontam uma vencedora.
- **Conjunto grande:** os dados não distinguem as estratégias. Isso é uma conclusão, não uma falha: escolha pelo critério mais simples (menos parâmetros, menor custo, maior capacidade) ou combine-as.

## Armadilhas

- **Escolher no mesmo período em que se compara.** Se você escolheu as 3 finalistas olhando os mesmos dados, o teste entre elas herda o viés de seleção. Inclua todas as candidatas no SPA/MCS ou compare num período que não foi usado para escolher.
- **Testar só o par mais promissor.** Rodar `teste_diferenca_sharpe` no par com maior diferença visível é seleção. Use `comparar_pares` ou o MCS.
- **Concluir "A é igual a B" a partir de p alto.** p alto significa "não deu para distinguir", não "são iguais". Reporte o IC: se ele vai de −0,8 a +1,1, a amostra não diz nada.
- **Ignorar custos e capacidade.** Sharpe estatisticamente maior com o triplo do giro pode perder depois de custos dobrados (passo 4). Rode a comparação também no cenário de custos dobrados.
- **Comparar períodos ou frequências diferentes.** Converta tudo para a mesma frequência e as mesmas datas.

## No relatório

Inclua a tabela lado a lado, a matriz de correlação e uma tabela de testes, por exemplo:

| Comparação | Diferença de Sharpe (IC 95%) | p-valor | Método |
|---|---|---|---|
| Nova x atual | 0,35 (−0,20; 0,90) | 0,21 | HAC (Ledoit-Wolf) |
| Melhor de 12 x buy and hold | | p_spa = 0,03 | SPA |
| MCS 90% | {nova, atual, combinada} | | T_max, bloco 10 |

Conclua com a decisão e o critério de desempate quando o teste não distingue as estratégias.
