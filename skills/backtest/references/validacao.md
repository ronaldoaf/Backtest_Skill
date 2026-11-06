# Divisão de dados, walk-forward, robustez e regimes (passos 5, 6, 7 e 9)

## Passo 5: separar os dados

Divida **cronologicamente**, nunca aleatoriamente (séries temporais têm autocorrelação).

```
|------ treino (in-sample) ------|-- validação --|-- holdout --|
          ~60%                         ~20%           ~20% (mais recente)
```

- **Treino:** desenvolver a ideia, escolher features e regras.
- **Validação:** escolher parâmetros e comparar variantes.
- **Holdout:** rodar **uma vez** com tudo congelado. Registrar data e resultado. Se o resultado decepcionar, a resposta honesta é "rejeitada" ou "voltar ao passo 1 com nova hipótese", não reajustar.
- Se o histórico for curto, prefira walk-forward + holdout pequeno a uma divisão estática.

### Purging e embargo
Quando o rótulo ou a feature de uma observação depende de uma janela (ex.: retorno dos próximos 5 dias), observações do treino próximas ao teste vazam informação.
- **Purging:** remova do treino as observações cujo horizonte do rótulo cruza o período de teste.
- **Embargo:** remova também um intervalo após o teste (ex.: 1% das observações ou o horizonte do rótulo).
- Para modelos de ML, considere **Combinatorial Purged Cross-Validation (CPCV)** (López de Prado), que gera vários caminhos fora da amostra.

## Passo 6: walk-forward

```
janela 1: [ otimiza ][ testa ]
janela 2:      [ otimiza ][ testa ]
janela 3:           [ otimiza ][ testa ]
...
resultado oficial = concatenação dos trechos "testa"
```

Decisões a documentar:
- **Janela ancorada** (treino cresce desde o início) ou **móvel** (tamanho fixo). Móvel adapta-se a mudanças de regime; ancorada é mais estável.
- Tamanho do treino (ex.: 3 a 5 anos para diário) e do teste (ex.: 6 a 12 meses). O teste deve ter trades suficientes.
- Critério de otimização (preferir Sharpe ou retorno/drawdown a retorno bruto) e o espaço de busca.

Métricas do walk-forward:
- **Eficiência walk-forward (WFE)** = desempenho anualizado fora da amostra / desempenho anualizado dentro da amostra. Abaixo de ~0,5 indica overfitting.
- **Estabilidade dos parâmetros** entre janelas: se pulam de um extremo a outro, a otimização está ajustando ruído.
- Percentual de janelas de teste lucrativas.

## Passo 7: robustez de parâmetros

1. Monte uma grade em torno dos parâmetros escolhidos (±20 a 50% em cada um).
2. Gere mapa de calor da métrica (Sharpe) para pares de parâmetros.
3. Procure **platô**: região ampla onde o desempenho é parecido. O parâmetro escolhido deve estar no **centro** do platô, não na borda nem num pico.
4. Indicadores simples de fragilidade:
   - mediana da grade muito abaixo do ponto escolhido;
   - fração da grade com Sharpe > 0 baixa (ex.: < 70%);
   - mudança de 10% num parâmetro derruba o resultado pela metade.

Outros testes de robustez:
- Perturbar os dados (ruído pequeno nos preços, remover aleatoriamente 5 a 10% dos trades, deslocar a data de início).
- Trocar a frequência de execução (ex.: rebalancear em outro dia da semana/mês).
- Retirar os N melhores trades: o resultado continua positivo?
- Menos parâmetros é melhor. Cada parâmetro livre é mais uma oportunidade de overfitting.

## Passo 9: regimes e mercados

Quebre o resultado oficial (fora da amostra) por:
- **Tendência:** mercado em alta, baixa e lateral (ex.: benchmark acima/abaixo da média de 200 dias e inclinação).
- **Volatilidade:** tercis da volatilidade realizada ou do índice de volatilidade.
- **Eventos de estresse:** 2008, 2020, crises locais, choques de juros.
- **Ano a ano** e **mês a mês** (tabela).
- **Lado:** compras vs. vendas, se aplicável.

Generalização (sem reotimizar):
- Ativos semelhantes (outras ações do setor, outros índices, outras moedas).
- Outros mercados/países.
- Períodos anteriores ao usado no desenvolvimento, se houver dados.

Interpretação: não é preciso ganhar em todo regime, mas é preciso **saber** onde perde e confirmar que isso é coerente com a hipótese do passo 1. Se o lucro inteiro vem de um único regime ou de poucos meses, diga isso explicitamente.
