"""Portfólio de estratégias: alocação de capital entre estratégias que operam em conjunto.

Dependência: apenas numpy (e `estatistica.py` e `comparacao.py`, na mesma pasta). Aceita
objetos do pandas sem importá-lo.

Pergunta que responde: "dado um conjunto de estratégias já validadas, quanto capital dar a
cada uma?", sem se enganar com a otimização dos pesos. Otimizar pesos é otimizar parâmetros:
os pesos ótimos dentro da amostra quase sempre decepcionam fora dela, e a referência honesta é
a alocação igual (1/N), que não estima nada.

Convenções (as mesmas de `comparacao.py`):
- `retornos`: DataFrame (uma coluna por estratégia), dict {nome: série 1D} ou matriz T x N (com
  `nomes`), retornos simples por período de cada estratégia sobre o capital alocado a ela,
  líquidos dos custos dela, **alinhados nas mesmas datas** e, de preferência, **fora da
  amostra** (o walk-forward concatenado de cada estratégia). Otimizar pesos sobre curvas
  dentro da amostra soma o overfitting de cada estratégia ao da alocação.
- `rf_por_periodo`: taxa livre de risco POR PERÍODO (número ou série alinhada). Em USD, o
  T-bill: `(1 + taxa_anual) ** (1 / 252) - 1` ou a série diária. O caixa não alocado rende rf, o
  caixa tomado emprestado (alavancagem) custa rf, e Sharpe e Kelly usam retornos em excesso.
- Pesos são frações do capital total. Nos métodos sem caixa somam 1; em `kelly` e com
  `vol_alvo`, a sobra (1 − soma) fica no caixa, rendendo rf.
- `periodos_por_ano`: 252 diário, 365 cripto, 52 semanal, 12 mensal.

Métodos de alocação (`METODOS`):
- "iguais": 1/N. A referência que todo método precisa bater fora da amostra.
- "vol_inversa": peso proporcional a 1 / volatilidade. Ignora correlações.
- "risk_parity": contribuições iguais ao risco (equal risk contribution). Usa a covariância.
- "hrp": Hierarchical Risk Parity (López de Prado, 2016). Agrupa estratégias parecidas antes
  de dividir o risco; não inverte a covariância, por isso é estável com muitas estratégias.
- "min_variancia": menor variância com os limites de peso. Usa só a covariância.
- "max_sharpe": maior Sharpe estimado. Usa médias, que são o insumo mais ruidoso: é o método
  mais propenso a overfitting. As médias são encolhidas para a média comum (`encolher_media`).
- "kelly": Kelly fracionado (padrão meio Kelly) em excesso de rf, com alavancagem limitada.
  A fração entra na otimização (sem limites ativos dá f × Kelly completo) e os limites valem
  para os pesos finais. A sobra fica no caixa. Também usa médias: mesmo cuidado de "max_sharpe".
Todos (menos "iguais") usam a covariância com encolhimento de Ledoit-Wolf
(`covariancia_ledoit_wolf`), que reduz o erro de estimação quando há poucos períodos por
estratégia.

Funções principais:
- `pesos_portfolio`: pesos de um método a partir de uma janela de retornos.
- `simular_portfolio`: retorno do portfólio com pesos que derivam entre rebalanceamentos e
  custo de rebalanceamento sobre o giro.
- `walk_forward_portfolio`: estima os pesos só com o passado, aplica no período seguinte e
  concatena (o resultado oficial). Mede também o Sharpe dentro da amostra e a eficiência, e
  devolve `pesos_atuais`, os pesos para operar hoje com os mesmos parâmetros.
- `analise_diversificacao`: correlações (inclusive em estresse), contribuição de risco,
  razão de diversificação e número efetivo de apostas.
- `comparar_metodos` / `formatar_comparacao`: walk-forward de vários métodos, tabela, teste de
  cada um contra 1/N, Model Confidence Set e alertas de overfitting.

Uso rápido:
    from portfolio import comparar_metodos, formatar_comparacao, analise_diversificacao

    df = pd.DataFrame({"momentum": r_mom, "reversao": r_rev, "carry": r_carry}).dropna()
    rf = tbill_diario.reindex(df.index)
    print(analise_diversificacao(df, rf_por_periodo=rf))
    res = comparar_metodos(df, janela=504, rebalancear_a_cada=21, custo_giro=0.001,
                           peso_max=0.5, rf_por_periodo=rf)
    print(formatar_comparacao(res))                  # inclui os pesos atuais de cada método
    res["resultados"]["risk_parity"]["pesos_atuais"]  # {nome: peso, "caixa": ...}

Execute `python portfolio.py` para uma demonstração com dados sintéticos.
"""
from __future__ import annotations

import itertools
import math
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comparacao import (  # noqa: E402
    _painel, _rf, bootstrap_diferenca_sharpe, model_confidence_set, tabela_comparativa,
    teste_diferenca_sharpe,
)
from estatistica import max_drawdown, sharpe  # noqa: E402

METODOS = ("iguais", "vol_inversa", "risk_parity", "hrp", "min_variancia", "max_sharpe", "kelly")
METODOS_COM_MEDIA = ("max_sharpe", "kelly")


# ---------------------------------------------------------------------------
# Covariância
# ---------------------------------------------------------------------------

def covariancia_ledoit_wolf(retornos) -> tuple[np.ndarray, float]:
    """Covariância com a correlação encolhida para a identidade (Ledoit e Wolf, 2004).

    Padroniza cada série, encolhe a matriz de correlação amostral para a identidade com a
    intensidade ótima de Ledoit-Wolf e volta às volatilidades amostrais. Preserva a
    volatilidade de cada estratégia e só puxa as correlações para zero, na medida do ruído.
    Retorna (covariância, intensidade entre 0 e 1). Intensidade alta = poucos períodos para o
    número de estratégias: as correlações estimadas são pouco confiáveis.
    """
    X = np.asarray(retornos, dtype=float)
    if X.ndim != 2 or X.shape[0] < 3:
        raise ValueError("covariancia_ledoit_wolf precisa de uma matriz T x N com T >= 3.")
    T, N = X.shape
    sd = X.std(axis=0, ddof=1)
    if np.any(sd <= 0):
        raise ValueError("Uma estratégia tem volatilidade zero na janela; a covariância não serve.")
    if N == 1:
        return np.array([[sd[0] ** 2]]), 0.0
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    S = Z.T @ Z / T                                        # correlação (ddof=0)
    alvo = np.eye(N) * np.trace(S) / N
    d2 = float(np.sum((S - alvo) ** 2))
    # (1/T²) Σ_t ‖z_t z_t' − S‖² = (Σ_t ‖z_t‖⁴ / T − ‖S‖²) / T
    b2 = float((np.sum(np.sum(Z ** 2, axis=1) ** 2) / T - np.sum(S ** 2)) / T)
    delta = 0.0 if d2 <= 0 else min(1.0, b2 / d2)
    C = delta * alvo + (1 - delta) * S
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)
    return C * np.outer(sd, sd), float(delta)


def _cov(R: np.ndarray, encolher_cov: bool) -> np.ndarray:
    if encolher_cov:
        return covariancia_ledoit_wolf(R)[0]
    S = np.atleast_2d(np.cov(R, rowvar=False))
    if np.any(np.diag(S) <= 0):
        raise ValueError("Uma estratégia tem volatilidade zero na janela; a covariância não serve.")
    return S


# ---------------------------------------------------------------------------
# Limites de peso e projeção
# ---------------------------------------------------------------------------

def _limites(peso_min, peso_max, N: int, soma: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    lo = np.broadcast_to(np.asarray(peso_min, dtype=float), (N,)).copy()
    hi = np.broadcast_to(np.asarray(peso_max, dtype=float), (N,)).copy()
    if np.any(lo > hi) or not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise ValueError("peso_min deve ser <= peso_max, ambos finitos.")
    if lo.sum() > soma + 1e-12 or hi.sum() < soma - 1e-12:
        raise ValueError(
            f"Limites impossíveis: com {N} estratégias, soma dos mínimos = {lo.sum():.3f} e soma "
            f"dos máximos = {hi.sum():.3f}, mas os pesos precisam somar {soma:g}. "
            f"Ex.: peso_max >= 1/N = {1 / N:.3f}.")
    return lo, hi


def _projetar(v: np.ndarray, lo: np.ndarray, hi: np.ndarray, soma: float) -> np.ndarray:
    """Projeção euclidiana em {lo <= w <= hi, sum(w) = soma} (busca exata nos pontos de quebra)."""
    quebras = np.sort(np.concatenate([v - lo, v - hi]))
    h = np.clip(v[None, :] - quebras[:, None], lo, hi).sum(axis=1)   # não crescente em tau
    k = int(np.searchsorted(-h, -soma, side="left"))
    if k == 0:
        tau = quebras[0]
    elif k >= len(quebras):
        tau = quebras[-1]
    else:
        t0, t1, h0, h1 = quebras[k - 1], quebras[k], h[k - 1], h[k]
        tau = t0 if h0 == h1 else t0 + (h0 - soma) * (t1 - t0) / (h0 - h1)
    return np.clip(v - tau, lo, hi)


def _projetar_ate(v, lo, hi, soma_max):
    """Projeção em {lo <= w <= hi, sum(w) <= soma_max}."""
    w = np.clip(v, lo, hi)
    return w if w.sum() <= soma_max else _projetar(v, lo, hi, soma_max)


def _limitar(w: np.ndarray, lo: np.ndarray, hi: np.ndarray, soma: float = 1.0) -> np.ndarray:
    """Limites nos métodos heurísticos: w_i' = clip(c × w_i, lo_i, hi_i), com c tal que a soma dê `soma`.

    Quem bate no limite fica nele e as demais mantêm as proporções entre si (o excesso é
    repartido proporcionalmente ao peso, não em partes iguais como na projeção euclidiana, que
    daria mais risco justamente às estratégias mais voláteis). Pesos devem ser positivos.
    """
    w = np.asarray(w, dtype=float)
    if np.any(w <= 0):
        return _projetar(w, lo, hi, soma)
    quebras = np.sort(np.concatenate([lo / w, hi / w]))
    h = np.clip(quebras[:, None] * w[None, :], lo, hi).sum(axis=1)   # não decrescente em c
    k = int(np.searchsorted(h, soma, side="left"))
    if k == 0:
        c = quebras[0]
    elif k >= len(quebras):
        c = quebras[-1]
    else:
        c0, c1, h0, h1 = quebras[k - 1], quebras[k], h[k - 1], h[k]
        c = c1 if h1 == h0 else c0 + (soma - h0) * (c1 - c0) / (h1 - h0)
    return np.clip(c * w, lo, hi)


def _quadratico(A: np.ndarray, b: np.ndarray, projetar, w0: np.ndarray,
                max_iter: int = 20000, tol: float = 1e-12) -> np.ndarray:
    """Minimiza 0,5 w'Aw − b'w no conjunto de `projetar` por gradiente projetado acelerado (FISTA).

    O momento é reiniciado quando aponta contra o gradiente (O'Donoghue e Candès, 2015), e a
    parada exige que w seja ponto fixo do gradiente projetado simples: com momento, dois
    iterados iguais num vértice não garantem o ótimo.
    """
    passo = 1.0 / float(np.linalg.eigvalsh(A)[-1])
    w = y = projetar(w0)
    t = 1.0
    for _ in range(max_iter):
        g = A @ y - b
        w_novo = projetar(y - passo * g)
        if np.max(np.abs(w_novo - w)) < tol:
            if np.max(np.abs(projetar(w_novo - passo * (A @ w_novo - b)) - w_novo)) < tol:
                return w_novo
            w, y, t = w_novo, w_novo, 1.0                 # parou no momento, não no ótimo
            continue
        if g @ (w_novo - w) > 0:                          # momento contra o gradiente: reinicia
            w, y, t = w_novo, w_novo, 1.0
            continue
        t_novo = (1 + math.sqrt(1 + 4 * t * t)) / 2
        y = w_novo + (t - 1) / t_novo * (w_novo - w)
        w, t = w_novo, t_novo
    return w


# ---------------------------------------------------------------------------
# Métodos de alocação
# ---------------------------------------------------------------------------

def _vol_inversa(S):
    v = 1 / np.sqrt(np.diag(S))
    return v / v.sum()


def _risk_parity(S, orcamento=None, max_iter: int = 10000, tol: float = 1e-12):
    """Contribuição ao risco proporcional ao orçamento (padrão: igual), por descida coordenada.

    Minimiza 0,5 x'Sx − Σ b_i log x_i (Griveau-Billion, Richard e Roncalli, 2013); a solução
    normalizada tem contribuições x_i (Sx)_i proporcionais a b_i.
    """
    N = len(S)
    b = np.full(N, 1 / N) if orcamento is None else np.asarray(orcamento, dtype=float)
    if b.shape != (N,) or np.any(b <= 0):
        raise ValueError("orcamento_risco deve ter um valor positivo por estratégia.")
    b = b / b.sum()
    x = _vol_inversa(S) / np.sqrt(np.diag(S)).mean()
    for _ in range(max_iter):
        x_ant = x.copy()
        for i in range(N):
            c = S[i] @ x - S[i, i] * x[i]
            x[i] = (-c + math.sqrt(c * c + 4 * S[i, i] * b[i])) / (2 * S[i, i])
        if np.max(np.abs(x - x_ant)) < tol * max(1.0, np.max(np.abs(x))):
            break
    return x / x.sum()


def _arvore_hrp(C: np.ndarray):
    """Árvore do agrupamento hierárquico (ligação simples) usado no HRP, como listas aninhadas."""
    D = np.sqrt(np.clip(0.5 * (1 - C), 0, None))
    D2 = np.sqrt(((D[:, None, :] - D[None, :, :]) ** 2).sum(axis=2))   # distância entre colunas
    grupos = {i: ([i], i) for i in range(len(C))}                     # id: (folhas, subárvore)
    proximo = len(C)
    while len(grupos) > 1:
        a, b = min(itertools.combinations(grupos, 2),
                   key=lambda par: D2[np.ix_(grupos[par[0]][0], grupos[par[1]][0])].min())
        (fa, ta), (fb, tb) = grupos.pop(a), grupos.pop(b)
        grupos[proximo] = (fa + fb, (ta, tb))
        proximo += 1
    return next(iter(grupos.values()))[1]


def _folhas(arvore) -> list[int]:
    return [arvore] if isinstance(arvore, int) else _folhas(arvore[0]) + _folhas(arvore[1])


def _hrp(S):
    """Hierarchical Risk Parity (López de Prado, 2016).

    Agrupa as estratégias pela correlação e divide o capital de cima para baixo: em cada ramo,
    os dois sub-ramos recebem capital inversamente proporcional à variância de cada um (com
    pesos de variância inversa dentro dele). Divide nos ramos da árvore, e não ao meio da lista
    ordenada como no artigo original, para não separar estratégias do mesmo grupo.
    """
    d = np.sqrt(np.diag(S))
    w = np.ones(len(S))

    def var_grupo(g):
        sub = S[np.ix_(g, g)]
        ivp = 1 / np.diag(sub)
        ivp /= ivp.sum()
        return float(ivp @ sub @ ivp)

    pilha = [_arvore_hrp(S / np.outer(d, d))]
    while pilha:
        no = pilha.pop()
        if isinstance(no, int):
            continue
        g1, g2 = _folhas(no[0]), _folhas(no[1])
        v1, v2 = var_grupo(g1), var_grupo(g2)
        alfa = 1 - v1 / (v1 + v2)
        w[g1] *= alfa
        w[g2] *= 1 - alfa
        pilha += [no[0], no[1]]
    return w / w.sum()


def _media_encolhida(mu: np.ndarray, encolher: float) -> np.ndarray:
    if not 0 <= encolher <= 1:
        raise ValueError("encolher_media deve estar entre 0 e 1.")
    return (1 - encolher) * mu + encolher * mu.mean()


def _maior_media(mu, lo, hi, soma: float = 1.0) -> np.ndarray:
    """Carteira viável de maior média (programa linear): parte dos mínimos e enche os maiores mu."""
    w = lo.astype(float).copy()
    resto = soma - w.sum()
    for i in np.argsort(-mu, kind="stable"):
        acrescimo = min(hi[i] - w[i], resto)
        w[i] += acrescimo
        resto -= acrescimo
    return w


def _max_sharpe(mu, S, lo, hi, max_iter: int = 200, tol: float = 1e-10):
    """Maior Sharpe em {lo <= w <= hi, soma 1}, por uma sequência de problemas quadráticos.

    No ótimo, w também resolve min 0,5 w'Sw − g mu'w no mesmo conjunto com g = w'Sw / mu'w
    (condições de KKT; o Sharpe não muda com a escala). Itera g ← variância / média do último
    w até estabilizar, partindo do melhor entre mínima variância, 1/N e a carteira de maior
    média (que garante achar uma carteira de média positiva sempre que ela existir, mesmo com
    1/N abaixo de rf). Sem nenhuma carteira viável de média positiva, devolve a mínima
    variância.
    """
    proj = lambda v: _projetar(v, lo, hi, 1.0)  # noqa: E731
    N = len(S)
    minvar = _quadratico(S, np.zeros(N), proj, np.full(N, 1 / N))
    sr = lambda w: (mu @ w) / math.sqrt(w @ S @ w)  # noqa: E731
    w = max([minvar, proj(np.full(N, 1 / N)), _maior_media(mu, lo, hi)], key=sr)
    if mu @ w <= 0:
        return minvar
    g = float(w @ S @ w / (mu @ w))
    for _ in range(max_iter):
        w_novo = _quadratico(S, g * mu, proj, w, tol=1e-13)
        if mu @ w_novo <= 0:
            break
        if sr(w_novo) >= sr(w) - 1e-15:
            w = w_novo
        g_novo = float(w_novo @ S @ w_novo / (mu @ w_novo))
        if abs(g_novo - g) <= tol * abs(g):
            break
        g = g_novo
    return w


def pesos_portfolio(retornos, metodo: str = "risk_parity", nomes=None, peso_min=0.0,
                    peso_max=1.0, rf_por_periodo=0.0, periodos_por_ano: int = 252,
                    encolher_cov: bool = True, encolher_media: float = 0.5,
                    fracao_kelly: float = 0.5, alavancagem_max: float = 1.0,
                    orcamento_risco=None) -> dict:
    """Pesos de um método de alocação estimados na janela `retornos` (todas as linhas).

    `peso_min`/`peso_max`: limites por estratégia (número, um valor por estratégia na ordem das
    colunas ou dict {nome: limite}). Nos métodos sem otimização ("iguais", "vol_inversa",
    "risk_parity", "hrp"), quem passa do limite fica nele e as demais mantêm as proporções do
    método; com limite ativo, as contribuições ao risco deixam de ser exatamente as do método.
    Limite máximo é a defesa mais simples contra concentração por erro de estimação: com 5
    estratégias, peso_max entre 0,3 e 0,4 é razoável.
    `encolher_media` (max_sharpe e kelly): puxa cada média para a média comum; 0 usa as médias
    amostrais, 1 iguala todas (max_sharpe vira a mínima variância). Padrão 0,5. Só reduz as
    DIFERENÇAS entre as médias; o nível, que é o que superdimensiona o Kelly, só é controlado
    por `fracao_kelly`.
    `fracao_kelly` (f, padrão 0,5): maximiza mu'w − w'Sw / (2f), que sem limites ativos dá
    exatamente f × Kelly completo. `alavancagem_max`: soma máxima dos pesos no Kelly (padrão 1,
    sem alavancagem); a sobra fica no caixa, rendendo rf. Os limites valem para os pesos finais.
    `orcamento_risco`: em "risk_parity", a fração do risco de cada estratégia (padrão igual;
    lista na ordem das colunas ou dict {nome: fração}).

    Retorna {"pesos": {nome: peso}, "caixa": 1 − soma, "vol_anual", "sharpe_dentro"
    (Sharpe dos pesos fixos na própria janela, em excesso de rf: otimista por construção)}.
    Para os pesos de hoje com os parâmetros de um walk-forward (inclusive `vol_alvo` e
    `ancorada`), use `walk_forward_portfolio(...)["pesos_atuais"]`.
    """
    if metodo not in METODOS:
        raise ValueError(f"metodo deve ser um de {METODOS}; recebeu {metodo!r}.")
    R, nomes, indice = _painel(retornos, nomes, minimo_colunas=1)
    T, N = R.shape
    rf = _rf(rf_por_periodo, T, indice, periodos_por_ano)
    peso_min, peso_max, orcamento_risco = _por_nome(nomes, peso_min, peso_max, orcamento_risco)
    w = _pesos(R, rf, metodo, peso_min, peso_max, encolher_cov, encolher_media,
               fracao_kelly, alavancagem_max, orcamento_risco)
    S = _cov(R, encolher_cov) if N > 1 else np.atleast_2d(R.var(ddof=1))
    rp = R @ w + (1 - w.sum()) * rf
    return {"pesos": dict(zip(nomes, map(float, w))), "caixa": float(1 - w.sum()),
            "vol_anual": float(math.sqrt(w @ S @ w * periodos_por_ano)),
            "sharpe_dentro": sharpe(rp - rf, periodos_por_ano)}


def _por_nome(nomes, *valores):
    """Converte dicts {nome: valor} (limites, orçamento de risco) em listas na ordem das colunas."""
    saida = []
    for v in valores:
        if isinstance(v, dict):
            faltam = [n for n in nomes if n not in v]
            sobram = [k for k in v if k not in nomes]
            if faltam or sobram:
                raise ValueError(f"Dict de limites ou orçamento deve ter exatamente as estratégias "
                                 f"{nomes}; faltam {faltam}, sobram {sobram}.")
            v = [float(v[n]) for n in nomes]
        saida.append(v)
    return saida


def _pesos(R, rf, metodo, peso_min, peso_max, encolher_cov, encolher_media,
           fracao_kelly, alavancagem_max, orcamento_risco) -> np.ndarray:
    T, N = R.shape
    if metodo == "kelly":
        if not 0 < fracao_kelly <= 1:
            raise ValueError("fracao_kelly deve estar em (0, 1]; meio Kelly = 0.5.")
        if alavancagem_max <= 0:
            raise ValueError("alavancagem_max deve ser positiva.")
        lo, hi = _limites(peso_min, peso_max, N, soma=min(alavancagem_max, np.sum(
            np.broadcast_to(np.asarray(peso_max, dtype=float), (N,)))))
    else:
        lo, hi = _limites(peso_min, peso_max, N)
    if N == 1 and metodo != "kelly":
        return np.ones(1)
    if metodo == "iguais":
        return _limitar(np.full(N, 1 / N), lo, hi)
    S = _cov(R, encolher_cov)
    proj = lambda v: _projetar(v, lo, hi, 1.0)  # noqa: E731
    if metodo == "vol_inversa":
        return _limitar(_vol_inversa(S), lo, hi)
    if metodo == "risk_parity":
        return _limitar(_risk_parity(S, orcamento_risco), lo, hi)
    if metodo == "hrp":
        return _limitar(_hrp(S), lo, hi)
    if metodo == "min_variancia":
        return _quadratico(S, np.zeros(N), proj, _vol_inversa(S))
    mu = _media_encolhida((R - rf[:, None]).mean(axis=0), encolher_media)
    if metodo == "max_sharpe":
        return _max_sharpe(mu, S, lo, hi)
    # Kelly fracionado: maximiza mu'w − w'Sw / (2f) (crescimento aproximado em excesso de rf,
    # com aversão 1/f), com os limites nos pesos finais e soma até alavancagem_max; sobra no caixa.
    # Sem limites ativos dá f × S⁻¹mu. Multiplicar o Kelly limitado por f, em vez disso,
    # deixaria a exposição sempre abaixo de f × alavancagem_max e violaria peso_min.
    projk = lambda v: _projetar_ate(v, lo, hi, alavancagem_max)  # noqa: E731
    return _quadratico(S / fracao_kelly, mu, projk, lo.copy())


# ---------------------------------------------------------------------------
# Simulação com deriva e custos de rebalanceamento
# ---------------------------------------------------------------------------

def _uma_passada(R, rf, alvos: dict, custo_giro: float, faixa: float):
    """Núcleo da simulação. `alvos`: {t: pesos-alvo}, aplicados no início do período t.

    Pesos são frações do capital; o caixa (1 − soma) rende rf. Entre rebalanceamentos os
    pesos derivam com os retornos. Custo = custo_giro × Σ|alvo − atual| (o caixa não paga).
    """
    T, N = R.shape
    w = np.zeros(N)
    saida, giros, custos, executados = np.zeros(T), np.zeros(T), np.zeros(T), []
    for t in range(T):
        if t in alvos:
            alvo = alvos[t]
            if not executados or np.max(np.abs(alvo - w)) > faixa:
                giros[t] = float(np.abs(alvo - w).sum())
                custos[t] = custo_giro * giros[t]
                w = alvo.copy()
                executados.append(t)
        caixa = 1.0 - w.sum()
        bruto = float(w @ R[t] + caixa * rf[t])
        saida[t] = (1 + bruto) * (1 - custos[t]) - 1
        if 1 + bruto > 0:
            w = w * (1 + R[t]) / (1 + bruto)
    return saida, giros, custos, executados


def simular_portfolio(retornos, pesos, nomes=None, rebalancear_a_cada: int | None = 21,
                      custo_giro: float = 0.001, faixa: float = 0.0, rf_por_periodo=0.0,
                      periodos_por_ano: int = 252) -> dict:
    """Retornos de um portfólio com pesos-alvo fixos, deriva entre rebalanceamentos e custos.

    `pesos`: dict {nome: peso} ou vetor na ordem das colunas. Soma < 1 deixa caixa rendendo rf;
    soma > 1 toma emprestado a rf.
    `rebalancear_a_cada`: períodos entre rebalanceamentos (None = só a alocação inicial).
    `custo_giro`: custo por unidade de capital movimentada entre estratégias (0,001 = 10 bps
    sobre o valor transferido). É custo adicional ao das estratégias: redimensionar uma
    estratégia exige operar as posições dela.
    `faixa`: banda sem operação; só rebalanceia se algum peso se afastou mais que isso do alvo.
    A alocação inicial também paga custo.
    """
    R, nomes, indice = _painel(retornos, nomes, minimo_colunas=1)
    T, N = R.shape
    rf = _rf(rf_por_periodo, T, indice, periodos_por_ano)
    w = np.array([pesos[n] for n in nomes], dtype=float) if isinstance(pesos, dict) \
        else np.asarray(pesos, dtype=float)
    if w.shape != (N,) or not np.all(np.isfinite(w)):
        raise ValueError(f"pesos deve ter {N} valores finitos.")
    _checar_custos(custo_giro, faixa)
    passo = T if rebalancear_a_cada is None else _checar_passo(rebalancear_a_cada)
    alvos = {t: w for t in range(0, T, passo)}
    r, giros, custos, executados = _uma_passada(R, rf, alvos, custo_giro, faixa)
    anos = T / periodos_por_ano
    return {"retornos": r, "indice": indice, "giro_anual": float(giros[1:].sum() / anos),
            "custo_anual": float(custos.sum() / anos), "n_rebalanceamentos": len(executados),
            "sharpe": sharpe(r - rf, periodos_por_ano)}


def _checar_custos(custo_giro, faixa):
    if not 0 <= custo_giro < 0.1:
        raise ValueError("custo_giro deve estar entre 0 e 0,1 (fração do valor movimentado; "
                         "0,001 = 10 bps).")
    if faixa < 0:
        raise ValueError("faixa deve ser >= 0.")


def _checar_passo(rebalancear_a_cada):
    if int(rebalancear_a_cada) != rebalancear_a_cada or rebalancear_a_cada < 1:
        raise ValueError("rebalancear_a_cada deve ser um inteiro >= 1.")
    return int(rebalancear_a_cada)


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

def walk_forward_portfolio(retornos, metodo: str = "risk_parity", nomes=None, janela: int = 504,
                           rebalancear_a_cada: int = 21, ancorada: bool = False,
                           custo_giro: float = 0.001, faixa: float = 0.0,
                           vol_alvo: float | None = None, rf_por_periodo=0.0,
                           periodos_por_ano: int = 252, **kw_metodo) -> dict:
    """Alocação fora da amostra: pesos estimados só com dados anteriores a cada rebalanceamento.

    No período t de rebalanceamento, os pesos usam as linhas [t − janela, t) (ou [0, t) se
    `ancorada`) e valem a partir do retorno de t. O resultado oficial é a série a partir de
    `janela` (os primeiros `janela` períodos só servem para estimar).
    `vol_alvo`: volatilidade anual desejada (ex.: 0.10). Escala os pesos pela vol estimada na
    janela, com soma limitada a `alavancagem_max` (padrão 1, sem alavancagem); a sobra fica no
    caixa. Não combina com "kelly" (o alvo de vol substituiria o tamanho de Kelly).
    `kw_metodo`: repassados a `pesos_portfolio` (peso_min, peso_max, encolher_media, ...).

    Retorna:
    - `retornos` (fora da amostra) e `indice` (as datas deles, se a entrada for do pandas);
    - `pesos`: lista com um dict por rebalanceamento previsto ("periodo", "data", um peso por
      estratégia, "caixa" e "executado"; com `faixa`, executado=False quando a carteira estava
      perto o bastante do alvo e não foi mexida);
    - `pesos_atuais`: os pesos para operar a partir de agora, estimados com as últimas linhas e
      os mesmos parâmetros (janela, ancorada, limites, vol_alvo), e `pesos_atuais_ate`, a última
      data usada;
    - `giro_anual`, `custo_anual`, `max_drawdown`, `sharpe_fora`, `sharpe_dentro` (média dos
      Sharpes dos pesos na própria janela de estimação, sem custo nem deriva: otimista) e
      `eficiencia` = sharpe_fora / sharpe_dentro (abaixo de ~0,5 indica overfitting da
      alocação; para 1/N, que não estima nada, só reflete a variação do Sharpe no tempo).
    """
    if metodo not in METODOS:
        raise ValueError(f"metodo deve ser um de {METODOS}; recebeu {metodo!r}.")
    R, nomes, indice = _painel(retornos, nomes, minimo_colunas=1)
    T, N = R.shape
    rf = _rf(rf_por_periodo, T, indice, periodos_por_ano)
    passo = _checar_passo(rebalancear_a_cada)
    _checar_custos(custo_giro, faixa)
    if int(janela) != janela or not max(10, 2 * N) <= janela <= T - passo:
        raise ValueError(
            f"janela deve ser um inteiro entre {max(10, 2 * N)} e T − rebalancear_a_cada = "
            f"{T - passo}; recebeu {janela}. Com {N} estratégias, use pelo menos 20 × N = "
            f"{20 * N} períodos (abaixo disso, `comparar_metodos` alerta).")
    janela = int(janela)
    if vol_alvo is not None and metodo == "kelly":
        raise ValueError("vol_alvo não combina com 'kelly': o alvo de volatilidade substituiria o "
                         "tamanho de Kelly. Use um ou outro (para limitar a exposição do Kelly, "
                         "use fracao_kelly e alavancagem_max).")
    if vol_alvo is not None and not vol_alvo > 0:
        raise ValueError("vol_alvo deve ser positiva (volatilidade anual, ex.: 0.10).")
    params = dict(peso_min=0.0, peso_max=1.0, encolher_cov=True, encolher_media=0.5,
                  fracao_kelly=0.5, alavancagem_max=1.0, orcamento_risco=None)
    desconhecidos = set(kw_metodo) - set(params)
    if desconhecidos:
        raise ValueError(f"Parâmetros desconhecidos: {sorted(desconhecidos)}.")
    params.update(kw_metodo)
    params["peso_min"], params["peso_max"], params["orcamento_risco"] = _por_nome(
        nomes, params["peso_min"], params["peso_max"], params["orcamento_risco"])

    def alvo(t):
        """Pesos-alvo estimados com as linhas anteriores a t."""
        ini = 0 if ancorada else t - janela
        Rj, rfj = R[ini:t], rf[ini:t]
        w = _pesos(Rj, rfj, metodo, **params)
        if vol_alvo is not None:
            S = _cov(Rj, params["encolher_cov"]) if N > 1 else np.atleast_2d(Rj.var(ddof=1))
            vol = math.sqrt(w @ S @ w * periodos_por_ano)
            if vol > 0 and w.sum() > 0:
                w = w * min(vol_alvo / vol, params["alavancagem_max"] / w.sum())
        rp = Rj @ w + (1 - w.sum()) * rfj
        return w, sharpe(rp - rfj, periodos_por_ano)

    alvos, historico, sr_dentro = {}, [], []
    for t in range(janela, T, passo):
        w, sr = alvo(t)
        alvos[t - janela] = w
        historico.append({"periodo": t, "data": indice[t] if indice is not None else t,
                          **dict(zip(nomes, map(float, w))), "caixa": float(1 - w.sum())})
        sr_dentro.append(sr)
    r, giros, custos, executados = _uma_passada(R[janela:], rf[janela:], alvos, custo_giro, faixa)
    executados = set(executados)
    for h in historico:
        h["executado"] = h["periodo"] - janela in executados
    w_hoje, _ = alvo(T)
    anos = (T - janela) / periodos_por_ano
    sr_fora = sharpe(r - rf[janela:], periodos_por_ano)
    sr_in = float(np.nanmean(sr_dentro))
    return {
        "metodo": metodo, "retornos": r, "rf": rf[janela:],
        "indice": indice[janela:] if indice is not None else None,
        "pesos": historico,
        "pesos_atuais": {**dict(zip(nomes, map(float, w_hoje))), "caixa": float(1 - w_hoje.sum())},
        "pesos_atuais_ate": indice[-1] if indice is not None else T - 1,
        "giro_anual": float(giros[1:].sum() / anos),
        "custo_anual": float(custos.sum() / anos), "sharpe_fora": sr_fora,
        "sharpe_dentro": sr_in,
        "eficiencia": float(sr_fora / sr_in) if sr_in > 0 else float("nan"),
        "max_drawdown": max_drawdown(r), "janela": janela, "nomes": nomes,
    }


# ---------------------------------------------------------------------------
# Diversificação
# ---------------------------------------------------------------------------

def analise_diversificacao(retornos, pesos=None, nomes=None, rf_por_periodo=0.0,
                           periodos_por_ano: int = 252, quantil_estresse: float = 0.10,
                           n_sim: int = 200, seed: int = 0) -> dict:
    """Quanto as estratégias realmente diversificam, com os `pesos` dados (padrão 1/N).

    - `correlacao` e `correlacao_media` (média fora da diagonal).
    - `perdas_conjuntas`: para cada par, a chance de uma estratégia ter um dia ruim dado que a
      outra teve. Dia ruim = um dos `quantil_estresse` × T piores dias da estratégia (padrão
      10%), contando só dias com perda: estratégias que ficam zeradas na maior parte dos dias
      têm menos dias ruins, e dias parados nunca contam (`dias_ruins` dá a contagem). Com
      estratégias independentes ≈ quantil; com a mesma aposta ≈ 1.
    - `pares_perdem_juntos`: pares (a, b, observado, esperado) que perdem juntos mais do que a
      dependência comum explica. "Esperado" vem de `n_sim` simulações com as mesmas
      distribuições de cada estratégia e a mesma correlação de postos (Spearman), sem
      dependência extra nas caudas; observado e esperado são a média dos dois sentidos do par.
      Entra o par cujo excesso passa `limiar_perdas_conjuntas`: o maior entre 10 pontos e o
      excesso que o acaso produz em 5% das simulações (o maior entre todos os pares, o que
      controla alarmes falsos com muitos pares e amostras curtas). Com menos de ~500 períodos
      o diagnóstico tem pouco poder: ausência de alerta não prova que não perdem juntas.
      `perdas_conjuntas_media` (fora da diagonal) e `perdas_conjuntas_normal` (o esperado,
      médio) resumem. Não use a correlação nos piores dias do portfólio: condicionar na soma
      cria correlação negativa artificial.
    - `contribuicao_risco`: fração da variância do portfólio que vem de cada estratégia (soma 1).
      Peso de 20% com 50% do risco é concentração escondida.
    - `razao_diversificacao` = Σ w_i σ_i / σ_portfolio (1 = nenhuma diversificação; √N com N
      estratégias independentes de mesma vol).
    - `n_efetivo_apostas`: 1 / Σ (contribuição de risco)². Entre 1 e N quando todas as
      contribuições são positivas; com correlação negativa uma contribuição pode ser negativa
      e o número sai do intervalo (prefira então `n_efetivo_autovalores`).
    - `n_efetivo_autovalores`: exp(entropia dos autovalores da correlação): quantos fatores
      independentes existem de fato, sem depender dos pesos.
    - `pares_redundantes`: pares com correlação acima de 0,8 (provavelmente a mesma aposta).
    """
    R, nomes, indice = _painel(retornos, nomes, minimo_colunas=2)
    T, N = R.shape
    _rf(rf_por_periodo, T, indice, periodos_por_ano)       # só confere datas e escala
    if pesos is None:
        w = np.full(N, 1 / N)
    else:
        w = np.array([pesos[n] for n in nomes], dtype=float) if isinstance(pesos, dict) \
            else np.asarray(pesos, dtype=float)
    if w.shape != (N,):
        raise ValueError(f"pesos deve ter {N} valores.")
    if not 0 < quantil_estresse < 0.5:
        raise ValueError("quantil_estresse deve estar entre 0 e 0,5.")
    if int(n_sim) != n_sim or n_sim < 20:
        raise ValueError("n_sim deve ser um inteiro >= 20.")
    S = np.cov(R, rowvar=False)
    sd = np.sqrt(np.diag(S))
    if np.any(sd <= 0):
        raise ValueError("Uma estratégia tem volatilidade zero; a correlação não está definida.")
    C = S / np.outer(sd, sd)
    fora = ~np.eye(N, dtype=bool)
    var_p = float(w @ S @ w)
    rc = w * (S @ w) / var_p
    autov = np.clip(np.linalg.eigvalsh(C), 1e-12, None)
    p = autov / autov.sum()
    caudas = _caudas_conjuntas(R, quantil_estresse, int(n_sim), np.random.default_rng(seed))
    pc, par_obs, par_esp, limiar = caudas["pc"], caudas["observado"], caudas["esperado"], caudas["limiar"]
    excesso_cauda = [(nomes[i], nomes[j], float(par_obs[i, j]), float(par_esp[i, j]))
                     for i, j in itertools.combinations(range(N), 2)
                     if np.isfinite(par_obs[i, j]) and par_obs[i, j] - par_esp[i, j] > limiar]
    redundantes = [(nomes[i], nomes[j], float(C[i, j]))
                   for i, j in itertools.combinations(range(N), 2) if C[i, j] > 0.8]
    return {
        "correlacao": {n: {m: float(C[i, j]) for j, m in enumerate(nomes)} for i, n in enumerate(nomes)},
        "correlacao_media": float(C[fora].mean()),
        "perdas_conjuntas": {n: {m: float(pc[i, j]) for j, m in enumerate(nomes)}
                             for i, n in enumerate(nomes)},
        "perdas_conjuntas_media": float(np.nanmean(pc[fora])),
        "perdas_conjuntas_normal": float(np.nanmean(par_esp[fora])),
        "pares_perdem_juntos": excesso_cauda,
        "limiar_perdas_conjuntas": float(limiar),
        "dias_ruins": dict(zip(nomes, map(int, caudas["dias_ruins"]))),
        "quantil_estresse": quantil_estresse,
        "contribuicao_risco": dict(zip(nomes, map(float, rc))),
        "razao_diversificacao": float(np.abs(w) @ sd / math.sqrt(var_p)),
        "n_efetivo_apostas": float(1 / np.sum(rc ** 2)),
        "n_efetivo_autovalores": float(math.exp(-np.sum(p * np.log(p)))),
        "pares_redundantes": redundantes,
        "vol_anual_portfolio": float(math.sqrt(var_p * periodos_por_ano)),
        "vol_anual_media_estrategias": float(sd.mean() * math.sqrt(periodos_por_ano)),
    }


def _dias_ruins(R: np.ndarray, q: float) -> np.ndarray:
    """Matriz T x N de 0/1: os round(q × T) piores dias de cada estratégia, só entre dias com perda."""
    T, N = R.shape
    k = max(1, int(round(q * T)))
    ruins = np.zeros((T, N))
    np.put_along_axis(ruins, np.argsort(R, axis=0, kind="stable")[:k], 1.0, axis=0)
    return ruins * (R < 0)


def _perdas_conjuntas(R: np.ndarray, q: float) -> np.ndarray:
    """P(j em dia ruim | i em dia ruim), por par (matriz N x N; NaN se i nunca perde)."""
    ruins = _dias_ruins(R, q)
    n = ruins.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n[:, None] > 0, (ruins.T @ ruins) / n[:, None], np.nan)


def _postos_medios(R: np.ndarray) -> np.ndarray:
    """Postos de cada coluna, com a média dos postos nos empates (para a correlação de Spearman)."""
    T, N = R.shape
    postos = np.empty((T, N))
    for i in range(N):
        ordem = np.argsort(R[:, i], kind="stable")
        _, inv, contagem = np.unique(R[ordem, i], return_inverse=True, return_counts=True)
        medio = np.cumsum(contagem) - (contagem - 1) / 2
        postos[ordem, i] = medio[inv]
    return postos


def _caudas_conjuntas(R: np.ndarray, q: float, n_sim: int, rng) -> dict:
    """Perdas conjuntas observadas e o esperado sem dependência extra nas caudas, com limiar.

    O esperado vem de uma cópula normal com a correlação de Spearman convertida
    (ρ = 2 sen(π ρ_s / 6)) e as distribuições empíricas de cada estratégia (cada coluna
    simulada é reordenada com os retornos observados). O limiar é o quantil 95% do maior
    excesso entre os pares nas simulações, com piso de 10 pontos.
    """
    T, N = R.shape
    pc = _perdas_conjuntas(R, q)
    Cs = np.corrcoef(_postos_medios(R), rowvar=False)
    C = 2 * np.sin(np.pi * Cs / 6)
    autov, V = np.linalg.eigh(C)
    A = V * np.sqrt(np.clip(autov, 1e-10, None))
    ordenados = np.sort(R, axis=0)
    sims = np.empty((n_sim, N, N))
    for b in range(n_sim):
        Z = rng.standard_normal((T, N)) @ A.T
        postos = np.argsort(np.argsort(Z, axis=0, kind="stable"), axis=0, kind="stable")
        sims[b] = _perdas_conjuntas(np.take_along_axis(ordenados, postos, axis=0), q)
    par = lambda M: (M + np.swapaxes(M, -1, -2)) / 2  # noqa: E731
    sim_par = par(sims)
    with warnings.catch_warnings():                      # estratégia sem dia de perda: NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        esperado = np.nanmean(sim_par, axis=0)
    iu = np.triu_indices(N, 1)
    excesso = (sim_par - esperado)[:, iu[0], iu[1]]
    validos = np.isfinite(excesso).all(axis=0)
    maior = excesso[:, validos].max(axis=1) if validos.any() else np.zeros(n_sim)
    return {"pc": pc, "observado": par(pc), "esperado": esperado,
            "limiar": max(0.10, float(np.quantile(maior, 0.95))),
            "dias_ruins": _dias_ruins(R, q).sum(axis=0)}


# ---------------------------------------------------------------------------
# Comparação de métodos e alertas
# ---------------------------------------------------------------------------

def comparar_metodos(retornos, metodos=("iguais", "vol_inversa", "risk_parity", "hrp",
                                        "min_variancia", "max_sharpe"),
                     nomes=None, rf_por_periodo=0.0, periodos_por_ano: int = 252,
                     n_sim: int = 2000, seed: int = 42, **kw_walk_forward) -> dict:
    """Walk-forward de cada método, teste contra 1/N, MCS e alertas de overfitting.

    Cada método (e cada combinação de parâmetros) testado é uma tentativa: registre-os (regra
    de ouro 4). "iguais" entra sempre como referência. O teste contra 1/N usa
    `bootstrap_diferenca_sharpe` (estudentizado) com menos de 120 períodos fora da amostra e
    `teste_diferenca_sharpe` (HAC) com mais; os p-valores são corrigidos por Holm. O MCS
    (Sharpe, 10%) diz quais métodos os dados não conseguem descartar.

    `kw_walk_forward`: repassados a `walk_forward_portfolio` (janela, rebalancear_a_cada,
    custo_giro, peso_max, vol_alvo, ...). "kelly" não aceita `vol_alvo`.

    Retorna:
    - `resultados`: {método: saída de `walk_forward_portfolio`}, com `retornos`, `indice`
      (datas), `pesos` (histórico), `pesos_atuais` (para operar) e métricas;
    - `tabela`: `tabela_comparativa` das séries fora da amostra, mais sharpe_dentro, eficiencia,
      giro_anual e custo_anual de cada método;
    - `contra_iguais`: uma linha por método (diferenca_sharpe, ic, p_valor, p_holm, teste);
    - `mcs`: saída de `model_confidence_set`; `diversificacao`: `analise_diversificacao` com 1/N;
    - `alertas` (lista de textos), `n_periodos_fora`, `rf_anual`.
    """
    from comparacao import holm
    metodos = list(dict.fromkeys(["iguais", *metodos]))
    R, nomes, indice = _painel(retornos, nomes, minimo_colunas=2)
    _rf(rf_por_periodo, len(R), indice, periodos_por_ano)
    resultados = {m: walk_forward_portfolio(retornos, m, nomes=nomes, rf_por_periodo=rf_por_periodo,
                                            periodos_por_ano=periodos_por_ano, **kw_walk_forward)
                  for m in metodos}
    series = {m: r["retornos"] for m, r in resultados.items()}
    rf = resultados["iguais"]["rf"]
    T = len(rf)
    tabela = tabela_comparativa(series, periodos_por_ano=periodos_por_ano, rf_por_periodo=rf)
    for m, r in resultados.items():
        tabela["metricas"][m].update({"sharpe_dentro": r["sharpe_dentro"],
                                      "eficiencia": r["eficiencia"], "giro_anual": r["giro_anual"],
                                      "custo_anual": r["custo_anual"]})
    contra = []
    for m in metodos[1:]:
        if T < 120:
            res = bootstrap_diferenca_sharpe(series[m], series["iguais"], periodos_por_ano,
                                             n_sim=n_sim, seed=seed, rf_por_periodo=rf)
            ic, metodo_teste = res["ic"], "bootstrap estudentizado"
        else:
            res = teste_diferenca_sharpe(series[m], series["iguais"], periodos_por_ano,
                                         rf_por_periodo=rf)
            ic, metodo_teste = res["ic95"], "HAC (Ledoit-Wolf)"
        contra.append({"metodo": m, "diferenca_sharpe": res["diferenca"], "ic": ic,
                       "p_valor": res["p_valor"], "teste": metodo_teste})
    if contra:
        for linha, p in zip(contra, holm([c["p_valor"] for c in contra])):
            linha["p_holm"] = float(p)
    mcs = model_confidence_set(series, n_sim=n_sim, seed=seed, rf_por_periodo=rf,
                               periodos_por_ano=periodos_por_ano) if len(series) > 1 else None
    diversificacao = analise_diversificacao(R, nomes=nomes, periodos_por_ano=periodos_por_ano)
    alertas = _alertas(R, resultados, contra, mcs, diversificacao, periodos_por_ano)
    return {"resultados": resultados, "tabela": tabela, "contra_iguais": contra, "mcs": mcs,
            "diversificacao": diversificacao, "alertas": alertas, "n_periodos_fora": T,
            "rf_anual": tabela["rf_anual"]}


def _alertas(R, resultados, contra, mcs, div, periodos_por_ano) -> list[str]:
    T, N = R.shape
    janela = next(iter(resultados.values()))["janela"]
    alertas = []
    if janela < 20 * N:
        alertas.append(
            f"Janela de {janela} períodos para {N} estratégias ({janela / N:.0f} por estratégia): "
            f"covariância e médias mal estimadas. Prefira métodos que não usam médias, limite os "
            f"pesos ou aumente a janela (>= 20 × N).")
    for m, r in resultados.items():
        if m != "iguais" and np.isfinite(r["eficiencia"]) and r["eficiencia"] < 0.5:
            alertas.append(f"'{m}': eficiência {r['eficiencia']:.2f} (Sharpe fora "
                           f"{r['sharpe_fora']:.2f} contra {r['sharpe_dentro']:.2f} dentro da "
                           f"janela). Os pesos se ajustam a ruído.")
        if m in METODOS_COM_MEDIA:
            alertas.append(f"'{m}' usa médias estimadas, o insumo mais ruidoso; só o adote se "
                           f"bater 1/N com folga fora da amostra e em vários períodos.")
    ganhou = [c["metodo"] for c in contra if c["p_holm"] < 0.05 and c["diferenca_sharpe"] > 0]
    if not ganhou:
        alertas.append("Nenhum método bateu 1/N com significância (p de Holm < 0,05). O normal é "
                       "ficar com 1/N ou com o método mais simples entre os que estão no MCS.")
    if mcs is not None and len(mcs["mcs"]) > 1:
        alertas.append(f"MCS (90%) com {len(mcs['mcs'])} métodos: {mcs['mcs']}. Os dados não "
                       f"distinguem esses métodos; desempate por simplicidade e giro.")
    if div["pares_redundantes"]:
        pares = ", ".join(f"{a}/{b} ({c:.2f})" for a, b, c in div["pares_redundantes"])
        alertas.append(f"Pares com correlação > 0,8 (amostra inteira): {pares}. Provavelmente a "
                       f"mesma aposta. Se decidir juntar ou tirar uma, decida com dados anteriores "
                       f"ao período de teste ou conte o novo conjunto como mais uma tentativa.")
    q = div["quantil_estresse"]
    if div["pares_perdem_juntos"]:
        pares = ", ".join(f"{a}/{b} ({o:.0%} contra {n:.0%} esperados)"
                          for a, b, o, n in div["pares_perdem_juntos"])
        alertas.append(f"Perdem juntas nos piores {q:.0%} dos dias mais do que a dependência comum "
                       f"explica: {pares}. A diversificação entre elas falha em crises; dimensione "
                       f"pelo cenário de estresse.")
    elif T < 500:
        alertas.append(f"Só {T} períodos: o diagnóstico de perdas conjuntas nos piores {q:.0%} tem "
                       f"pouco poder. A falta de alerta não mostra que as estratégias não caem "
                       f"juntas; olhe as crises conhecidas uma a uma.")
    for m, r in resultados.items():
        if r["custo_anual"] > 0 and r["giro_anual"] > 4:
            alertas.append(f"'{m}': giro de {r['giro_anual']:.1f}x ao ano entre estratégias "
                           f"(custo {r['custo_anual']:.2%} ao ano). Considere uma faixa sem "
                           f"operação ou rebalancear menos.")
    return alertas


def formatar_comparacao(res: dict, casas: int = 3) -> str:
    """Tabela Markdown dos métodos (fora da amostra), testes contra 1/N, MCS e alertas."""
    fmt = lambda v: f"{v:.{casas}f}" if isinstance(v, float) else str(v)  # noqa: E731
    metodos = list(res["tabela"]["metricas"])
    chaves = ["cagr", "volatilidade", "sharpe", "sharpe_dentro", "eficiencia", "max_drawdown",
              "giro_anual", "custo_anual"]
    linhas = [f"Walk-forward fora da amostra: {res['n_periodos_fora']} períodos; Sharpe em excesso "
              f"de {res['rf_anual']:.1%} ao ano. Eficiência de 1/N omitida: 1/N não estima nada.", "",
              "| Métrica | " + " | ".join(metodos) + " |", "|---" * (len(metodos) + 1) + "|"]
    for k in chaves:
        linhas.append(f"| {k} | " + " | ".join(
            "—" if (m == "iguais" and k in ("sharpe_dentro", "eficiencia"))
            else fmt(res["tabela"]["metricas"][m][k]) for m in metodos) + " |")
    resultados = res["resultados"]
    nomes = next(iter(resultados.values()))["nomes"]
    ate = next(iter(resultados.values()))["pesos_atuais_ate"]
    ate = f"{ate.date()}" if hasattr(ate, "date") else f"o período {ate}"
    linhas += ["", f"Pesos atuais para operar (estimados até {ate}, mesmos parâmetros do walk-forward):",
               "", "| Estratégia | " + " | ".join(metodos) + " |", "|---" * (len(metodos) + 1) + "|"]
    for n in [*nomes, "caixa"]:
        linhas.append(f"| {n} | " + " | ".join(f"{resultados[m]['pesos_atuais'][n]:.1%}"
                                               for m in metodos) + " |")
    if res["contra_iguais"]:
        linhas += ["", "| Método x 1/N | Diferença de Sharpe | IC | p | p Holm | Teste |",
                   "|---|---|---|---|---|---|"]
        for c in res["contra_iguais"]:
            linhas.append(f"| {c['metodo']} | {fmt(c['diferenca_sharpe'])} | "
                          f"({fmt(c['ic'][0])}; {fmt(c['ic'][1])}) | {fmt(c['p_valor'])} | "
                          f"{fmt(c['p_holm'])} | {c['teste']} |")
    if res["mcs"] is not None:
        linhas += ["", f"Model Confidence Set (90%, Sharpe): {res['mcs']['mcs']}"]
    d = res["diversificacao"]
    linhas += ["", f"Diversificação (1/N): correlação média {d['correlacao_media']:.2f}; perdas "
               f"conjuntas nos piores {d['quantil_estresse']:.0%} {d['perdas_conjuntas_media']:.0%} "
               f"(esperado sem dependência extra nas caudas: {d['perdas_conjuntas_normal']:.0%}); "
               f"razão de diversificação "
               f"{d['razao_diversificacao']:.2f}; apostas efetivas {d['n_efetivo_autovalores']:.1f}."]
    if res["alertas"]:
        linhas += ["", "Alertas:"] + [f"- {a}" for a in res["alertas"]]
    return "\n".join(linhas)


# ---------------------------------------------------------------------------
# Demonstração
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(7)
    T = 252 * 8
    rf = 1.04 ** (1 / 252) - 1
    fator = rng.normal(0, 0.008, T)                       # risco comum (ex.: bolsa)
    choque = np.where(rng.random(T) < 0.03, rng.normal(-0.018, 0.004, T), 0.0)  # dias de crise
    carry = rf + 0.0009 + 0.3 * fator + choque + rng.normal(0, 0.006, T)
    r = {
        "tendencia": rf + 0.0004 + 0.3 * fator + rng.normal(0, 0.007, T),
        "reversao": rf + 0.0003 - 0.2 * fator + rng.normal(0, 0.006, T),
        "carry": carry,
        "carry_2": carry + rng.normal(-0.0001, 0.003, T),  # quase a mesma aposta que carry
        "vol_vendida": rf + 0.0009 + 0.2 * fator + 0.8 * choque + rng.normal(0, 0.005, T),
    }
    print(formatar_comparacao(comparar_metodos(
        r, janela=504, rebalancear_a_cada=21, custo_giro=0.001, peso_max=0.4,
        rf_por_periodo=rf, n_sim=500)))
