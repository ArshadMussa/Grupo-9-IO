# -*- coding: utf-8 -*-
"""
Investigação Operacional - Projecto Temático 2 (Grupo 9)
Alocação do orçamento agrícola público entre nove políticas (Moçambique).

Implementação computacional do modelo e testes com dados estocásticos.
Autor: Falaque, Dalton Gomes Lauter

Dependências: numpy, scipy   (pip install numpy scipy)
Unidades: USD milhões.
"""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

# ======================================================================
# 1. DADOS (Projecto Temático 1 e Formulação Matemática do PT2)
# ======================================================================
nomes = ["I&D", "Extensão Rural", "Fertilizantes", "Sementes", "Irrigação",
         "Mecanização", "Electrificação", "Estradas Rurais", "Outros Apoios"]

O = np.array([16.30, 18.75, 10.60, 8.96, 22.82, 14.67, 12.71, 28.60, 29.59])  # investimento observado Oᵢ
b = np.array([8.5, 7.2, 6.8, 7.5, 9.0, 5.5, 4.8, 6.0, 5.0])                    # benefício por USD 1 M (bᵢ)
B = 163.0                                                                       # orçamento disponível
L = 0.30 * O                                                                    # limite mínimo Lᵢ = 0,30·Oᵢ
U = 1.50 * O                                                                    # limite máximo Uᵢ = 1,50·Oᵢ
n = len(O)

# ======================================================================
# 2. VERIFICAÇÃO DOS DADOS
# ======================================================================
Z_obs = b @ O                       # benefício da distribuição realmente observada

print("=== Verificação dos dados ===")
print(f"Soma de Oᵢ = {O.sum():.2f}")
print(f"Soma de Lᵢ = {L.sum():.2f}   |   Soma de Uᵢ = {U.sum():.2f}")
print(f"Z observado = {Z_obs:.2f}")
# O problema só é viável se os mínimos cabem no orçamento (ΣLᵢ <= B)
# e o orçamento só é restrição activa se não chega para todos os máximos (B < ΣUᵢ).
assert L.sum() <= B <= U.sum(), "Dados inconsistentes: rever Lᵢ, Uᵢ e B"

# ======================================================================
# 3. MODELO BASE (PL):  max Σ bᵢxᵢ   s.a.  Σ xᵢ <= B ;  Lᵢ <= xᵢ <= Uᵢ
# ======================================================================
# O linprog MINIMIZA; para maximizar Z minimiza-se -Z (daí o sinal de b).
res = linprog(c=-b,
              A_ub=np.ones((1, n)), b_ub=[B],       # restrição orçamental: x1 + ... + x9 <= B
              bounds=list(zip(L, U)),               # Lᵢ <= xᵢ <= Uᵢ para cada política
              method="highs")

assert res.status == 0, f"O solver não encontrou solução: {res.message}"

x_opt = res.x
Z_opt = -res.fun                                    # voltar a trocar o sinal
preco_sombra = -res.ineqlin.marginals[0]            # ganho por USD 1 M extra de orçamento

print("\n=== Solução óptima do PL base ===")
print(f"{'Política':<17}{'Observado':>10}{'Óptimo':>10}{'Estado':>12}")
for i in range(n):
    estado = ("máximo" if np.isclose(x_opt[i], U[i]) else
              "mínimo" if np.isclose(x_opt[i], L[i]) else "marginal")
    print(f"{nomes[i]:<17}{O[i]:>10.3f}{x_opt[i]:>10.3f}{estado:>12}")
print(f"\nZ* = {Z_opt:.2f}   (observado: {Z_obs:.2f})")
print(f"Ganho = {Z_opt - Z_obs:.2f}  (+{100 * (Z_opt - Z_obs) / Z_obs:.2f}%)")
print(f"Orçamento usado = {x_opt.sum():.2f} de {B:.0f}")
print(f"Preço-sombra do orçamento = {preco_sombra:.2f} por USD 1 M adicional")


# ======================================================================
# 4. MODELO ESTENDIDO (PLIM): políticas podem ser activadas ou não
# ======================================================================
F = 0.02 * O                        # custo fixo de activar a política i
G = [0, 1, 4, 6, 7]                 # grupo G (índices): I&D, Extensão, Irrigação, Electrificação, Estradas


def resolver_plim(b=b, B=B, K=7, custos_fixos=True, com_E6=True, gamma=None):
    """Resolve o PLIM. Vector de decisão z = [x1..x9, y1..y9]."""
    Fv = F if custos_fixos else np.zeros(n)

    c = np.concatenate([-b, np.zeros(n)])           # maximizar Σ bᵢxᵢ (os y não entram no objectivo)

    restricoes = [
        # E1: Σ xᵢ + Σ Fᵢ yᵢ <= B
        LinearConstraint(np.concatenate([np.ones(n), Fv]).reshape(1, -1), -np.inf, B),
        # E3: xᵢ - Lᵢ yᵢ >= 0   (se y=1, x >= L ; se y=0, x >= 0)
        LinearConstraint(np.hstack([np.eye(n), -np.diag(L)]), 0, np.inf),
        # E4: xᵢ - Uᵢ yᵢ <= 0   (se y=0, x <= 0 ; se y=1, x <= U)
        LinearConstraint(np.hstack([np.eye(n), -np.diag(U)]), -np.inf, 0),
        # E5: Σ yᵢ >= K
        LinearConstraint(np.concatenate([np.zeros(n), np.ones(n)]).reshape(1, -1), K, np.inf),
    ]
    if com_E6:                                      # E6: Σ_{i∈G} xᵢ >= 0,55·B
        linha = np.zeros(2 * n)
        linha[G] = 1
        restricoes.append(LinearConstraint(linha.reshape(1, -1), 0.55 * B, np.inf))
    if gamma is not None:                           # E7 (opcional): x7 >= γ·x5
        linha = np.zeros(2 * n)
        linha[6], linha[4] = 1, -gamma
        restricoes.append(LinearConstraint(linha.reshape(1, -1), 0, np.inf))

    r = milp(c,
             constraints=restricoes,
             integrality=np.concatenate([np.zeros(n), np.ones(n)]),   # 0 = contínua (x), 1 = inteira (y)
             bounds=Bounds(np.zeros(2 * n), np.concatenate([U, np.ones(n)])))
    if r.status != 0:
        return None
    return r.x[:n], np.round(r.x[n:]).astype(int), -r.fun


# --- 4.1 Validação: com y=1 obrigatório (K=9), sem custos fixos e sem E6, o PLIM tem de dar o PL
_, _, Z_teste = resolver_plim(K=9, custos_fixos=False, com_E6=False)
print("\n=== Validação do PLIM ===")
print(f"PLIM reduzido ao PL: Z = {Z_teste:.2f}  (tem de coincidir com Z* do PL = {Z_opt:.2f})")
assert abs(Z_teste - Z_opt) < 1e-6

# --- 4.2 PLIM de referência (K=7, custos fixos, E6)
x_m, y_m, Z_m = resolver_plim()
print("\n=== Solução do PLIM de referência (K=7) ===")
print(f"{'Política':<17}{'Activa':>8}{'Investimento':>14}")
for i in range(n):
    print(f"{nomes[i]:<17}{'sim' if y_m[i] else 'não':>8}{x_m[i]:>14.3f}")
print(f"\nZ* (PLIM) = {Z_m:.2f}   |   Z* (PL) = {Z_opt:.2f}")
print(f"Políticas activas: {y_m.sum()} de {n}")


# ======================================================================
# 5. CENÁRIOS, VARIANTES E ANÁLISE DE SENSIBILIDADE
# ======================================================================
def resolver_pl(b=b, B=B):
    """PL base para um orçamento B e índices b dados. Devolve (x, Z, preço-sombra)."""
    r = linprog(-b, A_ub=np.ones((1, n)), b_ub=[B], bounds=list(zip(L, U)), method="highs")
    return r.x, -r.fun, -r.ineqlin.marginals[0]


def inactivas(y):
    """Nomes das políticas desactivadas (y = 0), ou 'nenhuma'."""
    nomes_off = [nomes[i] for i in range(n) if y[i] == 0]
    return ", ".join(nomes_off) if nomes_off else "nenhuma"


# --- 5.1 Cenários de orçamento (conteção = 90% de 163 ; previsto = 100% de 227)
print("\n=== 5.1 Cenários de orçamento ===")
print(f"{'Cenário':<18}{'B':>8}{'Z (PL)':>11}{'Z (PLIM)':>11}   Inactivas no PLIM")
for nome_c, B_c in [("Contenção", 146.7), ("Base", 163.0), ("Previsto", 227.0)]:
    _, Z_pl_c, _ = resolver_pl(B=B_c)
    _, y_c, Z_plim_c = resolver_plim(B=B_c)
    print(f"{nome_c:<18}{B_c:>8.1f}{Z_pl_c:>11.2f}{Z_plim_c:>11.2f}   {inactivas(y_c)}")

# --- 5.2 Variantes das regras do modelo (B = 163)
print("\n=== 5.2 Variantes do PLIM (B = 163) ===")
print(f"{'Variante':<34}{'Z*':>10}{'Perda vs ref.':>15}   Inactivas")
variantes = [
    ("Referência (K=7, F=0,02·Oᵢ)", dict()),
    ("Sem custos fixos", dict(custos_fixos=False)),
    ("K = 8", dict(K=8)),
    ("K = 9 (todas activas)", dict(K=9)),
    ("E7 com γ = 0,25", dict(gamma=0.25)),
    ("E7 com γ = 0,50", dict(gamma=0.5)),
]
Z_ref = resolver_plim()[2]
for nome_v, kw in variantes:
    _, y_v, Z_v = resolver_plim(**kw)
    print(f"{nome_v:<34}{Z_v:>10.2f}{Z_ref - Z_v:>15.2f}   {inactivas(y_v)}")

# --- 5.3 Sensibilidade do PL base
lam = preco_sombra
i_marg = [i for i in range(n) if not (np.isclose(x_opt[i], L[i]) or np.isclose(x_opt[i], U[i]))][0]
B_inf = B - (x_opt[i_marg] - L[i_marg])       # até onde se pode cortar o orçamento
B_sup = B + (U[i_marg] - x_opt[i_marg])       # até onde se pode aumentar
print("\n=== 5.3 Sensibilidade do PL base ===")
print(f"Política marginal: {nomes[i_marg]}  (x = {x_opt[i_marg]:.3f})")
print(f"Preço-sombra λ = {lam:.2f}, válido para B entre {B_inf:.2f} e {B_sup:.2f}")
print(f"{'Política':<17}{'bᵢ':>6}{'Custo reduzido (bᵢ-λ)':>26}")
for i in range(n):
    print(f"{nomes[i]:<17}{b[i]:>6.1f}{b[i] - lam:>26.2f}")


# ======================================================================
# 6. TESTES COM DADOS ESTOCÁSTICOS (Monte Carlo)
# ======================================================================
# Pressupostos (escolhidos pelo grupo, NÃO retirados de dados):
#   - bᵢ ~ Normal(bᵢ central, (CV·bᵢ central)²), independentes e positivos; CV = 5%, 10%, 20%
#   - taxa de execução ρ ~ Normal(0,72 ; 0,08) truncada em [0,5 ; 1] e B = ρ·227
rng = np.random.default_rng(2026)       # semente fixa: resultados reproduzíveis
N = 2000                                # número de simulações


def sortear_b(cv, tamanho):
    """Sorteia 'tamanho' vectores de índices bᵢ com coeficiente de variação cv."""
    return np.clip(rng.normal(b, cv * b, size=(tamanho, n)), 0.5, None)


# --- 6.1 Incerteza nos índices bᵢ (PL, B = 163)
print("\n=== 6.1 Incerteza em bᵢ (PL, B = 163, N = 2000) ===")
print(f"{'CV de bᵢ':<10}{'E[Z] decisão fixa':>20}{'E[Z] sabendo bᵢ':>18}{'EVPI':>9}{'EVPI %':>9}")
freq_cv20 = None
for cv in (0.05, 0.10, 0.20):
    bs = sortear_b(cv, N)
    Z_fixa = bs @ x_opt                                      # solução do passo 3 avaliada em cada sorteio
    Z_sabendo = np.empty(N)
    estados = np.zeros((3, n))                               # [máximo, marginal, mínimo] por política
    for k in range(N):
        xk, Zk, _ = resolver_pl(b=bs[k])
        Z_sabendo[k] = Zk
        estados[0] += np.isclose(xk, U)
        estados[2] += np.isclose(xk, L)
        estados[1] += ~(np.isclose(xk, U) | np.isclose(xk, L))
    evpi = Z_sabendo.mean() - Z_fixa.mean()
    print(f"{cv:<10.0%}{Z_fixa.mean():>20.1f}{Z_sabendo.mean():>18.1f}{evpi:>9.2f}{100 * evpi / Z_sabendo.mean():>8.2f}%")
    if cv == 0.20:
        freq_cv20 = estados / N
        Z_fixa_20, Z_sabendo_20 = Z_fixa, Z_sabendo       # guardados para os gráficos (secção 7)

print("\nEstabilidade da estrutura óptima com CV = 20% (% das simulações):")
print(f"{'Política':<17}{'no máximo':>11}{'marginal':>10}{'no mínimo':>11}")
for i in range(n):
    print(f"{nomes[i]:<17}{100 * freq_cv20[0, i]:>10.1f}%{100 * freq_cv20[1, i]:>9.1f}%{100 * freq_cv20[2, i]:>10.1f}%")

# --- 6.2 PLIM sob incerteza em bᵢ (CV = 10%): que políticas o modelo desliga?
N_M = 600
bs = sortear_b(0.10, N_M)
desligada = np.zeros(n)
for k in range(N_M):
    _, yk, _ = resolver_plim(b=bs[k])
    desligada += (yk == 0)
print("\n=== 6.2 PLIM com CV = 10%: % das simulações em que a política é desactivada ===")
for i in np.argsort(-desligada):
    if desligada[i] > 0:
        print(f"{nomes[i]:<17}{100 * desligada[i] / N_M:>6.1f}%")

# --- 6.3 Incerteza na taxa de execução do orçamento (PL)
rho = rng.normal(0.72, 0.08, 20000)
rho = rho[(rho >= 0.5) & (rho <= 1.0)][:N]
B_s = rho * 227
Z_rho = np.array([resolver_pl(B=Bk)[1] for Bk in B_s])
Z_no_B_medio = resolver_pl(B=B_s.mean())[1]
fora = np.mean((B_s < B_inf) | (B_s > B_sup))
print("\n=== 6.3 Incerteza na execução do orçamento (B = ρ·227) ===")
print(f"B:   média {B_s.mean():.1f}, desvio-padrão {B_s.std(ddof=1):.1f}")
print(f"Z*:  média {Z_rho.mean():.1f}, desvio-padrão {Z_rho.std(ddof=1):.1f}, "
      f"percentis 5%-95%: {np.percentile(Z_rho, 5):.0f} a {np.percentile(Z_rho, 95):.0f}")
print(f"B fora de [{B_inf:.1f} ; {B_sup:.1f}] (onde λ = {lam:.1f}): {100 * fora:.1f}% das simulações")
print(f"E[Z*(B)] = {Z_rho.mean():.1f}  <  Z*(E[B]) = {Z_no_B_medio:.1f}   (diferença {Z_no_B_medio - Z_rho.mean():.2f})")


# ======================================================================
# 7. GRÁFICOS (gravados em ficheiros .png na pasta figuras_pt2)
# ======================================================================
try:
    import os
    import matplotlib
    matplotlib.use("Agg")                   # grava ficheiros, sem abrir janelas
    import matplotlib.pyplot as plt
except ImportError:
    print("\n=== 7. Gráficos: matplotlib não está instalada ===")
    print("Instale com:  sudo apt install python3-matplotlib   e volte a correr.")
else:
    PASTA = "figuras_pt2"
    os.makedirs(PASTA, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
    VERDE, CINZA, VERMELHO, AMARELO = "#1f6f50", "#9aa5b1", "#c0392b", "#f1c40f"
    idx = np.arange(n)

    # --- Fig. 1: distribuição observada vs óptima
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(idx - 0.2, O, 0.4, label="Observado (Oᵢ)", color=CINZA)
    ax.bar(idx + 0.2, x_opt, 0.4, label="Óptimo PL (x*ᵢ)", color=VERDE)
    ax.set_xticks(idx)
    ax.set_xticklabels(nomes, rotation=30, ha="right")
    ax.set_ylabel("USD milhões")
    ax.set_title(f"Distribuição observada vs óptima (Z = {Z_obs:,.1f} → Z* = {Z_opt:,.1f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{PASTA}/fig1_observado_vs_optimo.png", dpi=170)
    plt.close(fig)

    # --- Fig. 2 e 3: Z*(B) e preço-sombra em função do orçamento
    grelha_B = np.linspace(50, 250, 201)
    Z_pl_B = np.array([resolver_pl(B=Bk)[1] for Bk in grelha_B])
    Z_plim_B = np.array([resolver_plim(B=Bk)[2] for Bk in grelha_B])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(grelha_B, Z_pl_B, color=VERDE, lw=2.2, label="PL (modelo base)")
    ax.plot(grelha_B, Z_plim_B, color=VERMELHO, lw=1.8, ls="--", label="PLIM (K=7, F=0,02·Oᵢ, E6)")
    ax.axvspan(B_inf, B_sup, color=AMARELO, alpha=0.15, label=f"λ = {lam:.1f} válido ({B_inf:.1f}–{B_sup:.1f})")
    for nome_c, B_c in [("Contenção", 146.7), ("Base", 163.0), ("Previsto", 227.0)]:
        ax.axvline(B_c, color="k", lw=0.8, ls=":")
        ax.text(B_c, ax.get_ylim()[0] + 20, nome_c, rotation=90, va="bottom", ha="right", fontsize=8)
    ax.set_xlabel("Orçamento B (USD milhões)")
    ax.set_ylabel("Benefício óptimo Z*")
    ax.set_title("Valor óptimo em função do orçamento disponível")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{PASTA}/fig2_Z_vs_B.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.step(grelha_B, np.gradient(Z_pl_B, grelha_B), where="mid", color=VERDE, lw=2)
    ax.set_xlabel("Orçamento B (USD milhões)")
    ax.set_ylabel("Preço-sombra dZ*/dB")
    ax.set_title("Preço-sombra do orçamento (cada degrau = mudança da política marginal)")
    fig.tight_layout()
    fig.savefig(f"{PASTA}/fig3_preco_sombra.png", dpi=170)
    plt.close(fig)

    # --- Fig. 4: custo de impor a cobertura mínima K e a complementaridade γ
    Ks = list(range(1, 10))
    Z_K = [resolver_plim(K=k)[2] for k in Ks]
    gammas = np.linspace(0, 1.5, 16)
    Z_g = [resolver_plim(gamma=g)[2] for g in gammas]
    fig, eixos = plt.subplots(1, 2, figsize=(10, 4))
    eixos[0].plot(Ks, Z_K, "o-", color=VERDE)
    eixos[0].set_xlabel("Cobertura mínima K")
    eixos[0].set_ylabel("Z*")
    eixos[0].set_title("Custo da cobertura mínima (E5)")
    eixos[1].plot(gammas, Z_g, "o-", color=VERMELHO)
    eixos[1].set_xlabel("γ  (x₇ ≥ γ·x₅)")
    eixos[1].set_title("Custo da complementaridade (E7)")
    fig.tight_layout()
    fig.savefig(f"{PASTA}/fig4_custo_K_gamma.png", dpi=170)
    plt.close(fig)

    # --- Fig. 5: Monte Carlo em bᵢ (CV = 20%): benefício e arrependimento
    arrep = Z_sabendo_20 - Z_fixa_20
    fig, eixos = plt.subplots(1, 2, figsize=(10, 4))
    eixos[0].hist(Z_sabendo_20, bins=40, alpha=0.7, color=VERDE, label="Decisão com bᵢ conhecidos")
    eixos[0].hist(Z_fixa_20, bins=40, alpha=0.6, color=CINZA, label="Decisão fixa (passo 3)")
    eixos[0].set_xlabel("Benefício Z")
    eixos[0].set_title("Incerteza em bᵢ (CV = 20%, N = 2000)")
    eixos[0].legend(fontsize=8)
    eixos[1].hist(arrep, bins=40, color=VERMELHO, alpha=0.8)
    eixos[1].axvline(arrep.mean(), color="k", ls="--", label=f"média = {arrep.mean():.1f}")
    eixos[1].set_xlabel("Arrependimento (Z sabendo − Z decisão fixa)")
    eixos[1].set_title("Perda por ignorar a incerteza")
    eixos[1].legend()
    fig.tight_layout()
    fig.savefig(f"{PASTA}/fig5_montecarlo_b.png", dpi=170)
    plt.close(fig)

    # --- Fig. 6: estabilidade da estrutura óptima (CV = 20%)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(idx, freq_cv20[0], color=VERDE, label="no máximo")
    ax.bar(idx, freq_cv20[1], bottom=freq_cv20[0], color=AMARELO, label="marginal")
    ax.bar(idx, freq_cv20[2], bottom=freq_cv20[0] + freq_cv20[1], color=CINZA, label="no mínimo")
    ax.set_xticks(idx)
    ax.set_xticklabels(nomes, rotation=30, ha="right")
    ax.set_ylabel("Frequência")
    ax.set_title("Estrutura da solução óptima sob incerteza em bᵢ (CV = 20%)")
    ax.legend(ncol=3, loc="lower center")
    fig.tight_layout()
    fig.savefig(f"{PASTA}/fig6_estabilidade.png", dpi=170)
    plt.close(fig)

    # --- Fig. 7: incerteza na execução do orçamento
    fig, eixos = plt.subplots(1, 2, figsize=(10, 4))
    eixos[0].hist(B_s, bins=40, color=CINZA)
    eixos[0].axvspan(B_inf, B_sup, color=AMARELO, alpha=0.25, label="λ constante")
    eixos[0].set_xlabel("B = ρ·227")
    eixos[0].set_title("Orçamento sob incerteza na execução")
    eixos[0].legend()
    eixos[1].hist(Z_rho, bins=40, color=VERDE)
    eixos[1].axvline(Z_rho.mean(), color="k", ls="--", label=f"E[Z*] = {Z_rho.mean():.1f}")
    eixos[1].axvline(Z_no_B_medio, color=VERMELHO, ls=":", label=f"Z*(E[B]) = {Z_no_B_medio:.1f}")
    eixos[1].set_xlabel("Z*")
    eixos[1].set_title("Benefício óptimo (desigualdade de Jensen)")
    eixos[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{PASTA}/fig7_incerteza_rho.png", dpi=170)
    plt.close(fig)

    print(f"\n=== 7. Gráficos ===\nGravadas 7 figuras na pasta '{PASTA}'.")
