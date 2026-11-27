# -*- coding: utf-8 -*-
"""
Investigação Operacional - Projecto Temático 4 (Grupo 9)  |  Tema 4: Métodos de Pesquisa do Óptimo
Orçamento agrícola público de Moçambique (FAO/MAFAP) com retornos decrescentes.

Modelo:  max Z(x) = sum_i b_i * O_i^(1-g) * x_i^g   s.a.  sum x_i = B ;  L_i <= x_i <= U_i
(calibrado para que Z(O) = sum b_i O_i = 1 079,45, como no PT1; g -> 1 recupera o PL).
Métodos: secção áurea, Fibonacci, bissecção da derivada, condições KKT, gradiente projectado
com pesquisa linear unidimensional.  Validação: SciPy SLSQP e linprog (HiGHS).
Dependências: numpy, scipy, matplotlib.   Unidades: USD milhões.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog, minimize

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
nomes = ["I&D", "Extensão Rural", "Fertilizantes", "Sementes", "Irrigação",
         "Mecanização", "Electrificação", "Estradas Rurais", "Outros Apoios"]
O = np.array([16.30, 18.75, 10.60, 8.96, 22.82, 14.67, 12.71, 28.60, 29.59])
b = np.array([8.5, 7.2, 6.8, 7.5, 9.0, 5.5, 4.8, 6.0, 5.0])
B, GAM = 163.0, 0.70
L, U = 0.30 * O, 1.50 * O
n = 9
np.seterr(over='ignore')
R = {}                       # resultados exportados para o relatório

# ---------------------------------------------------------------- modelo
def f_i(x, g=GAM):  return b * O ** (1 - g) * x ** g
def Z(x, g=GAM):    return f_i(x, g).sum()
def grad(x, g=GAM): return g * b * (O / x) ** (1 - g)

def x_lam(lam, g=GAM, idx=slice(None)):
    """Solução de cada política para o preço-sombra lam (KKT + limites)."""
    return np.clip(O[idx] * (g * b[idx] / lam) ** (1 / (1 - g)), L[idx], U[idx])

def dual(lam, g=GAM):
    x = x_lam(lam, g)
    return f_i(x, g).sum() - lam * x.sum() + lam * B      # g(lam), convexa

def resolver(Bv, g=GAM, idx=slice(None)):
    """Referência exacta: bissecção em lam sobre sum x(lam) = Bv."""
    lo, hi = 1e-6, 1e3
    for _ in range(300):
        m = np.sqrt(lo * hi)
        if x_lam(m, g, idx).sum() > Bv: lo = m
        else: hi = m
    lam = np.sqrt(lo * hi)
    return lam, x_lam(lam, g, idx)

# ---------------------------------------------------------------- pesquisa unidimensional
class Contador:
    def __init__(s, fun): s.fun, s.n = fun, 0
    def __call__(s, t): s.n += 1; return s.fun(t)

def seccao_aurea(fun, a, bb, tol):
    """Minimiza fun unimodal; 1 avaliação por iteração. Devolve x, histórico, nº avaliações."""
    fn = Contador(fun); r = (np.sqrt(5) - 1) / 2
    x1, x2 = bb - r * (bb - a), a + r * (bb - a); f1, f2 = fn(x1), fn(x2); H = []
    k = 0
    while bb - a > tol:
        k += 1; H.append((k, a, bb, x1, x2, f1, f2))
        if f1 < f2: bb, x2, f2 = x2, x1, f1; x1 = bb - r * (bb - a); f1 = fn(x1)
        else:       a, x1, f1 = x1, x2, f2;  x2 = a + r * (bb - a); f2 = fn(x2)
    return 0.5 * (a + bb), H, fn.n, bb - a

def fibonacci(fun, a, bb, tol):
    """Pesquisa de Fibonacci: n escolhido de modo que (b-a)/F_n <= tol."""
    F = [1, 1]
    while F[-1] < (bb - a) / tol: F.append(F[-1] + F[-2])
    N = len(F) - 1
    fn = Contador(fun)
    x1, x2 = a + F[N-2] / F[N] * (bb - a), a + F[N-1] / F[N] * (bb - a); f1, f2 = fn(x1), fn(x2)
    for k in range(1, N - 1):
        if f1 < f2: bb, x2, f2 = x2, x1, f1; x1 = a + F[N-k-2] / F[N-k] * (bb - a); f1 = fn(x1)
        else:       a, x1, f1 = x1, x2, f2;  x2 = a + F[N-k-1] / F[N-k] * (bb - a); f2 = fn(x2)
    if f1 < f2: bb = x2
    else:       a = x1
    return 0.5 * (a + bb), fn.n, bb - a, N

def bisseccao(dfun, a, bb, tol):
    """Bissecção sobre a derivada (1 avaliação por iteração)."""
    k = 0
    while bb - a > tol:
        m = 0.5 * (a + bb); k += 1
        if dfun(m) > 0: bb = m
        else: a = m
    return 0.5 * (a + bb), k, bb - a

# ---------------------------------------------------------------- 1. optimo de referência + KKT
lam_ref, x_ref = resolver(B)
Z_ref = Z(x_ref)
print(f"[ref] lam*={lam_ref:.6f}  Z*={Z_ref:.4f}  soma x={x_ref.sum():.6f}")

# 1.1 secção áurea sobre a função dual g(lam)
LAM_A, LAM_B = 0.5, 12.0
lam_g, H_g, ev_g, w_g = seccao_aurea(dual, LAM_A, LAM_B, 1e-6)
x_g = x_lam(lam_g); Z_g = Z(x_g)
print(f"[aurea dual] lam={lam_g:.6f} Z={Z_g:.4f} g(lam)={dual(lam_g):.4f} aval={ev_g}")
R["aurea_dual"] = dict(lam=lam_g, Z=Z_g, g=dual(lam_g), aval=ev_g, gap=dual(lam_g) - Z_g)

# 1.2 tabela das primeiras iterações (réplica da folha de cálculo)
R["tabela_aurea"] = [dict(k=k, a=a, b=bb, x1=x1, x2=x2, f1=f1, f2=f2) for (k, a, bb, x1, x2, f1, f2) in H_g[:8]]

# 1.3 comparação dos três métodos (nº de avaliações)
comp = []
for tol in (1e-2, 1e-4, 1e-6):
    xa, Ha, ea, wa = seccao_aurea(dual, LAM_A, LAM_B, tol)
    xf, ef, wf, N = fibonacci(dual, LAM_A, LAM_B, tol)
    dg = lambda l: B - x_lam(l).sum()          # g'(lam) = B - sum x(lam)
    xb, eb, wb = bisseccao(dg, LAM_A, LAM_B, tol)
    comp.append(dict(tol=tol, aurea=ea, fib=ef, fibN=N, bis=eb, lam_a=xa, lam_f=xf, lam_b=xb, wa=wa, wf=wf, wb=wb))
    print(f"[tol {tol:g}] aurea {ea} | fib {ef} (F_{N}) | bissecção {eb} | lam: {xa:.6f} {xf:.6f} {xb:.6f}")
R["comparacao"] = comp

# curvas de convergência (largura do intervalo vs avaliações)
def larg_aurea(tol=1e-8):
    _, H, e, _ = seccao_aurea(dual, LAM_A, LAM_B, tol)
    return [(2 + k, h[2] - h[1]) for k, h in enumerate(H)]      # k-ésima iteração -> 2+k avaliações
def larg_fib():
    out = []
    for N in range(3, 30):
        F = [1, 1]
        for _ in range(N): F.append(F[-1] + F[-2])
        out.append((N, (LAM_B - LAM_A) / F[N]))
    return out
def larg_bis(m=40): return [(k, (LAM_B - LAM_A) / 2 ** k) for k in range(1, m)]
conv = dict(aurea=larg_aurea(), fib=larg_fib(), bis=larg_bis())

# 1.4 abordagem primal: perfil phi(x8) = max_{x_-8} Z  (Estradas Rurais como variável marginal)
def phi(t, g=GAM, k=7):
    idx = np.array([i for i in range(n) if i != k])
    lam, xr = resolver(B - t, g, idx)
    return f_i(np.array([t]), g)[0] * 0 + b[k] * O[k] ** (1 - g) * t ** g + (b[idx] * O[idx] ** (1 - g) * xr ** g).sum()
negphi = lambda t: -phi(t)
t_ast, H_p, ev_p, _ = seccao_aurea(negphi, L[7], U[7], 1e-6)
tF, evF, _, NF = fibonacci(negphi, L[7], U[7], 1e-6)
print(f"[perfil x8] aurea x8={t_ast:.5f} ({ev_p} aval)  fib x8={tF:.5f} ({evF} aval, F_{NF})  exacto={x_ref[7]:.5f}  phi={phi(t_ast):.4f}")
R["perfil_x8"] = dict(x8=t_ast, aval=ev_p, fib_x8=tF, fib_aval=evF, exacto=x_ref[7], phi=phi(t_ast))

# 1.5 KKT
f1 = grad(x_ref)
kkt = []
for i in range(n):
    est = "U" if abs(x_ref[i] - U[i]) < 1e-7 else "L" if abs(x_ref[i] - L[i]) < 1e-7 else "I"
    muU = max(f1[i] - lam_ref, 0) if est == "U" else 0.0
    muL = max(lam_ref - f1[i], 0) if est == "L" else 0.0
    res = f1[i] - lam_ref - muU + muL
    kkt.append(dict(nome=nomes[i], O=O[i], L=L[i], U=U[i], x=x_ref[i], est=est, marg=f1[i], muU=muU, muL=muL, res=res))
R["kkt"] = kkt
R["kkt_res_max"] = max(abs(k["res"]) for k in kkt)
R["kkt_comp_max"] = max(abs(k["muU"] * (U[i] - x_ref[i])) + abs(k["muL"] * (x_ref[i] - L[i])) for i, k in enumerate(kkt))
R["ref"] = dict(lam=lam_ref, Z=Z_ref, x=list(x_ref), Zobs=Z(O), budget=x_ref.sum())

# ---------------------------------------------------------------- 2. gradiente projectado com pesquisa linear
def proj(y):
    lo, hi = -1e3, 1e3
    for _ in range(200):
        m = 0.5 * (lo + hi)
        if np.clip(y - m, L, U).sum() > B: lo = m
        else: hi = m
    return np.clip(y - 0.5 * (lo + hi), L, U)

def gradiente(s, linha=True, it=150):
    x = O.copy(); hist = [(0, Z(x), np.linalg.norm(x - x_ref))]; ev = 0
    for k in range(1, it + 1):
        d = proj(x + s * grad(x)) - x
        if np.linalg.norm(d) < 1e-10: break
        if linha:
            th, _, e, _ = seccao_aurea(lambda t: -Z(x + t * d), 0.0, 1.0, 1e-6); ev += e
        else: th = 1.0
        x = x + th * d
        hist.append((k, Z(x), np.linalg.norm(x - x_ref)))
        if hist[-1][2] < 1e-7: break
    return x, hist, ev
xg1, hg1, evg1 = gradiente(10.0, True)     # passo grande + pesquisa linear
xg2, hg2, _ = gradiente(2.0, False)         # passo fixo prudente
xg3, hg3, _ = gradiente(10.0, False)        # passo fixo grande (oscila)
def it_tol(h, tol=1e-6):
    for k, z, d in h:
        if d < tol: return k
    return None
for nm, h, xx in (("linha s=10", hg1, xg1), ("fixo s=2", hg2, xg2), ("fixo s=10", hg3, xg3)):
    print(f"[grad {nm}] it={len(h)-1} it(1e-6)={it_tol(h)} Z={Z(xx):.4f} erro={h[-1][2]:.2e}")
print("[grad] avaliações de linha:", evg1)
R["gradiente"] = dict(it_linha=it_tol(hg1), aval=evg1, Z_linha=Z(xg1), erro_linha=hg1[-1][2],
                      it_fixo2=it_tol(hg2), erro_fixo10=hg3[-1][2], Z_fixo10=Z(xg3),
                      traj=[dict(k=k, Z=z, d=d) for k, z, d in hg1[:6]])

# ---------------------------------------------------------------- 3. validação externa
res = minimize(lambda x: -Z(x), O.copy(), jac=lambda x: -grad(x), method="SLSQP",
               bounds=list(zip(L, U)), constraints=[dict(type="eq", fun=lambda x: x.sum() - B, jac=lambda x: np.ones(n))],
               options=dict(ftol=1e-14, maxiter=500))
R["slsqp"] = dict(Z=-res.fun, dx=float(np.abs(res.x - x_ref).max()), ok=bool(res.success))
print(f"[SLSQP] Z={-res.fun:.4f} max|dx|={np.abs(res.x - x_ref).max():.2e}")
Zpl = -linprog(-b, A_ub=np.ones((1, n)), b_ub=[B], bounds=list(zip(L, U)), method="highs").fun
xpl = linprog(-b, A_ub=np.ones((1, n)), b_ub=[B], bounds=list(zip(L, U)), method="highs").x
lam99, x99 = resolver(B, 0.9999)
R["validacao_g1"] = dict(Zpl=Zpl, Z_g9999=Z(x99, 0.9999), dx=float(np.abs(x99 - xpl).max()))
print(f"[g->1] PL Z={Zpl:.4f}  NLP(g=0.9999) Z={Z(x99,0.9999):.4f} max|dx|={np.abs(x99-xpl).max():.3f}")
R["xpl"] = list(xpl)

# ---------------------------------------------------------------- 4. curvas em B
Bg = np.linspace(60, 244, 185)
Zb, lb = [], []
for Bv in Bg:
    l, x = resolver(Bv); Zb.append(Z(x)); lb.append(l)
Zb, lb = np.array(Zb), np.array(lb)
lpl = []
for Bv in Bg:
    r = linprog(-b, A_ub=np.ones((1, n)), b_ub=[Bv], bounds=list(zip(L, U)), method="highs"); lpl.append(-r.ineqlin.marginals[0])
tabB = []
for Bv in (80, 100, 130, 163, 195, 227):
    l, x = resolver(Bv); tabB.append(dict(B=Bv, Z=Z(x), lam=l, Ztot=Z(x) / Bv))
R["tabB"] = tabB
# retorno marginal decrescente: Z(B+10)-Z(B)
R["ganho10"] = [dict(B=Bv, dZ=Z(resolver(Bv + 10)[1]) - Z(resolver(Bv)[1])) for Bv in (100, 130, 163, 195)]

# ---------------------------------------------------------------- 5. sensibilidade a gamma
gg = np.round(np.arange(0.30, 0.9951, 0.01), 3)
Zg, Xg, lg = [], [], []
for g in gg:
    l, x = resolver(B, g); Zg.append(Z(x, g)); Xg.append(x); lg.append(l)
Zg, Xg = np.array(Zg), np.array(Xg)
def est_str(x): return "".join("U" if abs(x[i]-U[i])<1e-7 else "L" if abs(x[i]-L[i])<1e-7 else "i" for i in range(n))
R["gamas"] = []
for g in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
    l, x = resolver(B, g)
    R["gamas"].append(dict(g=g, lam=l, Z=Z(x, g), ganho=Z(x, g) - Z(O, g), est=est_str(x), x=list(x), nU=est_str(x).count("U"), nL=est_str(x).count("L")))
# limiares: primeiro g em que cada política atinge U (Extensão) / L (Outros Apoios ainda interior)
def limiar(i, alvo):
    gs = np.arange(0.30, 0.9999, 0.0005)
    for g in gs:
        x = resolver(B, g)[1]
        if (alvo == "U" and abs(x[i] - U[i]) < 1e-7) or (alvo == "L" and abs(x[i] - L[i]) < 1e-7): return round(float(g), 4)
R["limiares"] = dict(ext_U=limiar(1, "U"), fert_U=limiar(2, "U"), elec_L=limiar(6, "L"), outros_L=limiar(8, "L"),
                     mec_L=limiar(5, "L"))
print("[limiares]", R["limiares"])

# ---------------------------------------------------------------- 6. ponte com o PT3: plano plurianual
beta, T, Bp, Bmin, Bmax = 0.94, 4, 652.0, 135.0, 195.0
Zstar = lambda Bv: Z(resolver(Bv)[1])
Btot = lambda lam: x_lam(lam).sum()
def traj(nu):
    return np.array([np.clip(Btot(nu / beta ** t), Bmin, Bmax) for t in range(T)])
lo, hi = 1e-3, 50.0
for _ in range(200):
    m = np.sqrt(lo * hi)
    if traj(m).sum() > Bp: lo = m
    else: hi = m
nu = np.sqrt(lo * hi); Bt = traj(nu)
Vdisc = lambda path: sum(beta ** t * Zstar(path[t]) for t in range(T))
paths = {"Uniforme (163 M/ano)": np.array([163.] * 4), "Trajectória PD do PT3 (PL)": np.array([192., 176., 142., 142.]), "Óptimo com retornos decrescentes": Bt}
R["plurianual"] = [dict(nome=k, B=list(v), V=Vdisc(v)) for k, v in paths.items()]
R["plurianual_nu"] = nu
rs = minimize(lambda p: -Vdisc(p), np.array([163.] * 4), method="SLSQP", bounds=[(Bmin, Bmax)] * 4,
              constraints=[dict(type="eq", fun=lambda p: p.sum() - Bp)], options=dict(ftol=1e-13, maxiter=500))
R["plurianual_slsqp"] = dict(B=list(rs.x), V=-rs.fun, dB=float(np.abs(rs.x - Bt).max()))
print("[plurianual]", [(r['nome'], np.round(r['B'], 2), round(r['V'], 2)) for r in R["plurianual"]], R["plurianual_slsqp"])

# ---------------------------------------------------------------- figuras
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8f98", "axes.labelcolor": "#222", "xtick.color": "#444", "ytick.color": "#444",
                     "axes.grid": True, "grid.color": "#e6e8ec", "grid.linewidth": .6, "axes.axisbelow": True})
C1, C2, C3, GR = "#1F4E79", "#D9822B", "#2A9D8F", "#7a8088"
def save(fig, nm): fig.savefig(os.path.join(OUT, nm), dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)

# Fig 1: unimodalidade
fig, ax = plt.subplots(1, 2, figsize=(10, 3.7))
tt = np.linspace(L[7], U[7], 200); ax[0].plot(tt, [phi(t) for t in tt], color=C1, lw=2)
pts = [(h[3], -h[5]) for h in H_p[:7]] + [(h[4], -h[6]) for h in H_p[:7]]
ax[0].scatter([p[0] for p in pts], [p[1] for p in pts], s=22, color=C2, zorder=3, label="Pontos testados (secção áurea)")
ax[0].axvline(t_ast, color=GR, ls="--", lw=1); ax[0].annotate(f"óptimo x₈* = {t_ast:.2f}", (t_ast, phi(t_ast)), (t_ast + 3, phi(t_ast) - 6), fontsize=8, color="#222")
ax[0].set_xlabel("Estradas Rurais, x₈ (USD milhões)"); ax[0].set_ylabel("Perfil φ(x₈) = máx Z dado x₈"); ax[0].set_title("A. Perfil primal: função unimodal", loc="left", fontsize=10, fontweight="bold"); ax[0].legend(frameon=False, fontsize=8, loc="lower center")
ll = np.linspace(2.5, 8.5, 300); ax[1].plot(ll, [dual(l) for l in ll], color=C1, lw=2)
pd_ = [(h[3], h[5]) for h in H_g[:7]] + [(h[4], h[6]) for h in H_g[:7]]
ax[1].scatter([p[0] for p in pd_], [p[1] for p in pd_], s=22, color=C2, zorder=3, label="Pontos testados (secção áurea)")
ax[1].axvline(lam_g, color=GR, ls="--", lw=1); ax[1].annotate(f"λ* = {lam_g:.3f}", (lam_g, dual(lam_g)), (lam_g + .4, dual(lam_g) + 8), fontsize=8)
ax[1].set_xlabel("Preço-sombra do orçamento, λ"); ax[1].set_ylabel("Função dual g(λ)"); ax[1].set_title("B. Perfil dual: função convexa", loc="left", fontsize=10, fontweight="bold"); ax[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); save(fig, "fig1_unimodalidade.png")

# Fig 2: convergência
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for nm, d, c, ls in (("Bissecção (sobre g′)", conv["bis"], GR, "-"), ("Secção áurea", conv["aurea"], C1, "-"), ("Fibonacci", conv["fib"], C2, "--")):
    ax.semilogy([p[0] for p in d if p[0] <= 30], [p[1] for p in d if p[0] <= 30], color=c, lw=2, ls=ls, label=nm)
ax.axhline(1e-6, color="#bbb", lw=.8); ax.text(1, 1.5e-6, "tolerância 10⁻⁶", fontsize=8, color=GR)
ax.set_xlabel("Número de avaliações da função"); ax.set_ylabel("Largura do intervalo de incerteza"); ax.legend(frameon=False, fontsize=8)
save(fig, "fig2_convergencia.png")

# Fig 3: gradiente
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for h, nm, c, ls in ((hg3, "Passo fixo grande (s = 10; θ = 1)", GR, "-"), (hg2, "Passo fixo prudente (s = 2; θ = 1)", C2, "--"), (hg1, "s = 10 com pesquisa linear (secção áurea)", C1, "-")):
    ax[0].semilogy([p[0] for p in h], [max(Z_ref - p[1], 1e-12) for p in h], marker="o", ms=3.5, lw=1.8, color=c, ls=ls, label=nm)
    ax[1].semilogy([p[0] for p in h], [max(p[2], 1e-12) for p in h], marker="o", ms=3.5, lw=1.8, color=c, ls=ls, label=nm)
ax[0].set_xlabel("Iteração k"); ax[0].set_ylabel("Z* − Z(xₖ)"); ax[0].set_title("A. Erro no valor da função", loc="left", fontsize=10, fontweight="bold"); ax[0].legend(frameon=False, fontsize=8, loc="center right", bbox_to_anchor=(1.0, 0.55))
ax[1].set_xlabel("Iteração k"); ax[1].set_ylabel("‖xₖ − x*‖"); ax[1].set_title("B. Distância ao óptimo", loc="left", fontsize=10, fontweight="bold")
fig.tight_layout(); save(fig, "fig3_gradiente.png")

# Fig 4: alocações
fig, ax = plt.subplots(figsize=(10, 3.7)); w = .27; xs = np.arange(n)
ax.bar(xs - w, O, w - .03, color="#b9c2cf", label="Observado (PT1)"); ax.bar(xs, xpl, w - .03, color=C2, label="PL (γ = 1)"); ax.bar(xs + w, x_ref, w - .03, color=C1, label=f"NLP (γ = {GAM:.2f})")
ax.plot(xs + w, U, "_", color="#222", ms=11, mew=1.4, label="Limites Lᵢ e Uᵢ"); ax.plot(xs + w, L, "_", color="#222", ms=11, mew=1.4)
ax.set_xticks(xs); ax.set_xticklabels([s.replace(" ", "\n", 1) if len(s) > 10 else s for s in nomes], fontsize=8); ax.set_ylabel("USD milhões"); ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(.5, 1.13))
save(fig, "fig4_alocacoes.png")

# Fig 5: curvas em B
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(Bg, Zb, color=C1, lw=2); ax[0].axvline(163, color=GR, ls="--", lw=1); ax[0].text(166, Zb.min() + 15, "B = 163", fontsize=8, color=GR)
ax[0].set_xlabel("Orçamento B (USD milhões)"); ax[0].set_ylabel("Z*(B)"); ax[0].set_title("A. Benefício óptimo (côncavo e suave)", loc="left", fontsize=10, fontweight="bold")
ax[1].plot(Bg, lb, color=C1, lw=2, label="NLP (γ = 0,70): λ*(B) contínuo"); ax[1].step(Bg, lpl, where="mid", color=C2, lw=1.8, ls="--", label="PL (PT2): λ(B) em degraus")
ax[1].axvline(163, color=GR, ls="--", lw=1); ax[1].set_xlabel("Orçamento B (USD milhões)"); ax[1].set_ylabel("Preço-sombra (benefício / USD 1 M)"); ax[1].set_title("B. Retorno marginal do orçamento", loc="left", fontsize=10, fontweight="bold"); ax[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); save(fig, "fig5_curvas_B.png")

# Fig 6: sensibilidade a gamma
fig, ax = plt.subplots(1, 2, figsize=(10, 3.7))
ax[0].plot(gg, Zg, color=C1, lw=2, label="Z*(γ) — NLP"); ax[0].axhline(Zpl, color=C2, ls="--", lw=1.5); ax[0].text(.31, Zpl + 3, f"PL (γ = 1): {Zpl:,.2f}".replace(",", " "), fontsize=8, color="#222")
ax[0].axhline(Z(O), color=GR, ls=":", lw=1.5); ax[0].text(.31, Z(O) + 3, f"Observado: {Z(O):,.2f}".replace(",", " "), fontsize=8, color="#222")
ax[0].set_xlabel("Expoente de retorno γ"); ax[0].set_ylabel("Benefício óptimo Z*"); ax[0].set_title("A. Benefício vs. grau de retornos decrescentes", loc="left", fontsize=10, fontweight="bold")
for i, c, ls in ((1, C1, "-"), (8, C2, "-"), (7, C3, "-"), (5, GR, "--")):
    ax[1].plot(gg, Xg[:, i], color=c, lw=2, ls=ls); ax[1].text(.997, Xg[-1, i] + {7: .9, 1: -.9}.get(i, 0), nomes[i], fontsize=8, va="center", ha="left")
ax[1].set_xlim(.3, 1.12); ax[1].set_xlabel("Expoente de retorno γ"); ax[1].set_ylabel("Dotação óptima (USD milhões)"); ax[1].set_title("B. Quatro políticas «pivô»", loc="left", fontsize=10, fontweight="bold")
fig.tight_layout(); save(fig, "fig6_sensibilidade_gamma.png")

with open(os.path.join(OUT, "resultados_pt4.json"), "w", encoding="utf-8") as fh: json.dump(R, fh, ensure_ascii=False, indent=1, default=float)
print("OK")
