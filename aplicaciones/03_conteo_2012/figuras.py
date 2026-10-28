"""Figuras del conteo 2012: Beta con las casillas, p-valores e intervalos. Uso: python figuras.py"""
import json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import beta
B = Path(__file__).resolve().parent; R = B / "resultados"; F = B / "figuras"; F.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
# 1. Beta con 2012
Q = pd.read_csv(R / 'beta2012_replicas.csv'); s = json.load(open(R / 'beta2012_resumen.json'))
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})
f, axs = plt.subplots(1, 3, figsize=(10.5, 3.0))
for ax, n, col in zip(axs, (19, 99, 999), ('#D08000', '#B22222', '#3333B3')):
    q = Q.loc[Q.n == n, 'Q'].to_numpy(); k = s[f'n{n}']['k']; lo, hi = {19: (.6, 1), 99: (.8, .98), 999: (.86, .94)}[n]
    ax.hist(q, bins=40, range=(lo, hi), density=True, color='#BBBBBB', edgecolor='white', label='3,000 calibraciones')
    u = np.linspace(lo, hi, 400); ax.plot(u, beta(k, n + 1 - k).pdf(u), color=col, lw=2, label=f'Beta({k}, {n+1-k})')
    ax.axvline(.9, color='black', lw=.8, ls='--'); ax.set_title(f'n = {n}', loc='left')
    ax.set_xlabel('Cobertura de una calibración fija'); ax.set_yticks([]); ax.legend(frameon=False, fontsize=8, loc='upper left')
f.tight_layout(); f.savefig(F / 'beta_2012.png', dpi=220, bbox_inches='tight'); plt.close(f)
# 2. P-valores 2012: marginal contra por grupo
if (R / 'conteo2012_pvalores_curvas.csv').exists():
    C = pd.read_csv(R / 'conteo2012_pvalores_curvas.csv'); cols = [c for c in C if c.startswith('a')]; a = np.array([float(c[1:]) for c in cols])
    M = C.groupby(['esquema', 'grupo'])[cols].mean()
    plt.rcParams.update({'axes.grid': True, 'grid.alpha': .25, 'axes.spines.top': False, 'axes.spines.right': False})
    f, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, esq, tit in zip(axs, ['marginal', 'por grupo'], ['Un solo umbral (marginal)', 'Un umbral por tipo de sección']):
        ax.plot([0, .5], [0, .5], color='black', lw=.8, ls='--', label='uniforme')
        for g, col in [('todas', '#555555'), ('rurales', '#D08000'), ('urbanas', '#3333B3')]:
            ax.plot(a, M.loc[(esq, g)].to_numpy(), color=col, lw=2, label=g)
        ax.axvline(.1, color='#999999', lw=.6); ax.set_title(tit, loc='left', fontsize=10)
        ax.set_xlabel(r'$\alpha$'); ax.set_xlim(0, .5); ax.set_ylim(0, .72)
    axs[0].set_ylabel(r'Proporción con $p \leq \alpha$'); axs[0].legend(frameon=False, fontsize=8, loc='upper left')
    f.tight_layout(); f.savefig(F / 'conteo2012_pvalores.png', dpi=220, bbox_inches='tight'); plt.close(f)
    print(M[['a0.10']].round(4))
# 3. Intervalos de predicción en 2012: tres métodos sobre las mismas 40 casillas
if (R / 'intervalos2012_ejemplo.npz').exists():
    e2 = np.load(R / 'intervalos2012_ejemplo.npz'); yy = e2['y']; tp = e2['tipo']
    orden = np.lexsort(((e2['conformal_l'] + e2['conformal_u']) / 2, -(tp == 'rural').astype(int)))
    plt.rcParams.update({'axes.grid': False})
    f, axs = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
    for ax, k, tit in zip(axs, ['clásico', 'CQR', 'conformal_por_grupo'], ['Clásico (regresión lineal normal)', 'CQR', 'Conformal por tipo de sección']):
        l, u = e2[k + '_l'][orden], e2[k + '_u'][orden]; yv = yy[orden]; tt = tp[orden]; xs = np.arange(len(yv))
        cubre = (yv >= l) & (yv <= u)
        for i in xs:
            ax.plot([i, i], [l[i], u[i]], color='#D08000' if tt[i] == 'rural' else '#3333B3', lw=2.2, alpha=.75)
        ax.scatter(xs[cubre], yv[cubre], s=12, color='black', zorder=3)
        ax.scatter(xs[~cubre], yv[~cubre], s=22, color='#B22222', marker='x', zorder=3)
        ax.set_title(tit, loc='left', fontsize=10); ax.set_xticks([]); ax.set_xlabel('40 casillas no muestreadas')
    axs[0].set_ylabel('Proporción PRI-PVEM'); axs[0].set_ylim(-0.05, 1.0)
    from matplotlib.lines import Line2D
    axs[2].legend([Line2D([0], [0], color='#D08000', lw=2), Line2D([0], [0], color='#3333B3', lw=2), Line2D([0], [0], marker='x', color='#B22222', lw=0)],
                  ['rural', 'urbana o mixta', 'no cubierta'], frameon=False, fontsize=8, loc='upper right')
    f.tight_layout(); f.savefig(F / 'intervalos2012.png', dpi=220, bbox_inches='tight'); plt.close(f)
