"""Figuras de dígitos: ejemplos de conjuntos y el primer algoritmo. Uso: python figuras.py"""
import json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import beta
B = Path(__file__).resolve().parent; R = B / "resultados"; F = B / "figuras"; F.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
# 1. Ejemplos de imágenes con su conjunto conforme (partición 0)
e = np.load(R / 'imagenes_ejemplo.npz'); X, y, lev, S = e['X'], e['y'], e['lev'], e['S']
size = S.sum(1); cov = S[np.arange(len(y)), y]; rng = np.random.default_rng(3)
pick = lambda c: int(rng.choice(np.flatnonzero(c)))
sel = [pick((lev == 0) & (size == 1) & cov), pick((lev == 1) & (size == 1) & cov), pick((lev >= 1) & (size == 2) & cov),
       pick((lev >= 1) & (size == 3) & cov), pick((lev == 2) & (size >= 4) & cov), pick((size == 1) & ~cov)]
NOM = ['sin ruido', 'ruido medio', 'ruido alto']
f, axs = plt.subplots(1, 6, figsize=(12, 2.9))
for ax, i in zip(axs, sel):
    ax.imshow(X[i].reshape(8, 8), cmap='gray_r', vmin=0, vmax=16, interpolation='nearest'); ax.set_xticks([]); ax.set_yticks([])
    lab = '{' + ', '.join(str(c) for c in np.flatnonzero(S[i])) + '}'
    ax.set_title(NOM[lev[i]], fontsize=9, color='#555555')
    ax.set_xlabel(f'C(x) = {lab}\nverdad: {y[i]}', fontsize=11, color='#B22222' if not cov[i] else '#222222')
f.tight_layout(); f.savefig(F / 'imagenes_ejemplos.png', dpi=220, bbox_inches='tight'); plt.close(f)
# 2. Primer algoritmo: probabilidad de la clase correcta en calibración y un caso nuevo
plt.rcParams.update({'axes.grid': False, 'axes.spines.top': False, 'axes.spines.right': False})
pc = e['p_cal']; P = e['P']; t = 1 - float(e['q'])
cand = np.flatnonzero((lev >= 1) & (size == 3) & cov)
i = int(cand[np.argmax(np.abs(P[cand] - t).min(1))])   # el caso con márgenes más claros respecto al umbral
f, axs = plt.subplots(1, 3, figsize=(11, 3.0), gridspec_kw={'width_ratios': [2.2, 0.9, 2.2]})
ax = axs[0]
ax.hist(pc, bins=np.linspace(0, 1, 41), color='#BBBBBB', edgecolor='white')
ax.axvline(t, color='#B22222', lw=2); ax.set_xlabel(r'$\widehat p_{Y_i}(X_i)$: probabilidad de la clase correcta')
ax.set_yticks([]); ax.set_title('Calibración (500 imágenes)', loc='left', fontsize=10)
ax.text(t + .02, ax.get_ylim()[1] * .85, f'$\\widehat t={t:.2f}$\ncerca de 10% queda a la izquierda', color='#B22222', fontsize=9)
axs[1].imshow(X[i].reshape(8, 8), cmap='gray_r', vmin=0, vmax=16, interpolation='nearest'); axs[1].set_xticks([]); axs[1].set_yticks([])
axs[1].set_title('Imagen nueva', fontsize=10)
ax = axs[2]; cols = ['#3333B3' if P[i, c] >= t else '#BBBBBB' for c in range(10)]
ax.bar(range(10), P[i], color=cols); ax.axhline(t, color='#B22222', lw=2); ax.set_xticks(range(10))
ax.set_xlabel('Dígito'); ax.set_ylabel(r'$\widehat p_y(x)$')
lab = '{' + ', '.join(str(c) for c in np.flatnonzero(P[i] >= t)) + '}'
ax.set_title(f'Conjunto: {lab}   (verdad: {y[i]})', loc='left', fontsize=10)
f.tight_layout(); f.savefig(F / 'imagenes_algoritmo.png', dpi=220, bbox_inches='tight'); plt.close(f)
print('algoritmo', i, lab, y[i], round(t, 3))
