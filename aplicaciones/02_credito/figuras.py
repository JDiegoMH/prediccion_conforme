"""Figuras de crédito desde resultados/credito_resumen.csv (20 particiones, reejecución UCI).

Uso: python figuras.py
"""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

B = Path(__file__).resolve().parent; OUT = B / 'figuras'; OUT.mkdir(exist_ok=True)
S = pd.read_csv(B / 'resultados/credito_resumen.csv')
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.dpi': 180})
colors = ['#244C78', '#1C947D']
labels = ['Logística', 'Boosting']
C = S[S.metodo == 'conformal'].set_index('modelo').loc[['logística', 'boosting']]


def save(fig, name, footer='20 particiones 60/20/20 · reejecución con los datos UCI'):
    fig.text(.02, .012, footer, fontsize=9, color='#555555')
    fig.tight_layout(rect=(0, .04, 1, 1))
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / (name + '.' + ext), bbox_inches='tight')
    plt.close(fig)


fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, col, title in zip(ax, ['coverage', 'ambos'], ['Cobertura marginal', 'Ambas etiquetas (abstención)']):
    a.bar(labels, 100 * C[col], color=colors, yerr=100 * C[col + '_emc'], capsize=5)
    for i, v in enumerate(C[col]): a.text(i, 100 * v + 1, f'{100*v:.2f}%', ha='center')
    a.set_ylim(0, 105 if col == 'coverage' else 40); a.set_title(title); a.set_ylabel('%')
ax[0].axhline(90, ls='--', color='#AC5937', lw=1)
save(fig, 'credito_modelos')

fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(2); w = .3
for j, col in enumerate(['cov_paga', 'cov_incumple']):
    vals = 100 * C[col].to_numpy()
    ax.bar(x + (j - .5) * w, vals, w, label=['Paga', 'Incumple'][j], color=colors[j], yerr=100 * C[col + '_emc'], capsize=4)
    for xx, v in zip(x + (j - .5) * w, vals): ax.text(xx, v + 1.5, f'{v:.1f}%', ha='center')
ax.set_xticks(x, labels); ax.axhline(90, ls='--', color='#AC5937'); ax.set_ylim(0, 110)
ax.set_ylabel('Cobertura por clase (%)'); ax.legend(loc='lower left')
save(fig, 'credito_por_clase')

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(labels, 100 * C.error_singleton, color=colors, yerr=100 * C.error_singleton_emc, capsize=5)
for i, v in enumerate(C.error_singleton): ax.text(i, 100 * v + .3, f'{100*v:.2f}%', ha='center')
ax.set_ylabel('Error entre conjuntos de una etiqueta (%)'); ax.set_ylim(0, 18)
ax.set_title('90% de cobertura marginal no implica 90% de acierto al decidir')
save(fig, 'credito_abstencion')

fig, ax = plt.subplots(figsize=(10, 4.5))
rows = S[(S.modelo == 'logística') & (S.metodo.isin(['plug-in', 'conformal']))].set_index('metodo').loc[['plug-in', 'conformal']]
for j, col in enumerate(['coverage', 'ambos']):
    vals = 100 * rows[col].to_numpy()
    ax.bar(x + (j - .5) * .32, vals, .32, label=['Cobertura', 'Ambas etiquetas'][j], color=colors[j], yerr=100 * rows[col + '_emc'], capsize=4)
    for xx, v in zip(x + (j - .5) * .32, vals): ax.text(xx, v + 1, f'{v:.1f}%', ha='center')
ax.set_xticks(x, ['Logística plug-in', 'Logística conforme']); ax.set_ylim(0, 112); ax.set_ylabel('%'); ax.legend(loc='upper right', fontsize=10)
save(fig, 'credito_comparacion')
print('Figuras escritas en', OUT)
