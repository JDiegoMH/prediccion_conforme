"""Clasificador de imágenes (dígitos escritos a mano) con conjuntos conformes.

Datos: sklearn.datasets.load_digits (1,797 imágenes de 8x8, 10 clases). A cada imagen se le
asigna al azar un nivel de ruido gaussiano (0, 4 u 8 unidades de intensidad; escala 0-16),
el mismo mecanismo para entrenamiento, calibración y prueba, así que los datos siguen siendo
intercambiables. Modelo: regresión logística multinomial. Puntaje: 1 - p_y(x). alpha = 0.10.
100 particiones: 600 entrenamiento, 500 calibración, 697 prueba.
"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from conforme import cuantil
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'resultados'; OUT.mkdir(parents=True, exist_ok=True); ALPHA = .10; NIV = np.array([0, 4, 8]); NOM = ['sin ruido', 'ruido medio', 'ruido alto']
X0, y = load_digits(return_X_y=True)
rows, ej = [], None
for rep in range(100):
    rng = np.random.default_rng(rep); idx = rng.permutation(len(X0))
    lev = rng.integers(0, 3, len(X0)); X = np.clip(X0 + rng.normal(0, 1, X0.shape) * NIV[lev][:, None], 0, 16)
    tr, ca, te = idx[:600], idx[600:1100], idx[1100:]
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000)).fit(X[tr], y[tr])
    Pc, Pt = m.predict_proba(X[ca]), m.predict_proba(X[te])
    q = cuantil(1 - Pc[np.arange(len(ca)), y[ca]], ALPHA)
    S = (1 - Pt) <= q; size = S.sum(1); cov = S[np.arange(len(te)), y[te]]; acc = Pt.argmax(1) == y[te]
    r = dict(rep=rep, q=q, coverage=cov.mean(), tam_medio=size.mean(), vacio=(size == 0).mean(), uno=(size == 1).mean(),
             dos=(size == 2).mean(), tres_mas=(size >= 3).mean(), exactitud=acc.mean())
    for nom, g in [('t1', size == 1), ('t2', size == 2), ('t3', size >= 3)]:
        r[f'cov_{nom}'] = cov[g].mean() if g.any() else np.nan
    for l in range(3):
        g = lev[te] == l; r.update({f'cov_{l}': cov[g].mean(), f'tam_{l}': size[g].mean(), f'uno_{l}': (size[g] == 1).mean(), f'exact_{l}': acc[g].mean()})
    rows.append(r)
    if rep == 0: ej = dict(X=X[te], y=y[te], lev=lev[te], S=S, P=Pt, q=q, p_cal=Pc[np.arange(len(ca)), y[ca]])
R = pd.DataFrame(rows); R.to_csv(f'{OUT}/imagenes_replicas.csv', index=False)
res = pd.concat([R.drop(columns='rep').mean().rename('media'), (R.drop(columns='rep').std() / 10).rename('emc')], axis=1)
res.to_csv(f'{OUT}/imagenes_resumen.csv'); print(res.round(4))
np.savez(f'{OUT}/imagenes_ejemplo.npz', **ej)
