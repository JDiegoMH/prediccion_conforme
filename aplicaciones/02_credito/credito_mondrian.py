"""Crédito UCI: conformal marginal frente a conformal por clase (Mondrian).

Mismas 20 particiones 60/20/20, preprocesamiento y modelos que credito.py.
Marginal: un umbral con todos los puntajes de calibración.
Por clase: un umbral q_y con los puntajes de calibración de la clase y; la etiqueta y
entra al conjunto si 1 - p_y(x) <= q_y.
"""
import argparse, io, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from conforme import cuantil, validar_particion
CAT = ['SEX', 'EDUCATION', 'MARRIAGE']; YCOL = 'default payment next month'
ap = argparse.ArgumentParser(); B = Path(__file__).resolve().parent
ap.add_argument('--datos', type=Path, default=B / 'datos/credito_uci.zip'); ap.add_argument('--salida', type=Path, default=B / 'resultados')
ap.add_argument('--replicas', type=int, default=20); ap.add_argument('--semilla', type=int, default=20261008); ap.add_argument('--alpha', type=float, default=.1)
a = ap.parse_args()
with zipfile.ZipFile(a.datos) as z:
    name = [n for n in z.namelist() if n.lower().endswith('.xls')][0]
    d = pd.read_excel(io.BytesIO(z.read(name)), header=1, engine='xlrd')
d.columns = d.columns.str.strip(); y = d[YCOL].to_numpy(dtype=int); num = [c for c in d if c not in CAT + ['ID', YCOL]]
def resumen(sets, yt, modelo, metodo, rep, q0, q1):
    size = sets.sum(1); cov = sets[np.arange(len(yt)), yt]; single = size == 1
    return dict(rep=rep, modelo=modelo, metodo=metodo, coverage=cov.mean(), cov_paga=cov[yt == 0].mean(), cov_incumple=cov[yt == 1].mean(),
                solo0=(sets[:, 0] & ~sets[:, 1]).mean(), solo1=(sets[:, 1] & ~sets[:, 0]).mean(), ambos=(size == 2).mean(), vacio=(size == 0).mean(),
                error_singleton=(~cov[single]).mean(), q0=q0, q1=q1)
rows = []
for rep, ss in enumerate(np.random.SeedSequence(a.semilla).spawn(a.replicas)):
    idx = np.random.default_rng(ss).permutation(len(d)); n = len(d)
    tr, ca, te = idx[:int(.6*n)], idx[int(.6*n):int(.8*n)], idx[int(.8*n):]; validar_particion(tr, ca, te, n)
    train, cal, test = [d.iloc[i] for i in (tr, ca, te)]
    oh = OneHotEncoder(handle_unknown='ignore', drop='first', min_frequency=50).fit(train[CAT]); sc = StandardScaler().fit(train[num])
    des = lambda f: np.column_stack([np.ones(len(f)), oh.transform(f[CAT]).toarray(), sc.transform(f[num])])
    lg = LogisticRegression(C=np.inf, fit_intercept=False, max_iter=5000).fit(des(train), y[tr])
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=.05, categorical_features=[0, 1, 2], random_state=0).fit(train[CAT + num], y[tr])
    for modelo, Pc, Pt in [('logística', lg.predict_proba(des(cal)), lg.predict_proba(des(test))),
                           ('boosting', gb.predict_proba(cal[CAT + num]), gb.predict_proba(test[CAT + num]))]:
        sc_cal = 1 - Pc[np.arange(len(ca)), y[ca]]
        q = cuantil(sc_cal, a.alpha)
        rows.append(resumen((1 - Pt) <= q, y[te], modelo, 'marginal', rep, q, q))
        qy = [cuantil(sc_cal[y[ca] == c], a.alpha) for c in (0, 1)]
        rows.append(resumen((1 - Pt) <= np.array(qy)[None, :], y[te], modelo, 'por clase', rep, *qy))
    print(rep, end=' ', flush=True)
R = pd.DataFrame(rows); a.salida.mkdir(parents=True, exist_ok=True); R.to_csv(a.salida / 'credito_mondrian_replicas.csv', index=False)
g = R.drop(columns='rep').groupby(['modelo', 'metodo'], sort=False)
res = g.mean().join(g.std().div(np.sqrt(a.replicas)).add_suffix('_emc')); res.to_csv(a.salida / 'credito_mondrian_resumen.csv')
pd.set_option('display.width', 250); print(); print(res[['coverage', 'cov_paga', 'cov_incumple', 'solo0', 'solo1', 'ambos', 'vacio', 'error_singleton', 'q0', 'q1', 'coverage_emc', 'cov_incumple_emc']].round(4))
