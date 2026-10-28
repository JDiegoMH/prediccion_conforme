"""Comparación de intervalos de predicción en el conteo 2012 (mismo diseño que conteo2012.py).

Muestra aleatoria de 7,500 casillas: 3,750 entrenan y 3,750 calibran; se evalúa en las 135,600 no
muestreadas. alpha = 0.10, 20 sorteos. Métodos:
  1. Clásico: regresión lineal (OLS) con intervalo de predicción normal (t), entrenado con las 7,500.
  2. Cuantiles sin calibrar: boosting de regresión cuantílica (5% y 95%) entrenado con 3,750.
  3. CQR: los mismos cuantiles, calibrados con las otras 3,750.
  4. Conformal con residuo absoluto (boosting), un solo umbral.
  5. Conformal con residuo absoluto, un umbral por tipo de sección.
Uso: python intervalos2012.py --repo RUTA/tereom/quickcountmx
"""
import argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd, rdata
from scipy.stats import t as tdist
from sklearn.ensemble import HistGradientBoostingRegressor
from conforme import cuantil
ap = argparse.ArgumentParser(); ap.add_argument('--repo', type=Path, required=True)
ap.add_argument('--salida', type=Path, default=Path(__file__).resolve().parent / 'resultados'); a = ap.parse_args()
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    d = rdata.conversion.convert(rdata.parser.parse_file(a.repo / 'data/nal_2012.rda'), default_encoding='latin1', force_default_encoding=True)['nal_2012']
d.columns = [str(c) for c in d.columns]; d = d[d.total > 0].reset_index(drop=True)
y = (d.pri_pvem.astype(float) / d.total.astype(float)).to_numpy()
tipo = d.tipo_seccion.map({'U': 'urbana', 'M': 'mixta', 'R': 'rural'}).fillna('sin dato').to_numpy()
tcode = d.tipo_seccion.map({'U': 0, 'M': 1, 'R': 2}).fillna(3).astype(float).to_numpy()
num = np.column_stack([d.rural.astype(float), d.tamano_md.astype(float), d.tamano_gd.astype(float), d.casilla_ex.astype(float),
                       (d.casilla == 'S').astype(float), np.log1p(d.ln.astype(float))])
X = np.column_stack([d.edo_id.astype(float), tcode, num])
edo = pd.get_dummies(d.edo_id.astype(int), drop_first=True).to_numpy(float); tdum = pd.get_dummies(tcode, drop_first=True).to_numpy(float)
XL = np.column_stack([np.ones(len(d)), edo, tdum, num])
N = len(d); ALPHA = .10; GR = ['urbana', 'mixta', 'rural']; rows = []; ej = None
gb = lambda **k: HistGradientBoostingRegressor(max_iter=300, learning_rate=.05, categorical_features=[0, 1], **k)
for rep in range(20):
    rng = np.random.default_rng(2012 + rep); perm = rng.permutation(N); mu, te = perm[:7500], perm[7500:]; tr, ca = mu[:3750], mu[3750:]
    out = {}
    # 1. clásico
    A = XL[mu]; b, *_ = np.linalg.lstsq(A, y[mu], rcond=None); res = y[mu] - A @ b; dof = len(mu) - np.linalg.matrix_rank(A)
    s2 = res @ res / dof; Ainv = np.linalg.pinv(A.T @ A); h = np.einsum('ij,jk,ik->i', XL[te], Ainv, XL[te])
    half = tdist.ppf(1 - ALPHA / 2, dof) * np.sqrt(s2 * (1 + h)); pr = XL[te] @ b; out['clásico'] = (pr - half, pr + half)
    # 2-3. cuantiles y CQR
    lo = gb(loss='quantile', quantile=ALPHA / 2, random_state=rep).fit(X[tr], y[tr]); hi = gb(loss='quantile', quantile=1 - ALPHA / 2, random_state=rep).fit(X[tr], y[tr])
    lt, ht = lo.predict(X[te]), hi.predict(X[te]); out['cuantiles sin calibrar'] = (lt, ht)
    sc = np.maximum(lo.predict(X[ca]) - y[ca], y[ca] - hi.predict(X[ca])); q = cuantil(sc, ALPHA); out['CQR'] = (lt - q, ht + q)
    # 4-5. residuo absoluto
    f = gb(random_state=rep).fit(X[tr], y[tr]); sc = np.abs(y[ca] - f.predict(X[ca])); ft = f.predict(X[te]); q = cuantil(sc, ALPHA)
    out['conformal'] = (ft - q, ft + q)
    qg = np.full(len(te), np.nan)
    for g in GR: qg[tipo[te] == g] = cuantil(sc[tipo[ca] == g], ALPHA)
    out['conformal por grupo'] = (ft - qg, ft + qg)
    m = np.isin(tipo[te], GR)
    for k, (l, u) in out.items():
        c = (y[te] >= l) & (y[te] <= u); w = u - l
        r = dict(rep=rep, metodo=k, coverage=c[m].mean(), ancho=w[m].mean())
        for g in GR: r[f'cov_{g}'] = c[tipo[te] == g].mean(); r[f'ancho_{g}'] = w[tipo[te] == g].mean()
        rows.append(r)
    if rep == 0:
        sel = rng.choice(np.flatnonzero(m), 40, replace=False)
        ej = dict(y=y[te][sel], tipo=tipo[te][sel].astype(str), **{k.replace(' ', '_') + '_l': v[0][sel] for k, v in out.items()}, **{k.replace(' ', '_') + '_u': v[1][sel] for k, v in out.items()})
    print(rep, end=' ', flush=True)
a.salida.mkdir(parents=True, exist_ok=True)
R = pd.DataFrame(rows); R.to_csv(a.salida / 'intervalos2012_replicas.csv', index=False)
g = R.drop(columns='rep').groupby('metodo', sort=False); res = g.mean().join(g.std().div(np.sqrt(20)).add_suffix('_emc'))
res.to_csv(a.salida / 'intervalos2012_resumen.csv'); np.savez(a.salida / 'intervalos2012_ejemplo.npz', **ej)
pd.set_option('display.width', 250); print(); print(res[['coverage', 'ancho', 'cov_urbana', 'cov_mixta', 'cov_rural', 'ancho_urbana', 'ancho_rural', 'coverage_emc']].round(4))
