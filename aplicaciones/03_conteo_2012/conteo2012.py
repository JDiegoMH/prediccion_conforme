"""Conteo rápido 2012 como aplicación principal de split conformal.

Población: 143,100 casillas con votos (quickcountmx::nal_2012). En cada réplica se sortea una
muestra aleatoria simple de 7,500 casillas (como un conteo rápido); la mitad entrena un boosting
de regresión y la otra mitad calibra. Se predice la proporción de PRI-PVEM en TODAS las casillas
no muestreadas (135,600), que son intercambiables con la calibración por el sorteo.
Puntaje |y - f(x)|, alpha = 0.10, 20 réplicas.
Marginal: un solo umbral. Por grupo: un umbral por tipo de sección (urbana, mixta, rural).
También: p-valores conformes de las no muestreadas, global y por tipo de sección.
Uso: python conteo2012.py --repo RUTA/tereom/quickcountmx [--salida resultados]
"""
import argparse, json, warnings
from pathlib import Path
import numpy as np, pandas as pd, rdata
from sklearn.ensemble import HistGradientBoostingRegressor
from conforme import cuantil
ap = argparse.ArgumentParser(); ap.add_argument('--repo', type=Path, required=True)
ap.add_argument('--salida', type=Path, default=Path(__file__).resolve().parent / 'resultados'); a = ap.parse_args()
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    d = rdata.conversion.convert(rdata.parser.parse_file(a.repo / 'data/nal_2012.rda'), default_encoding='latin1', force_default_encoding=True)['nal_2012']
d.columns = [str(c) for c in d.columns]
d = d[d.total > 0].reset_index(drop=True)
y = (d.pri_pvem.astype(float) / d.total.astype(float)).to_numpy()
tipo = d.tipo_seccion.map({'U': 'urbana', 'M': 'mixta', 'R': 'rural'}).fillna('sin dato').to_numpy()
X = np.column_stack([d.edo_id.astype(float), d.tipo_seccion.map({'U': 0, 'M': 1, 'R': 2}).fillna(3).astype(float),
                     d.rural.astype(float), d.tamano_md.astype(float), d.tamano_gd.astype(float), d.casilla_ex.astype(float),
                     (d.casilla == 'S').astype(float), np.log1p(d.ln.astype(float))])
N = len(d); ALPHA = .10; GR = ['urbana', 'mixta', 'rural']; GRID = np.linspace(0, .5, 51)
rows, curvas = [], []
for rep in range(20):
    rng = np.random.default_rng(2012 + rep)
    perm = rng.permutation(N); mu = perm[:7500]; te = perm[7500:]
    tr, ca = mu[:3750], mu[3750:]
    f = HistGradientBoostingRegressor(max_iter=300, learning_rate=.05, categorical_features=[0, 1], random_state=rep).fit(X[tr], y[tr])
    sc, st = np.abs(y[ca] - f.predict(X[ca])), np.abs(y[te] - f.predict(X[te]))
    q = cuantil(sc, ALPHA); cov = st <= q
    r = dict(rep=rep, n_cal=len(ca), n_prueba=len(te), q=q, ancho=2 * q, coverage=cov.mean())
    qg = {}
    covg = np.zeros(len(te), bool)
    for g in GR:
        mc, mt = tipo[ca] == g, tipo[te] == g
        qg[g] = cuantil(sc[mc], ALPHA); covg[mt] = st[mt] <= qg[g]
        r[f'cov_{g}'] = cov[mt].mean(); r[f'covgrupo_{g}'] = covg[mt].mean(); r[f'ancho_grupo_{g}'] = 2 * qg[g]; r[f'n_cal_{g}'] = int(mc.sum())
    m = np.isin(tipo[te], GR); r['coverage_grupo'] = covg[m].mean(); r['ancho_grupo_medio'] = np.mean([2 * qg[g] for g in tipo[te][m]])
    rows.append(r)
    Ss = np.sort(sc); p = (1 + len(Ss) - np.searchsorted(Ss, st, side='left')) / (len(Ss) + 1)
    pg = np.full(len(te), np.nan)
    for g in GR:
        S2 = np.sort(sc[tipo[ca] == g]); mt = tipo[te] == g
        pg[mt] = (1 + len(S2) - np.searchsorted(S2, st[mt], side='left')) / (len(S2) + 1)
    for esquema, pv in [('marginal', p), ('por grupo', pg)]:
        for g, msk in [('todas', np.isin(tipo[te], GR)), ('urbanas', tipo[te] == 'urbana'), ('rurales', tipo[te] == 'rural')]:
            curvas.append(dict(rep=rep, esquema=esquema, grupo=g, **{f'a{x:.2f}': (pv[msk] <= x).mean() for x in GRID}))
    print(rep, end=' ', flush=True)
a.salida.mkdir(parents=True, exist_ok=True)
R = pd.DataFrame(rows); R.to_csv(a.salida / 'conteo2012_replicas.csv', index=False)
res = pd.concat([R.drop(columns='rep').mean().rename('media'), (R.drop(columns='rep').std() / np.sqrt(len(R))).rename('emc')], axis=1)
res.to_csv(a.salida / 'conteo2012_resumen.csv'); pd.DataFrame(curvas).to_csv(a.salida / 'conteo2012_pvalores_curvas.csv', index=False)
print(); print(res.round(4).to_string())
