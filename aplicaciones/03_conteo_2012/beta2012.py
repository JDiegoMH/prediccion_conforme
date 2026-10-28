"""Variación entre calibraciones con la población completa de casillas de 2012.

La población (cómputos 2012, quickcountmx::nal_2012) hace de distribución: cada caso nuevo
es una casilla al azar de la población restante. Con el modelo fijo (entrenado una vez),
Q = fracción de la población restante cuyo puntaje queda <= q̂ es la cobertura exacta
de esa calibración. Se repite la calibración 3,000 veces para n = 19, 99, 999.

Uso: python beta2012.py --repo RUTA/quickcountmx [--salida resultados]
(el objeto nal_2012 está en data/nal_2012.rda del repositorio tereom/quickcountmx).
"""
import argparse, math, json, warnings
from pathlib import Path
import numpy as np, pandas as pd, rdata
from scipy.stats import beta
from sklearn.ensemble import HistGradientBoostingRegressor
ap = argparse.ArgumentParser(); ap.add_argument('--repo', type=Path, required=True)
ap.add_argument('--salida', type=Path, default=Path(__file__).resolve().parent / 'resultados'); a = ap.parse_args()
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    d = rdata.conversion.convert(rdata.parser.parse_file(a.repo / 'data/nal_2012.rda'),
                                 default_encoding='latin1', force_default_encoding=True)['nal_2012']
d.columns = [str(c) for c in d.columns]
a.salida.mkdir(parents=True, exist_ok=True); OUT = str(a.salida)
d = d[d.total > 0].copy()
d['y'] = d.pri_pvem.astype(float) / d.total.astype(float)
d['tipo'] = d.tipo_seccion.map({'U': 0, 'M': 1, 'R': 2}).fillna(3)
d['especial'] = (d.casilla == 'S').astype(int)
d['log_ln'] = np.log1p(d.ln.astype(float))
X = d[['edo_id', 'tipo', 'rural', 'tamano_md', 'tamano_gd', 'casilla_ex', 'especial', 'log_ln']].astype(float).to_numpy()
y = d.y.to_numpy(); N = len(d)
rng = np.random.default_rng(20121)
perm = rng.permutation(N); tr, rest = perm[:2000], perm[2000:]
f = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, categorical_features=[0, 1], random_state=0).fit(X[tr], y[tr])
S = np.abs(y[rest] - f.predict(X[rest]))           # puntajes de toda la población restante
M = len(S); Ssort = np.sort(S)
ties = 1 - len(np.unique(S)) / M
ALPHA, R = 0.10, 3000
out = {'N_poblacion': int(N), 'N_restante': int(M), 'n_entrenamiento': 2000, 'fraccion_puntajes_repetidos': ties,
       'pri_nacional': float(d.pri_pvem.sum() / d.total.sum())}
rows = []
for n in (19, 99, 999):
    k = math.ceil((n + 1) * (1 - ALPHA))
    Q = np.empty(R)
    for r in range(R):
        cal = rng.choice(M, n, replace=False)
        q = np.partition(S[cal], k - 1)[k - 1]
        # cobertura para un caso nuevo al azar entre las casillas restantes fuera de la calibración
        cubiertas = np.searchsorted(Ssort, q, side='right') - (S[cal] <= q).sum()
        Q[r] = cubiertas / (M - n)
        rows.append(dict(n=n, rep=r, Q=Q[r], q=q))
    b = beta(k, n + 1 - k)
    out[f'n{n}'] = dict(k=k, media=Q.mean(), media_teorica=k / (n + 1), de=Q.std(ddof=1), de_teorica=b.std(),
                        p_menor_085=(Q < .85).mean(), p_menor_085_teorica=b.cdf(.85),
                        p_menor_088=(Q < .88).mean(), p_menor_088_teorica=b.cdf(.88))
pd.DataFrame(rows).to_csv(f'{OUT}/beta2012_replicas.csv', index=False)
json.dump(out, open(f'{OUT}/beta2012_resumen.json', 'w'), indent=2, ensure_ascii=False)
print(json.dumps(out, indent=2, ensure_ascii=False))
