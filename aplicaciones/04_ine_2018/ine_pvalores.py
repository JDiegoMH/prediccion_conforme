"""P-valores conformes de casillas pendientes (INE 2018) frente a un control intercambiable.

Corte 21:00. Llegada real: entrenamiento y calibración con las remesas recibidas hasta el corte;
p-valores de las reportantes tardías (respuesta y calibración en remesa, misma fuente).
Control: las mismas 6,286 reportantes repartidas al azar, ignorando la hora, en entrenamiento,
calibración y prueba de los mismos tamaños. Mismo modelo que ine.py. 20 réplicas.
p(y) = (1 + #{i: S_i >= s}) / (n + 1).

Uso: python ine_pvalores.py --repo RUTA/quickcountmx [--salida resultados]
"""
import argparse, json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from ine import cargar
ap = argparse.ArgumentParser(); ap.add_argument('--repo', type=Path, required=True)
ap.add_argument('--salida', type=Path, default=Path(__file__).resolve().parent / 'resultados'); a = ap.parse_args()
d, features, _ = cargar(a.repo)
d.columns = [str(c) for c in d.columns]   # rdata devuelve nombres de dos tipos de texto; sklearn exige uno
a.salida.mkdir(parents=True, exist_ok=True); OUT = str(a.salida)
def pvals(Scal, Snew):
    Ss = np.sort(Scal); n = len(Ss)
    return (1 + n - np.searchsorted(Ss, Snew, side='left')) / (n + 1)
def fit_scores(train, cal, test, rep):
    f = HistGradientBoostingRegressor(max_iter=300, learning_rate=.05, categorical_features=[4], random_state=rep).fit(train[features], train.y_rem)
    return np.abs(cal.y_rem.to_numpy() - f.predict(cal[features])), np.abs(test.y_rem.to_numpy() - f.predict(test[features]))
rep_all = d[d.y_rem.notna()]
t = 21; obs = rep_all[rep_all.h <= t]; late = rep_all[rep_all.h > t]
rows = []; GRID = np.linspace(0, .5, 51); curvas = []
for rep in range(20):
    rng = np.random.default_rng(1000 + rep)
    ix = rng.permutation(len(obs)); h = len(obs) // 2
    Sc, St = fit_scores(obs.iloc[ix[:h]], obs.iloc[ix[h:]], late, rep)
    p_llegada = pvals(Sc, St)
    jx = rng.permutation(len(rep_all)); a, b = h, len(obs)
    Sc2, St2 = fit_scores(rep_all.iloc[jx[:a]], rep_all.iloc[jx[a:b]], rep_all.iloc[jx[b:]], rep)
    p_control = pvals(Sc2, St2)
    for nombre, p, rur in [('llegada real', p_llegada, late.tipo_seccion.to_numpy()), ('control al azar', p_control, rep_all.iloc[jx[b:]].tipo_seccion.to_numpy())]:
        for grupo, msk in [('todas', np.ones(len(p), bool)), ('rurales', rur == 3), ('urbanas', rur == 1)]:
            curvas.append(dict(rep=rep, esquema=nombre, grupo=grupo, **{f'a{a:.2f}': (p[msk] <= a).mean() for a in GRID}))
        rows.append(dict(rep=rep, esquema=nombre, n_cal=len(Sc), n_prueba=len(p), p_le_010=(p <= .10).mean(), p_le_005=(p <= .05).mean(),
                         p_media=p.mean(), p_le_010_rural=(p[rur == 3] <= .10).mean(), p_le_010_urbana=(p[rur == 1] <= .10).mean()))
    if rep == 0: np.savez(f'{OUT}/ine_pvalores_ejemplo.npz', llegada=p_llegada, control=p_control)
pd.DataFrame(curvas).to_csv(f'{OUT}/ine_pvalores_curvas.csv', index=False)
R = pd.DataFrame(rows); R.to_csv(f'{OUT}/ine_pvalores_replicas.csv', index=False)
g = R.drop(columns='rep').groupby('esquema', sort=False); res = g.mean().join(g.std().div(np.sqrt(20)).add_suffix('_emc'))
res.to_csv(f'{OUT}/ine_pvalores_resumen.csv'); pd.set_option('display.width', 250); print(res.round(4))
