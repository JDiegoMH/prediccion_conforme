"""Crédito UCI: probabilidades plug-in y conjuntos conformes LAC.

El plug-in y su variante Wald son comparadores modelados, sin garantía
conforme finita. Crédito binario produce CONJUNTOS de etiquetas, no intervalos.
"""
import argparse
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import zipfile
from conforme import cuantil, validar_particion

BASE = Path(__file__).resolve().parent
CAT = ['SEX', 'EDUCATION', 'MARRIAGE']
YCOL = 'default payment next month'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datos', type=Path, default=BASE / 'datos/credito_uci.zip')
    parser.add_argument('--salida', type=Path, default=BASE / 'resultados')
    parser.add_argument('--replicas', type=int, default=20)
    parser.add_argument('--semilla', type=int, default=20261008)
    parser.add_argument('--alpha', type=float, default=.1)
    args = parser.parse_args()
    if args.replicas < 1 or not 0 < args.alpha < .5:
        parser.error('replicas >= 1 y 0 < alpha < 0.5')
    import numpy as np
    import pandas as pd
    from scipy.special import expit
    from scipy.stats import norm
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    if args.datos.suffix.lower() == '.zip':
        with zipfile.ZipFile(args.datos) as z:
            names = [n for n in z.namelist() if n.lower().endswith('.xls')]
            if len(names) != 1:
                raise ValueError('Se esperaba un único XLS UCI en el ZIP')
            d = pd.read_excel(io.BytesIO(z.read(names[0])), header=1, engine='xlrd')
    else:
        d = pd.read_excel(args.datos, header=1, engine='xlrd')
    d.columns = d.columns.str.strip()
    y = d[YCOL].to_numpy(dtype=int)
    if len(d) != 30000 or not np.isin(y, [0, 1]).all() or not d.ID.is_unique or d.isna().any().any():
        raise ValueError('No coincide con los 30,000 registros completos de crédito UCI')
    num = [c for c in d if c not in CAT + ['ID', YCOL]]
    args.salida.mkdir(parents=True, exist_ok=True)

    def plugin(p, lo=None, hi=None):
        lo = p if lo is None else lo
        hi = p if hi is None else hi
        return np.column_stack([~(lo >= 1 - args.alpha), ~(hi <= args.alpha)])

    def summary(sets, yt, model, method, rep, p, q=np.nan):
        size = sets.sum(1)
        covered = sets[np.arange(len(yt)), yt]
        single = size == 1
        return dict(rep=rep, modelo=model, metodo=method, coverage=covered.mean(),
                    cov_paga=covered[yt == 0].mean(), cov_incumple=covered[yt == 1].mean(),
                    solo0=(sets[:, 0] & ~sets[:, 1]).mean(), solo1=(sets[:, 1] & ~sets[:, 0]).mean(),
                    ambos=(size == 2).mean(), vacio=(size == 0).mean(),
                    error_singleton=(~covered[single]).mean() if single.any() else np.nan,
                    auc=roc_auc_score(yt, p), q=q)

    rows = []
    for rep, ss in enumerate(np.random.SeedSequence(args.semilla).spawn(args.replicas)):
        idx = np.random.default_rng(ss).permutation(len(d))
        n = len(d)
        tr, ca, te = idx[:int(.6*n)], idx[int(.6*n):int(.8*n)], idx[int(.8*n):]
        validar_particion(tr, ca, te, n)
        train, cal, test = [d.iloc[i] for i in (tr, ca, te)]
        # Todo preprocesamiento se ajusta exclusivamente en entrenamiento.
        oh = OneHotEncoder(handle_unknown='ignore', drop='first', min_frequency=50).fit(train[CAT])
        sc = StandardScaler().fit(train[num])
        def design(frame):
            return np.column_stack([np.ones(len(frame)), oh.transform(frame[CAT]).toarray(), sc.transform(frame[num])])
        Xtr, Xca, Xte = [design(f) for f in (train, cal, test)]
        lg = LogisticRegression(C=np.inf, fit_intercept=False, max_iter=5000).fit(Xtr, y[tr])
        b = lg.coef_.ravel()
        ptr = expit(Xtr @ b)
        information = (Xtr * (ptr * (1 - ptr))[:, None]).T @ Xtr
        V = np.linalg.inv(information)  # Fallar si no hay inversa; no ocultar singularidad.
        eta = Xte @ b
        variances = np.einsum('ij,jk,ik->i', Xte, V, Xte)
        if np.any(variances < -1e-10):
            raise ValueError('Varianza Wald negativa')
        se = np.sqrt(np.maximum(variances, 0))
        lo, hi = expit(eta - norm.ppf(.975)*se), expit(eta + norm.ppf(.975)*se)
        pte, pca = expit(eta), expit(Xca @ b)
        Pc = np.column_stack([1-pca, pca])
        q = cuantil(1 - Pc[np.arange(len(ca)), y[ca]], args.alpha)
        ls = (1 - np.column_stack([1-pte, pte])) <= q
        rows.extend([summary(plugin(pte), y[te], 'logística', 'plug-in', rep, pte),
                     summary(plugin(pte, lo, hi), y[te], 'logística', 'plug-in + IC Wald', rep, pte),
                     summary(ls, y[te], 'logística', 'conformal', rep, pte, q)])
        gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=.05,
                                           categorical_features=[0, 1, 2], random_state=0)
        cols = CAT + num
        gb.fit(train[cols], y[tr])
        Pc, Pt = gb.predict_proba(cal[cols]), gb.predict_proba(test[cols])
        qg = cuantil(1 - Pc[np.arange(len(ca)), y[ca]], args.alpha)
        gs = (1-Pt) <= qg
        rows.extend([summary(plugin(Pt[:, 1]), y[te], 'boosting', 'plug-in', rep, Pt[:, 1]),
                     summary(gs, y[te], 'boosting', 'conformal', rep, Pt[:, 1], qg)])
        if rep == 0:
            # Identificador interno de orden test: no atributos personales.
            examples = pd.DataFrame(dict(fila_test=np.arange(len(te)), y=y[te], log_p1=pte,
                log_q=q, log_incluye_0=ls[:, 0], log_incluye_1=ls[:, 1], gb_p1=Pt[:, 1],
                gb_q=qg, gb_incluye_0=gs[:, 0], gb_incluye_1=gs[:, 1]))
            examples.to_csv(args.salida / 'credito_ejemplos.csv', index=False)
        print(f'Réplica {rep+1}/{args.replicas} terminada', flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(args.salida / 'credito_replicas.csv', index=False)
    g = R.drop(columns='rep').groupby(['modelo', 'metodo'], sort=False)
    g.mean().join(g.std().div(np.sqrt(args.replicas)).add_suffix('_emc')).to_csv(args.salida / 'credito_resumen.csv')
    metadata = dict(estado='reejecutado desde XLS UCI', semilla=args.semilla, replicas=args.replicas,
        alpha=args.alpha, particion='60/20/20', sha256_datos=hashlib.sha256(args.datos.read_bytes()).hexdigest(),
        python=platform.python_version(), versiones={p:importlib.metadata.version(p) for p in
        ['numpy','pandas','scipy','scikit-learn','xlrd']},
        nota='EMC es error Monte Carlo entre particiones de base fija; con una réplica no es estimable.')
    (args.salida / 'procedencia.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+'\n')


if __name__ == '__main__':
    main()
