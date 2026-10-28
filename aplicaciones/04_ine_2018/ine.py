"""Experimento retrospectivo INE 2018, restringido a arrivals_tbl.
No reproduce ni certifica el conjunto exacto del corte oficial.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from conforme import cuantil

BASE = Path(__file__).resolve().parent
COMMIT_REFERIDO = '86c75437'
FILES = ['data/arrivals_tbl.rda', 'data/conteo_2018.rda',
         'data-raw/4-ConteoRapido18MUESTRA-ELECCION-PRESIDENCIAL.csv',
         'data-raw/REMESAS0100020000.txt']


def cargar(repo):
    import pandas as pd
    import numpy as np
    import rdata
    # El objeto conteo_2018 necesita latin-1; no silenciar errores ni usar c=None.
    def rda(relative, name):
        parsed = rdata.parser.parse_file(repo / relative)
        # conteo_2018 incluye marcas de codificación incompatibles con sus bytes.
        # Forzar latin-1 sólo para ese objeto; respetar marcas de arrivals_tbl.
        return rdata.conversion.convert(
            parsed, default_encoding='latin1',
            force_default_encoding=(name == 'conteo_2018'))[name]
    a = rda(FILES[0], 'arrivals_tbl').rename(columns={'CLAVE_CASILLA':'key'})
    c = rda(FILES[1], 'conteo_2018').rename(columns={'CLAVE_CASILLA':'key'})
    m = pd.read_csv(repo / FILES[2], dtype=str)
    r = pd.read_csv(repo / FILES[3], sep='|', skiprows=1, dtype={'TIPO_CASILLA':str})
    m['key'] = m.ID.str[1:3] + m.ID.str[5:]
    r['key'] = (r.iD_ESTADO.astype(int).map('{:02d}'.format) + r.SECCION.astype(int).map('{:04d}'.format)
        + r.TIPO_CASILLA.str.strip() + r.ID_CASILLA.astype(int).map('{:02d}'.format)
        + r.EXT_CONTIGUA.astype(int).map('{:02d}'.format))
    for name, frame in [('arrivals',a),('computo',c),('muestra',m),('remesa',r)]:
        if frame.key.isna().any() or not frame.key.is_unique:
            raise ValueError(f'Llaves ausentes o duplicadas en {name}')
    if len(a) != 7754 or len(m) != 7787 or not a.key.isin(m.key).all():
        raise ValueError('Universo distinto al documentado: revisar antes de comparar resultados')
    for frame, denominator in [(r,'TOTAL'),(c,'TOTAL_VOTOS_CALCULADOS')]:
        evaluable = frame.AMLO.notna() & frame.key.isin(a.key)
        if not (frame.loc[evaluable,denominator] > 0).all():
            raise ValueError(f'Denominador inválido en {denominator}')
        ratio = frame.loc[evaluable,'AMLO'] / frame.loc[evaluable,denominator]
        if not ratio.between(0,1).all():
            raise ValueError('Fracción de votos fuera de [0,1]')
    r['ts'] = pd.to_datetime(dict(year=r.ANIO,month=r.MES,day=r.DIA,
                                 hour=r.HORA,minute=r.MINUTOS,second=r.SEGUNDOS))
    # Conservar el día: una captura posterior a medianoche no es temprana.
    r['h'] = (r.ts - pd.Timestamp('2018-07-01')).dt.total_seconds()/3600
    r['y_rem'] = r.AMLO/r.TOTAL
    c['y_comp'] = c.AMLO/c.TOTAL_VOTOS_CALCULADOS.replace(0,np.nan)
    d = a.merge(r[['key','h','y_rem']],on='key',how='left',validate='one_to_one').merge(
        c[['key','y_comp']],on='key',how='left',validate='one_to_one')
    if len(d) != 7754:
        raise ValueError('La unión cambió el universo')
    no_reportantes = int((d.y_rem.isna() & d.y_comp.notna()).sum())
    if no_reportantes != 1468:
        raise ValueError(f'No reportantes evaluables: {no_reportantes}, se esperaban 1468')
    cen = ['VPH_INTER','VPH_PISOTI','VPH_LAVAD','VPH_REFRI','VPH_CEL']
    for v in cen:
        d[v+'_p'] = d[v]/d.TVIVPARHAB.replace(0,np.nan)
    d['edo'] = d.key.str[:2].astype(int)
    d['rural'] = (d.tipo_seccion == 3).astype(int)
    d['mixta'] = (d.tipo_seccion == 2).astype(int)
    d['esp'] = (d.tipo_casilla == 'S').astype(int)
    features = ['lista_nominal_log','rural','mixta','esp','edo']+[v+'_p' for v in cen]
    if np.isinf(d[features].to_numpy(dtype=float)).any():
        raise ValueError('Covariables infinitas')
    audit = dict(n_arrivals=len(a), n_muestra=len(m), excluidas_muestra=int((~m.key.isin(a.key)).sum()),
        no_reportantes_evaluables=no_reportantes, remesas_fuera_arrivals=int((~r.key.isin(a.key)).sum()),
        sha256={f:hashlib.sha256((repo/f).read_bytes()).hexdigest() for f in FILES})
    d.columns = [str(c) for c in d.columns]  # rdata 1.1 + pandas 3: nombres de columna con dos tipos de texto
    return d, features, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path, help='Checkout local cotecora-team-2/quickcountmx')
    parser.add_argument('--salida', type=Path, default=BASE/'resultados')
    parser.add_argument('--replicas', type=int, default=20)
    parser.add_argument('--semilla', type=int, default=1000)
    parser.add_argument('--alpha', type=float, default=.1)
    parser.add_argument('--solo-auditoria', action='store_true')
    args = parser.parse_args()
    if args.replicas < 1 or not 0 < args.alpha < 1:
        parser.error('replicas >= 1 y 0 < alpha < 1')
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor
    import importlib.metadata
    import platform
    d, features, audit = cargar(args.repo)
    try:
        commit = subprocess.check_output(['git','-C',str(args.repo),'rev-parse','HEAD'],text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'desconocido (copia sin .git); verificar hashes'
    audit.update(commit_observado=commit, commit_referido_fuente=COMMIT_REFERIDO,
                 semilla=args.semilla, replicas=args.replicas, alpha=args.alpha,
                 universo='7,754 casillas de arrivals_tbl; no toda la muestra diseñada',
                 python=platform.python_version(),
                 versiones={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','rdata']})
    args.salida.mkdir(parents=True,exist_ok=True)
    (args.salida/'auditoria.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    if args.solo_auditoria:
        print(json.dumps(audit,ensure_ascii=False,indent=2)); return
    rows = []
    for t in [20,21,22]:
        obs = d[(d.h <= t)&d.y_rem.notna()]
        tardias = d[(d.h > t)&d.y_rem.notna()]
        todas = d[~d.key.isin(obs.key)&d.y_comp.notna()]
        if len(todas)-len(tardias) != 1468:
            raise ValueError('La diferencia de pendientes no concuerda con el universo heredado')
        # El tercer brazo separa el cambio de medición del cambio de población.
        comunes = tardias[tardias.y_comp.notna()]
        variantes = [('remesa_remesa',tardias,'y_rem'),('remesa_computo',todas,'y_comp'),
                     ('remesa_computo_mismos_reportantes',comunes,'y_comp')]
        for rep in range(args.replicas):
            ix = np.random.default_rng(args.semilla+rep).permutation(len(obs))
            h = len(obs)//2
            train, cal = obs.iloc[ix[:h]], obs.iloc[ix[h:]]
            if set(train.key)&set(cal.key):
                raise ValueError('Fuga entre entrenamiento y calibración')
            f = HistGradientBoostingRegressor(max_iter=300,learning_rate=.05,
                categorical_features=[4],random_state=rep).fit(train[features],train.y_rem)
            q = cuantil(np.abs(cal.y_rem.to_numpy()-f.predict(cal[features])),args.alpha)
            for name, pend, ycol in variantes:
                if set(pend.key)&set(obs.key) or len(pend) == 0:
                    raise ValueError('Prueba vacía o solapada con observadas')
                covered = np.abs(pend[ycol].to_numpy()-f.predict(pend[features])) <= q
                rural = pend.rural.to_numpy()==1
                urbana = pend.tipo_seccion.to_numpy()==1
                rows.append(dict(corte=f'{t}:00',variante=name,rep=rep,n_cal=len(cal),n_pend=len(pend),
                    coverage=covered.mean(),cov_rural=covered[rural].mean(),cov_urbana=covered[urbana].mean(),
                    ancho=2*q,q=q))
            print(f'Corte {t}:00; réplica {rep+1}/{args.replicas}',flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(args.salida/'ine_replicas.csv',index=False)
    metrics=['coverage','cov_rural','cov_urbana','ancho','q']
    g=R.groupby(['corte','variante'],sort=False)
    out=g[['n_cal','n_pend']].first().join(g[metrics].mean()).join(
        g[metrics].std().div(np.sqrt(args.replicas)).add_suffix('_emc'))
    out.to_csv(args.salida/'ine_resumen.csv')


if __name__ == '__main__':
    main()
