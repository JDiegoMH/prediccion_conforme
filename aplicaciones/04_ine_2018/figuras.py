"""Figuras del INE 2018: cobertura por corte, curvas de p-valores y, opcionalmente, la deriva de llegada.

Uso: python figuras.py [--remesas RUTA/REMESAS0100020000.txt]
La deriva usa las 6,307 remesas completas, un universo distinto al de 7,754 casillas de arrivals_tbl.
Sin --remesas se conserva figuras/ine_deriva.png tal como se entregó.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent; R = BASE / 'resultados'; OUT = BASE / 'figuras'; OUT.mkdir(exist_ok=True)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--remesas', type=Path, help='REMESAS0100020000.txt: genera ine_deriva sólo si se entrega este archivo')
args = parser.parse_args()
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.dpi': 180})
colors = ['#244C78', '#1C947D']


def save(fig, name, footer='20 réplicas · universo arrivals_tbl: 7,754 casillas'):
    fig.text(.02, .012, footer, fontsize=9, color='#555555')
    fig.tight_layout(rect=(0, .04, 1, 1))
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / (name + '.' + ext), bbox_inches='tight')
    plt.close(fig)


I = pd.read_csv(R / 'ine_resumen.csv')
fig, ax = plt.subplots(figsize=(9, 4.5))
for i, (key, label) in enumerate([('remesa_remesa', 'Reportantes tardías → remesa'), ('remesa_computo', 'Pendientes evaluables → cómputo')]):
    r = I[I.variante == key]
    ax.errorbar(r.corte, 100 * r.coverage, yerr=100 * r.coverage_emc, color=colors[i], marker='o', capsize=5, label=label)
ax.axhline(90, ls='--', color='#AC5937', label='Referencia nominal 90%')
ax.set_ylim(82, 92); ax.set_ylabel('Cobertura (%)'); ax.legend(loc='lower left', fontsize=10)
ax.set_title('Universo: 7,754 casillas de arrivals_tbl')
save(fig, 'ine_cobertura')

# 3. Curvas de p-valores
C = pd.read_csv(R / 'ine_pvalores_curvas.csv'); cols = [c for c in C if c.startswith('a')]; a = np.array([float(c[1:]) for c in cols])
M = C.groupby(['esquema', 'grupo'])[cols].mean()
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .25})
f, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for ax, esq, tit in zip(axs, ['control al azar', 'llegada real'], ['Reportantes repartidas al azar', 'Llegada real (corte 21:00)']):
    ax.plot([0, .5], [0, .5], color='black', lw=.8, ls='--', label='uniforme')
    for g, col in [('todas', '#555555'), ('rurales', '#D08000'), ('urbanas', '#3333B3')]:
        ax.plot(a, M.loc[(esq, g)].to_numpy(), color=col, lw=2, label=g)
    ax.axvline(.1, color='#999999', lw=.6); ax.set_title(tit, loc='left', fontsize=10)
    ax.set_xlabel(r'$\alpha$'); ax.set_xlim(0, .5); ax.set_ylim(0, .72)
axs[0].set_ylabel(r'Proporción con $p \leq \alpha$'); axs[0].legend(frameon=False, fontsize=8, loc='upper left')
f.tight_layout(); f.savefig(OUT / 'ine_pvalores.png', dpi=220, bbox_inches='tight'); plt.close(f)

# Este gráfico usa TODAS las 6,307 remesas, no el universo arrivals_tbl de split.
if args.remesas is not None:
    r=pd.read_csv(args.remesas,sep='|',skiprows=1,dtype={'TIPO_CASILLA':str})
    required=['ANIO','MES','DIA','HORA','MINUTOS','SEGUNDOS','AMLO','TOTAL','TIPO_SECCION',
              'iD_ESTADO','SECCION','TIPO_CASILLA','ID_CASILLA','EXT_CONTIGUA']
    if not set(required).issubset(r.columns) or len(r)!=6307:
        raise ValueError('Se esperan las 6,307 remesas y todas las columnas documentadas')
    keys=['iD_ESTADO','SECCION','TIPO_CASILLA','ID_CASILLA','EXT_CONTIGUA']
    if r[keys].isna().any().any() or r.duplicated(keys).any():
        raise ValueError('Llaves ausentes o duplicadas en remesas')
    numeric=r[['AMLO','TOTAL','TIPO_SECCION']]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all() or not (r.TOTAL>0).all():
        raise ValueError('Datos ausentes/infinitos o denominadores no positivos')
    if not (r.AMLO/r.TOTAL).between(0,1).all() or not r.TIPO_SECCION.isin([1,2,3]).all():
        raise ValueError('Respuesta o tipo de sección fuera del dominio documentado')
    r['ts']=pd.to_datetime(dict(year=r.ANIO,month=r.MES,day=r.DIA,
                                hour=r.HORA,minute=r.MINUTOS,second=r.SEGUNDOS))
    r=r.sort_values('ts',kind='stable').reset_index(drop=True)
    # Conserva el día: 00:30 del día siguiente se dibuja como 24:30.
    r['h']=(r.ts-pd.Timestamp('2018-07-01')).dt.total_seconds()/3600
    cum=100*r.AMLO.cumsum()/r.TOTAL.cumsum()
    rural=100*(r.TIPO_SECCION==3).cumsum()/np.arange(1,len(r)+1)
    mask=np.arange(len(r))>=50  # Igual que la figura heredada: omitir primeros 50 puntos.
    fig,(a1,a2)=plt.subplots(2,1,figsize=(9,5.3),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    a1.plot(r.h[mask],cum[mask],color='#244C78',lw=2,label='Fracción acumulada en remesas')
    a1.axhline(53.19,color='#AC5937',ls='--',lw=1.2,label='Cómputo oficial: 53.19%')
    a1.axvline(22.5,color='#555555',lw=.9,ls=':',label='Referencia horaria: 22:30')
    a1.set(ylabel='% AMLO acumulado',title='Llegada de las 6,307 remesas registradas')
    a1.legend(frameon=False,fontsize=9,loc='lower right')
    a2.plot(r.h[mask],rural[mask],color='#1C947D',lw=2)
    a2.set(ylabel='% casillas de\nsección rural',xlabel='Hora registrada; horas desde el inicio del 1 de julio de 2018')
    a2.set_xticks(range(19,25),[f'{h}:00' for h in range(19,25)])
    save(fig,'ine_deriva','Fuente: remesas quickcountmx; referencia nacional: INE, cómputo presidencial, 6 jul. 2018')
    meta=dict(estado='gráfico regenerado desde remesas; no ajusta modelos',n_remesas=len(r),
        universo='6,307 remesas; distinto del universo arrivals_tbl: 7,754 casillas',
        sha256_remesas=hashlib.sha256(args.remesas.read_bytes()).hexdigest(),
        formula_voto='100 * cumsum(AMLO) / cumsum(TOTAL)',
        formula_rural='100 * cumsum(TIPO_SECCION == 3) / número acumulado de casillas',
        corte_2230='referencia horaria; no certifica identidad del conjunto oficial',
        referencia_oficial=53.19,fuente_oficial='https://centralelectoral.ine.mx/2018/07/06/da-conocer-ine-resultados-del-computo-de-la-eleccion-presidencial-2018/')
    (OUT/'ine_deriva_procedencia.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n')
else:
    print("ine_deriva no regenerada: proporcionar --remesas REMESAS0100020000.txt")
print('Figuras escritas en', OUT)
