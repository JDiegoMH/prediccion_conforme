# 02 · Crédito: conjuntos binarios y cobertura por clase

**Qué muestra:** con una respuesta binaria, el conjunto conforme es `{0}`, `{1}`, `{0,1}` o vacío. Una cobertura marginal del 90% puede esconder una cobertura muy baja en la clase minoritaria, y calibrar por clase (Mondrian) lo corrige.

- **Datos:** [UCI, Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (Yeh y Lien, 2009; DOI [10.24432/C55S3H](https://doi.org/10.24432/C55S3H)), con 30,000 clientes, de los cuales 22.1% incumple. Descarga el ZIP y guárdalo como `datos/credito_uci.zip`. La carpeta `datos/` no se versiona.
- **Diseño:** 20 particiones 60/20/20 con logística sin penalizar y boosting (`HistGradientBoostingClassifier`), α = 0.10. Los comparadores *plug-in* y *plug-in + IC de Wald* no tienen garantía en muestra finita.
- **Resultado (logística):**

| Método | Cobertura | Paga | Incumple |
|---|---|---|---|
| Marginal | 0.900 | 0.997 | **0.561** |
| Por clase | 0.903 | 0.903 | 0.905 |

```sh
python credito.py --replicas 20        # marginal y comparadores
python credito_mondrian.py             # marginal contra por clase
python figuras.py
```

La reejecución con numpy 2.5.4 y pandas 3.0.6 reproduce las coberturas. Algunas proporciones de tamaño de conjunto difieren en 1/120,000 respecto a `resultados/` (las versiones originales están en `procedencia.json`).

**Teoría:** sección "Crédito" de la guía.
