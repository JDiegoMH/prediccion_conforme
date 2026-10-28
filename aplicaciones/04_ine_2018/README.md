# 04 · INE 2018: cuando el orden de llegada rompe la intercambiabilidad

**Qué muestra:** la noche de la elección, las casillas que llegan primero no son una muestra al azar de las que faltan. Si se calibra con las que ya llegaron, la cobertura en las pendientes cae por debajo del 90%, sobre todo en las rurales. Los p-valores conformes permiten detectarlo.

- **Datos:** el repositorio [cotecora-team-2/quickcountmx](https://github.com/cotecora-team-2/quickcountmx) en el commit `86c75437`. Usa `data/arrivals_tbl.rda`, `data/conteo_2018.rda`, `data-raw/4-ConteoRapido18MUESTRA-ELECCION-PRESIDENCIAL.csv` y `data-raw/REMESAS0100020000.txt`.
- **Universo deliberado:** las 7,754 casillas de `arrivals_tbl`; 33 de las 7,787 de la muestra diseñada quedan fuera.
- **Resultado (cobertura en las pendientes, respuesta en remesa):**

| Corte | Cobertura | Rurales | Urbanas |
|---|---|---|---|
| 20:00 | 0.885 | 0.781 | 0.911 |
| 21:00 | 0.882 | 0.743 | 0.916 |
| 22:00 | 0.892 | 0.751 | 0.926 |

- **P-valores al corte de 21:00:** el 11.8% de las casillas tiene p ≤ 0.10 con la llegada real, contra 9.7% en un control con las mismas casillas repartidas al azar. En las rurales es 25.7%, pero en el control también es alto (24.2%): la mayor parte del exceso rural viene de usar un solo umbral para todas las casillas (cobertura marginal y no condicional), no del orden de llegada.

```sh
git clone https://github.com/cotecora-team-2/quickcountmx.git datos/quickcountmx
git -C datos/quickcountmx checkout 86c75437
python ine.py --repo datos/quickcountmx --replicas 20
python ine_pvalores.py --repo datos/quickcountmx
python figuras.py [--remesas datos/quickcountmx/data-raw/REMESAS0100020000.txt]
```

El experimento es retrospectivo: no reproduce el conjunto exacto del corte oficial. `figuras/ine_deriva.png` es la imagen original; solo se regenera si se pasa `--remesas`.

**Teoría:** secciones "Complemento: cuando el sorteo no basta" y "P-valores conformes" de la guía. Continúa en [09_martingalas](../09_martingalas/).
