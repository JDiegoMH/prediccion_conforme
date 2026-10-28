# 03 · Conteo rápido 2012: el caso principal

**Qué muestra:** en un conteo rápido el sorteo de casillas produce exactamente la intercambiabilidad que pide el teorema. La aplicación compara un umbral marginal contra un umbral por tipo de sección, contrasta distintos intervalos (clásico, cuantiles, CQR, conformal) y verifica la distribución Beta de la cobertura con una población real.

- **Datos:** `data/nal_2012.rda` del repositorio [tereom/quickcountmx](https://github.com/tereom/quickcountmx), con 143,100 casillas con votos. Respuesta: proporción PRI-PVEM.
- **Diseño:** se sortean 7,500 casillas; 3,750 entrenan un boosting de regresión y 3,750 calibran. Se evalúa en las 135,600 casillas no muestreadas, con 20 sorteos y α = 0.10.

| Método | Cobertura | Urbanas | Rurales | Ancho medio |
|---|---|---|---|---|
| Clásico (lineal normal) | 0.903 | 0.949 | 0.766 | 0.317 |
| Cuantiles sin calibrar | 0.809 | 0.818 | 0.791 | 0.246 |
| CQR | 0.901 | 0.918 | 0.861 | 0.309 |
| Conformal (un umbral) | 0.900 | 0.948 | 0.756 | 0.321 |
| Conformal por tipo de sección | 0.900 | 0.900 | 0.900 | 0.316 |

**Beta con 2012:** con el modelo fijo, se repite 3,000 veces la calibración para n = 19, 99 y 999. La media y la desviación estándar de la cobertura coinciden con Beta(k, n+1−k); por ejemplo, para n = 99 la desviación estándar es 0.0295, frente a 0.0299 en teoría.

```sh
git clone https://github.com/tereom/quickcountmx.git datos/quickcountmx
python conteo2012.py --repo datos/quickcountmx
python intervalos2012.py --repo datos/quickcountmx
python beta2012.py --repo datos/quickcountmx
python figuras.py
```

**Teoría:** secciones "Elecciones 2012" y "Variación entre calibraciones" de la guía.
