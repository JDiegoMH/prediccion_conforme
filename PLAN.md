# Plan de acción

Estado al 12 de octubre de 2026: las aplicaciones 01–04 funcionan y las 05–09 solo tienen su ficha. Este plan dice qué falta en cada una, en el orden sugerido.

## Criterio común de "terminado"

Una aplicación está terminada cuando tiene:

1. **Scripts reproducibles:** semilla fija, rutas relativas y datos que se descargan o se indica cómo obtenerlos (nunca versionados).
2. **Réplicas:** al menos 20 particiones o sorteos. Se reportan media y error Monte Carlo (EMC).
3. **Resultados y figuras:** los resultados en `resultados/`, las figuras en `figuras/` (con su `figuras.py`) y las versiones de los paquetes en un `procedencia.json`.
4. **README con resultados:** pregunta, diseño, tabla de resultados y comandos.
5. **Pruebas:** toda función nueva de `src/conforme/` tiene pruebas en `tests/`, incluida una simulación que compruebe su garantía.
6. **Teoría:** una sección nueva en `teoria/guia_teorica.tex`, con el enunciado y la demostración (o un esbozo con referencia, si la prueba es larga).

---

## 0 · Cerrar 03 y 04 (antes de lo nuevo)

**Objetivo:** que todo lo publicado se pueda reejecutar desde el repo.

1. Clonar `tereom/quickcountmx` y `cotecora-team-2/quickcountmx` (este último en el commit `86c75437`) en `datos/` de cada aplicación.
2. Correr `conteo2012.py`, `intervalos2012.py` y `beta2012.py`, y compararlos con `resultados/` (diferencia máxima por columna).
3. Correr `ine.py --solo-auditoria`, después `ine.py` e `ine_pvalores.py`. Esto confirma el arreglo de nombres de columna.
4. Si hay diferencias, documentarlas en el README, como en 02.

**Hecho cuando:** los cuatro README digan "reproducido desde el repo", con las versiones de los paquetes.

---

## 07 · Fraude: p-valores conformes + Benjamini-Hochberg

**Objetivo:** marcar transacciones sospechosas controlando la proporción de falsas alarmas (FDR ≤ q).

**Datos:** *Credit Card Fraud Detection* (ULB; Kaggle `mlg-ulb/creditcardfraud`, también en OpenML), con 284,807 transacciones y 492 fraudes (0.17%). Variables: `Time`, `Amount` y `V1`–`V28` (componentes principales).

**Pasos:**
1. Script `descargar.md` con instrucciones (Kaggle pide cuenta) y verificación del hash.
2. `fraude.py`, en cada réplica:
   - separar entrenamiento, calibración **solo con legítimas** y prueba (con fraudes);
   - puntaje A, sin etiquetas: `IsolationForest` entrenado con legítimas;
   - puntaje B, supervisado: `HistGradientBoostingClassifier` con p̂(fraude) como puntaje;
   - p-valores con `cf.pvalores` y rechazos con `cf.benjamini_hochberg` para q ∈ {0.05, 0.1, 0.2}.
3. Métricas: FDR observado, poder (fraudes detectados / fraudes totales) y número de marcadas. Comparar con un umbral fijo del clasificador (p̂ > 0.5), que no controla la FDR.
4. Figuras: FDR observado contra q (sobre la diagonal debe quedar ≤ q), y poder contra q para ambos puntajes.
5. Extensión opcional: selección conforme (Jin y Candès, 2023) sobre los datos de 02_credito, para elegir clientes "de bajo riesgo" con FDR controlada.

**Teoría a agregar:** p-valores conformes como prueba de valores atípicos; por qué son PRDS; por qué BH controla la FDR (Bates et al., 2023).

**Decisiones abiertas:** si la calibración usa una muestra temporal (por `Time`) o al azar. Al azar mantiene la garantía; temporal es más realista.

---

## 08 · Bayes contra conformal

**Objetivo:** mostrar cuándo la predictiva posterior cubre lo que promete y cuándo no, y cómo conformalizarla.

**Datos:** simulaciones. Después, aplicar lo mismo a 03 (2012) como caso real.

**Pasos:**
1. `simulacion.py` con regresión lineal bayesiana conjugada (normal–gamma inversa), que da una predictiva posterior t de forma cerrada.
2. Escenarios:
   - (a) modelo correcto;
   - (b) errores con colas pesadas (t₃);
   - (c) heterocedasticidad;
   - (d) previa mal centrada y fuerte;
   - (e) previas poco informativas, incluidas las de predictiva previa uniforme de la tesis.
3. Para n ∈ {10, 20, 50, 200, 1000} y 1,000 réplicas, comparar la cobertura frecuentista y el ancho de:
   - (i) el intervalo predictivo posterior al 90%;
   - (ii) split conformal con residuo absoluto;
   - (iii) *conformal Bayes* (Fong y Holmes, 2021): full conformal con la densidad predictiva posterior como puntaje, calculada por muestreo de importancia con muestras de la posterior.
4. Agregar `conformal_bayes` a `src/conforme/`, con una prueba de cobertura.
5. Figuras: cobertura contra n por escenario y ancho contra n.

**Teoría a agregar:** full conformal; por qué conformal Bayes conserva la garantía; con modelo correcto, la predictiva posterior cubre en promedio sobre la previa, no para cada θ.

**Decisiones abiertas:** si el escenario (e) se profundiza aquí o se deja para la tesis.

---

## 05 · Ingreso de los hogares (ENIGH): CQR

**Objetivo:** intervalos de ingreso que se ensanchen donde hay más incertidumbre, comparando CQR contra el residuo absoluto.

**Datos:** ENIGH más reciente (INEGI), tabla `concentradohogar`. Respuesta: `log(ing_cor)`. Covariables: tamaño del hogar, edad y escolaridad del jefe, tamaño de localidad (`tam_loc`), entidad, características de la vivienda.

**Pasos:**
1. `preparar.py` para descargar, unir y limpiar, y guardar un parquet en `datos/`. Excluir hogares con ingreso cero o documentar cómo se tratan.
2. **Diseño:** tratar la muestra como población y sortear dentro de ella, como en 2012. Eso da intercambiabilidad por construcción. Discutir en el README qué cambia con los factores de expansión.
3. `ingreso.py`, con 20 sorteos de entrenamiento, calibración y prueba, compara:
   - (i) residuo absoluto con boosting;
   - (ii) residuo escalado |y − f(x)| / σ̂(x);
   - (iii) CQR con boosting cuantílico (5% y 95%);
   - (iv) cuantiles sin calibrar;
   - (v) CQR con un umbral por tamaño de localidad.
4. Métricas: cobertura y ancho global, por decil de ingreso predicho, por tamaño de localidad y por entidad.
5. Figuras: cobertura y ancho por decil; ejemplo de 40 hogares como `intervalos2012.png`.

**Teoría a agregar:** CQR y su garantía, el residuo escalado y cobertura condicional aproximada contra exacta (imposibilidad de Vovk 2012 / Lei-Wasserman).

**Decisiones abiertas:** año de la ENIGH; si el intervalo se reporta en pesos (transformando de vuelta los extremos) o en logaritmo.

---

## 06 · Calidad del aire: series de tiempo con ACI

**Objetivo:** mostrar que split conformal pierde cobertura con estacionalidad y contingencias, y que ACI la recupera en el largo plazo.

**Datos:** SIMAT, CDMX (archivos anuales de datos horarios por contaminante y estación). Ozono horario de varios años en 3–5 estaciones. Alternativa: demanda eléctrica del CENACE.

**Pasos:**
1. `preparar.py` para descargar varios años, limpiar faltantes y formar la serie horaria por estación.
2. Pronóstico de la hora siguiente (o del máximo del día siguiente) con boosting sobre rezagos, hora, día de la semana y mes. Se entrena una vez con el primer año.
3. `aire.py` compara:
   - (i) split conformal calibrado una vez;
   - (ii) split con ventana móvil;
   - (iii) conformal ponderado con pesos que decaen en el tiempo (`cf.cuantil_ponderado`, Barber et al., 2023);
   - (iv) ACI con varios γ (`cf.aci`).
4. Métricas: cobertura global y en ventanas mensuales, ancho medio, y cobertura en días de contingencia ambiental.
5. Figuras: cobertura móvil por método, la trayectoria de α_t de ACI y un mes de intervalos sobre la serie.

**Teoría a agregar:** por qué falla la intercambiabilidad en series; la cota determinista de ACI sobre la frecuencia de errores; la cota de cobertura de Barber et al. en función de la distancia a la intercambiabilidad.

**Decisiones abiertas:** horizonte (una hora contra el día siguiente) y estaciones.

---

## 09 · Martingalas de intercambiabilidad

**Objetivo:** detectar en línea, mientras llegan las casillas en 2018, el momento en que deja de ser razonable la intercambiabilidad. Cierra la historia de 04.

**Datos:** los de 04: remesas de 2018 en su orden real de llegada.

**Pasos:**
1. Agregar a `src/conforme/` las funciones `pvalores_en_linea` (p-valores conformes aleatorizados de cada observación contra las anteriores), `martingala_potencia(p, eps)` y `martingala_mezcla(p)` (integral en ε). Pruebas: con datos i.i.d. la martingala rara vez pasa 1/δ; con un cambio de distribución la pasa.
2. `martingalas.py`, sobre las casillas en orden de llegada:
   - puntaje con un modelo fijo (entrenado con una muestra al azar ajena a la secuencia) o con la distancia a vecinos;
   - martingala acumulada y rechazo cuando supera 1/δ (δ = 0.05, 0.01), con su hora.
3. Control: las mismas casillas en 200 órdenes al azar. Reportar la proporción de falsas alarmas, que debe ser ≤ δ.
4. Figuras: log de la martingala contra hora, con la línea 1/δ y las curvas del control; superponer la deriva de `ine_deriva.png`.

**Teoría a agregar:** p-valores en línea independientes y uniformes bajo intercambiabilidad; la desigualdad de Ville; martingalas de potencia y de mezcla.

**Decisiones abiertas:** el puntaje (un modelo fijo es más interpretable; la distancia a vecinos no necesita entrenamiento).

---

## Publicación

1. Licencia: MIT para el código y CC BY 4.0 para la guía (sugerido).
2. Primer commit con 01–04 cerrados (paso 0) y subir el repo, público o privado.
3. Agregar integración continua de GitHub (correr `pytest` en cada push).
4. Cada aplicación nueva entra como su propio commit o PR, cuando cumpla el criterio común.
