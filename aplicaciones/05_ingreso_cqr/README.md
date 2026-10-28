# 05 · Ingreso de los hogares: intervalos adaptativos (CQR) · *pendiente*

**Pregunta:** el intervalo `f(x) ± q` tiene el mismo ancho para todos los hogares, pero la incertidumbre sobre el ingreso de un hogar de altos ingresos es mucho mayor. ¿Cómo obtener intervalos que se ensanchen donde hace falta, sin perder la garantía?

- **Método:** regresión cuantílica conformalizada, con `conforme.cqr`. Se compara con el residuo absoluto (`conforme.intervalo_absoluto`) y con el residuo escalado `|y − f(x)| / σ̂(x)`.
- **Datos:** ENIGH (INEGI), con el ingreso corriente trimestral del hogar (posiblemente en logaritmo) y covariables sociodemográficas y de vivienda.
- **Qué medir:** cobertura y ancho por decil de ingreso estimado, por entidad y por tamaño de localidad, para verlo como cobertura condicional aproximada.
- **Ojo:** los factores de expansión de la encuesta rompen el muestreo aleatorio simple. Conviene discutir si se calibra por estrato o con pesos (conecta con `cuantil_ponderado`).

**Referencias:** Romano, Patterson y Candès (2019), *Conformalized quantile regression*; Lei et al. (2018).

**Plan de acción:** [PLAN.md](../../PLAN.md#05--ingreso-de-los-hogares-enigh-cqr)
