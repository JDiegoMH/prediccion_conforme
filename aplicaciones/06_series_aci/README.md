# 06 · Calidad del aire: series de tiempo sin intercambiabilidad · *pendiente*

**Pregunta:** en una serie de tiempo los datos no son intercambiables: hay estacionalidad, tendencia y cambios de régimen. ¿Qué garantía se puede conservar?

- **Método:** inferencia conforme adaptativa (ACI) con `conforme.aci`, que garantiza una frecuencia de errores de largo plazo igual a α sin suponer intercambiabilidad. Se compara con split conformal ingenuo y con conformal ponderado con pesos que decaen en el tiempo.
- **Datos:** SIMAT de la CDMX (ozono o PM2.5 horario por estación). Alternativa: demanda eléctrica del CENACE.
- **Qué medir:** cobertura en ventanas móviles (por ejemplo mensuales), no solo la global; el comportamiento en contingencias ambientales; la sensibilidad a `gamma` y a la ventana.

**Referencias:** Gibbs y Candès (2021), *Adaptive conformal inference under distribution shift*; Barber, Candès, Ramdas y Tibshirani (2023), *Conformal prediction beyond exchangeability*.

**Plan de acción:** [PLAN.md](../../PLAN.md#06--calidad-del-aire-series-de-tiempo-con-aci)
