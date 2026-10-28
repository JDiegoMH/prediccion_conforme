# 09 · Detectar cuándo se rompe la intercambiabilidad · *pendiente*

**Pregunta:** en [04_ine_2018](../04_ine_2018/) la cobertura falla porque el orden de llegada no es aleatorio. ¿Se puede detectar en línea, conforme llegan las casillas, en qué momento deja de ser razonable el supuesto?

- **Método:** martingalas de intercambiabilidad. Con los p-valores conformes aleatorizados de cada observación nueva (`conforme.pvalores` con `rng`) se construye una martingala de apuestas (power o mixture martingale). Si su valor supera 1/δ, se rechaza la intercambiabilidad con nivel δ (desigualdad de Ville), en cualquier momento y sin corrección por mirar varias veces.
- **Datos:** la llegada de remesas de 2018 en su orden real, contra un control con orden aleatorio. Después, cualquier serie de 06.
- **Qué medir:** la hora a la que se rechaza la intercambiabilidad, la falsa alarma en el control y la relación con la deriva de `ine_deriva.png`.

**Referencias:** Vovk, Nouretdinov y Gammerman (2003), *Testing exchangeability on-line*; Vovk (2021), *Testing randomness online*.

**Plan de acción:** [PLAN.md](../../PLAN.md#09--martingalas-de-intercambiabilidad)
