# 08 · Bayes contra conformal · *pendiente*

**Pregunta:** el intervalo predictivo posterior bayesiano cubre bien si el modelo es correcto. ¿Qué pasa cuando no lo es? ¿Se puede conformalizar la predictiva posterior?

- **Método:**
  1. Simulación con modelo correcto: la predictiva posterior y la conformal cubren bien, y la bayesiana es más angosta.
  2. Modelo mal especificado (colas pesadas, heterocedasticidad, previa mal centrada): la posterior pierde cobertura y la conformal no.
  3. *Conformal Bayes:* usar la densidad predictiva posterior como puntaje.
  4. Efecto de la previa con n pequeño, incluidas las previas con predictiva previa uniforme de la tesis.
- **Datos:** simulaciones; después, aplicar lo mismo a 03 o 05.
- **Qué medir:** cobertura frecuentista y ancho, en función de n y del grado de mala especificación.

**Referencias:** Fong y Holmes (2021), *Conformal Bayesian computation*; Melluish, Saunders, Nouretdinov y Vovk (2001), *Comparing the Bayes and typicalness frameworks*.

**Plan de acción:** [PLAN.md](../../PLAN.md#08--bayes-contra-conformal)
