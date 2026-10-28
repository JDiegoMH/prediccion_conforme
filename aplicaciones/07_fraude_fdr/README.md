# 07 · Fraude: p-valores conformes y falsos descubrimientos · *pendiente*

**Pregunta:** entre miles de transacciones, ¿cuáles marcar como sospechosas controlando la proporción de falsas alarmas?

- **Método:** se calibra solo con transacciones legítimas. `conforme.pvalores` da un p-valor por transacción nueva y `conforme.benjamini_hochberg` decide cuáles marcar. Como los p-valores comparten la calibración son dependientes, pero son PRDS, así que BH sigue controlando la tasa de falsos descubrimientos.
- **Datos:** Credit Card Fraud Detection (ULB, en Kaggle), con 284,807 transacciones y 0.17% de fraudes. También sirve como segunda aplicación a [02_credito](../02_credito/).
- **Qué medir:** FDR y poder (proporción de fraudes detectados) según q; comparar con un umbral fijo del clasificador.
- **Extensión:** selección conforme (Jin y Candès, 2023): elegir clientes "de bajo riesgo" con control de falsos descubrimientos, usando el crédito.

**Referencias:** Bates, Candès, Lei, Romano y Sesia (2023), *Testing for outliers with conformal p-values*; Jin y Candès (2023), *Selection by prediction with conformal p-values*.

**Plan de acción:** [PLAN.md](../../PLAN.md#07--fraude-p-valores-conformes--benjamini-hochberg)
