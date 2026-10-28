# Predicción conforme: teoría y aplicaciones

La predicción conforme convierte cualquier modelo en un conjunto predictivo con cobertura garantizada en muestra finita, con un solo supuesto: la intercambiabilidad. Este repositorio reúne la teoría (con demostraciones), una pequeña biblioteca y aplicaciones con datos reales. Cada aplicación muestra una idea distinta.

## Contenido

- **[`teoria/`](teoria/guia_teorica.pdf)**: guía con todas las demostraciones: cobertura marginal, cuantil con corrección n+1, intercambiabilidad, la distribución Beta de la cobertura, cobertura por clase y p-valores conformes.
- **`src/conforme/`**: funciones comunes con pruebas en `tests/`.
- **`aplicaciones/`**:

| # | Aplicación | Idea | Estado |
|---|---|---|---|
| 01 | [Dígitos](aplicaciones/01_digitos/) | Primer algoritmo: conjuntos de clasificación (LAC) | ✅ |
| 02 | [Crédito](aplicaciones/02_credito/) | Respuesta binaria; marginal contra por clase (Mondrian) | ✅ |
| 03 | [Conteo rápido 2012](aplicaciones/03_conteo_2012/) | El sorteo da intercambiabilidad; umbral por grupo; CQR; verificación de la Beta | ✅ |
| 04 | [INE 2018](aplicaciones/04_ine_2018/) | Cuando el orden de llegada rompe la intercambiabilidad; p-valores | ✅ |
| 05 | [Ingreso (ENIGH)](aplicaciones/05_ingreso_cqr/) | Intervalos adaptativos con CQR | pendiente |
| 06 | [Calidad del aire](aplicaciones/06_series_aci/) | Series de tiempo: conformal adaptativo (ACI) | pendiente |
| 07 | [Fraude](aplicaciones/07_fraude_fdr/) | P-valores conformes + Benjamini-Hochberg | pendiente |
| 08 | [Bayes contra conformal](aplicaciones/08_bayes_vs_conforme/) | Predictiva posterior bajo mala especificación | pendiente |
| 09 | [Martingalas](aplicaciones/09_martingalas/) | Detectar en línea que se rompe la intercambiabilidad | pendiente |

El plan de acción para terminar 03–04 y desarrollar 05–09 está en [PLAN.md](PLAN.md).

## La biblioteca `conforme`

| Función | Qué hace |
|---|---|
| `cuantil`, `cuantil_ponderado` | Cuantil conforme con corrección n+1; versión ponderada (cambio de covariables) |
| `intervalo_absoluto`, `cqr` | Intervalos de regresión: residuo absoluto y regresión cuantílica conformalizada |
| `conjuntos_lac`, `conjuntos_aps`, `conjuntos_por_clase` | Conjuntos de clasificación: LAC, adaptativos (APS) y por clase |
| `umbrales_por_grupo` | Un umbral por grupo (Mondrian) |
| `pvalores`, `benjamini_hochberg` | P-valores conformes (deterministas o aleatorizados) y control de FDR |
| `aci` | Conformal adaptativo en línea (Gibbs y Candès, 2021) |
| `cobertura_beta`, `n_minimo`, `n_para_garantia` | Distribución de la cobertura de una calibración y tamaños de calibración |

```python
import conforme as cf
lo, hi, q = cf.intervalo_absoluto(pred_cal, y_cal, pred_test, alpha=0.1)
C, q = cf.conjuntos_lac(p_cal, y_cal, p_test, alpha=0.1)   # matriz booleana n x K
```

## Instalación

Con Python 3.11 o posterior:

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[aplicaciones,pruebas]'
.venv/bin/pytest
```

Los datos crudos no se versionan. Cada aplicación explica en su README cómo obtenerlos y guarda sus resultados en `resultados/` y sus figuras en `figuras/`.

Para recompilar la guía (necesita una distribución TeX con latexmk):

```sh
cd teoria && latexmk -pdf -outdir=.build guia_teorica.tex && cp .build/guia_teorica.pdf .
```

## Referencias base

- Angelopoulos, A. N. y Bates, S. (2023). Conformal prediction: A gentle introduction. *Foundations and Trends in Machine Learning*, 16(4).
- Angelopoulos, A. N., Barber, R. F. y Bates, S. *Theoretical Foundations of Conformal Prediction* (arXiv:2411.11824).
- Vovk, V., Gammerman, A. y Shafer, G. (2005; 2a ed. 2022). *Algorithmic Learning in a Random World*. Springer.

Autor: José Diego Martínez Hernández.
