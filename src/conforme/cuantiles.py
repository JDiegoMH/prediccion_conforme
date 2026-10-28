"""Cuantil conforme: corrección n+1, sin interpolación; versión ponderada."""
from decimal import Decimal, ROUND_CEILING
import math

import numpy as np


def _validar_alpha(alpha):
    if not 0 < alpha < 1:
        raise ValueError('alpha debe estar entre 0 y 1')


def rango_conforme(n, alpha):
    """k = ceil((n+1)(1-alpha)), con aritmética decimal para evitar deslices de coma flotante."""
    _validar_alpha(alpha)
    return int((Decimal(n + 1) * (1 - Decimal(str(alpha)))).to_integral_value(rounding=ROUND_CEILING))


def cuantil(scores, alpha=0.1):
    """k-ésimo puntaje ordenado con k = ceil((n+1)(1-alpha)); infinito si k > n.

    El conjunto {y : s(x, y) <= cuantil} cubre con probabilidad >= 1 - alpha si los
    puntajes de calibración y el del caso nuevo son intercambiables.
    """
    values = [float(x) for x in scores]
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError('Se requieren puntajes finitos y calibración no vacía')
    k = rango_conforme(len(values), alpha)
    return math.inf if k > len(values) else sorted(values)[k - 1]


def cuantil_ponderado(scores, pesos, peso_nuevo, alpha=0.1):
    """Cuantil 1 - alpha de sum_i p_i delta_{S_i} + p_{n+1} delta_{+inf} (Tibshirani et al., 2019).

    p_i = w_i / (sum_j w_j + w_{n+1}). Con pesos iguales coincide con `cuantil`.
    Para cambio de covariables, w(x) = dP_prueba/dP_calibración (x).
    """
    _validar_alpha(alpha)
    s = np.asarray(scores, dtype=float); w = np.asarray(pesos, dtype=float)
    if s.ndim != 1 or s.shape != w.shape or s.size == 0:
        raise ValueError('scores y pesos deben ser vectores del mismo tamaño, no vacíos')
    if not (np.isfinite(s).all() and np.isfinite(w).all() and (w >= 0).all()) or not (math.isfinite(peso_nuevo) and peso_nuevo >= 0):
        raise ValueError('Puntajes finitos y pesos finitos no negativos')
    total = w.sum() + peso_nuevo
    if total <= 0:
        raise ValueError('La suma de pesos debe ser positiva')
    orden = np.argsort(s, kind='stable')
    acumulado = np.cumsum(w[orden]) / total
    # Tolerancia relativa para que pesos iguales reproduzcan exactamente el cuantil sin pesos.
    i = np.searchsorted(acumulado, (1 - alpha) * (1 - 1e-12), side='left')
    return math.inf if i >= s.size else float(s[orden][i])
