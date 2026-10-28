"""Diagnósticos: distribución de la cobertura condicional a la calibración y tamaños mínimos."""
import math

from scipy.stats import beta

from .cuantiles import rango_conforme


def cobertura_beta(n, alpha=0.1):
    """Distribución de la cobertura de una calibración fija: Beta(k, n + 1 - k).

    Supone entrenamiento fijo, puntajes i.i.d. continuos y k = ceil((n+1)(1-alpha)) <= n.
    """
    k = rango_conforme(n, alpha)
    if k > n:
        raise ValueError(f'Con n = {n} y alpha = {alpha} el umbral es infinito')
    return beta(k, n + 1 - k)


def n_minimo(alpha=0.1):
    """Menor n con umbral finito: (n + 1)(1 - alpha) <= n, es decir n >= 1/alpha - 1."""
    return math.ceil(1 / alpha - 1 - 1e-12)


def n_para_garantia(alpha=0.1, epsilon=0.02, delta=0.05, n_max=100_000):
    """Menor n tal que P(cobertura < 1 - alpha - epsilon) <= delta bajo la Beta."""
    for n in range(n_minimo(alpha), n_max + 1):
        if cobertura_beta(n, alpha).cdf(1 - alpha - epsilon) <= delta:
            return n
    raise ValueError('No se alcanzó la garantía con n_max')
