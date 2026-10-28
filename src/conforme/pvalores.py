"""P-valores conformes y control de la tasa de falsos descubrimientos."""
import numpy as np


def pvalores(scores_cal, scores_test, rng=None):
    """p(s) = (1 + #{i : S_i >= s}) / (n + 1); valores altos del puntaje son raros.

    Con rng se usa la versión aleatorizada (#{S_i > s} + U (1 + #{S_i = s})) / (n + 1),
    exactamente uniforme bajo intercambiabilidad; sin rng el p-valor es superuniforme.
    """
    cal = np.sort(np.asarray(scores_cal, dtype=float)); s = np.asarray(scores_test, dtype=float)
    if cal.size == 0:
        raise ValueError('Calibración vacía')
    n = cal.size
    mayores = n - np.searchsorted(cal, s, side='right')
    if rng is None:
        iguales = np.searchsorted(cal, s, side='right') - np.searchsorted(cal, s, side='left')
        return (1 + mayores + iguales) / (n + 1)
    iguales = np.searchsorted(cal, s, side='right') - np.searchsorted(cal, s, side='left')
    return (mayores + rng.uniform(size=s.shape) * (1 + iguales)) / (n + 1)


def benjamini_hochberg(p, q=0.1):
    """Rechazos de Benjamini-Hochberg al nivel q. Devuelve un vector booleano.

    Los p-valores conformes que comparten calibración son PRDS, así que BH controla la
    tasa de falsos descubrimientos (Bates, Candès, Lei, Romano y Sesia, 2023).
    """
    p = np.asarray(p, dtype=float); m = p.size
    if not 0 < q < 1:
        raise ValueError('q debe estar entre 0 y 1')
    orden = np.argsort(p, kind='stable')
    debajo = p[orden] <= q * np.arange(1, m + 1) / m
    rechazos = np.zeros(m, dtype=bool)
    if debajo.any():
        rechazos[orden[: np.flatnonzero(debajo).max() + 1]] = True
    return rechazos
