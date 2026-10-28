"""Conjuntos conformes para clasificación: LAC, APS y por clase (Mondrian)."""
import numpy as np

from .cuantiles import cuantil


def _probabilidades(p, y=None):
    p = np.asarray(p, dtype=float)
    if p.ndim != 2:
        raise ValueError('Las probabilidades deben ser una matriz n x K')
    if y is None:
        return p
    y = np.asarray(y, dtype=int)
    if y.shape != (p.shape[0],) or y.min() < 0 or y.max() >= p.shape[1]:
        raise ValueError('Etiquetas fuera de 0..K-1 o de tamaño distinto')
    return p, y


def conjuntos_lac(p_cal, y_cal, p_test, alpha=0.1):
    """Puntaje 1 - p_y(x): y entra si p_y(x) >= 1 - q. Devuelve (matriz booleana, q)."""
    p_cal, y_cal = _probabilidades(p_cal, y_cal); p_test = _probabilidades(p_test)
    q = cuantil(1 - p_cal[np.arange(len(y_cal)), y_cal], alpha)
    return 1 - p_test <= q, q


def puntajes_aps(p, u=None):
    """Puntaje APS de todas las etiquetas: masa de las etiquetas más probables que y, más u * p_y.

    Con u ~ U(0,1) es la versión aleatorizada de Romano, Sesia y Candès (2020); con u = 1,
    la determinista. Devuelve una matriz n x K.
    """
    p = _probabilidades(p)
    u = np.ones(p.shape[0]) if u is None else np.asarray(u, dtype=float)
    orden = np.argsort(-p, axis=1, kind='stable')
    ordenadas = np.take_along_axis(p, orden, axis=1)
    antes = np.cumsum(ordenadas, axis=1) - ordenadas
    s = np.empty_like(p)
    np.put_along_axis(s, orden, antes + u[:, None] * ordenadas, axis=1)
    return s


def conjuntos_aps(p_cal, y_cal, p_test, alpha=0.1, rng=None):
    """Conjuntos adaptativos (APS). Si rng es None se usa la versión determinista (u = 1).

    Devuelve (matriz booleana, q).
    """
    p_cal, y_cal = _probabilidades(p_cal, y_cal); p_test = _probabilidades(p_test)
    u_cal = None if rng is None else rng.uniform(size=len(y_cal))
    u_test = None if rng is None else rng.uniform(size=p_test.shape[0])
    q = cuantil(puntajes_aps(p_cal, u_cal)[np.arange(len(y_cal)), y_cal], alpha)
    return puntajes_aps(p_test, u_test) <= q, q


def umbrales_por_grupo(scores, grupos, alpha=0.1):
    """Un cuantil conforme por grupo (Mondrian): cobertura >= 1 - alpha dentro de cada grupo."""
    scores = np.asarray(scores, dtype=float); grupos = np.asarray(grupos)
    if scores.shape != grupos.shape:
        raise ValueError('scores y grupos deben tener el mismo tamaño')
    return {g: cuantil(scores[grupos == g], alpha) for g in np.unique(grupos)}


def conjuntos_por_clase(p_cal, y_cal, p_test, alpha=0.1):
    """LAC con un umbral por clase: y entra si 1 - p_y(x) <= q_y. Devuelve (matriz booleana, q por clase)."""
    p_cal, y_cal = _probabilidades(p_cal, y_cal); p_test = _probabilidades(p_test)
    q = umbrales_por_grupo(1 - p_cal[np.arange(len(y_cal)), y_cal], y_cal, alpha)
    qv = np.array([q.get(k, np.inf) for k in range(p_test.shape[1])])
    return 1 - p_test <= qv[None, :], q


def cobertura_conjuntos(conjuntos, y):
    """Indicador de que la etiqueta verdadera está en el conjunto."""
    conjuntos = np.asarray(conjuntos, dtype=bool); y = np.asarray(y, dtype=int)
    return conjuntos[np.arange(len(y)), y]
