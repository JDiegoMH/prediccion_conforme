"""Predicción conforme en línea sin intercambiabilidad: inferencia conforme adaptativa (ACI)."""
import math

import numpy as np


def _cuantil_nivel(pasados, nivel):
    """Cuantil conforme de los puntajes pasados para cobertura objetivo `nivel`."""
    if nivel >= 1:
        return math.inf
    if nivel <= 0:
        return -math.inf
    n = len(pasados)
    k = math.ceil((n + 1) * nivel)
    return math.inf if k > n else float(np.sort(pasados)[k - 1])


def aci(scores, alpha=0.1, gamma=0.005, inicio=100, ventana=None):
    """ACI de Gibbs y Candès (2021) sobre una sucesión de puntajes.

    En el tiempo t el umbral es el cuantil de nivel 1 - alpha_t de los puntajes anteriores
    (o de los últimos `ventana`); después se observa s_t y
    alpha_{t+1} = alpha_t + gamma (alpha - error_t). La frecuencia de errores converge a
    alpha sin suponer intercambiabilidad. Devuelve un dict con umbral, error y alpha_t
    para t >= inicio.
    """
    s = np.asarray(scores, dtype=float)
    if not 0 < alpha < 1 or gamma <= 0 or inicio < 1 or s.size <= inicio:
        raise ValueError('Parámetros inválidos o serie demasiado corta')
    a = alpha; umbrales, errores, alphas = [], [], []
    for t in range(inicio, s.size):
        pasados = s[:t] if ventana is None else s[max(0, t - ventana):t]
        q = _cuantil_nivel(pasados, 1 - a)
        err = float(s[t] > q)
        umbrales.append(q); errores.append(err); alphas.append(a)
        a = a + gamma * (alpha - err)
    return dict(umbral=np.array(umbrales), error=np.array(errores), alpha_t=np.array(alphas))
