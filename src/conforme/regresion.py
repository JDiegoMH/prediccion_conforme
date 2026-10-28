"""Intervalos split conformes para regresión."""
import numpy as np

from .cuantiles import cuantil


def intervalo_absoluto(pred_cal, y_cal, pred_test, alpha=0.1):
    """Puntaje |y - f(x)|: intervalo f(x) +/- q, de ancho constante.

    Devuelve (inferior, superior, q).
    """
    pred_cal, y_cal, pred_test = (np.asarray(a, dtype=float) for a in (pred_cal, y_cal, pred_test))
    q = cuantil(np.abs(y_cal - pred_cal), alpha)
    return pred_test - q, pred_test + q, q


def cqr(inf_cal, sup_cal, y_cal, inf_test, sup_test, alpha=0.1):
    """Regresión cuantílica conformalizada (Romano, Patterson y Candès, 2019).

    Puntaje max(inf(x) - y, y - sup(x)); el intervalo [inf(x) - q, sup(x) + q] conserva
    el ancho variable de los cuantiles estimados. Devuelve (inferior, superior, q).
    """
    inf_cal, sup_cal, y_cal, inf_test, sup_test = (np.asarray(a, dtype=float) for a in (inf_cal, sup_cal, y_cal, inf_test, sup_test))
    q = cuantil(np.maximum(inf_cal - y_cal, y_cal - sup_cal), alpha)
    return inf_test - q, sup_test + q, q


def cubre(y, inferior, superior):
    """Indicador de que y cae en [inferior, superior]."""
    y = np.asarray(y, dtype=float)
    return (y >= inferior) & (y <= superior)
