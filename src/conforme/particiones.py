"""Particiones entrenamiento / calibración / prueba sin fugas."""
import numpy as np


def validar_particion(train, cal, test, n):
    groups = [set(map(int, x)) for x in (train, cal, test)]
    if any(len(g) != len(x) or not g for g, x in zip(groups, (train, cal, test))):
        raise ValueError('Bloque vacío o índices repetidos')
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Fuga: entrenamiento, calibración y prueba se solapan')
    if set.union(*groups) != set(range(n)):
        raise ValueError('La partición no cubre exactamente los datos')


def particion(n, n_train, n_cal, rng):
    """Permutación aleatoria en tres bloques disjuntos; el resto queda como prueba."""
    if n_train < 1 or n_cal < 1 or n_train + n_cal >= n:
        raise ValueError('Tamaños inválidos')
    idx = rng.permutation(n)
    tr, ca, te = idx[:n_train], idx[n_train:n_train + n_cal], idx[n_train + n_cal:]
    validar_particion(tr, ca, te, n)
    return tr, ca, te
