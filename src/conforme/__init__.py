"""Predicción conforme: funciones comunes a la teoría y las aplicaciones."""
from .cuantiles import cuantil, cuantil_ponderado, rango_conforme
from .regresion import intervalo_absoluto, cqr, cubre
from .clasificacion import (conjuntos_lac, conjuntos_aps, puntajes_aps, conjuntos_por_clase,
                            umbrales_por_grupo, cobertura_conjuntos)
from .pvalores import pvalores, benjamini_hochberg
from .series import aci
from .diagnosticos import cobertura_beta, n_minimo, n_para_garantia
from .particiones import particion, validar_particion

__version__ = '0.1.0'
