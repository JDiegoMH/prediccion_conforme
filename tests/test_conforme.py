import math

import numpy as np
import pytest

import conforme as cf


# Cuantil: casos límite heredados de la auditoría de la presentación.
def test_cuantil_casos_limite():
    assert cf.cuantil([.1] * 9, .1) == .1          # empates: inclusión <=
    assert math.isinf(cf.cuantil([.1] * 8, .1))     # k = n + 1
    assert cf.cuantil(range(19), .1) == 17          # k = 18, sin desliz de coma flotante
    assert cf.cuantil(range(99), .1) == 89
    for scores, alpha in [([], .1), ([math.nan], .1), ([1], 0), ([1], 1)]:
        with pytest.raises(ValueError):
            cf.cuantil(scores, alpha)


@pytest.mark.parametrize('n', [8, 9, 19, 99, 250])
@pytest.mark.parametrize('alpha', [.05, .1, .2])
def test_ponderado_con_pesos_iguales_es_el_cuantil(n, alpha):
    s = np.random.default_rng(n).normal(size=n)
    assert cf.cuantil_ponderado(s, np.ones(n), 1.0, alpha) == cf.cuantil(s, alpha)


def test_n_minimo():
    assert cf.n_minimo(.1) == 9 and cf.n_minimo(.05) == 19
    assert math.isfinite(cf.cuantil(range(9), .1)) and math.isinf(cf.cuantil(range(8), .1))


def _cobertura_regresion(metodo, reps=400, n=200, alpha=.1, seed=0):
    rng = np.random.default_rng(seed); cov = []
    for _ in range(reps):
        x = rng.uniform(0, 4, size=n + 1); y = x + rng.normal(scale=.2 + x, size=n + 1)  # heterocedástico
        f = x  # predictor fijo (ya entrenado)
        if metodo == 'absoluto':
            lo, hi, _ = cf.intervalo_absoluto(f[:n], y[:n], f[n:], alpha)
        else:
            sd = .2 + x
            lo, hi, _ = cf.cqr(f[:n] - sd[:n], f[:n] + sd[:n], y[:n], f[n:] - sd[n:], f[n:] + sd[n:], alpha)
        cov.append(cf.cubre(y[n:], lo, hi)[0])
    return np.mean(cov)


@pytest.mark.parametrize('metodo', ['absoluto', 'cqr'])
def test_cobertura_regresion(metodo):
    # Media de 400 indicadores con p >= 0.9: error estándar ~ 0.015.
    assert _cobertura_regresion(metodo) >= .9 - 3 * .015


def _probs(rng, n, k=5):
    logits = rng.normal(size=(n, k)) * 2
    p = np.exp(logits); p /= p.sum(1, keepdims=True)
    y = np.array([rng.choice(k, p=pi) for pi in p])
    return p, y


@pytest.mark.parametrize('metodo', ['lac', 'aps', 'aps_aleatorio', 'por_clase'])
def test_cobertura_clasificacion(metodo):
    rng = np.random.default_rng(1); p, y = _probs(rng, 6000)
    pc, yc, pt, yt = p[:1000], y[:1000], p[1000:], y[1000:]
    if metodo == 'lac':
        C, _ = cf.conjuntos_lac(pc, yc, pt)
    elif metodo == 'aps':
        C, _ = cf.conjuntos_aps(pc, yc, pt)
    elif metodo == 'aps_aleatorio':
        C, _ = cf.conjuntos_aps(pc, yc, pt, rng=rng)
    else:
        C, _ = cf.conjuntos_por_clase(pc, yc, pt)
    cov = cf.cobertura_conjuntos(C, yt)
    assert abs(cov.mean() - .9) < .03
    if metodo == 'por_clase':
        for k in range(5):
            assert cov[yt == k].mean() > .9 - .05


def test_puntaje_aps_ejemplo():
    s = cf.puntajes_aps(np.array([[.5, .3, .2]]))
    assert np.allclose(s, [[.5, .8, 1.0]])


def test_pvalores_superuniformes_y_bh():
    rng = np.random.default_rng(2)
    cal = rng.normal(size=500); nulos = rng.normal(size=4000)
    for r in (None, rng):
        p = cf.pvalores(cal, nulos, rng=r)
        assert ((p > 0) & (p <= 1)).all()
        assert (p <= .1).mean() < .1 + .03
    # Alternativas desplazadas: BH las encuentra controlando falsos descubrimientos.
    alt = rng.normal(loc=4, size=200)
    p = cf.pvalores(cal, np.r_[nulos[:1800], alt])
    rech = cf.benjamini_hochberg(p, .1)
    falsos = rech[:1800].sum() / max(rech.sum(), 1)
    assert rech[1800:].mean() > .8 and falsos < .2


def test_pvalor_formula():
    assert np.allclose(cf.pvalores([1, 2, 3, 4], [2.5, 0, 9]), [3 / 5, 5 / 5, 1 / 5])


def test_bh_ejemplo():
    assert cf.benjamini_hochberg([.001, .02, .03, .5], .1).tolist() == [True, True, True, False]


def test_aci_con_cambio_de_escala():
    rng = np.random.default_rng(3)
    s = np.abs(np.r_[rng.normal(size=2000), rng.normal(scale=3, size=2000)])  # sin intercambiabilidad
    r = cf.aci(s, alpha=.1, gamma=.01, inicio=200)
    assert abs(r['error'].mean() - .1) < .02


def test_beta_y_garantia():
    b = cf.cobertura_beta(19, .1)
    assert b.args == (18, 2) and math.isclose(b.mean(), 18 / 20)
    n = cf.n_para_garantia(.1, .02, .05)
    assert cf.cobertura_beta(n, .1).cdf(.88) <= .05 < cf.cobertura_beta(n - 1, .1).cdf(.88)


def test_particion():
    tr, ca, te = cf.particion(100, 50, 30, np.random.default_rng(0))
    assert (len(tr), len(ca), len(te)) == (50, 30, 20)
    with pytest.raises(ValueError):
        cf.validar_particion([0, 1], [1, 2], [3], 4)
