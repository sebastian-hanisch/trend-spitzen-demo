"""Orakel: GEV mit und ohne Lage-Trend (Likelihood, Optimum, Pegel, Delta-Intervall, Gumbel mit Trend in Lage und Skala) gegen scipy (nur Test).

Vorzeichen-Konvention der Linie: ξ > 0 Fréchet-Typ; `scipy.stats.genextreme` hat **c = −ξ** (hier in jedem Aufruf benannt)."""

import math

import numpy as np
import pytest

import tsp_models as M

stats = pytest.importorskip("scipy.stats")
optimize = pytest.importorskip("scipy.optimize")


def _reihe(rng, m, beta):
    return beta * np.arange(m) - np.log(-np.log(rng.random(m)))


@pytest.mark.parametrize("xi", [-0.4, -0.1, 0.0, 0.2, 0.5])
def test_nll_gleich_minus_summe_genextreme_logpdf_mit_c_gleich_minus_xi(xi):
    rng = np.random.default_rng(3)
    m = 40
    c = M.centered_time(m)
    a, b, s = 0.6, 0.02, 1.2
    x = stats.genextreme.ppf(rng.random(m), c=-xi, loc=a + b * c, scale=s)       # Werte aus der GEV selbst: immer im Träger
    lp = stats.genextreme.logpdf(x, c=-xi, loc=a + b * c, scale=s)
    assert np.all(np.isfinite(lp))
    assert M.gev_nll((a, b, math.log(s), xi), x, c) == pytest.approx(-float(np.sum(lp)), rel=1e-10)
    x0 = stats.genextreme.ppf(rng.random(m), c=-xi, loc=a, scale=s)
    lp0 = stats.genextreme.logpdf(x0, c=-xi, loc=a, scale=s)
    assert M.gev_nll((a, math.log(s), xi), x0) == pytest.approx(-float(np.sum(lp0)), rel=1e-10)


def test_gev_nll_ausserhalb_des_traegers_und_unter_xi_minus_1_ist_riesig():
    x = np.array([0.0, 1.0, 5.0])
    assert M.gev_nll((0.0, 0.0, -1.5), x) == M.HUGE
    assert M.gev_nll((0.0, 0.0, -0.3), x) == M.HUGE          # Endpunkt 3,33 < 5: Wert außerhalb des Trägers


@pytest.mark.parametrize("xi", [-0.3, 0.0, 0.25])
def test_pegel_gleich_genextreme_isf_und_cdf(xi):
    for p in (0.5, 0.1, 0.01, 0.001):
        lev = float(M.gev_level(p, 2.0, 1.5, xi))
        assert lev == pytest.approx(stats.genextreme.isf(p, c=-xi, loc=2.0, scale=1.5), rel=1e-9)
        assert stats.genextreme.cdf(lev, c=-xi, loc=2.0, scale=1.5) == pytest.approx(1 - p, abs=1e-10)


@pytest.mark.parametrize("m,beta", [(20, 0.0), (50, 0.02), (100, 0.04)])
def test_gev_optimum_ist_nicht_schlechter_als_nelder_mead_von_scipy(m, beta):
    rng = np.random.default_rng(40 + m)
    x = _reihe(rng, m, beta)
    c = M.centered_time(m)
    for trend in (False, True):
        f = M.fit_gev(x, trend)
        q = [f.a, f.b, math.log(f.sigma), f.xi] if trend else [f.a, math.log(f.sigma), f.xi]
        fun = (lambda z: M.gev_nll(z, x, c)) if trend else (lambda z: M.gev_nll(z, x))
        assert fun(q) == pytest.approx(f.nll, rel=1e-12)
        best = np.inf
        for k in range(4):
            q0 = np.array(q) + np.random.default_rng(k).normal(0, 0.15, len(q))
            if fun(q0) >= M.HUGE:
                continue
            r = optimize.minimize(fun, q0, method="Nelder-Mead", options=dict(xatol=1e-9, fatol=1e-12, maxiter=6000, maxfev=12000))
            best = min(best, r.fun)
        assert f.nll <= best + 1e-6, (trend, f.nll, best)


def test_gev_hat_hoechstens_die_negative_log_likelihood_der_gumbel_anpassung():
    rng = np.random.default_rng(8)
    for m in (20, 50):
        x = _reihe(rng, m, 0.02)
        assert M.fit_gev(x, True).nll <= M.fit_gumbel(x, True).nll + 1e-6
        assert M.fit_gev(x, False).nll <= M.fit_gumbel(x, False).nll + 1e-6


def test_pegel_der_anpassung_mit_trend_ist_der_pegel_im_letzten_jahr():
    rng = np.random.default_rng(9)
    x = _reihe(rng, 50, 0.03)
    f = M.fit_gev(x, True)
    loc_last = f.a + f.b * (50 - 1) / 2
    assert f.level() == pytest.approx(float(stats.genextreme.isf(0.01, c=-f.xi, loc=loc_last, scale=f.sigma)), rel=1e-9)


def test_delta_intervall_standardfehler_gleich_unabhaengiger_hesse_matrix_mit_anderer_schrittweite():
    rng = np.random.default_rng(10)
    x = _reihe(rng, 100, 0.0)
    f = M.fit_gev(x, False)
    ci = M.delta_interval(f, x)
    assert ci is not None
    th = np.array([f.a, math.log(f.sigma), f.xi])
    fn = lambda q: M.gev_nll(q, x)
    h = 2e-4
    H = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            ei, ej = np.eye(3)[i] * h, np.eye(3)[j] * h
            H[i, j] = (fn(th + ei + ej) - fn(th + ei - ej) - fn(th - ei + ej) + fn(th - ei - ej)) / (4 * h * h)
    g = optimize.approx_fprime(th, lambda q: float(M.gev_level(0.01, q[0], math.exp(q[1]), q[2])), 1e-6)
    se = math.sqrt(float(g @ np.linalg.inv(H) @ g))
    assert ci[2] == pytest.approx(se, rel=0.02)
    assert ci[0] == pytest.approx(f.level() - 1.959964 * ci[2], rel=1e-3) and ci[1] == pytest.approx(f.level() + 1.959964 * ci[2], rel=1e-3)
    assert ci[0] < f.level() < ci[1]


def test_gumbel_mit_trend_in_lage_und_skala_gleich_nelder_mead_von_scipy():
    rng = np.random.default_rng(11)
    m = 60
    c = M.centered_time(m)
    x = 0.5 + 0.03 * c - np.exp(0.2 + 0.01 * c) * np.log(-np.log(rng.random(m)))
    f = M.fit_gumbel_loc_logscale(x)
    ref = -float(np.sum(stats.gumbel_r.logpdf(x, loc=f.a + f.b * c, scale=np.exp(f.s0 + f.s1 * c))))
    assert f.nll == pytest.approx(ref, rel=1e-10)
    r = optimize.minimize(lambda q: M.gumbel_loc_logscale_nll(q, x, c), [f.a, f.b, f.s0, f.s1], method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-13, maxiter=6000, maxfev=12000))
    assert f.nll <= r.fun + 1e-6
    cl = (m - 1) / 2
    assert f.level() == pytest.approx(float(stats.gumbel_r.isf(0.01, loc=f.a + f.b * cl, scale=math.exp(f.s0 + f.s1 * cl))), rel=1e-9)
