"""Orakel: Gumbel-Anpassung mit und ohne Lage-Trend (Newton, eigener Gradient und eigene Hesse-Matrix) gegen scipy (nur Test) und gegen finite Differenzen.

Anderer Rechenweg: `scipy.stats.gumbel_r.logpdf` für die Likelihood, `scipy.optimize.minimize` (Nelder-Mead, ohne Gradient) für das Optimum, zentrale Differenzen für Gradient und Hesse-Matrix,
Quadratur für die Fisher-Information, `scipy.stats.chi2.ppf` für die Schwelle."""

import math

import numpy as np
import pytest

import tsp_models as M
import tsp_numeric as N

stats = pytest.importorskip("scipy.stats")
optimize = pytest.importorskip("scipy.optimize")
integrate = pytest.importorskip("scipy.integrate")


def _reihe(rng, m, beta):
    return beta * np.arange(m) - np.log(-np.log(rng.random(m)))


def test_negative_log_likelihood_gleich_summe_der_logpdf_von_scipy():
    rng = np.random.default_rng(1)
    for m, beta in ((5, 0.0), (20, 0.03), (60, -0.02)):
        x = _reihe(rng, m, beta)
        c = M.centered_time(m)
        for th in ((0.3, 0.02, 1.1), (-0.5, -0.04, 0.7)):
            ref = -float(np.sum(stats.gumbel_r.logpdf(x, loc=th[0] + th[1] * c, scale=th[2])))
            assert M._gumbel_nll(np.array(th), x, c) == pytest.approx(ref, rel=1e-12)
            assert M.gumbel_nll_grad_hess(np.array(th), x, c)[0] == pytest.approx(ref, rel=1e-12)


def test_gradient_und_hesse_matrix_gleich_zentralen_differenzen():
    rng = np.random.default_rng(2)
    for m in (8, 30, 80):
        x = _reihe(rng, m, 0.02)
        c = M.centered_time(m)
        th = np.array([0.4, 0.03, 0.9])
        f, g, H = M.gumbel_nll_grad_hess(th, x, c)
        h = 1e-5
        for i in range(3):
            e = np.zeros(3)
            e[i] = h
            gn = (M._gumbel_nll(th + e, x, c) - M._gumbel_nll(th - e, x, c)) / (2 * h)
            assert g[i] == pytest.approx(gn, rel=1e-6, abs=1e-5)
            Hn = (M.gumbel_nll_grad_hess(th + e, x, c)[1] - M.gumbel_nll_grad_hess(th - e, x, c)[1]) / (2 * h)
            assert np.allclose(H[i], Hn, rtol=1e-5, atol=1e-4)
        assert np.allclose(H, H.T)


@pytest.mark.parametrize("m,beta", [(5, 0.0), (10, 0.01), (20, 0.04), (50, 0.02), (100, 0.0), (150, 0.04)])
def test_optimum_gleich_nelder_mead_von_scipy_und_gradient_null(m, beta):
    rng = np.random.default_rng(100 + m)
    x = _reihe(rng, m, beta)
    c = M.centered_time(m)
    for trend in (True, False):
        f = M.fit_gumbel(x, trend)
        th = np.array([f.a, f.b, f.sigma])
        _v, g, _H = M.gumbel_nll_grad_hess(th, x, c)
        idx = [0, 1, 2] if trend else [0, 2]
        assert float(np.max(np.abs(g[idx]))) < 1e-7
        if trend:
            fun = lambda q: M._gumbel_nll(q, x, c)
            start = [f.a + 0.2, f.b - 0.01, f.sigma * 1.2]
        else:
            fun = lambda q: M._gumbel_nll(np.array([q[0], 0.0, q[1]]), x, c)
            start = [f.a + 0.2, f.sigma * 1.2]
        r = optimize.minimize(fun, start, method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-13, maxiter=4000, maxfev=8000))
        assert f.nll <= r.fun + 1e-9
        assert f.nll == pytest.approx(r.fun, abs=1e-7)
        assert f.a == pytest.approx(r.x[0], abs=1e-4) and f.sigma == pytest.approx(r.x[-1], abs=1e-4)
        if trend:
            assert f.b == pytest.approx(r.x[1], abs=1e-4)


def test_stationaere_gumbel_anpassung_gleich_scipy_gumbel_r_fit():
    rng = np.random.default_rng(5)
    x = _reihe(rng, 200, 0.0)
    f = M.fit_gumbel(x, False)
    loc, scale = stats.gumbel_r.fit(x)
    assert f.a == pytest.approx(loc, abs=1e-4) and f.sigma == pytest.approx(scale, abs=1e-4)


def test_likelihood_quotient_ist_nicht_negativ_und_schwelle_gleich_chi2_quantil():
    rng = np.random.default_rng(6)
    for _ in range(20):
        lr, f0, f1 = M.likelihood_ratio(_reihe(rng, 30, rng.choice([0.0, 0.03])))
        assert lr >= 0 and f1.nll <= f0.nll + 1e-9
    assert N.CHI2_95_DF1 == pytest.approx(stats.chi2.ppf(0.95, 1), abs=1e-12)
    assert N.Z95 == pytest.approx(stats.norm.ppf(0.975), abs=1e-12) and N.Z95 ** 2 == pytest.approx(N.CHI2_95_DF1, abs=1e-12)


def test_fisher_information_der_gumbel_verteilung_gleich_quadratur():
    pdf = lambda z: math.exp(-z - math.exp(-z))
    s_mu = lambda z: 1 - math.exp(-z)
    s_sg = lambda z: -1 + z * (1 - math.exp(-z))
    quad = lambda f: integrate.quad(lambda z: f(z) * pdf(z), -12, 60, limit=400, epsabs=1e-12, epsrel=1e-12)[0]
    info = M.gumbel_information()
    assert quad(lambda z: s_mu(z) ** 2) == pytest.approx(info["mu_mu"], abs=1e-8)
    assert quad(lambda z: s_mu(z) * s_sg(z)) == pytest.approx(info["mu_sigma"], abs=1e-8)
    assert quad(lambda z: s_sg(z) ** 2) == pytest.approx(info["sigma_sigma"], abs=1e-8)


def test_trennschaerfe_formel_gleich_scipy_norm_und_nicht_zentralem_chi2():
    for beta, m in ((0.02, 50), (0.01, 99), (0.04, 39), (0.005, 156)):
        lam = beta * beta * m * (m * m - 1) / 12
        ref = stats.norm.cdf(math.sqrt(lam) - stats.norm.ppf(0.975)) + stats.norm.cdf(-math.sqrt(lam) - stats.norm.ppf(0.975))
        assert M.power(beta, m) == pytest.approx(ref, abs=1e-9)
        # das nicht zentrale χ²₁ mit Verschiebung λ gibt dieselbe Wahrscheinlichkeit
        assert 1 - stats.ncx2.cdf(stats.chi2.ppf(0.95, 1), 1, lam) == pytest.approx(M.power(beta, m), abs=1e-8)
