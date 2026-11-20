"""Orakel: Mischungsquantil, Risiko, Design Life Level, Wiederkehrperiode und Rekorde (exakte Formeln) gegen andere Rechenwege:
scipy-`brentq` auf der Mischungsverteilungsfunktion, das Produkt der Jahresverteilungen, die empirische Verteilung von 10⁶ gezogenen Maxima (SplitMix64-Strom der Vorlage), Quadratur."""

import math

import numpy as np
import pytest

import tsp_models as M
import tsp_vecrng as V

optimize = pytest.importorskip("scipy.optimize")
stats = pytest.importorskip("scipy.stats")

ZELLEN = [(0.0, 50), (0.01, 30), (0.02, 50), (0.04, 100), (0.02, 3), (-0.02, 50)]


def _fmix(x, beta, m):
    return float(np.mean([stats.gumbel_r.cdf(x, loc=beta * t) for t in range(m)]))


@pytest.mark.parametrize("beta,m", ZELLEN)
def test_mischungsquantil_gleich_brentq_von_scipy(beta, m):
    ref = optimize.brentq(lambda x: _fmix(x, beta, m) - 0.99, -10, 80, xtol=1e-13)
    assert M.mixture_quantile(beta, m) == pytest.approx(ref, abs=1e-9)


@pytest.mark.parametrize("beta,m", ZELLEN)
def test_mischungsquantil_liegt_zwischen_erstem_und_letztem_pegel_und_beta_null_ist_stationaer(beta, m):
    q = M.mixture_quantile(beta, m)
    lo, hi = sorted((M.level_year(beta, 0), M.level_year(beta, m - 1)))
    assert lo - 1e-9 <= q <= hi + 1e-9
    if beta == 0:
        assert q == pytest.approx(M.Y100, abs=1e-10)


def test_mischungsquantil_gleich_empirischem_quantil_von_einer_million_gezogenen_maxima():
    beta, m, n = 0.02, 50, 1_000_000
    vs = V.VecStream(2026)
    t = vs.uniform(n)
    years = np.minimum((t * m).astype(np.int64), m - 1)
    x = beta * years + vs.gumbel(n)
    q_emp = float(np.quantile(x, 0.99))
    # in der Wahrscheinlichkeit gemessen: F_mix(q_emp) liegt nahe 0,99 (Standardfehler sqrt(0,99·0,01/n) = 1e-4)
    assert abs(_fmix(q_emp, beta, m) - 0.99) < 5 * math.sqrt(0.99 * 0.01 / n)
    assert q_emp == pytest.approx(M.mixture_quantile(beta, m), abs=0.05)


@pytest.mark.parametrize("beta,m", [(0.01, 20), (0.02, 50), (0.04, 100), (0.0, 50)])
def test_risiko_gleich_eins_minus_produkt_der_jahresverteilungen(beta, m):
    prod = float(np.prod([stats.gumbel_r.cdf(M.Y100, loc=beta * t) for t in range(m)]))      # Pegel von heute = Pegel des Jahres 0
    assert M.risk_in_years(beta, m) == pytest.approx(1 - prod, abs=1e-12)
    prod1 = float(np.prod([stats.gumbel_r.cdf(M.Y100, loc=beta * t) for t in range(1, m + 1)]))
    assert M.risk_in_years(beta, m, start=1) == pytest.approx(1 - prod1, abs=1e-12)


@pytest.mark.parametrize("beta,m,q", [(0.02, 50, 0.05), (0.04, 100, 0.01), (0.01, 25, 0.10), (0.0, 50, 0.05)])
def test_design_life_level_gleich_brentq_auf_dem_produkt_der_jahresverteilungen(beta, m, q):
    prod = lambda x: float(np.prod([stats.gumbel_r.cdf(x, loc=beta * t) for t in range(m)]))
    ref = optimize.brentq(lambda x: prod(x) - (1 - q), -5, 80, xtol=1e-13)
    assert M.design_life_level(beta, m, q) == pytest.approx(ref, abs=1e-9)
    assert prod(M.design_life_level(beta, m, q)) == pytest.approx(1 - q, abs=1e-12)


@pytest.mark.parametrize("beta,t", [(0.02, 0), (0.02, 25), (0.02, 100), (0.04, 50), (-0.01, 30)])
def test_wiederkehrperiode_des_heutigen_pegels_gleich_eins_durch_ueberschreitungswahrscheinlichkeit(beta, t):
    p_exc = float(stats.gumbel_r.sf(M.Y100, loc=beta * t))
    assert M.return_period_after(beta, t) == pytest.approx(1 / p_exc, rel=1e-9)


@pytest.mark.parametrize("beta,t", [(0.02, 0), (0.02, 50), (0.04, 25)])
def test_mittlere_wartezeit_bei_weiter_steigendem_trend_ist_kuerzer_als_der_kehrwert(beta, t):
    """E[W] = Σ_k P(W > k) mit P(W > k) = Π_{j<k} (1 − p_{t+j}) (Produkt der Jahresverteilungen aus scipy), Jahr t mitgezählt: kürzer als 1/p_t."""
    from claims_fragmente import mittlere_wartezeit
    ew, surv = 0.0, 1.0
    for j in range(2000):
        ew += surv
        surv *= float(stats.gumbel_r.cdf(M.Y100, loc=beta * (t + j)))
        if surv < 1e-16:
            break
    assert mittlere_wartezeit(beta, t) == pytest.approx(ew, rel=1e-9)
    assert ew < M.return_period_after(beta, t)


def test_mittlere_wartezeit_gegen_simulation_im_band():
    beta, t0, R, K = 0.02, 50, 40_000, 200
    x = (beta * (t0 + np.arange(K)) + V.VecStream(79).gumbel(R * K).reshape(R, K)) > M.Y100
    assert np.all(x.any(axis=1))
    w = x.argmax(axis=1) + 1.0
    from claims_fragmente import mittlere_wartezeit
    assert abs(float(w.mean()) - mittlere_wartezeit(beta, t0)) < 5 * float(w.std(ddof=1)) / math.sqrt(R)
    assert float(w.mean()) < M.return_period_after(beta, t0) - 5.0


def test_risiko_und_design_life_level_gegen_simulation_von_blocken_im_band():
    beta, m, nb = 0.02, 25, 200_000
    x = (beta * np.arange(m) + V.VecStream(77).gumbel(nb * m).reshape(nb, m)).max(axis=1)
    r = M.risk_in_years(beta, m)
    assert abs(float(np.mean(x > M.Y100)) - r) < 5 * math.sqrt(r * (1 - r) / nb)
    dll = M.design_life_level(beta, m, 0.05)
    assert abs(float(np.mean(x > dll)) - 0.05) < 5 * math.sqrt(0.05 * 0.95 / nb)


def test_rekordwahrscheinlichkeit_jedes_jahres_gleich_simulation():
    beta, m, R = 0.05, 12, 200_000
    X = beta * np.arange(m) + V.VecStream(78).gumbel(R * m).reshape(R, m)
    is_rec = X == np.maximum.accumulate(X, axis=1)
    lam = np.exp(beta * np.arange(m))
    exact = lam / np.cumsum(lam)
    se = np.sqrt(exact * (1 - exact) / R)
    assert np.all(np.abs(is_rec.mean(axis=0) - exact) < 5 * se + 1e-12)
    assert M.expected_records(beta, m) == pytest.approx(float(exact.sum()), abs=1e-12)
    assert M.expected_records(0.0, 7) == pytest.approx(M.harmonic(7), abs=1e-12)


def test_log_summe_laeuft_bei_grossem_beta_nicht_ueber():
    assert math.isfinite(M.log_sum_exp_years(5.0, 1000))
    assert M.risk_in_years(5.0, 1000) == pytest.approx(1.0, abs=1e-12)
    assert M.return_period_after(5.0, 1000) == pytest.approx(1.0, abs=1e-12)
