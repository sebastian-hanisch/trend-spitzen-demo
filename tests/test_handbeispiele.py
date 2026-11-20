"""Handbeispiele des Plans (H 1 bis H 6) als exakte Tests: Brüche und Ganzzahlen von Hand, Gegenproben per Aufzählung, Quadratur und Simulation.

Spielzeugtrend e^β = 2 (β = ln 2 ≈ 0,69 σ je Jahr, absichtlich groß): E_t = e^{−X_t} ist exponentialverteilt mit Rate λ_t = 2^t, damit werden Ordnungs- und Rekordwahrscheinlichkeiten Brüche.
Die App-Funktionen (`tsp_models`, `tsp_vehicle`) werden gegen diese Handrechnung gestellt, nicht gegen sich selbst."""

import itertools
import math
from fractions import Fraction as Fr

import numpy as np
import pytest

import tsp_models as M
import tsp_numeric as N
import tsp_vecrng as V
import tsp_vehicle as VH

LN2 = math.log(2.0)


def plackett_luce_increasing(lams):
    """P(X_0 < X_1 < … < X_{m−1}) für X_t = −ln E_t, E_t ~ Exp(λ_t): Π_{k≥1} λ_k / (λ_0 + … + λ_k)."""
    p = Fr(1)
    for k in range(1, len(lams)):
        p *= Fr(lams[k]) / sum(Fr(v) for v in lams[: k + 1])
    return p


def plackett_luce_order(lams, order):
    p = Fr(1)
    rest = list(order)
    for _ in range(len(order) - 1):
        p *= Fr(lams[rest[0]]) / sum(Fr(lams[i]) for i in rest)
        rest = rest[1:]
    return p


def n_records(seq):
    best, c = None, 0
    for v in seq:
        if best is None or v > best:
            c += 1
            best = v
    return c


# --- H 1: vier Jahresmaxima, alle steigend ----------------------------------------------------------------------------------------------------
def test_h1_ordnung_alle_vier_steigend_mit_und_ohne_trend(spielzeug):
    assert plackett_luce_increasing(spielzeug) == Fr(64, 315) == Fr(2, 3) * Fr(4, 7) * Fr(8, 15)
    assert plackett_luce_increasing([1, 1, 1, 1]) == Fr(1, 24)
    assert float(Fr(64, 315)) == pytest.approx(0.2032, abs=5e-5)


def test_h1_alle_24_ordnungen_summieren_sich_zu_eins_und_eine_ist_steigend(spielzeug):
    assert sum(plackett_luce_order(spielzeug, o) for o in itertools.permutations(range(4))) == 1
    assert sum(1 for o in itertools.permutations(range(4)) if list(o) == sorted(o)) == 1


def test_h1_drei_jahre_achtel_einundzwanzigstel_mit_quadratur():
    assert plackett_luce_increasing([1, 2, 4]) == Fr(8, 21)
    integrate = pytest.importorskip("scipy.integrate")
    f3 = lambda e2, e1, e0: 1 * np.exp(-1 * e0) * 2 * np.exp(-2 * e1) * 4 * np.exp(-4 * e2)
    val, _ = integrate.tplquad(f3, 0, np.inf, lambda e0: 0, lambda e0: e0, lambda e0, e1: 0, lambda e0, e1: e1, epsabs=1e-10, epsrel=1e-10)
    assert val == pytest.approx(8 / 21, abs=1e-7)


def test_h1_simulation_mit_dem_strom_der_vorlage_im_band_von_fuenf_standardfehlern():
    R = 400_000
    vs = V.VecStream(8)
    X = np.arange(4) * LN2 + vs.gumbel(R * 4).reshape(R, 4)
    inc = float(np.mean(np.all(np.diff(X, axis=1) > 0, axis=1)))
    p = 64 / 315
    assert abs(inc - p) < 5 * math.sqrt(p * (1 - p) / R)
    X0 = V.VecStream(9).gumbel(R * 4).reshape(R, 4)
    inc0 = float(np.mean(np.all(np.diff(X0, axis=1) > 0, axis=1)))
    assert abs(inc0 - 1 / 24) < 5 * math.sqrt(1 / 24 * 23 / 24 / R)


# --- H 2: Rekorde ---------------------------------------------------------------------------------------------------------------------------
def test_h2_erwartete_rekorde_ohne_trend_sind_die_harmonische_zahl():
    mean_rec = Fr(sum(n_records(o) for o in itertools.permutations(range(4))), 24)
    assert mean_rec == sum(Fr(1, k) for k in range(1, 5)) == Fr(25, 12)
    assert M.harmonic(4) == pytest.approx(25 / 12, abs=1e-15)
    assert M.expected_records(0.0, 4) == pytest.approx(25 / 12, abs=1e-14)


def test_h2_erwartete_rekorde_mit_spielzeugtrend_sind_97_35(spielzeug):
    p_rec = [Fr(spielzeug[k]) / sum(Fr(v) for v in spielzeug[: k + 1]) for k in range(4)]
    assert p_rec == [Fr(1), Fr(2, 3), Fr(4, 7), Fr(8, 15)]
    assert sum(p_rec) == Fr(97, 35)
    assert M.expected_records(LN2, 4) == pytest.approx(97 / 35, abs=1e-12)
    assert M.expected_records(LN2, 4) > M.harmonic(4)


def test_h2_simulation_der_rekorde_stimmt_mit_97_35():
    R = 200_000
    X = np.arange(4) * LN2 + V.VecStream(10).gumbel(R * 4).reshape(R, 4)
    rec = float(np.mean(np.sum(np.maximum.accumulate(X, axis=1) == X, axis=1)))
    assert abs(rec - 97 / 35) < 0.01


def test_count_records_zaehlt_neue_hoechstwerte():
    assert M.count_records([1, 3, 2, 5, 5, 4]) == 3
    assert M.count_records([]) == 0 and M.count_records([2.0]) == 1


# --- H 3: stationär heißt Mittel --------------------------------------------------------------------------------------------------------------
def test_h3_mischung_zweier_jahre_loest_eine_quadratische_gleichung():
    v = (-1 + math.sqrt(1 + 4 * 1.98)) / 2          # v² + v − 1,98 = 0 mit v = exp(−e^{−x})
    x_mix = -math.log(-math.log(v))
    assert x_mix == pytest.approx(5.0051, abs=5e-4)
    assert M.mixture_quantile(LN2, 2) == pytest.approx(x_mix, abs=1e-9)
    fmix = lambda x: 0.5 * (math.exp(-math.exp(-x)) + math.exp(-2 * math.exp(-x)))
    assert fmix(x_mix) == pytest.approx(0.99, abs=1e-12)
    first, last = M.level_year(LN2, 0), M.level_year(LN2, 1)
    assert first == pytest.approx(4.600149, abs=1e-6) and last == pytest.approx(5.293296, abs=1e-6)
    assert first < x_mix < last
    assert last - x_mix == pytest.approx(0.2882, abs=5e-4)


def test_h3_unter_der_mischung_kehrt_der_heutige_pegel_nur_alle_133_jahre_wieder():
    last = M.level_year(LN2, 1)
    p_mix = 0.5 * ((1 - math.exp(-math.exp(-last))) + 0.01)     # Mittel der Jahresraten 1/200 und 1/100
    assert 1 / p_mix == pytest.approx(133.0, abs=1.0)
    assert M.return_period_after(LN2, 1) == pytest.approx(1 / (1 - 0.99 ** 2), abs=1e-9)


# --- H 4: Risiko und Design Life Level --------------------------------------------------------------------------------------------------------
def test_h4_risiko_in_vier_jahren_ist_1_minus_099_hoch_15():
    assert M.risk_in_years(LN2, 4) == pytest.approx(float(1 - Fr(99, 100) ** 15), abs=1e-13)
    assert float(1 - Fr(99, 100) ** 15) == pytest.approx(0.1399, abs=5e-5)
    assert float(1 - Fr(99, 100) ** 4) == pytest.approx(0.0394, abs=5e-5)
    assert M.risk_in_years(0.0, 4) == pytest.approx(float(1 - Fr(99, 100) ** 4), abs=1e-13)
    assert sum(2 ** t for t in range(4)) == 15


def test_h4_risiko_per_simulation_im_band():
    R = 400_000
    xm = (np.arange(4) * LN2 + V.VecStream(81).gumbel(R * 4).reshape(R, 4)).max(axis=1)
    risk = 1 - 0.99 ** 15
    assert abs(float(np.mean(xm > M.Y100)) - risk) < 5 * math.sqrt(risk * (1 - risk) / R)


def test_h4_design_life_level_ist_ln15_minus_ln_von_minus_ln_095():
    dll = math.log(15) - math.log(-math.log(0.95))
    assert dll == pytest.approx(5.678, abs=5e-4)
    assert M.design_life_level(LN2, 4, 0.05) == pytest.approx(dll, abs=1e-12)
    assert M.design_life_level(LN2, 4, 0.05) - M.design_life_level(0.0, 4, 0.05) == pytest.approx(math.log(15 / 4), abs=1e-12)
    prod_cdf = lambda x: float(np.prod([math.exp(-math.exp(-(x - t * LN2))) for t in range(4)]))
    assert prod_cdf(dll) == pytest.approx(0.95, abs=1e-12)
    root = N.brent(lambda x: prod_cdf(x) - 0.95, 0, 30, xtol=1e-13)
    assert root == pytest.approx(dll, abs=1e-9)


def test_h4_wiederkehrperioden_100_50_25_13():
    T = [M.return_period_after(LN2, t) for t in range(4)]
    assert T[0] == pytest.approx(100, abs=1e-9)
    assert [round(v, 1) for v in T] == [100.0, 50.3, 25.4, 12.9]
    assert T[1] == pytest.approx(1 / (1 - 0.99 ** 2), abs=1e-9)


# --- H 5: Trennschärfe ------------------------------------------------------------------------------------------------------------------------
def test_h5_summe_der_zentrierten_quadrate_ist_m_mal_m2_minus_1_durch_12():
    tc = [Fr(t) - Fr(3, 2) for t in range(4)]
    assert sum(v * v for v in tc) == Fr(4 * 15, 12) == 5 and sum(tc) == 0
    assert M.sum_centered_squares(4) == 5.0
    assert M.sum_centered_squares(50) == pytest.approx(50 * 2499 / 12, abs=1e-9)


def test_h5_lambda_und_trennschaerfe_bei_beta_002_m_50():
    lam = Fr(2, 100) ** 2 * Fr(50 * 2499, 12)
    assert lam == Fr(833, 200) and float(lam) == pytest.approx(4.165, abs=1e-12)
    assert M.noncentrality(0.02, 50) == pytest.approx(4.165, abs=1e-9)
    assert M.power(0.02, 50) == pytest.approx(0.532, abs=1e-3)
    # die Zelle D = 1,0 σ (β = 1/49) ist eine andere Zelle: 0,549, nicht 0,532
    assert M.power(1 / 49, 50) == pytest.approx(0.549, abs=1e-3) and M.power(1 / 49, 50) - M.power(0.02, 50) > 0.015


@pytest.mark.parametrize("D,m,gemessen,R", [(0.49, 50, 0.194, 500), (0.98, 50, 0.514, 500), (1.96, 50, 0.984, 500), (0.99, 100, 0.828, 500), (1.16, 30, 0.474, 500), (0.76, 20, 0.184, 500)])
def test_h5_formel_gegen_die_messungen_der_vormessung_im_band(D, m, gemessen, R):
    pf = M.power(D / (m - 1), m)
    assert abs(pf - gemessen) < 3 * math.sqrt(pf * (1 - pf) / R) + 0.03


def test_h5_jahre_bis_80_prozent():
    assert M.years_for_power(0.01) == 99
    assert M.power(0.01, 98) < 0.8 <= M.power(0.01, 99)
    assert [M.years_for_power(b) for b in (0.005, 0.01, 0.02, 0.04)] == [156, 99, 62, 39]
    assert M.years_for_power(0.0) is None


# --- H 6: Vehikel A ----------------------------------------------------------------------------------------------------------------------------
def test_h6_skalen_sind_brueche_und_verdreifachen_sich():
    rhos = [Fr(7, 10), Fr(3, 4), Fr(4, 5), Fr(17, 20), Fr(9, 10)]
    scale = [1 / (1 - r) for r in rhos]
    assert scale == [Fr(10, 3), Fr(4), Fr(5), Fr(20, 3), Fr(10)] and scale[-1] / scale[0] == 3
    for r, s in zip(rhos, scale):
        assert VH.gumbel_approximation(float(r))[1] == pytest.approx(float(s), abs=1e-12)


def test_h6_lagen_und_100_tage_pegel_der_rechnung():
    rhos = (0.70, 0.75, 0.80, 0.85, 0.90)
    mu = [VH.gumbel_approximation(r)[0] for r in rhos]
    assert all(abs(a - b) < 0.006 for a, b in zip(mu, [9.27, 9.94, 10.52, 10.59, 8.35]))
    lev = [VH.approximate_level(r) for r in rhos]
    assert all(abs(a - b) < 0.1 for a, b in zip(lev, [24.6, 28.3, 33.5, 41.3, 54.4]))
    assert max(mu) - min(mu) < 2.5 and lev[-1] / lev[0] > 2.0


def test_h6_randverteilung_der_wartezeit_ist_rho_mal_exponential():
    for rho in (0.7, 0.9):
        assert float(VH.tail_probability(rho, 0.0)) == pytest.approx(rho, abs=1e-15)
        assert float(VH.tail_probability(rho, 5.0)) == pytest.approx(rho * math.exp(-(1 - rho) * 5), abs=1e-15)
