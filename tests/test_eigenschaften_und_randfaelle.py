"""Eigenschaften (je 20 bis 50 Zufallsinstanzen, fester Zufalls-Seed) und Randfälle: m = 1, 2, 3, β < 0, β = 0, sehr großes β, q und p an den Rändern, Reihe ohne Streuung, ρ ≥ 1."""

import math

import numpy as np
import pytest

import tsp_evaluation as EV
import tsp_models as M
import tsp_scenario as S
import tsp_vecrng as V


def _instanzen(n, seed=2026):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield float(rng.choice([0.0, 0.005, 0.01, 0.02, 0.04, 0.08])), int(rng.integers(5, 160)), float(rng.choice([0.01, 0.05, 0.1]))


# --- Eigenschaften der exakten Größen -----------------------------------------------------------------------------------------------------------
def test_beta_null_gibt_mischung_gleich_pegel_und_risiko_gleich_stationaer():
    for _b, m, q in _instanzen(30):
        assert M.mixture_quantile(0.0, m) == pytest.approx(M.Y100, abs=1e-9)
        assert M.risk_in_years(0.0, m) == pytest.approx(1 - 0.99 ** m, abs=1e-12)
        assert M.design_life_level(0.0, m, q) == pytest.approx(math.log(m) - math.log(-math.log1p(-q)), abs=1e-12)


def test_mischung_liegt_zwischen_erstem_und_letztem_pegel():
    for b, m, _q in _instanzen(40):
        lo, hi = M.level_year(b, 0), M.level_year(b, m - 1)
        assert lo - 1e-9 <= M.mixture_quantile(b, m) <= hi + 1e-9


def test_wiederkehrperiode_des_heutigen_pegels_faellt_in_t_und_risiko_waechst_in_m_und_beta():
    for b, m, _q in _instanzen(30):
        if b == 0:
            continue
        T = [M.return_period_after(b, t) for t in range(0, 101, 10)]
        assert T[0] == pytest.approx(100.0, abs=1e-9) and all(x > y for x, y in zip(T, T[1:]))
        assert M.risk_in_years(b, m + 1) >= M.risk_in_years(b, m) and M.risk_in_years(b * 1.5, m) >= M.risk_in_years(b, m)
        assert 1 - M.risk_in_years(b, 25) > 1 - M.risk_in_years(b, 26) > 0      # echt wachsend, solange das Risiko nicht auf 1 gerundet ist
    assert M.risk_in_years(0.02, 50, start=1) > M.risk_in_years(0.02, 50)          # die Jahre 1 bis m liegen höher als 0 bis m−1


def test_design_life_level_waechst_in_beta_und_in_eins_minus_q():
    for b, m, _q in _instanzen(30):
        assert M.design_life_level(b, m, 0.01) > M.design_life_level(b, m, 0.05) > M.design_life_level(b, m, 0.10)
        if b > 0:
            assert M.design_life_level(b, m, 0.05) > M.design_life_level(0.0, m, 0.05)
            assert M.design_life_level(2 * b, m, 0.05) > M.design_life_level(b, m, 0.05)


def test_trennschaerfe_waechst_in_m_und_im_betrag_von_beta_und_ist_symmetrisch_in_beta():
    for b, m, _q in _instanzen(40):
        if b == 0:
            assert M.power(0.0, m) == pytest.approx(0.05, abs=1e-3)           # ohne Trend: Niveau (asymptotisch)
            continue
        assert M.power(b, m + 5) >= M.power(b, m) and M.power(1.5 * b, m) >= M.power(b, m)
        assert M.power(b, 12) < M.power(b, 17) and M.power(b, 12) < M.power(1.5 * b, 12)
        assert M.power(-b, m) == pytest.approx(M.power(b, m), abs=1e-15)
        assert 0.05 <= M.power(b, m) <= 1.0


def test_rekorde_ohne_trend_im_mittel_harmonische_zahl_im_band():
    m, R = 30, 20_000
    X = V.VecStream(31).gumbel(R * m).reshape(R, m)
    rec = np.sum(np.maximum.accumulate(X, axis=1) == X, axis=1)
    se = float(rec.std(ddof=1)) / math.sqrt(R)
    assert abs(float(rec.mean()) - M.harmonic(m)) < 5 * se


# --- Eigenschaften der Anpassungen ---------------------------------------------------------------------------------------------------------------
def _reihen(n=20, seed=3):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        m = int(rng.integers(12, 70))
        yield m, 0.03 * np.arange(m) - np.log(-np.log(rng.random(m)))


def test_lage_aequivarianz_verschieben_der_reihe_verschiebt_pegel_und_lage():
    for m, x in _reihen():
        f, g = M.fit_gumbel(x, True), M.fit_gumbel(x + 7.5, True)
        assert g.a == pytest.approx(f.a + 7.5, abs=1e-6) and g.b == pytest.approx(f.b, abs=1e-8) and g.sigma == pytest.approx(f.sigma, abs=1e-8)
        assert g.level() == pytest.approx(f.level() + 7.5, abs=1e-6)


def test_skalierung_skaliert_pegel_trend_und_skala():
    for m, x in _reihen():
        f, g = M.fit_gumbel(x, True), M.fit_gumbel(3.0 * x, True)
        assert g.sigma == pytest.approx(3 * f.sigma, rel=1e-6) and g.b == pytest.approx(3 * f.b, rel=1e-6, abs=1e-9) and g.level() == pytest.approx(3 * f.level(), rel=1e-6)
        assert M.likelihood_ratio(3.0 * x)[0] == pytest.approx(M.likelihood_ratio(x)[0], abs=1e-6)       # die Statistik ist skaleninvariant


def test_umkehren_der_zeit_kehrt_das_vorzeichen_von_beta_um():
    for m, x in _reihen():
        f, g = M.fit_gumbel(x, True), M.fit_gumbel(x[::-1], True)
        assert g.b == pytest.approx(-f.b, abs=1e-7) and g.a == pytest.approx(f.a, abs=1e-7) and g.sigma == pytest.approx(f.sigma, abs=1e-7)
        assert M.likelihood_ratio(x[::-1])[0] == pytest.approx(M.likelihood_ratio(x)[0], abs=1e-7)


def test_gev_lage_aequivarianz_und_negative_log_likelihood_nur_ein_teil_der_parameter():
    for m, x in list(_reihen(8))[:8]:
        f, g = M.fit_gev(x, False), M.fit_gev(x + 4.0, False)
        assert g.level() == pytest.approx(f.level() + 4.0, abs=2e-4)
        assert g.nll == pytest.approx(f.nll, abs=1e-6)


def test_anpassung_mit_trend_hat_nie_eine_schlechtere_likelihood_als_ohne():
    for m, x in _reihen(30, seed=4):
        lr, f0, f1 = M.likelihood_ratio(x)
        assert f1.nll <= f0.nll + 1e-9 and lr >= 0


# --- Randfälle ---------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("m,trend", [(1, False), (2, True), (1, True), (0, False)])
def test_zu_kurze_reihe_gibt_meldung_statt_zahl_fuer_gumbel(m, trend):
    with pytest.raises(M.FitError, match="mindestens"):
        M.fit_gumbel(np.arange(m, dtype=float), trend)


@pytest.mark.parametrize("m,trend", [(3, False), (3, True), (4, True)])
def test_gev_braucht_mehr_jahre_als_freie_parameter(m, trend):
    with pytest.raises(M.FitError, match="mindestens"):
        M.fit_gev(np.array([0.1, 1.2, 0.4, 2.0, 0.9])[:m], trend)


def test_drei_jahre_gumbel_mit_trend_und_vier_jahre_gev_ohne_trend_rechnen():
    x = S.gumbel_series(4, 0.02, 3)
    assert M.fit_gumbel(x[:3], True).sigma > 0
    assert math.isfinite(M.fit_gev(x, False).level())


def test_reihe_ohne_streuung_und_exakte_gerade_werden_abgewiesen():
    with pytest.raises(M.FitError, match="gleich"):
        M.fit_gumbel(np.full(20, 3.0), True)
    with pytest.raises(M.FitError, match="Gerade"):
        M.fit_gumbel(0.5 * np.arange(20.0), True)


def test_analyse_mit_zu_kurzer_reihe_wirft_fit_error_mit_meldung():
    with pytest.raises(M.FitError):
        EV.analyse(EV.Settings(m=2))


def test_analyse_mit_drei_jahren_meldet_gev_nicht_moeglich_statt_zu_rechnen():
    a = EV.analyse(EV.Settings(m=3, beta=0.01))
    assert a.gev_stat is None and "mindestens" in a.gev_error and a.interval is None


def test_fallender_trend_ist_spiegelbildlich_zum_steigenden():
    for m in (10, 50):
        up, down = M.mixture_quantile(0.02, m), M.mixture_quantile(-0.02, m)
        assert M.level_year(-0.02, m - 1) < M.level_year(-0.02, 0) and down < M.Y100
        lo, hi = sorted((M.level_year(-0.02, 0), M.level_year(-0.02, m - 1)))
        assert lo <= down <= hi and up > M.Y100                                    # die Mischung liegt zwischen erstem und letztem Jahr
        assert M.risk_in_years(-0.02, m) < M.risk_in_years(0.0, m) < M.risk_in_years(0.02, m)


def test_sehr_grosses_beta_laeuft_nicht_ueber():
    assert math.isfinite(M.design_life_level(5.0, 1000, 0.05))
    assert M.return_period_after(5.0, 1000) == 1.0
    assert M.expected_records(5.0, 200) == pytest.approx(1.0 + sum(1 - math.exp(-5.0) for _ in range(199)), rel=1e-3)


def test_wahrscheinlichkeiten_an_den_raendern():
    for p in (0.0, 1.0):
        with pytest.raises(ValueError):
            M.level_year(0.02, 5, p)
    for q in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            M.design_life_level(0.02, 50, q)
    assert M.level_year(0.0, 0, 0.5) == pytest.approx(-math.log(math.log(2)), abs=1e-12)    # Median der Standard-Gumbel-Verteilung


def test_stationaere_gev_am_rand_bei_zehn_jahren_kommt_vor_aber_selten():
    """ξ̂ am Rand −1 bei m = 10 ohne Trend: Anteil der Läufe (Messung dieser Suite, R = 400, Strom der Vorlage); die Zahl in der README ist ein Band, kein Fixwert."""
    X = V.VecStream(55).gumbel(400 * 10).reshape(400, 10)
    edge = sum(M.fit_gev(x, False).edge for x in X)
    assert 0 <= edge / 400 < 0.25


def test_zufalls_seeds_am_rand_der_regler_rechnen():
    for seed in (0, 999999):
        a = EV.analyse(EV.Settings(seed=seed))
        assert math.isfinite(a.lr) and a.records >= 1
