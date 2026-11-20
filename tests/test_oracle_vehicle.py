"""Orakel Vehikel A (M/M/1-Wartezeit, Lindley-Rekursion) und Vehikel B (Gewichtstabelle, Wochenstruktur).

Vehikel A: (1) die Randverteilung jedes Schritts ist im Gleichgewicht exakt ρ·e^{−(1−ρ)x}; (2) eine zweite Umsetzung der Rekursion mit Schleifen über dieselben Gleichverteilten gibt dieselben Tagesmaxima;
(3) die Gumbel-Näherung der Rechnung hat den dokumentierten Wert. Vehikel B: eigene Aufzählung aus der Definition (`vehikel_b_def.py`) gegen die kopierte Tabelle und gegen die eingefrorenen Wochenpegel."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

import tsp_models as M
import tsp_reference as REF
import tsp_vecrng as V
import tsp_vehicle as VH
import tsp_vehicle_b as VB

import vehikel_b_def as D

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("rho", [0.5, 0.7, 0.9])
def test_ein_kunde_pro_tag_hat_die_exakte_gleichgewichtsverteilung(rho):
    days = 300_000
    mx = VH.lindley_days(np.full(days, rho), 1, V.VecStream(11).uniform)
    for x in (0.0, 1.0, 5.0, 15.0):
        exact = rho * math.exp(-(1 - rho) * x)
        assert abs(float(np.mean(mx > x)) - exact) < 5 * math.sqrt(exact * (1 - exact) / days) + 1e-12


def _lindley_schleife(rho, n, days, seed):
    """Zweite Umsetzung: je Tag eine Schleife über die Kunden, dieselben Gleichverteilten in derselben Reihenfolge wie `lindley_days`."""
    vs = V.VecStream(seed)
    out = np.empty(days)
    u0 = vs.uniform(2 * days)
    w = np.array([(-math.log1p(-u0[days + d]) / (1 - rho)) if u0[d] < rho else 0.0 for d in range(days)])
    mx = w.copy()
    for _ in range(n - 1):
        u = vs.uniform(2 * days)
        for d in range(days):
            s = -math.log1p(-u[d])
            a = -math.log1p(-u[days + d]) / rho
            w[d] = max(0.0, w[d] + s - a)
            mx[d] = max(mx[d], w[d])
    out[:] = mx
    return out


@pytest.mark.parametrize("rho,n,days", [(0.7, 5, 40), (0.9, 30, 25), (0.5, 2, 60)])
def test_lindley_rekursion_gleich_schleifenfassung_mit_denselben_zufallszahlen(rho, n, days):
    a = VH.lindley_days(np.full(days, rho), n, V.VecStream(5).uniform)
    b = _lindley_schleife(rho, n, days, 5)
    assert np.allclose(a, b, rtol=0, atol=1e-12)


def test_lindley_mit_verschiedener_last_je_tag_nutzt_die_last_des_tages():
    rho = np.array([0.5] * 2000 + [0.9] * 2000)
    mx = VH.lindley_days(rho, 8, V.VecStream(6).uniform)
    assert mx[2000:].mean() > 2 * mx[:2000].mean()


def test_tagesmaximum_stationaer_der_letzte_schritt_hat_die_exakte_randverteilung():
    """Die Wartezeit jedes Kunden ist im Gleichgewicht ρ·e^{−(1−ρ)x}; das Maximum über n Kunden liegt über dem einzelnen Wert: P(max > x) ≥ ρ e^{−(1−ρ)x}."""
    rho, days = 0.8, 200_000
    mx = VH.lindley_days(np.full(days, rho), 20, V.VecStream(12).uniform)
    for x in (2.0, 6.0, 10.0):
        exact = rho * math.exp(-(1 - rho) * x)
        assert float(np.mean(mx > x)) >= exact - 5 * math.sqrt(exact * (1 - exact) / days)


def test_gumbel_naeherung_loest_die_gleichung_exp_minus_n_theta_rho_e_hoch_minus_gx():
    for rho in (0.7, 0.9):
        g = 1 - rho
        for p in (0.5, 0.01):
            x = VH.approximate_level(rho, p)
            assert math.exp(-256 * g * g * rho * math.exp(-g * x)) == pytest.approx(1 - p, abs=1e-12)


def test_ungueltige_last_wird_abgewiesen():
    for rho in (0.0, 1.0, 1.2, -0.1):
        with pytest.raises(ValueError):
            VH.gumbel_approximation(rho)
        with pytest.raises(ValueError):
            VH.tail_probability(rho, 1.0)
    with pytest.raises(ValueError):
        VH.lindley_days(np.array([0.5, 1.0]), 3, V.VecStream(1).uniform)


# --- Vehikel B ---------------------------------------------------------------------------------------------------------------------------------
def test_tabelle_gleich_eigener_aufzaehlung_aus_der_definition():
    schritte, mit_gewicht, summe = D.aufzaehlung()
    assert (schritte, mit_gewicht, len(summe)) == (4704, 2376, 1224)
    tab = {(round(mu, 10), s): w for w, mu, s in VB.components()}
    assert set(tab) == set(summe)
    assert max(abs(tab[k] - summe[k]) for k in tab) < 1e-12
    assert sum(w for w, _mu, _s in VB.components()) == pytest.approx(1.0, abs=1e-12)


def test_pegel_der_ganzen_woche_ist_287_158():
    assert VB.level(1e-5) == pytest.approx(287.158, abs=5e-4)
    assert VB.survival(VB.level(1e-5)) == pytest.approx(1e-5, rel=1e-9)


def _wochen_komponenten(we):
    """Eigene Aufzählung mit festem Wochentag-Typ (we = 0: Werktage, 1: Wochenende), Gewichte auf 1 normiert (Schleifen unabhängig vom Erzeuger)."""
    xg, wg = np.polynomial.legendre.leggauss(48)
    comps = []
    for art in range(4):
        for kunde in range(12):
            for zoll, pz in (((0, 0.75), (1, 0.25)) if art == 0 else ((0, 1.0),)):
                knoten = [(1.0 + 9.0 * (x + 1) / 2, w / 2) for x, w in zip(xg, wg)] if art == 1 else \
                         [(0.5 + 5.5 * (x + 1) / 2, w / 2) for x, w in zip(xg, wg)] if art == 2 else [(0.0, 1.0)]
                for vor, wv in knoten:
                    comps.append((D.ART_P[art] / 12 * pz * wv, math.log(D._median(art, zoll, we, vor)) + D.UK[kunde] * D.U_SCALE[art], D.S_BASE[art]))
    return comps


def test_wochenpegel_gleich_eigener_aufzaehlung_und_gemisch_gleich_ganze_woche():
    wd, we = _wochen_komponenten(0), _wochen_komponenten(1)
    lv_wd, lv_we = VB.level(1e-5, wd), VB.level(1e-5, we)
    w = REF.weekly()
    assert w["werktage"] == pytest.approx(lv_wd, rel=1e-6) and w["wochenende"] == pytest.approx(lv_we, rel=1e-6)
    assert w["faktor"] == pytest.approx(lv_we / lv_wd, rel=1e-6) and w["faktor"] == pytest.approx(1.35, abs=0.005)
    x = VB.level(1e-5)
    assert 5 / 7 * VB.survival(x, wd) + 2 / 7 * VB.survival(x, we) == pytest.approx(1e-5, rel=1e-6)


def test_gewichtstabelle_hat_quelle_und_stand_je_feld():
    q = json.loads((ROOT / "tsp_mixture_quelle.json").read_text(encoding="utf-8"))
    assert q["stand"] and q["quelle"] and q["sha256"]
    for feld in ("beschreibung", "n_komponenten", "summe_gewichte", "komponenten"):
        assert q["felder"][feld]["quelle"] and q["felder"][feld]["stand"]
    assert M.Y100 > 0
