"""Zufall und Szenario: die Vorlage SplitMix64, der vektorisierte Strom (bitgleich), abgeleitete Zufalls-Seeds, Gumbel-Ziehung, Reihen und Blöcke; festgehaltene Werte (plattformfest)."""

import math

import numpy as np
import pytest

import tsp_rng as R
import tsp_scenario as S
import tsp_vecrng as V


def test_vektorisierter_strom_gibt_dieselben_gleichverteilten_wie_die_vorlage():
    for seed in (0, 1, 35, 2 ** 63 + 12345, 2 ** 64 - 1):
        rng = R.SplitMix64(seed)
        ref = np.array([rng.random() for _ in range(3000)])
        vs = V.VecStream(seed)
        assert np.array_equal(vs.uniform(1000), ref[:1000])
        assert np.array_equal(vs.uniform(2000), ref[1000:])          # der Strom läuft über Aufrufe hinweg weiter


def test_vektorisierte_gumbel_werte_gleich_der_vorlage_bis_auf_numpy_log():
    vs = V.VecStream(777)
    g = vs.gumbel(300)
    rr = R.SplitMix64(777)
    ref = np.array([S.gumbel(rr) for _ in range(300)])
    assert np.allclose(g, ref, rtol=0, atol=1e-12)


def test_gumbel_aus_vorlage_hat_mittel_und_streuung_der_gumbel_verteilung():
    rr = R.SplitMix64(4)
    x = np.array([S.gumbel(rr) for _ in range(6000)])
    assert abs(x.mean() - 0.5772156649) < 5 * (math.pi / math.sqrt(6)) / math.sqrt(x.size)
    assert abs(x.std(ddof=1) - math.pi / math.sqrt(6)) < 0.05
    xs = np.sort(x)
    ks = float(np.max(np.abs(np.arange(1, xs.size + 1) / xs.size - np.exp(-np.exp(-xs)))))
    assert ks < 1.95 / math.sqrt(xs.size)                         # Kolmogorov-Smirnov zum Niveau 0,001 (1,95/√n)


def test_gumbel_zieht_neu_bei_exponentialwert_null():
    class Fake:
        def __init__(self):
            self.vals = [0.0, 1.0]

        def expovariate(self, _mean):
            return self.vals.pop(0)
    assert S.gumbel(Fake()) == 0.0         # −ln(1) = 0, der Wert 0 wurde verworfen


def test_abgeleitete_zufalls_seeds_sind_deterministisch_und_verschieden():
    assert S.derive_seed(27, 1) == S.derive_seed(27, 1)
    assert S.derive_seed(27, 1) == 16616101746815609346 and S.derive_seed(27, 2) == 7316528170013307371
    seeds = {S.derive_seed(27, t) for t in range(2000)} | {S.derive_seed(28, t) for t in range(2000)}
    assert len(seeds) == 4000
    assert S.derive_seed(27, 1, 2) != S.derive_seed(27, 2, 1)


def test_reihe_ist_festgehalten_und_wiederholbar():
    ref = [0.11190314813881412, -0.08794801658604302, 2.3052188537351754, -0.008705810820142518, 0.4970880150980405]
    x = S.gumbel_series(5, 0.02, 27)
    assert np.allclose(x, ref, rtol=0, atol=1e-12)
    assert np.array_equal(S.gumbel_series(50, 0.02, 27)[:5], x)           # gleiche Zahlen, längere Reihe: gleicher Anfang
    assert np.array_equal(S.gumbel_series(50, 0.02, 27), S.gumbel_series(50, 0.02, 27))
    assert not np.array_equal(S.gumbel_series(50, 0.02, 28), S.gumbel_series(50, 0.02, 27))


def test_trend_verschiebt_dasselbe_rauschen_um_beta_mal_t():
    a, b = S.gumbel_series(30, 0.0, 9), S.gumbel_series(30, 0.04, 9)
    assert np.allclose(b - a, 0.04 * np.arange(30), atol=1e-12)


def test_bloecke_sind_die_maxima_von_jahr_fuer_jahr_gezogenen_nutzungsdauern():
    mx = S.block_maxima(3, 10, 0.02, 5)
    rng = R.SplitMix64(5)
    ref = [max(0.02 * t + S.gumbel(rng) for t in range(10)) for _ in range(3)]
    assert np.allclose(mx, ref, rtol=0, atol=1e-12)
    assert mx.shape == (3,)


def test_vorlage_ist_exakt_und_plattformfest_exact_log_gleich_decimal():
    assert R.exact_log(1.0) == 0.0
    assert R.exact_log(math.e) == pytest.approx(1.0, abs=1e-15)
    assert R.exact_log(0.5) == pytest.approx(-math.log(2), abs=1e-15)
