"""Literaturwert: nur die Definition des Design Life Level nach Rootzén/Katz 2013 (Water Resour. Res. 49; W/lokal gelesen): der Design Life Level zum Risiko q ist das (1−q)-Quantil der
Verteilung des Maximums über die Nutzungsdauer. Hier als Test gegen eine Aufzählung (Produkt der Jahresverteilungen) und gegen die Simulation.

Bewusst **kein** Test aus Slater u. a. 2021 oder Serinaldi/Kilsby 2015 (nicht an der Tabelle bzw. nur im Auszug geprüft) und **keiner** aus dem Beispiel 90/70 Jahre der Einleitung von Rootzén/Katz
(gehört zu einem anderen Modell)."""

import math

import numpy as np
import pytest

import tsp_models as M
import tsp_vecrng as V


@pytest.mark.parametrize("beta,m,q", [(0.02, 50, 0.05), (0.04, 30, 0.10), (0.0, 50, 0.05), (0.01, 100, 0.01)])
def test_design_life_level_ist_das_eins_minus_q_quantil_des_maximums_ueber_die_nutzungsdauer(beta, m, q):
    dll = M.design_life_level(beta, m, q)
    prod = float(np.prod([math.exp(-math.exp(-(dll - beta * t))) for t in range(m)]))        # P(Maximum ≤ DLL) = Produkt der Jahresverteilungsfunktionen
    assert prod == pytest.approx(1 - q, abs=1e-12)


def test_design_life_level_simulation_ueberschreitung_gleich_q_im_band():
    beta, m, q, nb = 0.02, 40, 0.05, 150_000
    x = (beta * np.arange(m) + V.VecStream(2013).gumbel(nb * m).reshape(nb, m)).max(axis=1)
    assert abs(float(np.mean(x > M.design_life_level(beta, m, q))) - q) < 5 * math.sqrt(q * (1 - q) / nb)


def test_ohne_trend_ist_der_design_life_level_das_stationaere_quantil_des_maximums_aus_m_jahren():
    assert M.design_life_level(0.0, 50, 0.05) == pytest.approx(math.log(50) - math.log(-math.log(0.95)), abs=1e-12)
    # stationär: F^m(x) = 1 − q ⇒ der Pegel zur Überschreitungswahrscheinlichkeit 1 − (1 − q)^{1/m} je Jahr
    p_year = 1 - (1 - 0.05) ** (1 / 50)
    assert M.level_year(0.0, 0, p_year) == pytest.approx(M.design_life_level(0.0, 50, 0.05), abs=1e-9)
